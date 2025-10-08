# /home/artemis/project/graph_eqa_swagat/spatial_experiment/planners/vlm_planner_gemini_spatial_v2.py

import json
import os
import time
import base64
import mimetypes
import re
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import google.generativeai as genai

from graph_eqa.utils.data_utils import get_latest_image
from graph_eqa.envs.utils import pos_normal_to_habitat

# ---------- Gemini API ----------
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    raise RuntimeError("GOOGLE_API_KEY environment variable not set!")


def _b64encode_image(p: str) -> str:
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _safe_latest_image(out_dir: Path) -> Optional[str]:
    try:
        p = get_latest_image(out_dir)
        p = None if p is None else str(Path(p))
        return p if p and Path(p).exists() else None
    except Exception:
        return None


def _safe_float(x, d=0.0):
    try:
        return float(x)
    except Exception:
        return d


# ---------- Parse distance + relation from question ----------
_REL_MAP = {
    "in front of": "front",
    "front of": "front",
    "behind": "behind",
    "to the left of": "left",
    "left of": "left",
    "to the right of": "right",
    "right of": "right",
}
def _parse_q(q: str) -> Tuple[Optional[float], Optional[str]]:
    ql = q.lower()
    m = re.search(r"(\d+(\.\d+)?)\s*meters?", ql)
    dist = float(m.group(1)) if m else None
    rel = None
    for k, v in _REL_MAP.items():
        if k in ql:
            rel = v
            break
    return dist, rel


# ---------- Direction bucket in global HABITAT frame (Y-up; -Z forward; +X right) ----------
def _dir_bucket(ref_hab: np.ndarray, obj_hab: np.ndarray) -> str:
    d = obj_hab - ref_hab
    dx, _, dz = float(d[0]), float(d[1]), float(d[2])
    if abs(dx) >= abs(dz):
        return "right" if dx > 0 else "left"
    return "front" if dz < 0 else "behind"


def _fmt_row(cid, name, inst_name, dist, db, dx, dy, dz):
    return f"{cid}\t{inst_name}\t{name}\t{dist:.3f}\t{db}\t{dx:.3f}\t{dy:.3f}\t{dz:.3f}"


def _format_table(rows: List[str]) -> str:
    header = "graph_id\tdeclared_name\tclass\tdist_m\tdir_bucket\tdx\tdy\tdz"
    return "\n".join([header] + rows) if rows else "(no candidates yet)"


class VLMPlannerEQAGeminiSpatialV2:
    """
    v2 planner (graph-first):
      • Builds a per-step candidate table from the *scene graph* (and dataset candidates if provided).
      • Distances & direction buckets are computed relative to the reference in HABITAT frame.
      • Declared target must be an instance-like string (e.g., 'step_306'), drawn from the allowed list.
      • Navigation uses object/frontier node ids (same as v1).
      • Full logs -> llm_outputs_v2.txt
    """

    def __init__(
        self,
        cfg,
        sg_sim,
        question: str,
        ground_truth_target: Dict[str, Any],
        output_path: Path,
        reference_object: Optional[Dict[str, Any]] = None,
        reference_room: Optional[Dict[str, Any]] = None,
        candidate_targets: Optional[List[Dict[str, Any]]] = None,   # optional dataset candidates
    ):
        self._question = str(question)
        self._ground_truth_target = dict(ground_truth_target or {})
        self._output_path = Path(output_path)
        self._use_image = bool(getattr(cfg, "use_image", True))
        self._add_history = bool(getattr(cfg, "add_history", False))
        self._history = ""
        self._t = 0
        self.sg_sim = sg_sim

        self.reference_object = dict(reference_object or {})
        self.ref_name = str(self.reference_object.get("name", "object")).lower()
        # reference position is provided in HABITAT coords by the dataset
        self.ref_pos_hab = np.asarray(self.reference_object.get("position", [0, 0, 0]), np.float32)

        self.dataset_candidates = list(candidate_targets or [])

        self._outputs_to_save = [
            f'Question: {self._question}\nGround Truth: {self._ground_truth_target.get("name","?")} '
            f'({self._ground_truth_target.get("id","?")})\n'
        ]

        # Neutral, global-frame instruction
        self.model = genai.GenerativeModel(
            model_name="models/gemini-2.5-pro-preview-03-25",
            system_instruction=(
                "You are an expert robot navigator in a 3D indoor environment.\n"
                "GLOBAL FRAME (Habitat/HM3D): Y-up, -Z is forward, +X is right.\n"
                "Bucket rules (relative to the REFERENCE OBJECT center):\n"
                "  front  => candidate.z < ref.z\n"
                "  behind => candidate.z > ref.z\n"
                "  left   => candidate.x < ref.x\n"
                "  right  => candidate.x > ref.x\n"
                "Distance = Euclidean between centers.\n"
                "Use the numeric table for distance/direction; use the image for visual sanity checks.\n"
                "Navigate with Goto_object_node_step or Goto_frontier_node_step when unsure.\n"
                "When declaring, choose exactly one *declared_name* from the allowed list."
            ),
        )

    # ---------- Helpers ----------
    @property
    def t(self) -> int:
        return self._t

    def _collect_declared_names(self, obj_ids: List[str], obj_class_names: List[str]) -> List[str]:
        """
        Derive instance-like names from the scene graph if available: e.g., 'rug_288'.
        Fallback to class if no instance id can be found.
        """
        declared = []
        G = getattr(self.sg_sim, "filtered_netx_graph", None)
        nodes = getattr(G, "nodes", {}) if G is not None else {}
        for oid, base in zip(obj_ids, obj_class_names):
            inst = None
            if G is not None and oid in nodes:
                attrs = nodes[oid]
                inst = attrs.get("instance_id") or attrs.get("hm3d_instance_id") or attrs.get("ins_id")
            base_lc = str(base).lower()
            if inst is not None and str(inst).strip() != "":
                declared.append(f"{base_lc}_{inst}")
            else:
                declared.append(base_lc)
        # dedup preserve order
        return list(dict.fromkeys([d if d else "unknown" for d in declared])) or ["unknown"]

    def _graph_candidates_table(self) -> Tuple[str, List[str], List[str]]:
        """
        Build a per-step table from sg_sim:
          - graph ids (for navigation)
          - declared names (for final answer)
        Also merge dataset candidate ids when provided.
        """
        obj_ids = list(getattr(self.sg_sim, "object_node_ids", []) or [])
        obj_names = list(getattr(self.sg_sim, "object_node_names", []) or [])

        # Keep schema enums non-empty
        if not obj_ids:
            obj_ids = ["no_objects_available"]
            obj_names = ["unknown"]

        declared_from_graph = self._collect_declared_names(obj_ids, obj_names)
        # Merge dataset candidate ids (e.g., 'step_306') into allowed declarations
        ds_ids = [str(c.get("id")) for c in self.dataset_candidates if c and c.get("id")]
        allowed_declared = list(dict.fromkeys(declared_from_graph + ds_ids)) or ["unknown"]

        rows = []
        # compose rows using graph objects (positions via sg_sim)
        for oid, cls_name, decl_name in zip(obj_ids, obj_names, declared_from_graph):
            try:
                pos_norm = self.sg_sim.get_position_from_id(oid)  # NORMAL frame
                if pos_norm is None:
                    continue
                pos_hab = np.asarray(pos_normal_to_habitat(np.array(pos_norm, np.float32)), np.float32)
                pos_hab[1] = self.ref_pos_hab[1]  # equalize height for fair dx/dz
                delta = pos_hab - self.ref_pos_hab
                dx, dy, dz = float(delta[0]), float(delta[1]), float(delta[2])
                dist = float(np.linalg.norm(delta))
                bucket = _dir_bucket(self.ref_pos_hab, pos_hab)
                rows.append(_fmt_row(oid, str(cls_name).lower(), decl_name, dist, bucket, dx, dy, dz))
            except Exception:
                continue

        # also include dataset candidates (if not already present) just for visibility
        for it in self.dataset_candidates:
            try:
                pos = np.asarray(it.get("position", [0, 0, 0]), np.float32)
                delta = pos - self.ref_pos_hab
                dx, dy, dz = float(delta[0]), float(delta[1]), float(delta[2])
                dist = float(np.linalg.norm(delta))
                bucket = _dir_bucket(self.ref_pos_hab, pos)
                inst_name = str(it.get("id"))  # e.g., 'step_306'
                cls_name = str(it.get("name", "object")).lower()
                # prefix 'DS:' to note this row is from dataset, not graph-id navigable
                rows.append(_fmt_row(f"DS:{inst_name}", cls_name, inst_name, dist, bucket, dx, dy, dz))
            except Exception:
                continue

        rows_sorted = sorted(rows, key=lambda r: _safe_float(r.split("\t")[3], 9e9))
        return _format_table(rows_sorted), obj_ids, allowed_declared

    # ---------- Schema ----------
    def _planner_schema(self, frontier_ids: List[str], room_ids: List[str],
                        object_ids: List[str], declared_names: List[str]) -> genai.protos.Schema:
        frontier_ids = frontier_ids or ["no_frontiers_available"]
        room_ids = room_ids or ["room_unknown"]
        object_ids = object_ids or ["no_objects_available"]
        declared_names = declared_names or ["unknown"]

        frontier_step = genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "explanation_frontier": genai.protos.Schema(type=genai.protos.Type.STRING),
                "frontier_id": genai.protos.Schema(type=genai.protos.Type.STRING, enum=frontier_ids),
            },
            required=["explanation_frontier", "frontier_id"],
        )
        object_step = genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "explanation_room": genai.protos.Schema(type=genai.protos.Type.STRING),
                "room_id": genai.protos.Schema(type=genai.protos.Type.STRING, enum=room_ids),
                "explanation_obj": genai.protos.Schema(type=genai.protos.Type.STRING),
                "object_id": genai.protos.Schema(type=genai.protos.Type.STRING, enum=object_ids),
            },
            required=["explanation_room", "explanation_obj", "room_id", "object_id"],
        )

        next_action = genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "Goto_object_node_step": object_step,
                "Goto_frontier_node_step": frontier_step,
            },
            description="Choose only one of the two steps.",
        )

        target_declaration = genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "explanation": genai.protos.Schema(type=genai.protos.Type.STRING),
                "declared_target_object_id": genai.protos.Schema(
                    type=genai.protos.Type.STRING,
                    enum=declared_names,
                    description="Return the *declared_name* (e.g., 'step_306' or 'rug_288')."
                ),
                "confidence_level": genai.protos.Schema(type=genai.protos.Type.NUMBER),
                "is_confident": genai.protos.Schema(type=genai.protos.Type.BOOLEAN),
            },
            required=["explanation", "declared_target_object_id", "confidence_level", "is_confident"],
        )

        response_schema = genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "thought_process": genai.protos.Schema(type=genai.protos.Type.STRING),
                "next_action": next_action,
                "target_declaration": target_declaration,
            },
            required=["thought_process", "target_declaration"],
        )
        return response_schema

    # ---------- Prompt ----------
    def _state_prompt(self) -> str:
        dist_m, rel = _parse_q(self._question)
        table_str, obj_ids, declared_names = self._graph_candidates_table()

        # cache for schema call
        self._last_obj_ids = obj_ids
        self._last_declared_names = declared_names

        s = (
            f"Timestep t={self._t}\n"
            f"QUESTION: {self._question}\n"
            f"REFERENCE OBJECT: name={self.ref_name}, pos(HAB)={np.round(self.ref_pos_hab,3).tolist()}\n\n"
            f"CANDIDATES (graph + dataset) — authoritative for distance/direction:\n{table_str}\n\n"
            "SELECTION:\n"
            "- Use bucket rules and choose an allowed declared_name matching the question.\n"
            "- If a distance is given, prefer ±0.35 m tolerance; otherwise choose the closest that matches the direction.\n"
        )
        if dist_m is not None or rel is not None:
            s += f"\nParsed question: distance≈{dist_m} m, relation={rel}\n"
        if self._add_history and self._history:
            s += f"\nHISTORY: {self._history}\n"
        return s

    def _update_history(self, step_info: Optional[Dict[str, Any]]) -> None:
        action_str = "No navigation action."
        if step_info:
            k = list(step_info.keys())[0]
            if "object" in k:
                action_str = f"Goto object: {step_info[k].get('object_id','NA')}"
            elif "frontier" in k:
                action_str = f"Goto frontier: {step_info[k].get('frontier_id','NA')}"
        self._history += f"\n[Action(t={self._t})]: {action_str}"

    # ---------- Model Call ----------
    def _call_model(self, response_schema: genai.protos.Schema, current_state_prompt: str):
        parts = [{"text": current_state_prompt}]
        if self._use_image:
            img = _safe_latest_image(self._output_path)
            if img:
                mime = mimetypes.guess_type(img)[0] or "image/png"
                parts.append({"inline_data": {"mime_type": mime, "data": _b64encode_image(img)}})
        contents = [{"role": "user", "parts": parts}]

        for attempt in range(3):
            try:
                resp = self.model.generate_content(
                    contents,
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json",
                        temperature=0.2,
                        response_schema=response_schema,
                    ),
                )
                text = getattr(resp, "text", None)
                if not text and getattr(resp, "candidates", None):
                    cand = resp.candidates[0]
                    if cand and cand.content and cand.content.parts:
                        maybe = getattr(cand.content.parts[0], "text", None)
                        if maybe:
                            text = maybe
                data = json.loads(text)
                return data.get("next_action"), data.get("target_declaration")
            except Exception as e:
                print(f"API Error (attempt {attempt+1}/3): {e}. Retrying in 10s...")
                time.sleep(10)
        return None, None

    # ---------- Public step ----------
    def get_next_action(self):
        # Build per-step prompt + enums
        state_prompt = self._state_prompt()

        fr_ids = list(getattr(self.sg_sim, "frontier_node_ids", []) or [])
        rm_ids = list(getattr(self.sg_sim, "room_node_ids", []) or [])
        schema = self._planner_schema(
            frontier_ids=fr_ids or ["no_frontiers_available"],
            room_ids=rm_ids or ["room_unknown"],
            object_ids=self._last_obj_ids or ["no_objects_available"],
            declared_names=self._last_declared_names or ["unknown"],
        )

        nav_step_info, target_declaration = self._call_model(schema, state_prompt)

        # Log
        self._outputs_to_save.append(
            f"--- Timestep: {self._t} ---\n"
            f"State Prompt (trunc):\n{state_prompt[:1200]}\n\n"
            f"VLM Step: {json.dumps(nav_step_info, ensure_ascii=False)}\n"
            f"VLM Declaration: {json.dumps(target_declaration, ensure_ascii=False)}\n"
        )
        try:
            with open(self._output_path / "llm_outputs_v2.txt", "w") as f:
                f.write("\n".join(self._outputs_to_save))
        except Exception:
            pass

        # Convert nav step → pose id
        target_pose, target_id = None, None
        if nav_step_info:
            step_type = list(nav_step_info.keys())[0]
            if "Goto_object_node_step" in step_type:
                oid = nav_step_info[step_type].get("object_id")
                if oid and oid in (self._last_obj_ids or []):
                    target_id = oid
                    try:
                        target_pose = self.sg_sim.get_position_from_id(target_id)
                    except Exception:
                        target_pose = None
            elif "Goto_frontier_node_step" in step_type:
                fid = nav_step_info[step_type].get("frontier_id")
                if fid and fid in (list(getattr(self.sg_sim, "frontier_node_ids", []) or [])):
                    target_id = fid
                    try:
                        target_pose = self.sg_sim.get_position_from_id(target_id)
                    except Exception:
                        target_pose = None

        if self._add_history:
            self._update_history(nav_step_info)

        self._t += 1
        conf = _safe_float((target_declaration or {}).get("confidence_level", 0.0), 0.0)
        conf = max(0.0, min(1.0, conf))
        is_conf = bool((target_declaration or {}).get("is_confident", False)) and (conf >= 0.8)

        return target_pose, target_id, is_conf, conf, (target_declaration or {})
