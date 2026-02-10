# /home/artemis/project/graph_eqa_swagat/spatial_experiment/planners/vlm_planner_gemini_spatial.py

import json
import time
import base64
import os
import mimetypes
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import google.generativeai as genai

from graph_eqa.utils.data_utils import get_latest_image


# ---------- Gemini API setup ----------
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    raise RuntimeError("GOOGLE_API_KEY environment variable not set!")


def _b64encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _safe_latest_image(output_path: Path) -> Optional[str]:
    """Return latest image path or None if none exist or any error occurs."""
    try:
        p = get_latest_image(output_path)
        if p is None:
            return None
        p = Path(p)
        return str(p) if p.exists() else None
    except Exception:
        return None


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _basename(s: str) -> str:
    return str(s).split("_")[0].strip().lower()


# ---------- Structured output schema ----------
def _planner_schema(
    frontier_ids: List[str],
    room_ids: List[str],
    object_ids: List[str],
    declared_names: List[str],
) -> genai.protos.Schema:
    """
    - For navigation: model chooses internal ids (object_ids, frontier_ids, room_ids).
    - For declaration: model returns instance-like name from declared_names (e.g., 'tray_286').
    """
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

    target_declaration = genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "explanation": genai.protos.Schema(type=genai.protos.Type.STRING),
            "declared_target_object_id": genai.protos.Schema(
                type=genai.protos.Type.STRING,
                enum=declared_names,  # instance-like names such as 'tray_286' (fallback to class if unknown)
                description="Return the instance-like name (e.g., 'tray_286'). If not available, return the class label.",
            ),
            "confidence_level": genai.protos.Schema(type=genai.protos.Type.NUMBER),
            "is_confident": genai.protos.Schema(type=genai.protos.Type.BOOLEAN),
        },
        required=["explanation", "declared_target_object_id", "confidence_level", "is_confident"],
    )

    step = genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "Goto_frontier_node_step": frontier_step,
            "Goto_object_node_step": object_step,
        },
        description="Choose only one of 'Goto_frontier_node_step' or 'Goto_object_node_step'.",
    )

    response_schema = genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "thought_process": genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="Describe your reasoning about the scene, the question, and what action to take next.",
            ),
            "next_action": step,
            "target_declaration": target_declaration,
        },
        required=["thought_process", "target_declaration"],
    )
    return response_schema


class VLMPlannerEQAGeminiSpatial:
    """
    VLM-based planner for spatial reasoning + target identification.
    - Navigation 'object_id'/'frontier_id'/'room_id' use internal graph ids.
    - Final 'declared_target_object_id' uses instance-like *name* like 'tray_286' (if available).
    """

    def __init__(self, cfg, sg_sim, question, ground_truth_target, output_path: Path):
        self._question = str(question)
        self._ground_truth_target = ground_truth_target
        self._output_path = Path(output_path)
        self._use_image = bool(getattr(cfg, "use_image", True))
        self._add_history = bool(getattr(cfg, "add_history", False))
        self._history = ""
        self.full_plan = ""
        self._t = 0
        self.sg_sim = sg_sim
        self._outputs_to_save = [
            f'Question: {self._question}\nGround Truth: {self._ground_truth_target["name"]} '
            f'({self._ground_truth_target["id"]})\n'
        ]

        # Provide system instruction via model arg (avoid 'system' content role)
        self.model = genai.GenerativeModel(
            model_name="models/gemini-2.5-pro-preview-03-25",
            system_instruction=self.agent_role_prompt,
        )

    # ---------- Helpers ----------
    @property
    def t(self) -> int:
        return self._t

    @property
    def agent_role_prompt(self) -> str:
        return (
            "You are an expert robot navigator in a 3D indoor environment. Your goal is to explore the "
            "environment to find and identify a specific object based on a spatial query.\n\n"
            "TASK:\n"
            "Answer questions like “What object is 2 meters to the left of the sofa?”. Navigate, inspect objects, "
            "and build a mental map.\n\n"
            "INFORMATION:\n"
            "1) SCENE GRAPH: JSON of explored rooms, objects, frontiers (unexplored). Use it as your primary map.\n"
            "2) CURRENT IMAGE: Your current view for visual confirmation.\n"
            "3) AGENT STATE: Your current location within the scene graph.\n\n"
            "ACTIONS (when not confident):\n"
            "- Goto_object_node_step: move to a specific object already in the graph.\n"
            "- Goto_frontier_node_step: explore an unexplored area.\n\n"
            "ANSWERING:\n"
            "Always fill `target_declaration`. If uncertain, set is_confident=false and give your best guess. "
            "When absolutely certain you have visually located the correct object, set is_confident=true to end."
        )

    def _collect_declared_names(self, obj_ids: List[str], obj_class_names: List[str]) -> List[str]:
        """
        Prefer instance-like names such as 'tray_286' using node attributes:
        - instance_id / hm3d_instance_id / ins_id
        Fallback to the class name if no instance id is available.
        """
        declared = []
        G = getattr(self.sg_sim, "filtered_netx_graph", None)
        for oid, base in zip(obj_ids, obj_class_names):
            inst = None
            if G is not None and oid in getattr(G, "nodes", {}):
                attrs = G.nodes[oid]
                inst = attrs.get("instance_id") or attrs.get("hm3d_instance_id") or attrs.get("ins_id")
            if inst is not None and str(inst).strip() != "":
                declared.append(f"{base}_{inst}")
            else:
                declared.append(str(base))
        # Ensure non-empty and de-duplicate while preserving order
        if not declared:
            declared = ["unknown"]
        # Deduplicate
        dedup = list(dict.fromkeys([d if d else "unknown" for d in declared]))
        return dedup

    def _current_candidates(self) -> Dict[str, List[str]]:
        """
        Pull candidate ids/names fresh each timestep.
        Returns dict with keys: object_ids, object_names (class labels), declared_names (instance-like),
        frontier_ids, room_ids.
        """
        obj_ids = list(getattr(self.sg_sim, "object_node_ids", []) or [])
        obj_names = list(getattr(self.sg_sim, "object_node_names", []) or [])
        fr_ids = list(getattr(self.sg_sim, "frontier_node_ids", []) or [])
        rm_ids = list(getattr(self.sg_sim, "room_node_ids", []) or [])

        # Fallbacks to avoid empty enums (Gemini requires non-empty enum lists)
        if not fr_ids:
            fr_ids = ["no_frontiers_available"]
        if not obj_ids:
            # provide a dummy to keep schema valid; planner will ignore it later
            obj_ids = ["no_objects_available"]
            obj_names = ["unknown"]

        declared_names = self._collect_declared_names(obj_ids, obj_names)

        return {
            "object_ids": obj_ids,
            "object_names": obj_names,
            "declared_names": declared_names,
            "frontier_ids": fr_ids,
            "room_ids": rm_ids if rm_ids else ["room_unknown"],
        }

    def _state_prompt(self, scene_graph: str, agent_state: str) -> str:
        s = (
            f"Timestep t={self.t}:\n"
            f"CURRENT AGENT STATE: {agent_state}.\n"
            f"SCENE GRAPH: {scene_graph}.\n"
        )
        if self._add_history and self._history:
            s += f"HISTORY: {self._history}\n"
        # Also record what IDs are available now (helps debugging)
        cand = self._current_candidates()
        s += (
            f"AVAILABLE OBJECT IDS: {cand['object_ids']}\n"
            f"AVAILABLE OBJECT CLASS NAMES: {cand['object_names']}\n"
            f"AVAILABLE DECLARE NAMES: {cand['declared_names']}\n"
            f"AVAILABLE FRONTIERS: {cand['frontier_ids']}\n"
            f"AVAILABLE ROOMS: {cand['room_ids']}\n"
        )
        return s

    def _update_history(self, step_info: Optional[Dict[str, Any]]) -> None:
        action_str = "No navigation action taken."
        if step_info:
            step_type = list(step_info.keys())[0]
            if "object" in step_type:
                action_str = f"Goto object: {step_info[step_type].get('object_id', 'NA')}"
            elif "frontier" in step_type:
                action_str = f"Goto frontier: {step_info[step_type].get('frontier_id', 'NA')}"
        self._history += f"\n[Action(t={self.t})]: {action_str}"

    # ---------- Model Call ----------
    def _call_model(
        self, response_schema: genai.protos.Schema, current_state_prompt: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        parts = [{
            "text": (
                f"QUESTION: {self._question}\n\n"
                f"CURRENT STATE:\n{current_state_prompt}\n\n"
                "Provide JSON that follows the given schema."
            )
        }]

        # Attach latest image only if it exists
        if self._use_image:
            img_path = _safe_latest_image(self._output_path)
            if img_path:
                mime = mimetypes.guess_type(img_path)[0] or "image/png"
                parts.append({"inline_data": {"mime_type": mime, "data": _b64encode_image(img_path)}})

        contents = [{"role": "user", "parts": parts}]

        # Retry wrapper
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
                print(f"API Error (attempt {attempt+1}/3): {e}. Retrying in 15s...")
                time.sleep(15)

        print("Error: Max retries exceeded for Gemini API call.")
        return None, None

    # ---------- Public: one planning step ----------
    def get_next_action(self):
        # 1) Build prompt/state
        agent_state = self.sg_sim.get_current_semantic_state_str()
        current_state_prompt = self._state_prompt(self.sg_sim.scene_graph_str, agent_state)

        # 2) Build schema from *current* candidates
        cand = self._current_candidates()
        schema = _planner_schema(
            frontier_ids=cand["frontier_ids"],
            room_ids=cand["room_ids"],
            object_ids=cand["object_ids"],
            declared_names=cand["declared_names"],
        )

        # 3) Call model
        nav_step_info, target_declaration = self._call_model(schema, current_state_prompt)

        # If API failed, keep runner alive
        if target_declaration is None:
            return None, None, False, 0.0, {"error": "API Failure"}

        # 4) Log what VLM said
        self._outputs_to_save.append(
            f"--- Timestep: {self._t} ---\n"
            f"Agent State: {agent_state}\n"
            f"VLM Thought: {nav_step_info}\n"
            f"VLM Declaration: {target_declaration}\n"
            f"Candidates snapshot: {cand}\n"
        )
        self.full_plan = "\n".join(self._outputs_to_save)
        try:
            with open(self._output_path / "llm_outputs.txt", "w") as f:
                f.write(self.full_plan)
        except Exception:
            pass

        # 5) Convert nav step → pose if valid
        target_pose, target_id = None, None

        if nav_step_info:
            step_type = list(nav_step_info.keys())[0]

            if "object" in step_type:
                obj_id = nav_step_info[step_type].get("object_id")
                # stale guard: ensure id still exists this step
                if obj_id in cand["object_ids"] and obj_id != "no_objects_available":
                    target_id = obj_id
            elif "frontier" in step_type:
                fr_id = nav_step_info[step_type].get("frontier_id")
                if fr_id in cand["frontier_ids"] and fr_id != "no_frontiers_available":
                    target_id = fr_id

            if target_id:
                try:
                    target_pose = self.sg_sim.get_position_from_id(target_id)
                except Exception:
                    target_pose = None  # node may have vanished; skip move this step

        # 6) (optional) keep short history for model
        if self._add_history:
            self._update_history(nav_step_info)

        # 7) Advance time
        self._t += 1

        # 8) Confidence/termination guards
        raw_conf = target_declaration.get("confidence_level", 0.0)
        conf = _safe_float(raw_conf, 0.0)
        # Clamp to [0, 1] in case model outputs e.g. 5.0
        conf = max(0.0, min(1.0, conf))

        is_conf_claim = bool(target_declaration.get("is_confident", False))

        # Only allow termination if:
        # - confidence high enough, AND
        # - declared id is in the current declared_names (prevents "door" when no door exists)
        declared_id = str(target_declaration.get("declared_target_object_id", "")).strip()
        in_declared_pool = declared_id in cand["declared_names"]

        is_confident = is_conf_claim and (conf >= 0.8) and in_declared_pool

        return target_pose, target_id, is_confident, conf, target_declaration
