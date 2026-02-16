#!/usr/bin/env python3
# /home/artemis/project/graph_eqa_swagat/spatial_experiment/planners/vlm_planner_msp.py

from __future__ import annotations

import os
import json
import math
import base64
import mimetypes
from pathlib import Path
from typing import Optional, Any, Dict, List, Tuple

import numpy as np
import google.generativeai as genai

# Graph EQA / Habitat Imports
from graph_eqa.envs.utils import pos_normal_to_habitat
from graph_eqa.utils.data_utils import get_latest_image

# MSP Imports
from spatial_experiment.msp.pdf import combined_logpdf as _combined_logpdf

# -----------------------------------------------------------------------------
# Configuration & Setup
# -----------------------------------------------------------------------------

if "GOOGLE_API_KEY" not in os.environ:
    raise RuntimeError("GOOGLE_API_KEY must be set.")

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# IMPORTANT: keep a single global model instance
gemini_model = genai.GenerativeModel(
    model_name="models/gemini-2.5-pro",
)

# -----------------------------------------------------------------------------
# IO Helpers
# -----------------------------------------------------------------------------

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _safe_latest_image(out_path: Path) -> Optional[str]:
    img = get_latest_image(Path(out_path))
    return str(img) if img else None


def _write_jsonl(path: Path, record: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        print(f"[MSP] Failed to write jsonl log {path}: {e}")


def _write_json(path: Path, obj: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
    except Exception as e:
        print(f"[MSP] Failed to write json {path}: {e}")


# -----------------------------------------------------------------------------
# TRACE HELPERS (NEW)
# -----------------------------------------------------------------------------

def _write_trace_step(out_dir: Path, t: int, trace: Dict[str, Any]) -> None:
    _write_json(out_dir / f"trace_step_{t:03d}.json", trace)


def _append_trace_txt(out_dir: Path, lines: List[str]) -> None:
    try:
        with open(out_dir / "llm_outputs_smart.txt", "a") as f:
            for ln in lines:
                f.write(ln.rstrip() + "\n")
            f.write("\n")
    except Exception as e:
        print(f"[TRACE] Could not append trace txt: {e}")


def _shorten(s: str, n: int = 260) -> str:
    s = str(s or "")
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def _summarize_rank_delta(scored: List[Dict[str, Any]], k: int = 5) -> Dict[str, Any]:
    top = scored[:k]
    if len(top) < 2:
        gap = None
    else:
        gap = float(top[0]["msp_score"] - top[1]["msp_score"])
    return {
        "topk": [
            {
                "id": str(x.get("id", "")),
                "name": x.get("name", ""),
                "score": float(x.get("msp_score", 0.0)),
                "pos_hab": x.get("position", None),
            }
            for x in top
        ],
        "gap_1_2": gap,
    }


# -----------------------------------------------------------------------------
# Math / Geometry Helpers (V8-style)
# -----------------------------------------------------------------------------

def _wrap_angle(angle: float) -> float:
    """Wrap angle to [0, 2π)."""
    two_pi = 2.0 * math.pi
    return (angle % two_pi + two_pi) % two_pi


def _circular_blend(theta_a: float, theta_b: float, w_b: float) -> float:
    """
    Blend angles on the unit circle.
    Returns angle close to weighted sum of unit vectors:
      (1-w_b)*a + w_b*b
    """
    w_b = float(np.clip(w_b, 0.0, 1.0))
    w_a = 1.0 - w_b
    x = w_a * math.cos(theta_a) + w_b * math.cos(theta_b)
    y = w_a * math.sin(theta_a) + w_b * math.sin(theta_b)
    return _wrap_angle(math.atan2(y, x))


def _camera_theta_to_world(vlm_theta: float, agent_yaw: float) -> float:
    """
    Camera / agent frame (egocentric):
      0.0 rad = straight ahead (center of image)
      +π/2    = left of image
      -π/2    = right of image
      π       = behind (opposite of forward)

    The VLM now returns θ in this egocentric frame.
    We convert to world by simply adding the agent yaw.
    """
    theta_world = agent_yaw + vlm_theta
    return _wrap_angle(theta_world)


def _estimate_object_front_yaw_from_agent(anchor_pos_hab: np.ndarray, agent_pos_hab: np.ndarray) -> float:
    """
    Fallback heuristic (NOT intrinsic):
      - Compute yaw(object -> agent)
      - Assume object functional front points roughly toward open space / agent,
        so yaw_front = yaw_obj_to_agent + π
    """
    anchor = np.asarray(anchor_pos_hab, dtype=np.float32)
    agent = np.asarray(agent_pos_hab, dtype=np.float32)
    dx = agent[0] - anchor[0]
    dz = agent[2] - anchor[2]
    yaw_obj_to_agent = math.atan2(dz, dx)
    return _wrap_angle(yaw_obj_to_agent + math.pi)


def _predicate_offset(question_or_predicate: str) -> float:
#     """
#     Map linguistic predicate to offset in object-front frame:
#       0 rad  = in front (along functional front)
#       +π/2   = left of front
#       -π/2   = right of front
#       π      = behind
#     """
#     q = (question_or_predicate or "").lower()
#     if "left" in q:
#         return +math.pi / 2.0
#     if "right" in q:
#         return -math.pi / 2.0
#     if "behind" in q or "back of" in q or "backside" in q:
#         return math.pi
    return 0.0  # in front / front of


def _parse_q_dist(question: str) -> float:
    import re
    m = re.search(r"(\d+(?:\.\d+)?)\s*meters?", (question or "").lower())
    return float(m.group(1)) if m else 1.0


def _unit_dir_from_theta_phi(theta: float, phi: float) -> np.ndarray:
    """
    Convert spherical (theta azimuth, phi elevation) to unit vector in habitat/world:
        x = cos(theta)*sin(phi)
        y = cos(phi)
        z = sin(theta)*sin(phi)
    With phi=pi/2 -> level plane (y=0).
    """
    st = math.sin(phi)
    x = math.cos(theta) * st
    y = math.cos(phi)
    z = math.sin(theta) * st
    v = np.array([x, y, z], dtype=np.float32)
    n = float(np.linalg.norm(v) + 1e-8)
    return v / n


# -----------------------------------------------------------------------------
# STEP 1: Spatial kernel (VLM + intrinsic/front prior)
# -----------------------------------------------------------------------------

def get_vlm_spatial_kernel_params(
    image_path: Optional[str],
    question: str,
    anchor_name: str,
    anchor_pos_hab: np.ndarray,
    agent_pos_hab: np.ndarray,
    agent_yaw: float,
    anchor_front_yaw_world: Optional[float] = None,
    log_jsonl_path: Optional[Path] = None,
    step_t: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Returns:
      theta_world, phi, kappa, reasoning, debug

    CHANGE:
      - No predicate_offset heuristic.
      - No fusion/blending.
      - When VLM succeeds, we trust VLM direction entirely.
      - We keep an intrinsic-yaw prior ONLY as fallback (no image / VLM failure).
    """
    anchor_pos_hab = np.asarray(anchor_pos_hab, dtype=np.float32)
    agent_pos_hab = np.asarray(agent_pos_hab, dtype=np.float32)

    # Fallback PRIOR (ONLY for when VLM can't run)
    if anchor_front_yaw_world is not None:
        yaw_front_world = _wrap_angle(float(anchor_front_yaw_world))
        front_source = "csv_anchor_yaw"
    else:
        yaw_front_world = _estimate_object_front_yaw_from_agent(anchor_pos_hab, agent_pos_hab)
        front_source = "heuristic_obj_to_agent"

    theta_prior = _wrap_angle(yaw_front_world)
    phi_prior = math.pi / 2.0  # level

    # If no image -> fallback prior only
    if not image_path or not os.path.exists(str(image_path)):
        out = {
            "theta": theta_prior,
            "phi": phi_prior,
            "kappa": 6.0 if anchor_front_yaw_world is None else 10.0,
            "reasoning": (
                "No image available; used intrinsic/front prior ONLY as fallback. "
                f"(front_source={front_source})"
            ),
            "debug": {
                "theta_prior": theta_prior,
                "yaw_front_world": yaw_front_world,
                "front_source": front_source,
                "used_vlm": False,
            },
        }
        if log_jsonl_path:
            _write_jsonl(log_jsonl_path, {
                "type": "kernel",
                "t": step_t,
                "image_path": image_path,
                "question": question,
                "anchor_name": anchor_name,
                "used_image": False,
                "result": out,
            })
        return out

    # VLM prompt: still enforce "intrinsic front" concept
#     sys_prompt = """
# SYSTEM: You are a Spatial Affordance Reasoning Engine.

# CRITICAL RULE:
# - "Functional front" is INTRINSIC to the REFERENCE OBJECT, NOT camera-relative.
# - Do NOT define front as "toward right of image" or "toward camera".
# - The camera pose can be arbitrary; the object's intrinsic front stays the same.

# Examples of intrinsic front:
# - Sofa/chair: direction a seated person faces (out from seating surface).
# - Fridge: door-facing side is the front.
# - TV/monitor: screen-facing side is the front.
# - Sink: side you stand at to use it.

# Your job:
# Given the question and the reference object in the image, output the direction
# (for the queried location relative to the object) in CAMERA coordinates.

# CAMERA COORDS (Unit Circle from Top-Down):
# THETA (azimuth):
#   0.0 rad  = Right side of image
#   1.57 rad = CENTER of image (Forward / Deep into scene)
#   3.14 rad = Left side of image
#   4.71 rad = Behind camera

# PHI (elevation):
#   1.57 rad = level
#   0.0 rad  = above / on top
#   3.14 rad = below / under

# Return JSON only.
# """
    sys_prompt = """
SYSTEM: You are a Geometric Orientation Engine.

YOUR GOAL:
Identify the **INTRINSIC FRONT VECTOR** of the Reference Object relative to the Camera.

CRITICAL RULES:
1. **IGNORE THE DISTANCE** (e.g., "3 meters"). Your job is ONLY orientation (Angle), not destination.
2. **DO NOT LOOK FOR THE TARGET.** If the user asks "what is 3 meters in front", DO NOT look for objects 3 meters away.
3. **ONLY OUTPUT THE FACE ORIENTATION.** Just tell me which direction the object is "facing" (its functional front).

DEFINITION OF "INTRINSIC FRONT":
- Sofa/Chair: The direction your knees point when you sit on it.
- TV/Monitor: The direction the screen is projecting light.
- Cabinet: The direction the drawers open.

CAMERA COORDINATES (Egocentric, top-down):
THETA (azimuth):
  0.00 rad  = Straight ahead (center of the image)
  +1.57 rad = To the LEFT side of the image
  -1.57 rad (or 4.71) = To the RIGHT side of the image
  3.14 rad = Directly behind the camera (opposite of forward)

PHI (elevation):
  1.57 rad = level
  0.0 rad  = above / on top
  3.14 rad = below / under

IMPORTANT:
- Use this θ definition. Do NOT revert to any other convention.
- You are only deciding orientation, not distance.

Return JSON only.
"""

    schema = genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "reasoning": genai.protos.Schema(type=genai.protos.Type.STRING),
            "theta_radians": genai.protos.Schema(type=genai.protos.Type.NUMBER),
            "phi_radians": genai.protos.Schema(type=genai.protos.Type.NUMBER),
            "kappa": genai.protos.Schema(type=genai.protos.Type.NUMBER),
        },
        required=["reasoning", "theta_radians", "phi_radians", "kappa"],
    )

    mime = mimetypes.guess_type(str(image_path))[0] or "image/png"

    prior_hint = ""
    if anchor_front_yaw_world is not None:
        prior_hint = (
            f"\nNOTE: Intrinsic anchor-front yaw in WORLD coords is available: "
            f"{float(anchor_front_yaw_world):.4f} rad. "
            f"Use this as an intrinsic-front cue if the object is visible.\n"
        )

    # messages = [
    #     {
    #         "role": "user",
    #         "parts": [
    #             {
    #                 "text": (
    #                     f"{sys_prompt}\n{prior_hint}\n"
    #                     f"Query: {question}\n"
    #                     f"Reference Object: {anchor_name}\n"
    #                     f"Task: Output Theta/Phi/Kappa in CAMERA frame as JSON."
    #                 )
    #             },
    #             {
    #                 "inline_data": {
    #                     "mime_type": mime,
    #                     "data": encode_image(str(image_path)),
    #                 }
    #             },
    #         ],
    #     }
    # ]

        sanitized_query = f"Where is the intrinsic front of the {anchor_name}?"
    
    # If your system supports "Left/Right", you can map it dynamically:
    # if "left" in question.lower(): sanitized_query = f"Where is the Left side of {anchor_name}?"
    # elif "right" in question.lower(): sanitized_query = f"Where is the Right side of {anchor_name}?"
    # else: sanitized_query = f"Where is the Intrinsic Front of {anchor_name}?"

    messages = [
        {
            "role": "user",
            "parts": [
                {
                    "text": (
                        f"{sys_prompt}\n{prior_hint}\n"
                        f"Reference Object: {anchor_name}\n"
                        # CRITICAL CHANGE: Don't show the full question.
                        # Show a request for pure orientation.
                        f"Task: {sanitized_query}\n" 
                        f"Instruction: Output the Theta/Phi direction of that face relative to the camera."
                    )
                },
                {
                    "inline_data": {
                        "mime_type": mime,
                        "data": encode_image(str(image_path)),
                    }
                },
            ],
        }
    ]

    raw_text = ""
    try:
        resp = gemini_model.generate_content(
            messages,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.2,
                response_schema=schema,
            ),
        )
        raw_text = resp.text
        d = json.loads(resp.text)

        theta_cam = float(d["theta_radians"])
        phi = float(d["phi_radians"])
        kappa = float(d["kappa"])
        reasoning = d.get("reasoning", "")
        # Enforce a meaningful directional concentration:
        # - If model gives 0 or nonsense, fall back to a reasonable default.
        if not math.isfinite(kappa) or kappa <= 0.1:
            kappa = 5.0
            
        # Trust VLM: convert camera theta to world theta and return directly
        theta_world = _camera_theta_to_world(theta_cam, agent_yaw)
        print(
            f"[MSP-VLM-ANGLE] question='{question}', "
            f"anchor='{anchor_name}', "
            f"agent_yaw={agent_yaw:.3f}, "
            f"theta_cam={theta_cam:.3f}, "
            f"theta_world={theta_world:.3f}, "
            f"phi={phi:.3f}, kappa={kappa:.3f}"
        )


        out = {
            "theta": float(theta_world),
            "phi": float(phi),
            "kappa": float(kappa),
            "reasoning": reasoning,
            "debug": {
                "theta_cam": theta_cam,
                "theta_world": theta_world,
                "agent_yaw": float(agent_yaw),
                "front_source": front_source,
                "anchor_front_yaw_world": (float(anchor_front_yaw_world) if anchor_front_yaw_world is not None else None),
                "theta_prior_fallback_only": theta_prior,
                "used_vlm": True,
            },
        }

        if log_jsonl_path:
            _write_jsonl(log_jsonl_path, {
                "type": "kernel",
                "t": step_t,
                "image_path": image_path,
                "question": question,
                "anchor_name": anchor_name,
                "used_image": True,
                "messages": messages[0]["parts"][0]["text"],
                "raw_response_text": raw_text,
                "parsed": d,
                "result": out,
            })

        return out

    except Exception as e:
        # Fallback to prior (still no predicate_offset)
        out = {
            "theta": theta_prior,
            "phi": phi_prior,
            "kappa": 6.0 if anchor_front_yaw_world is None else 10.0,
            "reasoning": f"VLM call failed; fallback to intrinsic/front prior ONLY. Error: {e}",
            "debug": {
                "theta_prior": theta_prior,
                "yaw_front_world": yaw_front_world,
                "front_source": front_source,
                "used_vlm": False,
            },
        }
        if log_jsonl_path:
            _write_jsonl(log_jsonl_path, {
                "type": "kernel_error",
                "t": step_t,
                "image_path": image_path,
                "question": question,
                "anchor_name": anchor_name,
                "raw_response_text": raw_text,
                "error": str(e),
                "result": out,
            })
        return out


# -----------------------------------------------------------------------------
# STEP 2: MSP scoring core
# -----------------------------------------------------------------------------

class MSPEngineSmart:
    def __init__(self):
        pass

    def _get_metric_semantic_params(
        self,
        anchor_pos_hab: np.ndarray,
        candidate_pos_hab: np.ndarray,
        candidate_size: Optional[List[float]],
        distance_m: float,
    ) -> Dict[str, float]:
        pos = np.asarray(candidate_pos_hab, dtype=np.float32)
        size = candidate_size or [0.5, 0.5, 0.5]
        w, d, h = [float(x) for x in size[:3]]
        max_dim = max(w, d, h)

        return {
            "mu_x": float(pos[0]),
            "mu_y": float(pos[1]),
            "mu_z": float(pos[2]),
            "sigma_s": 0.5 * max_dim,

            "x0": float(anchor_pos_hab[0]),
            "y0": float(anchor_pos_hab[1]),
            "z0": float(anchor_pos_hab[2]),

            "d0": float(distance_m),
            "sigma_m": 0.3 * max_dim,
        }

    def score_point(
        self,
        point_hab: np.ndarray,
        anchor_pos_hab: np.ndarray,
        kernel_params: Dict[str, Any],
        question_dist: float,
        candidate_size: Optional[List[float]] = None,
    ) -> float:
        point_hab = np.asarray(point_hab, dtype=np.float32)
        anchor_pos_hab = np.asarray(anchor_pos_hab, dtype=np.float32)
        params = {
            **self._get_metric_semantic_params(
                anchor_pos_hab=anchor_pos_hab,
                candidate_pos_hab=point_hab,
                candidate_size=candidate_size or [0.5, 0.5, 0.5],
                distance_m=question_dist,
            ),
            "theta0": float(kernel_params["theta"]),
            "phi0": float(kernel_params["phi"]),
            "kappa": float(kernel_params["kappa"]),
        }
        logp = float(
            _combined_logpdf(
                np.array([point_hab[0]]),
                np.array([point_hab[1]]),
                np.array([point_hab[2]]),
                params,
            )[0]
        )
        return logp

    def score_candidates(
        self,
        objects: List[Dict[str, Any]],
        frontiers: List[Dict[str, Any]],
        anchor_pos_hab: np.ndarray,
        kernel_params: Dict[str, Any],
        question_dist: float,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:

        anchor_pos_hab = np.asarray(anchor_pos_hab, dtype=np.float32)

        scored_objects: List[Dict[str, Any]] = []
        for obj in objects:
            pos = np.asarray(obj["position"], dtype=np.float32)
            params = {
                **self._get_metric_semantic_params(anchor_pos_hab, pos, obj.get("size", None), question_dist),
                "theta0": float(kernel_params["theta"]),
                "phi0": float(kernel_params["phi"]),
                "kappa": float(kernel_params["kappa"]),
            }
            logp = float(
                _combined_logpdf(
                    np.array([pos[0]]), np.array([pos[1]]), np.array([pos[2]]), params
                )[0]
            )
            scored_objects.append({**obj, "msp_score": logp})

        scored_frontiers: List[Dict[str, Any]] = []
        for fr in frontiers:
            pos = np.asarray(fr["position"], dtype=np.float32)
            params = {
                **self._get_metric_semantic_params(anchor_pos_hab, pos, fr.get("size", None), question_dist),
                "theta0": float(kernel_params["theta"]),
                "phi0": float(kernel_params["phi"]),
                "kappa": float(kernel_params["kappa"]),
            }
            logp = float(
                _combined_logpdf(
                    np.array([pos[0]]), np.array([pos[1]]), np.array([pos[2]]), params
                )[0]
            )
            scored_frontiers.append({**fr, "msp_score": logp})

        scored_objects.sort(key=lambda x: x["msp_score"], reverse=True)
        scored_frontiers.sort(key=lambda x: x["msp_score"], reverse=True)
        return scored_objects, scored_frontiers


# -----------------------------------------------------------------------------
# STEP 3: Planner — VLM sees scored candidates + point guess, logs everything
# -----------------------------------------------------------------------------

class VLMPlannerMSP_Smart:
    def __init__(self, cfg, sg_sim, question, gt=None, out_path=".", **kwargs):
        self.cfg = cfg
        self.sg_sim = sg_sim
        self._question = question
        self._out_path = Path(out_path)
        self._t = 0
        self._history = ""
        self._outputs_to_save = [f"Question: {question}\n"]

        # MODE RESOLUTION (robust):
        # Support legacy graph_eqa values: "msp_point", "msp_object"
        # Support new values: "where", "which"
        # Prefer vlm.msp_nobnn.mode when present.
        raw_answer_mode = str(getattr(cfg, "answer_mode", "") or "").lower().strip()

        # 1) Prefer nested MSP config if present
        nested_mode = ""
        try:
            nested_mode = str(getattr(getattr(cfg, "msp_nobnn", None), "mode", "") or "").lower().strip()
        except Exception:
            nested_mode = ""

        # normalize mapping
        def _normalize_mode(m: str) -> str:
            m = (m or "").lower().strip()
            mapping = {
                "msp_point": "where",
                "msp_where": "where",
                "point": "where",
                "where": "where",

                "msp_object": "which",
                "msp_which": "which",
                "object": "which",
                "which": "which",

                # some older configs used "eqa" for object-choice behavior
                "eqa": "which",
            }
            return mapping.get(m, m)

        resolved = _normalize_mode(nested_mode) if nested_mode else _normalize_mode(raw_answer_mode)

        # final guard (do NOT hard-crash on legacy values)
        if resolved not in ["where", "which"]:
            # last fallback: if they used "msp_*" but unknown, default to where
            resolved = "where"

        self.answer_mode: str = resolved

        # Runner-provided hints from CSV:
        self._anchor_label: Optional[str] = kwargs.get("anchor_label", None)
        self._anchor_center_hab: Optional[np.ndarray] = kwargs.get("anchor_center_hab", None)
        if self._anchor_center_hab is not None:
            self._anchor_center_hab = np.asarray(self._anchor_center_hab, dtype=np.float32)

        # NEW: intrinsic anchor front yaw (world) from dataset (e.g., ann_yaw_rad)
        self._anchor_front_yaw_world: Optional[float] = kwargs.get("anchor_front_yaw_world", None)
        if self._anchor_front_yaw_world is not None:
            self._anchor_front_yaw_world = float(self._anchor_front_yaw_world)

        self.msp_engine = MSPEngineSmart()

        # Logging paths
        self._vlm_calls_path = self._out_path / "vlm_calls.jsonl"

        print(f"\n[MSP SMART INIT] mode={self.answer_mode} Q: {self._question}")
        print(f"  - Anchor label hint: {self._anchor_label}")
        print(f"  - Anchor center hint (hab): {self._anchor_center_hab}")
        print(f"  - Anchor front yaw world (intrinsic): {self._anchor_front_yaw_world}")

        _write_jsonl(self._vlm_calls_path, {
            "type": "planner_init",
            "mode": self.answer_mode,
            "question": self._question,
            "anchor_label_hint": self._anchor_label,
            "anchor_center_hint": (self._anchor_center_hab.tolist() if self._anchor_center_hab is not None else None),
            "anchor_front_yaw_world": self._anchor_front_yaw_world,
        })

        # Ensure llm_outputs_smart exists and starts cleanly for the run directory.
        # (We still append per step; this makes the file more readable.)
        try:
            with open(self._out_path / "llm_outputs_smart.txt", "w") as f:
                f.write(f"Question: {self._question}\n\n")
        except Exception:
            pass

    @property
    def t(self):
        return self._t

    def _get_scene_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        objects: List[Dict[str, Any]] = []
        for oid, name in zip(self.sg_sim.object_node_ids, self.sg_sim.object_node_names):
            try:
                pos_norm = self.sg_sim.get_position_from_id(oid)
                if pos_norm is None:
                    continue
                pos_hab = np.asarray(
                    pos_normal_to_habitat(np.asarray(pos_norm, dtype=np.float32)),
                    dtype=np.float32,
                )
                objects.append(
                    {
                        "id": str(oid),
                        "name": str(name).lower(),
                        "position": pos_hab.tolist(),
                        "type": "object",
                        "size": [0.5, 0.5, 0.5],
                    }
                )
            except Exception:
                continue

        frontiers: List[Dict[str, Any]] = []
        for fid in getattr(self.sg_sim, "frontier_node_ids", []) or []:
            try:
                pos_norm = self.sg_sim.get_position_from_id(fid)
                if pos_norm is None:
                    continue
                pos_hab = np.asarray(
                    pos_normal_to_habitat(np.asarray(pos_norm, dtype=np.float32)),
                    dtype=np.float32,
                )
                frontiers.append(
                    {
                        "id": str(fid),
                        "name": "frontier",
                        "position": pos_hab.tolist(),
                        "type": "frontier",
                        "size": [0.5, 0.5, 0.5],
                    }
                )
            except Exception:
                continue

        return objects, frontiers

    # -------------------------
    # NEW: Anchor resolution with debug trace
    # -------------------------

    def _resolve_anchor(
        self, objects: List[Dict[str, Any]]
    ) -> Tuple[Optional[Dict[str, Any]], str, Dict[str, Any]]:
        """
        Resolve anchor when multiple instances exist:
          1) filter by label substring match (case-insensitive)
          2) choose closest to anchor_center_hab (from dataset CSV)
        Returns (anchor_obj, anchor_name, debug_dict)
        """
        dbg: Dict[str, Any] = {
            "anchor_label_hint": self._anchor_label,
            "anchor_center_hab_hint": (self._anchor_center_hab.tolist() if self._anchor_center_hab is not None else None),
            "match_strategy": None,
            "num_objects": len(objects),
            "candidates": [],
            "chosen": None,
        }

        if not self._anchor_label:
            dbg["match_strategy"] = "no_label_hint"
            return None, "unknown_anchor", dbg

        label = self._anchor_label.strip().lower()

        candidates = [o for o in objects if label in (o.get("name") or "")]
        dbg["match_strategy"] = "substring_label"
        if not candidates:
            label2 = label.replace(" ", "")
            candidates = [o for o in objects if label2 in (o.get("name") or "").replace(" ", "")]
            dbg["match_strategy"] = "substring_label_no_spaces"

        if not candidates:
            dbg["match_strategy"] = str(dbg["match_strategy"]) + "_no_match"
            return None, label, dbg

        # log candidate list (cap for file size)
        for o in candidates[:30]:
            dbg["candidates"].append({
                "id": str(o.get("id","")),
                "name": o.get("name",""),
                "pos_hab": o.get("position", None),
            })

        # If no center hint, pick first (stable)
        if self._anchor_center_hab is None:
            best = candidates[0]
            dbg["chosen"] = {"id": str(best.get("id","")), "name": best.get("name",""), "pos_hab": best.get("position")}
            return best, best.get("name", label), dbg

        # Otherwise pick closest to csv center
        c0 = np.asarray(self._anchor_center_hab, dtype=np.float32)
        best = None
        best_d = 1e9
        for o in candidates:
            p = np.asarray(o["position"], dtype=np.float32)
            d = float(np.linalg.norm(p - c0))
            if d < best_d:
                best_d = d
                best = o

        dbg["match_strategy"] = str(dbg["match_strategy"]) + "_closest_to_csv_center"
        dbg["chosen"] = {
            "id": str(best.get("id","")) if best else "",
            "name": best.get("name","") if best else "",
            "pos_hab": best.get("position") if best else None,
            "dist_to_center": float(best_d) if best else None,
        }
        return best, best.get("name", label) if best else label, dbg

    # -------------------------
    # Selector VLM (unchanged behavior, better guardrails + better prompt)
    # -------------------------

    def _build_selector_prompt(
        self,
        agent_state: str,
        anchor_name: str,
        anchor_pos_hab: np.ndarray,
        kernel: Dict[str, Any],
        dist_m: float,
        top_objects: List[Dict[str, Any]],
        top_frontiers: List[Dict[str, Any]],
        point_guess: Optional[Dict[str, Any]],
    ) -> str:
        """
        Build a prompt that shows:
          - kernel info
          - top objects w/ scores and center coords
          - top frontiers w/ scores and coords
          - optional point guess (WHERE mode)
        """
        if self.answer_mode == "where":
            mode_rules = (
                "MODE=WHERE: You may choose a coordinate point (POINT_GUESS) OR an object id.\n"
                "If the query is about a location in space, POINT_GUESS is usually correct.\n"
            )
        else:
            mode_rules = (
                "MODE=WHICH: You MUST choose an OBJECT from TOP OBJECTS.\n"
                "POINT_GUESS is NOT allowed.\n"
                "If nothing matches, choose the best available object with low confidence and explain.\n"
            )

        obj_lines = []
        for o in top_objects:
            p = o.get("position", [0, 0, 0])
            obj_lines.append(
                f"- id={o['id']} name={o.get('name','')} score={o.get('msp_score',0.0):.3f} "
                f"center_xyz_hab=[{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}]"
            )

        fr_lines = []
        for f in top_frontiers:
            p = f.get("position", [0, 0, 0])
            fr_lines.append(
                f"- id={f['id']} score={f.get('msp_score',0.0):.3f} xyz_hab=[{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}]"
            )

        point_block = "POINT_GUESS: none\n"
        if point_guess is not None:
            pg = point_guess["target_xyz_hab"]
            point_block = (
                f"POINT_GUESS: id=POINT_GUESS score={point_guess['msp_score']:.3f} "
                f"xyz_hab=[{pg[0]:.3f},{pg[1]:.3f},{pg[2]:.3f}]\n"
            )

        return f"""
You are a selector for a robot spatial query system.

{mode_rules}

Question: {self._question}

Anchor:
- name: {anchor_name}
- anchor_xyz_hab: [{anchor_pos_hab[0]:.3f},{anchor_pos_hab[1]:.3f},{anchor_pos_hab[2]:.3f}]
- requested_distance_m: {dist_m:.3f}

Spatial Kernel (world):
- theta_world: {kernel['theta']:.4f}
- phi: {kernel['phi']:.4f}
- kappa: {kernel['kappa']:.3f}
- reasoning: {kernel.get('reasoning','')}

Candidates (ranked by MSP logpdf):
{point_block}

TOP OBJECTS:
{chr(10).join(obj_lines) if obj_lines else "- none"}

TOP FRONTIERS:
{chr(10).join(fr_lines) if fr_lines else "- none"}

History:
{self._history}

Current state:
{agent_state}

Task:
Decide whether to:
- answer now (and what the answer is),
- or move (goto_object/goto_frontier),
- or lookaround.

Output STRICT JSON only:
- thought: string
- action_type: one of ["goto_frontier","goto_object","lookaround","answer"]
- chosen_id: string
    * WHERE: "POINT_GUESS" or an object id (from TOP OBJECTS)
    * WHICH: must be an object id from TOP OBJECTS
- target_xyz_hab: [x,y,z] only if chosen_id=="POINT_GUESS", else []
- answer_text: string (what you would say)
- confidence: float in [0,1]
"""

    def _call_selector_llm(self, prompt: str) -> Tuple[Dict[str, Any], str]:
        selector_schema = genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "thought": genai.protos.Schema(type=genai.protos.Type.STRING),
                "action_type": genai.protos.Schema(
                    type=genai.protos.Type.STRING,
                    enum=["goto_frontier", "goto_object", "lookaround", "answer"],
                ),
                "chosen_id": genai.protos.Schema(type=genai.protos.Type.STRING),
                "target_xyz_hab": genai.protos.Schema(
                    type=genai.protos.Type.ARRAY,
                    items=genai.protos.Schema(type=genai.protos.Type.NUMBER),
                ),
                "answer_text": genai.protos.Schema(type=genai.protos.Type.STRING),
                "confidence": genai.protos.Schema(type=genai.protos.Type.NUMBER),
            },
            required=["thought", "action_type", "chosen_id", "confidence"],
        )

        messages = [{"role": "user", "parts": [{"text": prompt}]}]
        resp = gemini_model.generate_content(
            messages,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.1,
                response_schema=selector_schema,
            ),
        )
        return json.loads(resp.text), resp.text

    # -------------------------
    # Optional: room constraint hook (safe placeholder)
    # -------------------------

    def _get_anchor_room_id(self, anchor_object_id: str) -> Optional[str]:
        """
        If sg_sim exposes room mapping, plug it here.
        Current safe behavior: return None and log that room constraint is not enforced.
        """
        # Example (if you add this later):
        # fn = getattr(self.sg_sim, "get_room_for_object_id", None)
        # if callable(fn):
        #     return fn(anchor_object_id)
        return None

    # -------------------------
    # Main step
    # -------------------------

    def get_next_action(self, agent_yaw_rad: float = 0.0, agent_pos_hab: Optional[np.ndarray] = None):
        if agent_pos_hab is None:
            agent_pos_hab = np.array([0, 0, 0], dtype=np.float32)
        else:
            agent_pos_hab = np.asarray(agent_pos_hab, dtype=np.float32)

        # 1) Gather scene data
        objects, frontiers = self._get_scene_data()
        img_path = _safe_latest_image(self._out_path)

        # 2) Resolve anchor (+debug)
        anchor_obj, anchor_name, anchor_dbg = self._resolve_anchor(objects)

        # Default fallback if no anchor
        if anchor_obj is None:
            _write_jsonl(self._vlm_calls_path, {
                "type": "no_anchor",
                "t": self._t,
                "question": self._question,
                "mode": self.answer_mode,
                "anchor_resolution": anchor_dbg,
                "num_objects": len(objects),
                "num_frontiers": len(frontiers),
            })

            # Trace + txt narration for no-anchor
            trace = {
                "t": int(self._t),
                "mode": self.answer_mode,
                "question": self._question,
                "agent": {"pos_hab": agent_pos_hab.tolist(), "yaw_world_rad": float(agent_yaw_rad)},
                "anchor_resolution": anchor_dbg,
                "anchor": None,
                "kernel": None,
                "scoring": None,
                "selector_output": None,
                "guardrails": None,
                "decision_rationale": {"recommended": "explore", "note": "Anchor not resolved."},
                "room_constraint": {"anchor_room_id": None, "room_constraint_enforced": False},
            }
            _write_trace_step(self._out_path, self._t, trace)

            txt = [
                f"--- STEP {self._t} ---",
                f"Q: {self._question}",
                f"Mode: {self.answer_mode}",
                f"Agent: pos_hab={agent_pos_hab.tolist()} yaw={float(agent_yaw_rad):.3f} rad",
                "Anchor resolution FAILED:",
                f"- label_hint={anchor_dbg.get('anchor_label_hint')}",
                f"- strategy={anchor_dbg.get('match_strategy')}",
                f"- candidates={len(anchor_dbg.get('candidates', []))}",
            ]

            if len(frontiers) == 0:
                plan = {
                    "thought": "No anchor resolved and no frontiers available. Lookaround.",
                    "action_type": "lookaround",
                    "chosen_id": "",
                    "target_xyz_hab": [],
                    "answer_text": "",
                    "confidence": 0.0,
                }
                txt.append("Decision: lookaround (no frontiers).")
                _append_trace_txt(self._out_path, txt)

                self._history += f"[t={self._t}] action=lookaround\n"
                self._t += 1
                return None, None, False, 0.0, plan

            fid = str(frontiers[0]["id"])
            plan = {
                "thought": "Anchor not found yet; explore a frontier.",
                "action_type": "goto_frontier",
                "chosen_id": fid,
                "target_xyz_hab": [],
                "answer_text": "",
                "confidence": 0.0,
            }
            txt.append(f"Decision: goto_frontier chosen={fid} (anchor unresolved).")
            _append_trace_txt(self._out_path, txt)

            self._history += f"[t={self._t}] action=goto_frontier chosen_id={fid}\n"
            self._t += 1
            return self.sg_sim.get_position_from_id(fid), fid, False, 0.0, plan

        anchor_pos = np.asarray(anchor_obj["position"], dtype=np.float32)

        # 2.5) Room constraint hook (currently not enforced)
        anchor_room_id = self._get_anchor_room_id(str(anchor_obj.get("id", "")))
        room_trace = {
            "anchor_room_id": anchor_room_id,
            "room_constraint_enforced": False,
            "note": "Room constraint not enforced in this build (no room mapping available).",
        }

        # 3) Kernel VLM (logged) — uses intrinsic yaw if provided
        kernel = get_vlm_spatial_kernel_params(
            image_path=img_path,
            question=self._question,
            anchor_name=anchor_name,
            anchor_pos_hab=anchor_pos,
            agent_pos_hab=agent_pos_hab,
            agent_yaw=agent_yaw_rad,
            anchor_front_yaw_world=self._anchor_front_yaw_world,
            log_jsonl_path=self._vlm_calls_path,
            step_t=self._t,
        )

        dist_m = _parse_q_dist(self._question)

        # 4) MSP score objects + frontiers
        msp_objects, msp_frontiers = self.msp_engine.score_candidates(
            objects=objects,
            frontiers=frontiers,
            anchor_pos_hab=anchor_pos,
            kernel_params=kernel,
            question_dist=dist_m,
        )

        # 5) Point guess (WHERE mode): anchor + dist * direction (always computed/logged)
        dir_world = _unit_dir_from_theta_phi(float(kernel["theta"]), float(kernel["phi"]))
        point_xyz = (anchor_pos + float(dist_m) * dir_world).astype(np.float32)

        point_logp = self.msp_engine.score_point(
            point_hab=point_xyz,
            anchor_pos_hab=anchor_pos,
            kernel_params=kernel,
            question_dist=dist_m,
            candidate_size=[0.5, 0.5, 0.5],
        )

        point_guess = {
            "id": "POINT_GUESS",
            "target_xyz_hab": point_xyz.tolist(),
            "msp_score": float(point_logp),
        }

        # Top-K candidates to show VLM
        K_OBJ = int(getattr(self.cfg, "selector_topk_objects", 12))
        K_FR  = int(getattr(self.cfg, "selector_topk_frontiers", 8))
        top_objects = msp_objects[:K_OBJ]
        top_frontiers = msp_frontiers[:K_FR]

        # Save exact selector context for auditing
        selector_context = {
            "t": self._t,
            "mode": self.answer_mode,
            "question": self._question,
            "agent": {
                "pos_hab": agent_pos_hab.tolist(),
                "yaw_world_rad": float(agent_yaw_rad),
            },
            "anchor_resolution": anchor_dbg,
            "anchor": {
                "id": anchor_obj["id"],
                "name": anchor_name,
                "pos_hab": anchor_pos.tolist(),
                "anchor_front_yaw_world": self._anchor_front_yaw_world,
                "anchor_room_id": anchor_room_id,
            },
            "kernel": kernel,
            "dist_m": dist_m,
            "point_guess": point_guess,
            "top_objects": [
                {"id": o["id"], "name": o.get("name",""), "msp_score": float(o.get("msp_score",0.0)), "pos_hab": o.get("position")}
                for o in top_objects
            ],
            "top_frontiers": [
                {"id": f["id"], "msp_score": float(f.get("msp_score",0.0)), "pos_hab": f.get("position")}
                for f in top_frontiers
            ],
        }
        selector_context_path = self._out_path / f"selector_context_step_{self._t}.json"
        _write_json(selector_context_path, selector_context)

        # 6) Selector VLM call (logged)
        agent_state_str = self.sg_sim.get_current_semantic_state_str()
        prompt = self._build_selector_prompt(
            agent_state=agent_state_str,
            anchor_name=anchor_name,
            anchor_pos_hab=anchor_pos,
            kernel=kernel,
            dist_m=dist_m,
            top_objects=top_objects,
            top_frontiers=top_frontiers,
            point_guess=point_guess,  # show it always; WHICH prompt explicitly forbids selecting it
        )

        raw_text = ""
        try:
            plan, raw_text = self._call_selector_llm(prompt)
            _write_jsonl(self._vlm_calls_path, {
                "type": "selector",
                "t": self._t,
                "mode": self.answer_mode,
                "prompt": prompt,
                "raw_response_text": raw_text,
                "parsed": plan,
            })
        except Exception as e:
            _write_jsonl(self._vlm_calls_path, {
                "type": "selector_error",
                "t": self._t,
                "mode": self.answer_mode,
                "error": str(e),
                "raw_response_text": raw_text,
            })

            # fallback: choose best object (or frontier if none)
            if len(top_objects) > 0:
                plan = {
                    "thought": f"Selector failed; fallback to best object. Error={e}",
                    "action_type": "answer" if self.answer_mode == "which" else "goto_object",
                    "chosen_id": str(top_objects[0]["id"]),
                    "target_xyz_hab": [],
                    "answer_text": f"Fallback best object: {top_objects[0].get('name','')}",
                    "confidence": 0.2,
                }
            elif len(top_frontiers) > 0:
                plan = {
                    "thought": f"Selector failed; fallback to best frontier. Error={e}",
                    "action_type": "goto_frontier",
                    "chosen_id": str(top_frontiers[0]["id"]),
                    "target_xyz_hab": [],
                    "answer_text": "",
                    "confidence": 0.0,
                }
            else:
                plan = {
                    "thought": f"Selector failed; lookaround. Error={e}",
                    "action_type": "lookaround",
                    "chosen_id": "",
                    "target_xyz_hab": [],
                    "answer_text": "",
                    "confidence": 0.0,
                }

        # 7) Enforce mode constraints (hard guardrail)
        chosen_id = str(plan.get("chosen_id", "")).strip()
        action = str(plan.get("action_type", "goto_frontier")).strip()

        allowed_object_ids = set([str(o["id"]) for o in top_objects])
        allowed_frontier_ids = set([str(f["id"]) for f in top_frontiers])

        if self.answer_mode == "which":
            # POINT_GUESS is forbidden; chosen_id must be in object ids
            if chosen_id == "POINT_GUESS" or chosen_id not in allowed_object_ids:
                if len(top_objects) > 0:
                    forced = top_objects[0]
                    plan["thought"] = f"[guardrail WHICH] Forced to best object because chosen_id={chosen_id} invalid. " + plan.get("thought","")
                    plan["chosen_id"] = str(forced["id"])
                    plan["target_xyz_hab"] = []
                    chosen_id = str(forced["id"])
                    if action == "answer":
                        plan["answer_text"] = plan.get("answer_text","") or f"{forced.get('name','object')} (id={forced['id']})"
                    else:
                        plan["action_type"] = "goto_object"
                        action = "goto_object"
                else:
                    plan["thought"] = f"[guardrail WHICH] No objects available; switching to lookaround."
                    plan["action_type"] = "lookaround"
                    plan["chosen_id"] = ""
                    plan["target_xyz_hab"] = []
                    action = "lookaround"
                    chosen_id = ""
        else:
            # WHERE mode: if chosen_id is POINT_GUESS, ensure target_xyz_hab is filled
            if chosen_id == "POINT_GUESS":
                plan["target_xyz_hab"] = point_guess["target_xyz_hab"]
            else:
                # if it chose a non-existent id, snap to best frontier
                if chosen_id and (chosen_id not in allowed_object_ids) and (chosen_id not in allowed_frontier_ids):
                    if len(top_frontiers) > 0:
                        plan["thought"] = f"[guardrail WHERE] Invalid chosen_id={chosen_id}. Forced to best frontier."
                        plan["action_type"] = "goto_frontier"
                        plan["chosen_id"] = str(top_frontiers[0]["id"])
                        plan["target_xyz_hab"] = []
                        action = "goto_frontier"
                        chosen_id = str(top_frontiers[0]["id"])

        # 8) Convert chosen to target_pose (for goto actions)
        target_pose = None
        target_id = None

        if action == "goto_object":
            target_id = chosen_id
            if target_id:
                try:
                    target_pose = self.sg_sim.get_position_from_id(target_id)
                except Exception as e:
                    print(f"[MSP] Failed to get pose for object {target_id}: {e}")

        elif action == "goto_frontier":
            target_id = chosen_id
            if target_id:
                try:
                    target_pose = self.sg_sim.get_position_from_id(target_id)
                except Exception as e:
                    print(f"[MSP] Failed to get pose for frontier {target_id}: {e}")

        # 9) Decide confidence / termination
        conf = float(plan.get("confidence", 0.0))
        is_answer = (action == "answer")

        best_obj = top_objects[0] if len(top_objects) > 0 else None
        plan["selector"] = {
            "mode": self.answer_mode,
            "chosen_id": plan.get("chosen_id", ""),
            "answer_type": ("point" if plan.get("chosen_id","") == "POINT_GUESS" else "object"),
            "confidence": conf,
            "point_guess": point_guess,
            "best_object": (
                {
                    "id": str(best_obj["id"]),
                    "name": best_obj.get("name",""),
                    "msp_score": float(best_obj.get("msp_score",0.0)),
                    "target_xyz_hab": best_obj.get("position"),
                } if best_obj else None
            ),
            "topk_objects": [
                {
                    "id": str(o["id"]),
                    "name": o.get("name",""),
                    "msp_score": float(o.get("msp_score",0.0)),
                    "target_xyz_hab": o.get("position"),
                } for o in top_objects[:8]
            ],
            "topk_frontiers": [
                {
                    "id": str(f["id"]),
                    "msp_score": float(f.get("msp_score",0.0)),
                    "target_xyz_hab": f.get("position"),
                } for f in top_frontiers[:6]
            ],
        }

        # -------------------------
        # NEW: Comprehensive step trace (JSON + readable txt)
        # -------------------------

        kernel_trace = {
            "used_intrinsic_front": (self._anchor_front_yaw_world is not None),
            "anchor_front_yaw_world": self._anchor_front_yaw_world,
            "kernel_theta_world": float(kernel.get("theta", 0.0)),
            "kernel_phi": float(kernel.get("phi", math.pi / 2.0)),
            "kappa": float(kernel.get("kappa", 0.0)),
            "reasoning": kernel.get("reasoning", ""),
            "debug": kernel.get("debug", {}),
            "image_path": img_path,
        }

        score_trace = {
            "dist_m": float(dist_m),
            "point_guess": {"xyz_hab": point_xyz.tolist(), "score": float(point_logp)},
            "objects": _summarize_rank_delta(msp_objects, k=8),
            "frontiers": _summarize_rank_delta(msp_frontiers, k=6),
        }

        obj_gap = score_trace["objects"]["gap_1_2"]
        decision_rationale = {
            "object_gap_1_2": obj_gap,
            "recommended": "answer" if (obj_gap is not None and obj_gap > 2.0) else "explore",
            "note": "Heuristic recommendation; selector LLM makes final decision.",
        }

        guardrail_applied = ("[guardrail" in str(plan.get("thought", "")))

        trace = {
            "t": int(self._t),
            "mode": self.answer_mode,
            "question": self._question,
            "agent": {
                "pos_hab": agent_pos_hab.tolist(),
                "yaw_world_rad": float(agent_yaw_rad),
            },
            "anchor_resolution": anchor_dbg,
            "anchor": {
                "id": str(anchor_obj.get("id", "")),
                "name": anchor_name,
                "pos_hab": anchor_pos.tolist(),
            },
            "kernel": kernel_trace,
            "scoring": score_trace,
            "selector_prompt_path": str(selector_context_path),
            "selector_output": {
                "action_type": action,
                "chosen_id": chosen_id,
                "confidence": conf,
                "answer_text": plan.get("answer_text", ""),
            },
            "guardrails": {
                "applied": bool(guardrail_applied),
                "thought": plan.get("thought", ""),
            },
            "decision_rationale": decision_rationale,
            "room_constraint": room_trace,
        }
        _write_trace_step(self._out_path, self._t, trace)

        # Readable narration
        kdbg = kernel_trace.get("debug", {}) or {}
        txt: List[str] = []
        txt.append(f"--- STEP {self._t} ---")
        txt.append(f"Q: {self._question}")
        txt.append(f"Mode: {self.answer_mode}")
        txt.append(f"Agent: pos_hab={agent_pos_hab.tolist()} yaw={float(agent_yaw_rad):.3f} rad")

        txt.append("Anchor resolution:")
        txt.append(f"- label_hint={anchor_dbg.get('anchor_label_hint')}")
        txt.append(f"- strategy={anchor_dbg.get('match_strategy')}")
        cand_list = anchor_dbg.get("candidates", []) or []
        if cand_list:
            txt.append(f"- candidates_found={len(cand_list)} (showing up to 5):")
            for c in cand_list[:5]:
                txt.append(f"  * {c.get('id')} name={c.get('name')} pos={c.get('pos_hab')}")
        txt.append(f"- chosen={anchor_dbg.get('chosen')}")

        txt.append("Kernel:")
        txt.append(f"- used_intrinsic_front={kernel_trace['used_intrinsic_front']} front_yaw_world={kernel_trace['anchor_front_yaw_world']}")
        txt.append(f"- front_source={kdbg.get('front_source')}")
        txt.append(f"- theta_prior={kdbg.get('theta_prior')} yaw_front_world={kdbg.get('yaw_front_world')}")
        txt.append(f"- theta_final(world)={kernel_trace['kernel_theta_world']:.4f} phi={kernel_trace['kernel_phi']:.4f} kappa={kernel_trace['kappa']:.2f}")
        if "theta_vlm_world" in kdbg:
            txt.append(f"- vlm_theta_world={kdbg.get('theta_vlm_world'):.4f} fusion_weight={kdbg.get('fusion_weight'):.3f}")
        txt.append(f"- vlm_reasoning: {_shorten(kernel_trace.get('reasoning',''), 260)}")

        txt.append("MSP scoring:")
        txt.append(f"- point_guess xyz={score_trace['point_guess']['xyz_hab']} score={score_trace['point_guess']['score']:.3f}")
        txt.append(f"- top object gap(1-2)={obj_gap}")
        for r in score_trace["objects"]["topk"][:3]:
            txt.append(f"  * obj {r['id']} name={r['name']} score={r['score']:.3f} pos={r['pos_hab']}")
        for r in score_trace["frontiers"]["topk"][:2]:
            txt.append(f"  * frontier {r['id']} score={r['score']:.3f} pos={r['pos_hab']}")

        txt.append("Room constraint:")
        txt.append(f"- anchor_room_id={room_trace.get('anchor_room_id')} enforced={room_trace.get('room_constraint_enforced')}")
        txt.append(f"- note={room_trace.get('note')}")

        txt.append("Decision:")
        txt.append(f"- heuristic_recommendation={decision_rationale.get('recommended')} (gap={obj_gap})")
        txt.append(f"- selector_action={action} chosen_id={chosen_id} conf={conf:.2f}")
        if plan.get("answer_text"):
            txt.append(f"- answer_text: {plan.get('answer_text')}")
        if guardrail_applied:
            txt.append(f"- guardrail_applied: {_shorten(plan.get('thought',''), 220)}")
        else:
            txt.append(f"- selector_thought: {_shorten(plan.get('thought',''), 220)}")

        _append_trace_txt(self._out_path, txt)

        # Keep your compact history line as well (nice for quick greps)
        self._history += (
            f"[t={self._t}] mode={self.answer_mode} action={action} chosen={plan.get('chosen_id','')} "
            f"conf={conf:.2f} kernel(theta={kernel['theta']:.2f},kappa={kernel['kappa']:.1f}) "
            f"front_yaw_world={self._anchor_front_yaw_world} "
            f"pg_score={point_guess['msp_score']:.2f} "
            f"best_obj={(best_obj.get('name','') if best_obj else 'none')} "
            f"best_obj_score={(best_obj.get('msp_score',0.0) if best_obj else 0.0):.2f}\n"
        )

        # Also keep the previous consolidated file writer (harmless, but now we also append)
        self._outputs_to_save.append(self._history)
        try:
            with open(self._out_path / "llm_outputs_smart_compact.txt", "w") as f:
                f.write("\n".join(self._outputs_to_save))
        except Exception as e:
            print(f"[MSP PLANNER] Could not write compact logs: {e}")

        self._t += 1

        is_confident = is_answer and (conf >= 0.90)
        return target_pose, target_id, is_confident, conf, plan