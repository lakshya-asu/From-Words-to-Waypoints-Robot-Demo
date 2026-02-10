"""
PLANNER V8: VLM PREDICATE (DEBUG MODE)
Includes aggressive logging to diagnose how VLM + object-front reasoning
produce theta0 / phi0 in the MSP PDF.
"""
from __future__ import annotations
import json
import os
import base64
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import google.generativeai as genai
import pandas as pd
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from graph_eqa.envs.utils import pos_normal_to_habitat
from graph_eqa.utils.data_utils import get_latest_image
from spatial_experiment.msp.pdf import combined_logpdf as _combined_logpdf

# --- HELPERS -----------------------------------------------------------------

def _b64encode_image(p):
    try:
        with open(p, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"[IMAGE ERROR] Could not encode {p}: {e}")
        return None

def _safe_latest_image(p):
    from graph_eqa.utils.data_utils import get_latest_image
    img = get_latest_image(p)
    if img:
        return img
    print(f"[VLM WARNING] No images found in {p}")
    return None

def _parse_q_dist_only(question: str) -> float:
    import re
    ql = question.lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*meters?", ql)
    return float(m.group(1)) if m else 0.0

def _wrap_angle(angle: float) -> float:
    """Wrap angle to [0, 2π)."""
    two_pi = 2.0 * math.pi
    return (angle % two_pi + two_pi) % two_pi

def _camera_theta_to_world(vlm_theta: float, agent_yaw: float) -> float:
    """
    Map VLM theta (camera frame) to world frame.

    Camera frame in the prompt:
      - 0.0 rad  = right side of the image
      - π/2 rad  = forward (into the image)
      - π rad    = left side
      - 3π/2 rad = toward the camera

    Habitat/agent yaw is defined such that 0 rad ≈ "agent forward".
    So we first rotate VLM theta so that 0 rad = camera forward,
    then add the agent yaw.
    """
    # shift so that 0 rad corresponds to "forward"
    theta_forward_frame = vlm_theta - (math.pi / 2.0)
    return _wrap_angle(agent_yaw + theta_forward_frame)

def _estimate_object_front_yaw(reference_hab: np.ndarray, agent_pos_hab: np.ndarray) -> float:
    """
    Heuristic estimate of the reference object's functional FRONT direction in world yaw.

    Assumption:
      If the agent is looking at the front of the object, then the object's front
      roughly points from the object toward the agent.

    We compute yaw(object -> agent) and then flip it by π to get yaw(object_front).
    """
    ref = np.asarray(reference_hab, dtype=np.float32)
    agent = np.asarray(agent_pos_hab, dtype=np.float32)
    dx = agent[0] - ref[0]
    dz = agent[2] - ref[2]
    yaw_obj_to_agent = math.atan2(dz, dx)  # direction object -> agent
    yaw_front_world = _wrap_angle(yaw_obj_to_agent + math.pi)  # front points toward agent
    return yaw_front_world

def _predicate_offset(question: str) -> float:
    """
    Map a linguistic spatial predicate into an offset (in radians) in the object-centric frame.

    Object frame:
      - 0 rad   = along the object's functional front
      - +π/2    = to the LEFT of that front
      - -π/2    = to the RIGHT of that front
      - ±π      = BEHIND the object
    """
    q = question.lower()
    if "left" in q:
        return +math.pi / 2.0
    if "right" in q:
        return -math.pi / 2.0
    if "behind" in q or "back of" in q or "backside" in q:
        return math.pi
    # default: "in front / near / at", no extra offset
    return 0.0

def get_vlm_spatial_params(model, image_path, question, anchor_name):
    # 1. DEBUG: Print Inputs
    print(f"\n[VLM CALL] Query: '{question}' | Ref: '{anchor_name}'")

    # CHANGED: System Prompt to force Object-Centric Reasoning -> Camera Output
    sys_prompt = """
    SYSTEM: You are a Spatial Affordance Reasoning Engine.
    Your goal is to calculate the precise 3D direction vector (Theta, Phi) from the Reference Object to the Target Location.
    
    ### STEP 1: ANALYZE FUNCTIONAL FRONT (Object View)
    Look at the Reference Object. Determine its "Functional Front" based on utility:
    - Chair/Sofa: The side a person sits on.
    - TV/Monitor: The screen side.
    - Cabinet/Fridge: The door side.
    - Bed: The foot of the bed.
    
    ### STEP 2: APPLY THE SPATIAL PREDICATE
    Interpret the query (e.g., "Left of", "Behind") RELATIVE to that Functional Front.
    - "Left of Chair" means to the left-hand side of a person sitting in it.
    - "Behind Chair" means behind the backrest.
    
    ### STEP 3: CONVERT TO CAMERA COORDINATES (Crucial)
    Now, translate that direction into the CAMERA'S perspective (as seen in the image).
    Output the final angle relative to the image frame.
    
    ### COORDINATE DEFINITIONS (Camera Frame)
    * THETA (Horizontal Azimuth):
      - 0.0 rad  = To the Right side of the image.
      - 1.57 rad = Straight Forward (Deep into the image / Away from camera).
      - 3.14 rad = To the Left side of the image.
      - 4.71 rad = Toward the camera.
      
    * PHI (Vertical Elevation):
      - 1.57 rad = LEVEL (Default). Use this for all standard relationships (Left/Right/Front/Behind).
      - 0.0 rad  = ABOVE / ON TOP. Only use if query implies height (e.g. "Above", "On top").
      - 3.14 rad = BELOW. Only use if query implies "Below" or "Under".
      
    * KAPPA (Concentration):
      - High (>20): Specific direction (e.g. "Directly in front").
      - Low (5-10): Vague direction (e.g. "Somewhere to the left").
    """
    
    spatial_schema = genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "reasoning": genai.protos.Schema(type=genai.protos.Type.STRING),
            "theta_radians": genai.protos.Schema(type=genai.protos.Type.NUMBER),
            "phi_radians": genai.protos.Schema(type=genai.protos.Type.NUMBER),
            "kappa": genai.protos.Schema(type=genai.protos.Type.NUMBER)
        },
        required=["reasoning", "theta_radians", "phi_radians", "kappa"]
    )
    
    prompt = f"{sys_prompt}\nQuery: {question}\nRef Object: {anchor_name}\nTask: Output Theta/Phi/Kappa."
    parts = [{"text": prompt}]
    
    # 2. DEBUG: Check Image
    has_image = False
    if image_path and os.path.exists(image_path):
        b64_data = _b64encode_image(image_path)
        if b64_data:
            mime = mimetypes.guess_type(image_path)[0] or "image/png"
            parts.append({"inline_data": {"mime_type": mime, "data": b64_data}})
            has_image = True
            print(f"[VLM CALL] Attached Image: {image_path}")
    
    if not has_image:
        print("[VLM CRITICAL WARNING] Calling VLM without an image! Results will be hallucinations.")

    try:
        resp = model.generate_content(
            [{"role": "user", "parts": parts}],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.2,
                response_schema=spatial_schema
            )
        )
        d = json.loads(resp.text)
        print(
            f"[VLM RESPONSE] Theta(cameras): {d['theta_radians']:.3f}, "
            f"Phi(cameras): {d['phi_radians']:.3f}, "
            f"Kappa: {d['kappa']:.3f}, "
            f"Reason: {d.get('reasoning', '')[:80]}..."
        )
        return d["theta_radians"], d["phi_radians"], d["kappa"], d.get("reasoning", "")
    except Exception as e:
        print(f"[VLM ERROR] API Failed: {e}") 
        return None, None, None, str(e)

# ===================== MSP Bridge V7 =========================================

@dataclass
class MSPConfig:
    enabled: bool = True
    mode: str = "prior"
    model_dir: Optional[str] = None
    metric_run: Optional[str] = None
    semantic_run: Optional[str] = None
    predicate_run: Optional[str] = None 
    semantic_ohe: Optional[str] = None
    device: Optional[str] = "cuda"
    alpha: float = 0.6
    vlm_temp: float = 1.0
    eps: float = 1e-9

def _try_import_msp_stack():
    try:
        import torch
        import pyro
        from metric_semantic_predicate.training.model_io import load_model
        from metric_semantic_predicate.dataset.feature_utils import prepare_dataset
        from metric_semantic_predicate.models.bnn_metric_model import (
            bnn_predict_metric, BayesianNN_metric, guide_bnn_metric
        )
        from metric_semantic_predicate.models.bnn_semantic_model import (
            bnn_predict_sem, BayesianNNSem, guide_bnn_sem
        )
        return {
            "torch": torch, "pyro": pyro,
            "load_model": load_model, "prepare_dataset": prepare_dataset,
            "bnn_predict_metric": bnn_predict_metric, "BayesianNN_metric": BayesianNN_metric,
            "guide_bnn_metric": guide_bnn_metric,
            "bnn_predict_sem": bnn_predict_sem, "BayesianNNSem": BayesianNNSem,
            "guide_bnn_sem": guide_bnn_sem,
        }
    except Exception:
        return None

class MSPBridgeV8:
    def __init__(self, cfg: MSPConfig, vlm_model):
        self.cfg = cfg
        self.vlm_model = vlm_model
        self._stack = _try_import_msp_stack()
        self._sem_ohe_cols = None
        self._ready = False
        self._printed_coord_help = False  # print coordinate-system explainer once
        self._try_init_models()

    def _try_init_models(self):
        if not self.cfg.enabled or not self._stack:
            return
        try:
            with open(self.cfg.semantic_ohe, "r") as f:
                import json as _json
                self._sem_ohe_cols = _json.load(f)
            self._ready = True
        except Exception:
            pass

    def _get_metric_semantic_params(self, anchor, candidate, distance_m):
        # Placeholder for BNN logic – we still expose clear geometry params.
        w, d, h = [float(x) for x in (candidate.get("size") or [0.5]*3)[:3]]
        max_dim = max(w, d, h)
        return {
            "mu_x": float(candidate["position"][0]),
            "mu_y": float(candidate["position"][1]),
            "mu_z": float(candidate["position"][2]),
            "sigma_s": 0.5 * max_dim,
            "x0": float(anchor[0]),
            "y0": float(anchor[1]),
            "z0": float(anchor[2]),
            "d0": float(distance_m),
            "sigma_m": 0.3 * max_dim
        }

    def score_candidates(
        self,
        question: str,
        reference_hab: np.ndarray,
        dataset_candidates,
        sg_objects,
        image_path,
        ref_name: str,
        agent_yaw: float,
        agent_pos_hab: Optional[np.ndarray] = None,
    ):
        """
        Core reasoning step:
          1) Get VLM angles in camera frame.
          2) Estimate object-front yaw in world frame.
          3) Add predicate offset (left/right/behind) in object frame → prior theta.
          4) Map VLM theta to world frame and fuse with prior.
          5) Feed final (theta0, phi0, kappa) into MSP PDF to score candidates.
        """
        if not self._printed_coord_help:
            print("\n[COORD DEBUG] Coordinate systems (printed once):")
            print("  - Habitat/world frame: agent yaw from HabitatInterface.get_heading_angle().")
            print("  - Camera frame (VLM): theta=0 right, π/2 forward, π left, 3π/2 back; phi≈π/2 level.")
            print("  - Object-front frame: 0 along functional front of reference object.")
            print("  - theta0 (MSP): world-azimuth of target relative to reference object.")
            self._printed_coord_help = True

        dist_m = _parse_q_dist_only(question)
        rows: List[Dict[str, Any]] = []

        # 1. Ask VLM for Angles (camera frame)
        vlm_theta, vlm_phi, vlm_kappa, reasoning = get_vlm_spatial_params(
            self.vlm_model, image_path, question, ref_name
        )

        if agent_pos_hab is None:
            print("[MSP V7 WARNING] agent_pos_hab is None; using reference_hab as proxy for agent position.")
            agent_pos_hab = np.asarray(reference_hab, dtype=np.float32)
        else:
            agent_pos_hab = np.asarray(agent_pos_hab, dtype=np.float32)

        reference_hab = np.asarray(reference_hab, dtype=np.float32)

        print("\n[MSP V7] ---- Angle reasoning for this step ----")
        print(f"[MSP V7] Question: {question}")
        print(f"[MSP V7] Reference object name: {ref_name}")
        print(f"[MSP V7] Reference position (hab): {reference_hab}")
        print(f"[MSP V7] Agent position (hab): {agent_pos_hab}")
        print(f"[MSP V7] Agent yaw (world): {agent_yaw:.3f} rad")

        if vlm_theta is not None and vlm_phi is not None and vlm_kappa is not None:
            print(
                f"[MSP V7] Raw VLM (camera frame): "
                f"theta={vlm_theta:.3f}, phi={vlm_phi:.3f}, kappa={vlm_kappa:.3f}"
            )
        else:
            print("[MSP V7] VLM returned no angles; will default to prior-only object-front reasoning.")

        # 2. Estimate object-front orientation in world frame
        yaw_front_world = _estimate_object_front_yaw(reference_hab, agent_pos_hab)
        print(f"[MSP V7] Estimated object FRONT yaw (world): {yaw_front_world:.3f} rad")

        # 3. Predicate offset in object-front frame
        offset = _predicate_offset(question)
        print(f"[MSP V7] Predicate offset (object frame): {offset:.3f} rad")
        theta_prior_world = _wrap_angle(yaw_front_world + offset)
        phi_prior_world = math.pi / 2.0  # level by default
        print(f"[MSP V7] PRIOR theta (world) from object-front+predicate: {theta_prior_world:.3f} rad")
        print(f"[MSP V7] PRIOR phi   (world): {phi_prior_world:.3f} rad")

        # 4. Map VLM theta from camera → world and fuse with prior, if we have VLM output
        if vlm_theta is not None and vlm_phi is not None and vlm_kappa is not None:
            theta_vlm_world = _camera_theta_to_world(vlm_theta, agent_yaw)
            print(f"[MSP V7] VLM theta mapped to WORLD: {theta_vlm_world:.3f} rad")

            # Simple fusion between object-front prior and VLM: 0.5 / 0.5 blend
            fusion_weight = 0.5
            theta0_world = _wrap_angle(
                (1.0 - fusion_weight) * theta_prior_world + fusion_weight * theta_vlm_world
            )
            phi0_world = vlm_phi  # keep VLM's vertical angle if provided
            kappa_final = vlm_kappa
            print(f"[MSP V7] FINAL theta0 (world, fused): {theta0_world:.3f} rad")
            print(f"[MSP V7] FINAL phi0   (from VLM):    {phi0_world:.3f} rad")
            print(f"[MSP V7] FINAL kappa  (from VLM):    {kappa_final:.3f}")
        else:
            # VLM failed → use only object-front prior
            theta0_world = theta_prior_world
            phi0_world = phi_prior_world
            kappa_final = 5.0  # diffuse belief
            print("[MSP V7] Using PRIOR-only angles (no VLM).")
            print(f"[MSP V7] FINAL theta0 (world): {theta0_world:.3f} rad")
            print(f"[MSP V7] FINAL phi0   (world): {phi0_world:.3f} rad")
            print(f"[MSP V7] FINAL kappa  (default): {kappa_final:.3f}")

        # 5. Build candidate list
        all_cands = []
        for c in (dataset_candidates or []):
            all_cands.append({**c, "source": "dataset", "cls": str(c.get("name","obj")).lower()})
        for (gid, cls, pos) in (sg_objects or []):
            all_cands.append({
                "name": cls,
                "id": gid,
                "position": pos.tolist(),
                "size":[0.5]*3,
                "source":"graph",
                "cls": str(cls).lower()
            })

        print(f"[MSP V7] Scoring {len(all_cands)} candidates with combined_logpdf...")

        for c in all_cands:
            pos = np.asarray(c.get("position"), np.float32)
            base_params = self._get_metric_semantic_params(reference_hab, c, dist_m)
            params = {
                **base_params,
                "theta0": theta0_world,
                "phi0": phi0_world,
                "kappa": kappa_final
            }
            
            logp = float(_combined_logpdf(
                np.array([pos[0]]), np.array([pos[1]]), np.array([pos[2]]), params
            )[0])
            
            rows.append({
                "declared": c.get("id") or c.get("name"),
                "logp": logp,
                "theta": params["theta0"],
                "phi": params["phi0"],
                "kappa": params["kappa"],
                "source": c["source"],
                "pos": pos
            })
            
        rows.sort(key=lambda x: x["logp"], reverse=True)

        if rows:
            print("[MSP V7] Top-3 candidates by logp:")
            for r in rows[:3]:
                print(
                    f"  - {r['declared']} (src={r['source']}): "
                    f"logp={r['logp']:.3f}, theta={r['theta']:.3f}, phi={r['phi']:.3f}"
                )
        else:
            print("[MSP V7] No candidates to score!")

        best_declared = rows[0]["declared"] if rows else None
        return rows, best_declared, reasoning

# ===================== Planner V7 ============================================

class VLMPlannerEQAGeminiSpatialV8:
    def __init__(self, cfg, sg_sim, question, gt, out_path, **kwargs):
        try:
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
            self.model = genai.GenerativeModel("models/gemini-3-pro-preview")
        except Exception:
            raise RuntimeError("API Key missing or Gemini configuration failed")

        self.msp = MSPBridgeV8(MSPConfig(**cfg.msp), vlm_model=self.model)
        self._out_path = Path(out_path)
        self._question = question
        self._t = 0
        self.sg_sim = sg_sim
        self.ref_pos_hab = np.asarray(
            kwargs.get("reference_object",{}).get("position",[0,0,0]),
            np.float32
        )
        self.ref_name = str(
            kwargs.get("reference_object",{}).get("name","object")
        ).lower()
        self.dataset_candidates = kwargs.get("candidate_targets", [])
        self._outputs_to_save = [f'Question: {self._question}\n']

        print("\n[VLM PLANNER V7 INIT]")
        print(f"  - Reference object name: {self.ref_name}")
        print(f"  - Reference object position (hab): {self.ref_pos_hab}")
        print(f"  - # dataset candidates: {len(self.dataset_candidates)}")

    @property
    def t(self) -> int:
        return self._t

    def _current_graph_objects(self):
        out = []
        oids = list(getattr(self.sg_sim, "object_node_ids", []) or [])
        nms  = list(getattr(self.sg_sim, "object_node_names", []) or [])
        for oid, nm in zip(oids, nms):
            try:
                pos_norm = self.sg_sim.get_position_from_id(oid)
                if pos_norm is None:
                    continue
                pos_hab = np.asarray(
                    pos_normal_to_habitat(np.array(pos_norm, np.float32)),
                    np.float32
                )
                out.append((oid, str(nm).lower(), pos_hab))
            except Exception:
                continue
        print(f"[VLM PLANNER V7] Retrieved {len(out)} graph objects from SceneGraphSim.")
        return out

    def get_next_action(
        self,
        agent_yaw_rad: float = 0.0,
        agent_pos_hab: Optional[np.ndarray] = None
    ):
        """
        Single planner step:
          - Gathers current SceneGraph objects.
          - Fetches latest image.
          - Calls MSPBridgeV7.score_candidates with full angle reasoning stack.
          - Logs top candidates and saves CSV + text logs.
        """
        graph_objs = self._current_graph_objects()
        
        img_path = _safe_latest_image(self._out_path)
        
        rows, best_id, reasoning = self.msp.score_candidates(
            self._question,
            self.ref_pos_hab,
            self.dataset_candidates,
            graph_objs,
            image_path=img_path,
            ref_name=self.ref_name,
            agent_yaw=agent_yaw_rad,
            agent_pos_hab=agent_pos_hab
        )
        
        if rows:
            self._log_params_to_csv(rows, "v8_gemini")

        # LOGGING
        top_cands_str = ""
        if rows:
            top5 = rows[:5]
            top_cands_str = "\n".join(
                [f"  {r['declared']}: logp={r['logp']:.2f} (src={r['source']})"
                 for r in top5]
            )

        log_entry = (
            f"\n--- Step {self._t} ---\n"
            f"VLM Reasoning (truncated): {reasoning[:200] if isinstance(reasoning, str) else reasoning}\n"
            f"Agent Yaw (world): {agent_yaw_rad:.3f}\n"
            f"Top Candidates:\n{top_cands_str}\n"
            f"Selected: {best_id}\n"
        )
        self._outputs_to_save.append(log_entry)
        
        try:
            with open(self._out_path / "llm_outputs_v8.txt", "w") as f:
                f.write("\n".join(self._outputs_to_save))
        except Exception as e:
            print(f"[VLM PLANNER V7] Error saving LLM logs: {e}")

        target_pose, target_id = None, None
        is_confident = False
        
        if rows:
            if rows[0]["source"] == "graph":
                target_id = best_id
                try:
                    target_pose = self.sg_sim.get_position_from_id(target_id)
                except Exception:
                    target_pose = None
            
            if len(rows) > 1 and (rows[0]["logp"] - rows[1]["logp"] > 2.0):
                is_confident = True
        
        self._t += 1
        return target_pose, target_id, is_confident, 0.9, {"declared_target_object_id": best_id}

    def _log_params_to_csv(self, rows, mode):
        csv_path = self._out_path / "params_comparison.csv"
        data = []
        for r in rows:
            data.append({
                "step": self._t,
                "mode": mode,
                "candidate": r["declared"],
                "theta": r["theta"],
                "phi": r["phi"],
                "kappa": r["kappa"],
                "logp": r["logp"]
            })
        df = pd.DataFrame(data)
        df.to_csv(csv_path, mode='a', header=not csv_path.exists(), index=False)
