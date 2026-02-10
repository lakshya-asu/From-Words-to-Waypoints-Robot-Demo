"""
PLANNER V7: VLM PREDICATE (DEBUG MODE)
Includes aggressive logging to diagnose why VLM outputs are static.
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

# --- HELPERS ---
def _b64encode_image(p):
    try:
        with open(p, "rb") as f: return base64.b64encode(f.read()).decode("utf-8")
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
                response_mime_type="application/json", temperature=0.2, response_schema=spatial_schema
            )
        )
        d = json.loads(resp.text)
        print(f"[VLM RESPONSE] Theta: {d['theta_radians']:.3f}, Phi: {d['phi_radians']:.3f}, Reason: {d.get('reasoning', '')[:50]}...")
        return d["theta_radians"], d["phi_radians"], d["kappa"], d.get("reasoning", "")
    except Exception as e:
        print(f"[VLM ERROR] API Failed: {e}") 
        return None, None, None, str(e)

# ===================== MSP Bridge V7 =====================
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
        from metric_semantic_predicate.models.bnn_metric_model import bnn_predict_metric, BayesianNN_metric, guide_bnn_metric
        from metric_semantic_predicate.models.bnn_semantic_model import bnn_predict_sem, BayesianNNSem, guide_bnn_sem
        return {
            "torch": torch, "pyro": pyro,
            "load_model": load_model, "prepare_dataset": prepare_dataset,
            "bnn_predict_metric": bnn_predict_metric, "BayesianNN_metric": BayesianNN_metric, "guide_bnn_metric": guide_bnn_metric,
            "bnn_predict_sem": bnn_predict_sem, "BayesianNNSem": BayesianNNSem, "guide_bnn_sem": guide_bnn_sem,
        }
    except: return None

class MSPBridgeV7:
    def __init__(self, cfg: MSPConfig, vlm_model):
        self.cfg = cfg
        self.vlm_model = vlm_model
        self._stack = _try_import_msp_stack()
        self._sem_ohe_cols = None
        self._ready = False
        self._try_init_models()

    def _try_init_models(self):
        if not self.cfg.enabled or not self._stack: return
        try:
            with open(self.cfg.semantic_ohe, "r") as f:
                import json as _json
                self._sem_ohe_cols = _json.load(f)
            self._ready = True
        except: pass

    def _get_metric_semantic_params(self, anchor, candidate, distance_m):
        # Placeholder for BNN logic
        w, d, h = [float(x) for x in (candidate.get("size") or [0.5]*3)[:3]]
        max_dim = max(w,d,h)
        return {
            "mu_x": float(candidate["position"][0]), "mu_y": float(candidate["position"][1]), "mu_z": float(candidate["position"][2]),
            "sigma_s": 0.5 * max_dim,
            "x0": float(anchor[0]), "y0": float(anchor[1]), "z0": float(anchor[2]),
            "d0": float(distance_m), "sigma_m": 0.3 * max_dim
        }

    def score_candidates(self, question, reference_hab, dataset_candidates, sg_objects, image_path, ref_name, agent_yaw):
        dist_m = _parse_q_dist_only(question)
        rows = []

        # 1. Ask VLM for Angles
        vlm_theta, vlm_phi, vlm_kappa, reasoning = get_vlm_spatial_params(self.vlm_model, image_path, question, ref_name)
        
        # 3. DEBUG: Check for Fallback
        if vlm_theta is None: 
            print("[VLM FAIL] Using Default Fallback Values")
            vlm_theta, vlm_phi, vlm_kappa = 1.57, 1.57, 5.0 
            reasoning = f"VLM Failed. Error: {reasoning}"

        # 2. Convert to Global
        global_theta = vlm_theta + agent_yaw
        global_phi = vlm_phi 

        # Merge Cands
        all_cands = []
        for c in (dataset_candidates or []): all_cands.append({**c, "source": "dataset", "cls": str(c.get("name","obj")).lower()})
        for (gid, cls, pos) in (sg_objects or []): all_cands.append({"name": cls, "id": gid, "position": pos.tolist(), "size":[0.5]*3, "source":"graph", "cls": str(cls).lower()})

        for c in all_cands:
            pos = np.asarray(c.get("position"), np.float32)
            base_params = self._get_metric_semantic_params(reference_hab, c, dist_m)
            params = {**base_params, "theta0": global_theta, "phi0": global_phi, "kappa": vlm_kappa}
            
            logp = float(_combined_logpdf(np.array([pos[0]]), np.array([pos[1]]), np.array([pos[2]]), params)[0])
            
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
        return rows, (rows[0]["declared"] if rows else None), reasoning

# ===================== Planner V7 =====================
class VLMPlannerEQAGeminiSpatialV7:
    def __init__(self, cfg, sg_sim, question, gt, out_path, **kwargs):
        try:
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
            self.model = genai.GenerativeModel("models/gemini-3-pro-preview")
        except: raise RuntimeError("API Key missing")

        self.msp = MSPBridgeV7(MSPConfig(**cfg.msp), vlm_model=self.model)
        self._out_path = Path(out_path)
        self._question = question
        self._t = 0
        self.sg_sim = sg_sim
        self.ref_pos_hab = np.asarray(kwargs.get("reference_object",{}).get("position",[0,0,0]), np.float32)
        self.ref_name = str(kwargs.get("reference_object",{}).get("name","object")).lower()
        self.dataset_candidates = kwargs.get("candidate_targets", [])
        self._outputs_to_save = [f'Question: {self._question}\n']

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
                if pos_norm is None: continue
                pos_hab = np.asarray(pos_normal_to_habitat(np.array(pos_norm, np.float32)), np.float32)
                out.append((oid, str(nm).lower(), pos_hab))
            except: continue
        return out

    def get_next_action(self, agent_yaw_rad=0.0):
        graph_objs = self._current_graph_objects()
        
        # CORRECT VARIABLE NAME USED HERE
        img_path = _safe_latest_image(self._out_path)
        
        rows, best_id, reasoning = self.msp.score_candidates(
            self._question, self.ref_pos_hab, self.dataset_candidates, graph_objs,
            image_path=img_path, ref_name=self.ref_name, agent_yaw=agent_yaw_rad
        )
        
        if rows:
            self._log_params_to_csv(rows, "v7_gemini")

        # LOGGING
        top_cands_str = ""
        if rows:
            top5 = rows[:5]
            top_cands_str = "\n".join([f"  {r['declared']}: logp={r['logp']:.2f} (src={r['source']})" for r in top5])

        log_entry = (
            f"\n--- Step {self._t} ---\n"
            f"VLM Reasoning: {reasoning}\n"
            f"Agent Yaw: {agent_yaw_rad:.3f}\n"
            f"Top Candidates:\n{top_cands_str}\n"
            f"Selected: {best_id}\n"
        )
        self._outputs_to_save.append(log_entry)
        
        try:
            with open(self._out_path / "llm_outputs_v7.txt", "w") as f:
                f.write("\n".join(self._outputs_to_save))
        except Exception as e:
            print(f"Error saving LLM logs: {e}")

        target_pose, target_id = None, None
        is_confident = False
        
        if rows:
            if rows[0]["source"] == "graph":
                target_id = best_id
                try: target_pose = self.sg_sim.get_position_from_id(target_id)
                except: pass
            
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