"""
PLANNER V6: PURE BNN PREDICATE
Uses trained BNN models for all spatial parameters.
"""
from __future__ import annotations
import json
import os
import time
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

# --- HELPER FUNCTIONS ---
def _b64encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def _safe_latest_image(output_path: Path) -> Optional[str]:
    try:
        p = get_latest_image(output_path)
        if p is None: return None
        p = Path(p)
        return str(p) if p.exists() else None
    except Exception: return None

def _safe_float(x, default=0.0) -> float:
    try: return float(x)
    except Exception: return default

def _parse_q(question: str) -> Tuple[Optional[float], Optional[str]]:
    import re
    ql = question.lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*meters?", ql)
    dist = float(m.group(1)) if m else None
    for k, v in {
        "in front of": "front", "front of": "front", "behind": "behind",
        "to the left of": "left", "left of": "left",
        "to the right of": "right", "right of": "right",
        "above": "above", "on top of": "above", "below": "below"
    }.items():
        if k in ql: return dist, v
    return dist, None

def _relation_to_angles(rel: Optional[str]) -> Tuple[float, float]:
    # Hardcoded heuristic angles for V6 (BNN Fallback/Training labels)
    if rel == "left":   return (np.pi, np.pi/2)
    if rel == "right":  return (0.0,   np.pi/2)
    if rel == "front":  return (np.pi/2, np.pi/2)
    if rel == "behind": return (3*np.pi/2, np.pi/2)
    if rel == "above":  return (0.0, 0.0)      # Phi=0 is Up
    if rel == "below":  return (0.0, np.pi)    # Phi=pi is Down
    return (0.0, np.pi/2)

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
        from metric_semantic_predicate.models.bnn_predicate_model import (
            bnn_predict_predicate, BayesianNNPred, guide_bnn_pred
        )
        return {
            "torch": torch, "pyro": pyro,
            "load_model": load_model, "prepare_dataset": prepare_dataset,
            "bnn_predict_metric": bnn_predict_metric, "BayesianNN_metric": BayesianNN_metric, "guide_bnn_metric": guide_bnn_metric,
            "bnn_predict_sem": bnn_predict_sem, "BayesianNNSem": BayesianNNSem, "guide_bnn_sem": guide_bnn_sem,
            "bnn_predict_predicate": bnn_predict_predicate, "BayesianNNPred": BayesianNNPred, "guide_bnn_pred": guide_bnn_pred,
        }
    except Exception:
        return None

# ===================== MSP Bridge V6 (BNN) =====================
@dataclass
class MSPConfig:
    enabled: bool = True
    mode: str = "prior"
    model_dir: Optional[str] = None
    metric_run: Optional[str] = None
    semantic_run: Optional[str] = None
    predicate_run: Optional[str] = None
    semantic_ohe: Optional[str] = None
    device: Optional[str] = None
    alpha: float = 0.6
    vlm_temp: float = 1.0
    eps: float = 1e-9

class MSPBridgeV6:
    def __init__(self, cfg: MSPConfig):
        self.cfg = cfg
        self._stack = _try_import_msp_stack()
        self._sem_ohe_cols = None
        self._ready = False
        self._try_init_models()

    def _try_init_models(self):
        if not self.cfg.enabled or not self._stack: return
        env = os.environ
        self.cfg.model_dir     = self.cfg.model_dir     or env.get("MSP_MODEL_DIR")
        self.cfg.metric_run    = self.cfg.metric_run    or env.get("MSP_METRIC_RUN")
        self.cfg.semantic_run  = self.cfg.semantic_run  or env.get("MSP_SEMANTIC_RUN")
        self.cfg.predicate_run = self.cfg.predicate_run or env.get("MSP_PREDICATE_RUN")
        self.cfg.semantic_ohe  = self.cfg.semantic_ohe  or env.get("MSP_SEM_OHE")

        req = [self.cfg.model_dir, self.cfg.metric_run, self.cfg.semantic_run, self.cfg.predicate_run, self.cfg.semantic_ohe]
        if not all(req): return

        try:
            with open(self.cfg.semantic_ohe, "r") as f:
                import json as _json
                self._sem_ohe_cols = _json.load(f)
            self._ready = True
        except Exception: return

    def _predict_params_models(self, *, anchor, candidate, distance_m, rel) -> Optional[Dict[str, float]]:
        try:
            st = self._stack
            torch, pyro = st["torch"], st["pyro"]
            from collections import OrderedDict
            row = OrderedDict({
                "x0": float(anchor[0]), "y0": float(anchor[1]), "z0": float(anchor[2]),
                "metric": float(distance_m if distance_m is not None else 0.0),
                "semantic": str(candidate.get("cls", "object")).lower(),
                "width": float((candidate.get("size") or [0.5,0.5,0.5])[0]),
                "depth": float((candidate.get("size") or [0.5,0.5,0.5])[1]),
                "height": float((candidate.get("size") or [0.5,0.5,0.5])[2]),
                "predicate": str(rel or ""),
            })

            # Features
            X_metric, *_ = st["prepare_dataset"]([row], ["x0","y0","z0","metric"], target_cols=["P_combined"], fit=False)
            X_sem,    *_ = st["prepare_dataset"]([row], ["semantic","width","height","depth"], target_cols=["P_combined"],
                                                 fit=False, categorical_cols=["semantic"], ohe_fit_columns=self._sem_ohe_cols)
            X_pred,   *_ = st["prepare_dataset"]([row], ["predicate","width","height","depth"], target_cols=["P_combined"], fit=False)

            device = torch.device(self.cfg.device) if self.cfg.device else (
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
            X_metric_t = torch.tensor(X_metric, dtype=torch.float32, device=device)
            X_sem_t    = torch.tensor(X_sem,    dtype=torch.float32, device=device)
            X_pred_t   = torch.tensor(X_pred,   dtype=torch.float32, device=device)

            # Nets & Guides (simplified for inference)
            net_metric = st["BayesianNN_metric"](input_dim=X_metric_t.shape[1], hidden_dim=32,  output_dim=2).to(device)
            net_sem    = st["BayesianNNSem"](   input_dim=X_sem_t.shape[1],    hidden_dim=32,  output_dim=4).to(device)
            net_pred   = st["BayesianNNPred"](  input_dim=X_pred_t.shape[1],   hidden_dim=128, output_dim=3).to(device)

            # Load
            pyro.clear_param_store()
            st["guide_bnn_metric"](torch.zeros(1, X_metric_t.shape[1], device=device), torch.zeros(1, 2, device=device), net_metric)
            st["load_model"](self.cfg.metric_run, save_dir=self.cfg.model_dir)
            state_metric = pyro.get_param_store().get_state()

            pyro.clear_param_store()
            st["guide_bnn_sem"](torch.zeros(1, X_sem_t.shape[1], device=device), torch.zeros(1, 4, device=device), net_sem)
            st["load_model"](self.cfg.semantic_run, save_dir=self.cfg.model_dir)
            state_sem = pyro.get_param_store().get_state()

            pyro.clear_param_store()
            st["guide_bnn_pred"](torch.zeros(1, X_pred_t.shape[1], device=device), torch.zeros(1, 3, device=device), net_pred)
            st["load_model"](self.cfg.predicate_run, save_dir=self.cfg.model_dir)
            state_pred = pyro.get_param_store().get_state()

            # Predict
            pyro.get_param_store().set_state(state_metric)
            mean_metric, _ = st["bnn_predict_metric"](X_metric_t, net_metric)
            pyro.get_param_store().set_state(state_sem)
            mean_sem, _    = st["bnn_predict_sem"](X_sem_t, net_sem)
            pyro.get_param_store().set_state(state_pred)
            mean_pred, _   = st["bnn_predict_predicate"](X_pred_t, net_pred)

            d0, sigma_m = float(mean_metric[0,0]), float(max(1e-12, mean_metric[0,1]))
            mu_x, mu_y, mu_z, sigma_s = (float(mean_sem[0,0]), float(mean_sem[0,1]),
                                         float(mean_sem[0,2]), float(max(1e-12, mean_sem[0,3])))
            theta0, phi0, logk = float(mean_pred[0,0]), float(mean_pred[0,1]), float(mean_pred[0,2])
            kappa = float(np.exp(logk))

            return {
                "mu_x": mu_x, "mu_y": mu_y, "mu_z": mu_z, "sigma_s": sigma_s,
                "x0": float(anchor[0]), "y0": float(anchor[1]), "z0": float(anchor[2]),
                "d0": d0, "sigma_m": sigma_m,
                "theta0": theta0, "phi0": phi0, "kappa": kappa
            }
        except Exception:
            return None

    def _analytic_params(self, *, anchor, candidate, distance_m, rel) -> Dict[str, float]:
        # Fallback if BNN not ready
        w, d, h = [float(x) for x in (candidate.get("size") or [0.5, 0.5, 0.5])[:3]]
        max_dim = max(1e-6, w, d, h)
        theta0, phi0 = _relation_to_angles(rel)
        return {
            "mu_x": float(candidate["position"][0]), "mu_y": float(candidate["position"][1]), "mu_z": float(candidate["position"][2]),
            "sigma_s": 0.5 * max_dim,
            "x0": float(anchor[0]), "y0": float(anchor[1]), "z0": float(anchor[2]),
            "d0": float(distance_m if distance_m else 0.0), "sigma_m": 0.3 * max_dim,
            "theta0": float(theta0), "phi0": float(phi0), "kappa": 10.0 / max_dim
        }

    def score_candidates(self, question, reference_hab, dataset_candidates, sg_objects):
        dist_m, rel = _parse_q(question)
        rows = []
        
        all_cands = []
        for c in (dataset_candidates or []): all_cands.append({**c, "source": "dataset", "cls": str(c.get("name","obj")).lower()})
        for (gid, cls, pos) in (sg_objects or []): all_cands.append({"name": cls, "id": gid, "position": pos.tolist(), "size":[0.5]*3, "source":"graph", "cls": str(cls).lower()})

        for c in all_cands:
            pos = np.asarray(c.get("position", [0,0,0]), np.float32)
            if self._ready:
                params = self._predict_params_models(anchor=reference_hab, candidate=c, distance_m=dist_m, rel=rel)
            else:
                params = self._analytic_params(anchor=reference_hab, candidate=c, distance_m=dist_m, rel=rel)
            
            if not params: continue
            
            logp = float(_combined_logpdf(np.array([pos[0]]), np.array([pos[1]]), np.array([pos[2]]), params)[0])
            
            rows.append({
                "declared": c.get("id") or c.get("name"),
                "logp": logp,
                "theta": params["theta0"],
                "phi": params["phi0"],
                "kappa": params["kappa"],
                "source": c["source"]
            })
            
        rows.sort(key=lambda x: x["logp"], reverse=True)
        return rows, rows[0]["declared"] if rows else None

# ===================== Planner V6 =====================
class VLMPlannerEQAGeminiSpatialV6:
    def __init__(self, cfg, sg_sim, question, gt, out_path, **kwargs):
        self.msp = MSPBridgeV6(MSPConfig(
            enabled=True,
            model_dir=cfg.msp.model_dir,
            metric_run=cfg.msp.metric_run,
            semantic_run=cfg.msp.semantic_run,
            predicate_run=cfg.msp.predicate_run,
            semantic_ohe=cfg.msp.semantic_ohe,
            device="cuda"
        ))
        self._out_path = Path(out_path)
        self._question = question
        self._t = 0
        self.sg_sim = sg_sim
        self.ref_pos_hab = np.asarray(kwargs.get("reference_object",{}).get("position",[0,0,0]), np.float32)
        self.dataset_candidates = kwargs.get("candidate_targets", [])

    def _current_graph_objects(self) -> List[Tuple[str, str, np.ndarray]]:
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
        
        # 1. Score with Pure BNN
        rows, best = self.msp.score_candidates(
            self._question, self.ref_pos_hab, self.dataset_candidates, graph_objs
        )
        
        # 2. Log BNN Params
        if rows:
            self._log_params_to_csv(rows, "v6_bnn")

        # 3. Simple navigation logic (Go to best candidate)
        target_pose, target_id = None, None
        is_confident = False
        conf_level = 0.0
        
        if rows:
            best_cand = rows[0]
            if best_cand["source"] == "graph":
                target_id = best_cand["declared"]
                try: target_pose = self.sg_sim.get_position_from_id(target_id)
                except: pass
            
            if len(rows) > 1:
                gap = rows[0]["logp"] - rows[1]["logp"]
                conf_level = min(1.0, gap / 5.0)
                if conf_level > 0.8: is_confident = True
        
        self._t += 1
        return target_pose, target_id, is_confident, conf_level, {"declared_target_object_id": best}

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