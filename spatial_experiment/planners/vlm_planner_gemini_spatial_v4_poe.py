# /home/artemis/project/graph_eqa_swagat/spatial_experiment/planners/vlm_planner_gemini_spatial_v4.py
"""
Gemini planner (v4) with MSP bridge + numeric PoE:

- Adds mode: "poe"  → combine MSP log-likelihoods with VLM candidate probabilities:
      score_i = alpha * log_msp_i + (1 - alpha) * log(p_vlm_i + eps)
  choose argmax, softmax(scores) for confidence.

- Back-compatible with v3 modes: "prior", "fallback", "hard".

Dependencies: same as v3.
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

from graph_eqa.envs.utils import pos_normal_to_habitat
from graph_eqa.utils.data_utils import get_latest_image

# Combined PDF (vectorized logpdf preferred)
from spatial_experiment.msp.pdf import combined_logpdf as _combined_logpdf

import math
import pandas as pd

# set non-interactive backend BEFORE importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------- Optional MSP model stack (Pyro/Torch) ----------
def _try_import_msp_stack():
    try:
        import torch  # noqa: F401
        import pyro   # noqa: F401
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
            "torch": __import__("torch"),
            "pyro": __import__("pyro"),
            "load_model": load_model,
            "prepare_dataset": prepare_dataset,
            "bnn_predict_metric": bnn_predict_metric,
            "BayesianNN_metric": BayesianNN_metric,
            "guide_bnn_metric": guide_bnn_metric,
            "bnn_predict_sem": bnn_predict_sem,
            "BayesianNNSem": BayesianNNSem,
            "guide_bnn_sem": guide_bnn_sem,
            "bnn_predict_predicate": bnn_predict_predicate,
            "BayesianNNPred": BayesianNNPred,
            "guide_bnn_pred": guide_bnn_pred,
        }
    except Exception:
        return None


# ---------- Small helpers ----------
def _b64encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def _entropy(p, eps=1e-12):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def _safe_latest_image(output_path: Path) -> Optional[str]:
    try:
        p = get_latest_image(output_path)
        if p is None:
            return None
        p = Path(p)
        return str(p) if p.exists() else None
    except Exception:
        return None

def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _parse_q(question: str) -> Tuple[Optional[float], Optional[str]]:
    """Distance (meters) + relation bucket ('left','right','front','behind')."""
    import re
    ql = question.lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*meters?", ql)
    dist = float(m.group(1)) if m else None
    for k, v in {
        "in front of": "front", "front of": "front",
        "behind": "behind",
        "to the left of": "left", "left of": "left",
        "to the right of": "right", "right of": "right",
    }.items():
        if k in ql:
            return dist, v
    return dist, None

def _relation_to_angles(rel: Optional[str]) -> Tuple[float, float]:
    """
    Map 'left','right','front','behind' to spherical (theta, phi) in global frame:
    +X right, +Z behind, -Z front, Y up (phi=pi/2 in-plane).
    """
    if rel == "left":   return (np.pi, np.pi/2)
    if rel == "right":  return (0.0,   np.pi/2)
    if rel == "front":  return (np.pi/2, np.pi/2)
    if rel == "behind": return (3*np.pi/2, np.pi/2)
    return (0.0, np.pi/2)

def _dir_bucket(ref_hab: np.ndarray, obj_hab: np.ndarray) -> str:
    d = obj_hab - ref_hab
    dx, dz = float(d[0]), float(d[2])
    if abs(dx) >= abs(dz):
        return "right" if dx > 0 else "left"
    return "front" if dz < 0 else "behind"

def _round3(v): return np.round(v, 3).tolist()


# ===================== MSP Bridge =====================
@dataclass
class MSPConfig:
    enabled: bool = True
    # v3: "prior" | "fallback" | "hard"
    # v4: add "poe"
    mode: str = "prior"
    model_dir: Optional[str] = None
    metric_run: Optional[str] = None
    semantic_run: Optional[str] = None
    predicate_run: Optional[str] = None
    semantic_ohe: Optional[str] = None
    device: Optional[str] = None   # "cuda" | "cpu" | None (auto)

    # ---- NEW (PoE) ----
    alpha: float = 0.6     # weight for MSP log-likelihood
    vlm_temp: float = 1.0  # temperature on VLM distribution
    eps: float = 1e-9      # numerical floor for log()

class MSPBridge:
    """
    Loads MSP submodels if available; otherwise falls back to analytic parameterization.
    Predicts per-candidate combined-PDF scores evaluated at candidate centers (Habitat / DS == world).
    """

    def __init__(self, cfg: MSPConfig):
        self.cfg = cfg
        self._stack = None
        self._sem_ohe_cols = None
        self._ready = False
        self._try_init_models()

    @property
    def ready(self) -> bool:
        return self._ready

    def _try_init_models(self):
        if not self.cfg.enabled:
            return
        # read env defaults if missing
        env = os.environ
        self.cfg.model_dir     = self.cfg.model_dir     or env.get("MSP_MODEL_DIR")
        self.cfg.metric_run    = self.cfg.metric_run    or env.get("MSP_METRIC_RUN")
        self.cfg.semantic_run  = self.cfg.semantic_run  or env.get("MSP_SEMANTIC_RUN")
        self.cfg.predicate_run = self.cfg.predicate_run or env.get("MSP_PREDICATE_RUN")
        self.cfg.semantic_ohe  = self.cfg.semantic_ohe  or env.get("MSP_SEM_OHE")

        # quick sanity
        req = [self.cfg.model_dir, self.cfg.metric_run, self.cfg.semantic_run, self.cfg.predicate_run, self.cfg.semantic_ohe]
        if not all(req):
            return

        stack = _try_import_msp_stack()
        if stack is None:
            return

        # Load OHE column list
        try:
            with open(self.cfg.semantic_ohe, "r") as f:
                import json as _json
                self._sem_ohe_cols = _json.load(f)
        except Exception:
            return

        self._stack = stack
        self._ready = True

    # ---------- Heuristic parameterization (fallback) ----------
    def _analytic_params(self, *, anchor, candidate, distance_m, rel) -> Dict[str, float]:
        w, d, h = [float(x) for x in (candidate.get("size") or [0.5, 0.5, 0.5])[:3]]
        max_dim = max(1e-6, w, d, h)
        sigma_s = 0.5 * max_dim
        sigma_m = 0.3 * max_dim
        kappa   = 10.0 / max_dim
        theta0, phi0 = _relation_to_angles(rel)
        return {
            "mu_x": float(candidate["position"][0]),
            "mu_y": float(candidate["position"][1]),
            "mu_z": float(candidate["position"][2]),
            "sigma_s": float(sigma_s),
            "x0": float(anchor[0]),
            "y0": float(anchor[1]),
            "z0": float(anchor[2]),
            "d0": float(distance_m if distance_m is not None else 0.0),
            "sigma_m": float(sigma_m),
            "theta0": float(theta0),
            "phi0": float(phi0),
            "kappa": float(kappa),
        }

    # ---------- Pyro/Torch prediction per candidate ----------
    def _predict_params_models(self, *, anchor, candidate, distance_m, rel) -> Optional[Dict[str, float]]:
        try:
            st = self._stack
            torch, pyro = st["torch"], st["pyro"]
            from collections import OrderedDict
            row = OrderedDict({
                "x0": float(anchor[0]), "y0": float(anchor[1]), "z0": float(anchor[2]),
                "metric": float(distance_m if distance_m is not None else 0.0),
                "semantic": str(candidate.get("name", "object")).lower(),
                "width": float((candidate.get("size") or [0.5,0.5,0.5])[0]),
                "depth": float((candidate.get("size") or [0.5,0.5,0.5])[1]),
                "height": float((candidate.get("size") or [0.5,0.5,0.5])[2]),
                "predicate": str(rel or ""),
            })

            # Features (mirror training)
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

            # Nets
            net_metric = st["BayesianNN_metric"](input_dim=X_metric_t.shape[1], hidden_dim=32,  output_dim=2).to(device)
            net_sem    = st["BayesianNNSem"](   input_dim=X_sem_t.shape[1],    hidden_dim=32,  output_dim=4).to(device)
            net_pred   = st["BayesianNNPred"](  input_dim=X_pred_t.shape[1],   hidden_dim=128, output_dim=3).to(device)

            # Param stores
            pyro.clear_param_store()
            _ = st["guide_bnn_metric"](torch.zeros(1, X_metric_t.shape[1], device=device),
                                       torch.zeros(1, 2, device=device), net_metric)
            st["load_model"](self.cfg.metric_run, save_dir=self.cfg.model_dir)
            state_metric = pyro.get_param_store().get_state()

            pyro.clear_param_store()
            _ = st["guide_bnn_sem"](torch.zeros(1, X_sem_t.shape[1], device=device),
                                    torch.zeros(1, 4, device=device), net_sem)
            st["load_model"](self.cfg.semantic_run, save_dir=self.cfg.model_dir)
            state_sem = pyro.get_param_store().get_state()

            pyro.clear_param_store()
            _ = st["guide_bnn_pred"](torch.zeros(1, X_pred_t.shape[1], device=device),
                                     torch.zeros(1, 3, device=device), net_pred)
            st["load_model"](self.cfg.predicate_run, save_dir=self.cfg.model_dir)
            state_pred = pyro.get_param_store().get_state()

            # Predict means
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

            x0, y0, z0 = float(anchor[0]), float(anchor[1]), float(anchor[2])
            return {
                "mu_x": mu_x, "mu_y": mu_y, "mu_z": mu_z, "sigma_s": sigma_s,
                "x0": x0, "y0": y0, "z0": z0, "d0": d0, "sigma_m": sigma_m,
                "theta0": theta0, "phi0": phi0, "kappa": kappa
            }
        except Exception:
            return None

    def score_candidates(
        self,
        *,
        question: str,
        reference_hab: np.ndarray,
        dataset_candidates: List[Dict[str, Any]],
        sg_objects: List[Tuple[str, str, np.ndarray]],  # (graph_id, class_name, pos_hab)
    ):
        """
        Returns:
          - msp_rows: List[dict] with fields: {'declared','graph_id','class','pos','size','logp'}
          - best_declared: str | None
        """
        dist_m, rel = _parse_q(question)
        rows: List[Dict[str, Any]] = []

        def eval_logp(params, pos_hab):
            x, y, z = float(pos_hab[0]), float(pos_hab[1]), float(pos_hab[2])
            return float(_combined_logpdf(np.array([x]), np.array([y]), np.array([z]), params)[0])

        # dataset candidates
        for c in (dataset_candidates or []):
            pos = np.asarray(c.get("position", [0,0,0]), np.float32)
            if pos.shape[0] != 3 or not np.isfinite(pos).all():
                continue
            params = (self._predict_params_models(anchor=reference_hab, candidate=c, distance_m=dist_m, rel=rel)
                      if self._ready else
                      self._analytic_params(anchor=reference_hab, candidate=c, distance_m=dist_m, rel=rel))
            if params is None:
                continue
            logp = eval_logp(params, pos)
            rows.append({
                "declared": str(c.get("id") or c.get("name") or "unknown"),
                "graph_id": None,
                "class": str(c.get("name","object")).lower(),
                "pos": pos, "size": c.get("size"),
                "logp": logp, "source": "dataset"
            })

        # SG objects
        for (gid, cls, pos) in (sg_objects or []):
            fake_cand = {"name": cls, "position": pos.tolist(), "size": [0.5,0.5,0.5]}
            params = (self._predict_params_models(anchor=reference_hab, candidate=fake_cand, distance_m=dist_m, rel=rel)
                      if self._ready else
                      self._analytic_params(anchor=reference_hab, candidate=fake_cand, distance_m=dist_m, rel=rel))
            if params is None:
                continue
            logp = eval_logp(params, pos)
            rows.append({
                "declared": str(cls).lower(),
                "graph_id": gid,
                "class": str(cls).lower(),
                "pos": pos, "size": None,
                "logp": logp, "source": "graph"
            })

        if not rows:
            return [], None
        rows.sort(key=lambda r: r["logp"], reverse=True)
        best_declared = rows[0]["declared"]
        return rows, best_declared


# ===================== Planner (Gemini + MSP + PoE) =====================
class VLMPlannerEQAGeminiSpatialV4:
    def __init__(self, cfg, sg_sim, question, ground_truth_target, output_path: Path,
                 reference_object: Optional[Dict[str, Any]] = None,
                 reference_room: Optional[Dict[str, Any]] = None,
                 candidate_targets: Optional[List[Dict[str, Any]]] = None):
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
        self.ref_pos_hab = np.asarray(self.reference_object.get("position", [0,0,0]), np.float32)

        self.gt_id_str = None
        try:
            self.gt_id_str = str(self._ground_truth_target.get("id")).strip()
        except Exception:
            self.gt_id_str = None

        # start from raw candidates and drop the one whose id == GT id
        raw_candidates = list(candidate_targets or [])
        if self.gt_id_str:
            raw_candidates = [c for c in raw_candidates if str(c.get("id")).strip() != self.gt_id_str]
        self.dataset_candidates = raw_candidates
        

        # MSP bridge (+PoE params)
        msp_cfg_dict = getattr(cfg, "msp", {}) or {}
        self.msp = MSPBridge(MSPConfig(
            enabled=bool(msp_cfg_dict.get("enabled", True)),
            mode=str(msp_cfg_dict.get("mode", "prior")).lower(),
            model_dir=msp_cfg_dict.get("model_dir"),
            metric_run=msp_cfg_dict.get("metric_run"),
            semantic_run=msp_cfg_dict.get("semantic_run"),
            predicate_run=msp_cfg_dict.get("predicate_run"),
            semantic_ohe=msp_cfg_dict.get("semantic_ohe"),
            device=msp_cfg_dict.get("device"),
            alpha=float(msp_cfg_dict.get("alpha", 0.6)),
            vlm_temp=float(msp_cfg_dict.get("vlm_temp", 1.0)),
            eps=float(msp_cfg_dict.get("eps", 1e-9)),
        ))

        # VLM (Gemini) — used in modes prior/fallback/poe
        if self.msp.cfg.mode in ("prior", "fallback", "poe"):
            try:
                genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
            except KeyError:
                raise RuntimeError("GOOGLE_API_KEY environment variable not set!")
            self.model = genai.GenerativeModel(
                model_name="models/gemini-2.5-pro-preview-03-25",
                system_instruction=(
                    "You are an expert robot navigator in a 3D indoor environment.\n"
                    "Global frame: Y-up, -Z forward, +X right.\n"
                    "You will receive a numeric table with MSP (Bayesian) prior scores (log-probability) per candidate.\n"
                    "Prefer candidates with the highest logp that also match the relation bucket.\n"
                ),
            )

        self._outputs_to_save: List[str] = [
            f'Question: {self._question}\nGround Truth: {self._ground_truth_target.get("name","?")} '
            f'({self._ground_truth_target.get("id","?")})\n'
        ]
        self.full_plan = ""

    # --- Gemini helpers ---
    def _current_graph_objects(self) -> List[Tuple[str, str, np.ndarray]]:
        out = []
        oids = list(getattr(self.sg_sim, "object_node_ids", []) or [])
        nms  = list(getattr(self.sg_sim, "object_node_names", []) or [])
        for oid, nm in zip(oids, nms):
            try:
                pos_norm = self.sg_sim.get_position_from_id(oid)
                if pos_norm is None:
                    continue
                pos_hab = np.asarray(pos_normal_to_habitat(np.array(pos_norm, np.float32)), np.float32)
                out.append((oid, str(nm).lower(), pos_hab))
            except Exception:
                continue
        return out

    def _format_msp_table(self, rows: List[Dict[str, Any]]) -> str:
        head = "declared\tgraph_id\tclass\tlogp\tdir_bucket\tdist_m\tdx\tdy\tdz"
        lines = [head]
        for r in rows:
            ref = self.ref_pos_hab
            pos = r["pos"]
            d = pos - ref
            dx, dy, dz = float(d[0]), float(d[1]), float(d[2])
            dist = float(np.linalg.norm(d))
            bucket = _dir_bucket(ref, pos)
            lines.append(
                f"{r['declared']}\t{r['graph_id'] or '—'}\t{r['class']}\t"
                f"{r['logp']:.3f}\t{bucket}\t{dist:.3f}\t{dx:.3f}\t{dy:.3f}\t{dz:.3f}"
            )
        return "\n".join(lines)

    def _allowed_declared_names(self, obj_ids: List[str], obj_class_names: List[str]) -> List[str]:
        declared = []
        G = getattr(self.sg_sim, "filtered_netx_graph", None)
        nodes = getattr(G, "nodes", {}) if G is not None else {}
        for oid, base in zip(obj_ids, obj_class_names):
            inst = None
            if G is not None and oid in nodes:
                attrs = nodes[oid]
                inst = attrs.get("instance_id") or attrs.get("hm3d_instance_id") or attrs.get("ins_id")
            base_lc = str(base).lower()
            declared.append(f"{base_lc}_{inst}" if inst is not None and str(inst).strip() else base_lc)
        ds_ids = [str(c.get("id")).strip() for c in self.dataset_candidates if c and c.get("id")]
        if getattr(self, "gt_id_str", None):
            ds_ids = [i for i in ds_ids if i != self.gt_id_str]
        out = list(dict.fromkeys([*(d or "unknown" for d in declared), *ds_ids]))
        return out or ["unknown"]

    def _call_gemini(self, table_str: str, allowed_declared: List[str]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        parts = [{"text": (
            f"Timestep t={self._t}\n"
            f"QUESTION: {self._question}\n"
            f"REFERENCE OBJECT: name={self.ref_name}, pos(HAB)={_round3(self.ref_pos_hab)}\n\n"
            f"MSP PRIOR TABLE (higher logp is better):\n{table_str}\n\n"
            "Choose a declared_name from the allowed list that best matches relation + distance, "
            "using the MSP prior as guidance.\n"
        )}]
        if self._use_image:
            img_path = _safe_latest_image(self._output_path)
            if img_path:
                mime = mimetypes.guess_type(img_path)[0] or "image/png"
                parts.append({"inline_data": {"mime_type": mime, "data": _b64encode_image(img_path)}})

        contents = [{"role": "user", "parts": parts}]

        frontier_ids = list(getattr(self.sg_sim, "frontier_node_ids", []) or []) or ["no_frontiers_available"]
        room_ids     = list(getattr(self.sg_sim, "room_node_ids", []) or [])     or ["room_unknown"]
        object_ids   = list(getattr(self.sg_sim, "object_node_ids", []) or [])    or ["no_objects_available"]

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
                    enum=allowed_declared,
                    description="Return declared_name (e.g., 'step_306' or 'lamp_31' or class label if no instance)."
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

        for attempt in range(2):
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
                print(f"[Gemini] error: {e} (attempt {attempt+1}/2)")
                time.sleep(6)
        return None, None

    # ---- NEW: request VLM probabilities over the same candidate order ----
    def _vlm_probs_over_candidates(self, table_str: str, declared_names: List[str]) -> Optional[np.ndarray]:
        """
        Ask Gemini to produce a probability distribution over 'declared_names'
        (same order as MSP rows). Returns np.array [N] summing to 1, or None on failure.
        """
        N = len(declared_names)
        schema = genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "probs": genai.protos.Schema(
                    type=genai.protos.Type.ARRAY,
                    items=genai.protos.Schema(type=genai.protos.Type.NUMBER),
                    description=f"Return {N} probabilities in the same order as 'declared_names' below. Must sum to 1.0."
                ),
                "declared_names": genai.protos.Schema(
                    type=genai.protos.Type.ARRAY,
                    items=genai.protos.Schema(type=genai.protos.Type.STRING),
                    description="Echo the candidate names in order (optional).",
                ),
            },
            required=["probs"],
        )

        parts = [{
            "text": (
                f"Timestep t={self._t}\n"
                f"QUESTION: {self._question}\n"
                f"REFERENCE OBJECT: name={self.ref_name}, pos(HAB)={_round3(self.ref_pos_hab)}\n\n"
                f"MSP PRIOR TABLE (higher logp is better):\n{table_str}\n\n"
                "Produce a probability for each candidate in the SAME ORDER as 'declared_names'. "
                "Use both the relation and the MSP logp as signals. "
                "Probabilities must be non-negative and sum to 1.0."
            )
        }]
        if self._use_image:
            img_path = _safe_latest_image(self._output_path)
            if img_path:
                mime = mimetypes.guess_type(img_path)[0] or "image/png"
                parts.append({"inline_data": {"mime_type": mime, "data": _b64encode_image(img_path)}})

        contents = [{"role": "user", "parts": [*parts, {"text": f"declared_names:\n{json.dumps(declared_names)}"}]}]

        for attempt in range(2):
            try:
                resp = self.model.generate_content(
                    contents,
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json",
                        temperature=0.2,
                        response_schema=schema,
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
                probs = np.asarray(data["probs"], dtype=np.float64).reshape(-1)
                if probs.size != len(declared_names):
                    return None
                probs = np.clip(probs, 0.0, None)
                s = probs.sum()
                if not np.isfinite(s) or s <= 0:
                    return None
                return (probs / s).astype(np.float64)
            except Exception as e:
                print(f"[Gemini probs] error: {e} (attempt {attempt+1}/2)")
                time.sleep(3)
        return None

    def _dump_poe_debug(
        self,
        *,
        step_idx: int,
        rows: list,                # [{declared, graph_id, class, pos, size, logp, source}, ...] (your MSP rows)
        p_msp: np.ndarray,         # aligned to rows
        p_vlm: np.ndarray,         # aligned to rows
        p_post: np.ndarray,        # aligned to rows
        logp_msp: np.ndarray,      # aligned to rows
        p_prev: Optional[np.ndarray], # aligned to rows or None
        w_msp: float,
        w_vlm: float,
        T_msp: float,
        T_vlm: float,
        out_dir: Path,
    ):
        out_dir = Path(out_dir) / "poe_debug"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build tidy table (sorted by final posterior desc)
        order = np.argsort(-p_post)
        data = []
        for rank, idx in enumerate(order):
            r = rows[idx]
            data.append({
                "rank": rank + 1,
                "declared": r["declared"],
                "class": r.get("class"),
                "source": r.get("source"),
                "msp_logp": float(logp_msp[idx]),
                "msp_prob": float(p_msp[idx]),
                "vlm_prob": float(p_vlm[idx]),
                "post_prob": float(p_post[idx]),
                "prev_prob": float(p_prev[idx]) if p_prev is not None else None,
            })
        df = pd.DataFrame(data, columns=[
            "rank","declared","class","source",
            "msp_logp","msp_prob","vlm_prob","post_prob","prev_prob"
        ])

        # Save CSV + JSON
        stem = f"step_{step_idx:03d}"
        csv_path = out_dir / f"{stem}.csv"
        json_path = out_dir / f"{stem}.json"
        df.to_csv(csv_path, index=False)
        with open(json_path, "w") as f:
            json.dump({
                "weights": {"w_msp": w_msp, "w_vlm": w_vlm},
                "temperatures": {"T_msp": T_msp, "T_vlm": T_vlm},
                "entropy": {
                    "msp": _entropy(p_msp),
                    "vlm": _entropy(p_vlm),
                    "posterior": _entropy(p_post),
                    "previous": (_entropy(p_prev) if p_prev is not None else None),
                },
                "order": [int(i) for i in order],
            }, f, indent=2)

        # Quick bar plot (top-K for readability)
        K = min(12, len(order))
        top_idx = order[:K]
        labels = [rows[i]["declared"] for i in top_idx]
        x = np.arange(K)
        width = 0.25

        plt.figure(figsize=(max(6, K*0.6), 4.5))
        plt.bar(x - width, p_msp[top_idx], width, label="MSP p(c)")
        plt.bar(x,          p_vlm[top_idx], width, label="VLM p(c)")
        plt.bar(x + width,  p_post[top_idx], width, label="Posterior p(c)")

        plt.xticks(x, labels, rotation=45, ha="right")
        plt.ylabel("Probability")
        plt.title(f"Step {step_idx}: Expert probs vs Posterior (top-{K})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{stem}_bars.png", dpi=150)
        plt.close()

        # Optional: append posterior trace of best
        best_name = labels[0]
        with open(out_dir / "posterior_trace.txt", "a") as f:
            f.write(json.dumps({
                "step": step_idx,
                "best_declared": best_name,
                "best_post": float(p_post[top_idx[0]])
            }) + "\n")

    # ---------- Public step ----------
    @property
    def t(self) -> int:
        return self._t

    def get_next_action(self):
        # 1) MSP prior over (graph objects + dataset candidates)
        graph_objs = self._current_graph_objects()
        msp_rows, msp_best = self.msp.score_candidates(
            question=self._question,
            reference_hab=self.ref_pos_hab,
            dataset_candidates=self.dataset_candidates,
            sg_objects=graph_objs,
        )
        if getattr(self, "gt_id_str", None):
            msp_rows = [
                r for r in msp_rows
                if not (r.get("source") == "dataset" and str(r.get("declared")).strip() == self.gt_id_str)
            ]
        table_str = self._format_msp_table(msp_rows) if msp_rows else "(no MSP rows)"
        allowed_declared = self._allowed_declared_names(
            [g[0] for g in graph_objs], [g[1] for g in graph_objs]
        )

        nav_step_info = None
        target_declaration = None

        mode = self.msp.cfg.mode
        alpha = float(self.msp.cfg.alpha)
        eps   = float(self.msp.cfg.eps)
        temp  = float(self.msp.cfg.vlm_temp)
        T_msp = 1.0  # temperature for turning MSP log-likelihoods into probabilities

        declared_list = [r["declared"] for r in msp_rows]

        if mode == "poe":
            # 1) VLM probability over candidates (same order)
            p_vlm = self._vlm_probs_over_candidates(table_str, declared_list) if msp_rows else None
            if p_vlm is None and msp_rows:
                p_vlm = np.ones(len(msp_rows), dtype=np.float64) / max(1, len(msp_rows))

            # optional VLM temperature
            if p_vlm is not None and temp and np.isfinite(temp) and temp > 0 and temp != 1.0:
                logits = np.log(np.clip(p_vlm, eps, None)) / temp
                z = np.exp(logits - logits.max())
                p_vlm = z / (z.sum() + eps)

            # 2) MSP log-likelihoods
            log_msp = np.array([r["logp"] for r in msp_rows], dtype=np.float64) if msp_rows else np.array([])

            # Convert MSP log-likelihoods to a probability vector for visualization
            if msp_rows:
                lm_shift = (log_msp / max(T_msp, eps)) - (log_msp / max(T_msp, eps)).max()
                z_msp = np.exp(lm_shift)
                p_msp = z_msp / (z_msp.sum() + eps)

                # 3) PoE in log-space
                score = alpha * log_msp + (1.0 - alpha) * np.log(np.clip(p_vlm, eps, None))
                i_star = int(np.argmax(score))

                # Softmax over scores → posterior probs
                score_shift = score - score.max()
                z_post = np.exp(score_shift)
                p_post = z_post / (z_post.sum() + eps)

                conf = float(p_post[i_star])
                target_declaration = {
                    "explanation": f"Numeric PoE: alpha={alpha}, temp={temp}.",
                    "declared_target_object_id": declared_list[i_star],
                    "confidence_level": conf,
                    "is_confident": bool(conf >= 0.8),
                }

                # Debug dumps
                try:
                    self._dump_poe_debug(
                        step_idx=self._t,
                        rows=msp_rows,
                        p_msp=p_msp,
                        p_vlm=p_vlm,
                        p_post=p_post,
                        logp_msp=log_msp,
                        p_prev=None,            # hook for continuity prior if added later
                        w_msp=float(alpha),
                        w_vlm=float(1.0 - alpha),
                        T_msp=float(T_msp),
                        T_vlm=float(temp),
                        out_dir=self._output_path,
                    )
                except Exception as e:
                    print(f"[PoE debug] failed to dump step {self._t}: {e}")

            else:
                target_declaration = {
                    "explanation": "No MSP rows; cannot compute PoE.",
                    "declared_target_object_id": "unknown",
                    "confidence_level": 0.0,
                    "is_confident": False,
                }

        elif mode == "hard":
            target_declaration = {
                "explanation": "Chosen by MSP argmax (hard mode).",
                "declared_target_object_id": msp_best or "unknown",
                "confidence_level": 0.95 if msp_best else 0.2,
                "is_confident": bool(msp_best is not None),
            }

        else:
            # v3 behavior: ask Gemini for next_action + declaration
            nav_step_info, target_declaration = self._call_gemini(table_str, allowed_declared)
            if (mode == "fallback") and (not target_declaration or not target_declaration.get("declared_target_object_id")):
                target_declaration = {
                    "explanation": "MSP fallback: VLM missing/invalid.",
                    "declared_target_object_id": msp_best or "unknown",
                    "confidence_level": 0.85 if msp_best else 0.2,
                    "is_confident": bool(msp_best is not None),
                }

        # 3) logging
        self._outputs_to_save.append(
            f"--- Timestep: {self._t} (mode={mode}) ---\n"
            f"MSP top rows (head):\n{table_str[:1200]}\n\n"
            f"VLM Step: {json.dumps(nav_step_info, ensure_ascii=False)}\n"
            f"VLM/MSP Declaration: {json.dumps(target_declaration, ensure_ascii=False)}\n"
        )
        try:
            with open(self._output_path / "llm_outputs_v4.txt", "w") as f:
                f.write("\n".join(self._outputs_to_save))
        except Exception:
            pass

        # 4) optional navigation from VLM (if provided)
        target_pose, target_id = None, None
        if nav_step_info:
            key = list(nav_step_info.keys())[0]
            if "Goto_object_node_step" in key:
                oid = nav_step_info[key].get("object_id")
                if oid and oid in (list(getattr(self.sg_sim, "object_node_ids", []) or [])):
                    target_id = oid
                    try:
                        target_pose = self.sg_sim.get_position_from_id(target_id)
                    except Exception:
                        target_pose = None
            elif "Goto_frontier_node_step" in key:
                fid = nav_step_info[key].get("frontier_id")
                if fid and fid in (list(getattr(self.sg_sim, "frontier_node_ids", []) or [])):
                    target_id = fid
                    try:
                        target_pose = self.sg_sim.get_position_from_id(target_id)
                    except Exception:
                        target_pose = None

        # 5) confidence + termination
        self._t += 1
        conf = _safe_float((target_declaration or {}).get("confidence_level", 0.0), 0.0)
        conf = max(0.0, min(1.0, conf))
        is_conf = bool((target_declaration or {}).get("is_confident", False)) and (conf >= 0.8)

        return target_pose, target_id, is_conf, conf, (target_declaration or {})
