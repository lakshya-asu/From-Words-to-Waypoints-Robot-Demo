# /home/artemis/project/graph_eqa_swagat/spatial_experiment/planners/vlm_planner_gemini_spatial_v3.py
"""
Gemini planner (v3) with MSP bridge:
- Computes per-candidate combined-PDF scores using MSP (Bayesian) models if available,
  else falls back to analytic parameterization.
- Modes:
    * msp.mode = "prior"    -> pass a ranked table to Gemini and still ask the VLM
    * msp.mode = "fallback" -> ask Gemini; if invalid/low-confidence, use MSP argmax
    * msp.mode = "hard"     -> skip Gemini, use MSP argmax only (ablation)
- The "declared_target_object_id" remains an instance-like string (e.g., "step_306")
  chosen from the allowed pool (graph-derived + dataset candidate ids).
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

# VLM
import google.generativeai as genai

# Graph-EQA utils
from graph_eqa.envs.utils import pos_normal_to_habitat
from graph_eqa.utils.data_utils import get_latest_image

# ---- Combined PDF (vectorized logpdf preferred) ----
from spatial_experiment.msp.pdf import combined_logpdf as _combined_logpdf

# ---------- Optional MSP model stack (Pyro/Torch). Safe to import lazily ----------
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
    if rel == "front":  return (np.pi/2, np.pi/2)   # -Z forward → theta=+π/2 if we param in +Z
    if rel == "behind": return (3*np.pi/2, np.pi/2)
    # default neutral
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
    mode: str = "prior"   # "prior" | "fallback" | "hard"
    model_dir: Optional[str] = None
    metric_run: Optional[str] = None
    semantic_run: Optional[str] = None
    predicate_run: Optional[str] = None
    semantic_ohe: Optional[str] = None
    device: Optional[str] = None   # "cuda" | "cpu" | None (auto)

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
        self.cfg.model_dir = self.cfg.model_dir or env.get("MSP_MODEL_DIR")
        self.cfg.metric_run = self.cfg.metric_run or env.get("MSP_METRIC_RUN")
        self.cfg.semantic_run = self.cfg.semantic_run or env.get("MSP_SEMANTIC_RUN")
        self.cfg.predicate_run = self.cfg.predicate_run or env.get("MSP_PREDICATE_RUN")
        self.cfg.semantic_ohe = self.cfg.semantic_ohe or env.get("MSP_SEM_OHE")

        # quick sanity
        req = [self.cfg.model_dir, self.cfg.metric_run, self.cfg.semantic_run, self.cfg.predicate_run, self.cfg.semantic_ohe]
        if not all(req):
            # leave as heuristic-only
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
        # sizes: candidate["size"] = [w, d, h]
        w, d, h = [float(x) for x in (candidate.get("size") or [0.5, 0.5, 0.5])[:3]]
        max_dim = max(1e-6, w, d, h)
        sigma_s = 0.5 * max_dim
        sigma_m = 0.3 * max_dim
        kappa = 10.0 / max_dim
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
        # Build a tiny 1-row dataframe in-place (no pandas dependency)
        try:
            st = self._stack
            torch, pyro = st["torch"], st["pyro"]
            from collections import OrderedDict
            # features expected by training
            row = OrderedDict({
                "x0": float(anchor[0]), "y0": float(anchor[1]), "z0": float(anchor[2]),
                "metric": float(distance_m if distance_m is not None else 0.0),
                "semantic": str(candidate.get("name", "object")).lower(),
                "width": float((candidate.get("size") or [0.5,0.5,0.5])[0]),
                "depth": float((candidate.get("size") or [0.5,0.5,0.5])[1]),
                "height": float((candidate.get("size") or [0.5,0.5,0.5])[2]),
                "predicate": str(rel or ""),
            })

            # --- prepare features (mirror training utils) ---
            # metric
            X_metric, *_ = st["prepare_dataset"](
                [row], ["x0","y0","z0","metric"], target_cols=["P_combined"], fit=False
            )
            # semantic (OHE)
            X_sem, *_ = st["prepare_dataset"](
                [row], ["semantic","width","height","depth"], target_cols=["P_combined"],
                fit=False, categorical_cols=["semantic"], ohe_fit_columns=self._sem_ohe_cols
            )
            # predicate
            X_pred, *_ = st["prepare_dataset"](
                [row], ["predicate","width","height","depth"], target_cols=["P_combined"], fit=False
            )

            device = torch.device(self.cfg.device) if self.cfg.device else (
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )

            X_metric_t = torch.tensor(X_metric, dtype=torch.float32, device=device)
            X_sem_t    = torch.tensor(X_sem,    dtype=torch.float32, device=device)
            X_pred_t   = torch.tensor(X_pred,   dtype=torch.float32, device=device)

            # nets
            net_metric = st["BayesianNN_metric"](input_dim=X_metric_t.shape[1], hidden_dim=32, output_dim=2).to(device)
            net_sem    = st["BayesianNNSem"](   input_dim=X_sem_t.shape[1],    hidden_dim=32, output_dim=4).to(device)
            net_pred   = st["BayesianNNPred"](  input_dim=X_pred_t.shape[1],   hidden_dim=128, output_dim=3).to(device)

            # restore param stores via guides, then load states
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

            # predict means
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

            # anchor is explicitly the reference center
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
          - best_declared: str | None (instance-like id if available)
        """
        dist_m, rel = _parse_q(question)

        rows: List[Dict[str, Any]] = []

        # helper: evaluate logpdf at a habitat/world point
        def eval_logp(params, pos_hab):
            x, y, z = float(pos_hab[0]), float(pos_hab[1]), float(pos_hab[2])
            return float(_combined_logpdf(np.array([x]), np.array([y]), np.array([z]), params)[0])

        # 1) score dataset candidates (have instance-like IDs + sizes)
        for c in dataset_candidates or []:
            pos = np.asarray(c.get("position", [0,0,0]), np.float32)
            if pos.shape[0] != 3 or not np.isfinite(pos).all():
                continue
            # per-candidate params (size, semantic)
            if self._ready:
                params = self._predict_params_models(anchor=reference_hab, candidate=c, distance_m=dist_m, rel=rel)
            else:
                params = self._analytic_params(anchor=reference_hab, candidate=c, distance_m=dist_m, rel=rel)
            if params is None:
                continue
            logp = eval_logp(params, pos)
            rows.append({
                "declared": str(c.get("id") or c.get("name") or "unknown"),
                "graph_id": None,  # not a navigable id
                "class": str(c.get("name","object")).lower(),
                "pos": pos, "size": c.get("size"),
                "logp": logp, "source": "dataset"
            })

        # 2) score current SG objects (no instance-like id; we allow class name as 'declared')
        for (gid, cls, pos) in sg_objects or []:
            fake_cand = {"name": cls, "position": pos.tolist(), "size": [0.5,0.5,0.5]}  # size unknown → neutral
            if self._ready:
                params = self._predict_params_models(anchor=reference_hab, candidate=fake_cand, distance_m=dist_m, rel=rel)
            else:
                params = self._analytic_params(anchor=reference_hab, candidate=fake_cand, distance_m=dist_m, rel=rel)
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

        # pick best by logp
        rows.sort(key=lambda r: r["logp"], reverse=True)
        best_declared = rows[0]["declared"]
        return rows, best_declared


# ===================== Planner (Gemini + MSP) =====================
class VLMPlannerEQAGeminiSpatialV3:
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
        # dataset stores HABITAT coords for reference center
        self.ref_pos_hab = np.asarray(self.reference_object.get("position", [0,0,0]), np.float32)

        self.dataset_candidates = list(candidate_targets or [])

        # MSP bridge
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
        ))

        # VLM (Gemini) — can be bypassed by msp.mode == "hard"
        if self.msp.cfg.mode != "hard":
            try:
                genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
            except KeyError:
                raise RuntimeError("GOOGLE_API_KEY environment variable not set!")
            self.model = genai.GenerativeModel(
                model_name="models/gemini-2.5-pro-preview-03-25",
                system_instruction=(
                    "You are an expert robot navigator in a 3D indoor environment.\n"
                    "Global frame: Y-up, -Z forward, +X right.\n"
                    "You will receive a numeric table with MSP (Bayesian) prior scores (log-probability) "
                    "per candidate. Prefer candidates with the highest logp **that match the relation bucket**.\n"
                    "When declaring, choose exactly one *declared_name* from the allowed list.\n"
                ),
            )

        self._outputs_to_save: List[str] = [
            f'Question: {self._question}\nGround Truth: {self._ground_truth_target.get("name","?")} '
            f'({self._ground_truth_target.get("id","?")})\n'
        ]
        self.full_plan = ""

    # --- Gemini helpers ---
    def _current_graph_objects(self) -> List[Tuple[str, str, np.ndarray]]:
        """Return list of (graph_id, class_name, pos_hab)."""
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
        # collect instance-like names from graph if present
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
        # merge dataset candidate ids (e.g., 'step_306')
        ds_ids = [str(c.get("id")) for c in self.dataset_candidates if c and c.get("id")]
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
        table_str = self._format_msp_table(msp_rows) if msp_rows else "(no MSP rows)"
        allowed_declared = self._allowed_declared_names(
            [g[0] for g in graph_objs], [g[1] for g in graph_objs]
        )

        nav_step_info = None
        target_declaration = None

        # 2) Choose according to mode
        mode = self.msp.cfg.mode
        if mode == "hard":
            # Skip Gemini altogether
            target_declaration = {
                "explanation": "Chosen by MSP argmax (hard mode).",
                "declared_target_object_id": msp_best or "unknown",
                "confidence_level": 0.95 if msp_best else 0.2,
                "is_confident": bool(msp_best is not None),
            }
        else:
            # ask Gemini with MSP table as prior context
            nav_step_info, target_declaration = self._call_gemini(table_str, allowed_declared)

            # fallback if needed
            if (mode == "fallback") and (not target_declaration or not target_declaration.get("declared_target_object_id")):
                target_declaration = {
                    "explanation": "MSP fallback: VLM missing/invalid.",
                    "declared_target_object_id": msp_best or "unknown",
                    "confidence_level": 0.85 if msp_best else 0.2,
                    "is_confident": bool(msp_best is not None),
                }

        # 3) logging
        self._outputs_to_save.append(
            f"--- Timestep: {self._t} ---\n"
            f"MSP top rows (head):\n{table_str[:1200]}\n\n"
            f"VLM Step: {json.dumps(nav_step_info, ensure_ascii=False)}\n"
            f"VLM/MSP Declaration: {json.dumps(target_declaration, ensure_ascii=False)}\n"
        )
        try:
            with open(self._output_path / "llm_outputs_v3.txt", "w") as f:
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
