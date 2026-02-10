#!/usr/bin/env python3
"""
PLANNER V9 (NO-BNN): MSP + optional VLM
Two modes:
  - mode="which": rank discrete object candidates by MSP logp
  - mode="where": compute PDF over space, save .npz + heatmap, return region summary

Design:
  - All MSP math stays in code (combined_logpdf).
  - VLM is OPTIONAL:
      * predicate params (theta/phi/kappa)
      * region explanation in natural language (using nearby objects)
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from graph_eqa.envs.utils import pos_normal_to_habitat
from graph_eqa.utils.data_utils import get_latest_image

from spatial_experiment.msp.pdf import combined_logpdf as _combined_logpdf

from spatial_experiment.msp_nobnn.core import (
    MSPNoBNNConfig,
    MSPNoBNNEngine,
    SceneObject,
    parse_query_frame,
)
from spatial_experiment.msp_nobnn.region import (
    RegionGridConfig,
    compute_region_posterior,
    summarize_region,
    save_region_artifacts,
)
from spatial_experiment.msp_nobnn.vlm import (
    GeminiVLMConfig,
    GeminiVLMClient,
    PredicateParams,
)


def _safe_latest_image(out_path: Path) -> Optional[str]:
    img = get_latest_image(str(out_path))
    if img:
        return img
    print(f"[VLM WARNING] No images found in {out_path}")
    return None


class VLMPlannerEQAGeminiSpatialV9_NoBNN:
    """
    Drop-in planner replacement.

    Required env:
      - GOOGLE_API_KEY set (if use_vlm_predicate/use_vlm_explain True)

    kwargs:
      - candidate_targets: list of dicts with position/size/name (optional)
      - (optional) reference_object hint (name/position) but we prefer parsing anchor from question
    """

    def __init__(self, cfg, sg_sim, question, gt, out_path, **kwargs):
        self._out_path = Path(out_path)
        self._question = question
        self._t = 0
        self.sg_sim = sg_sim

        # ---- config ---------------------------------------------------------
        # Planner config expects:
        # cfg.msp.mode in {"which","where"}
        # cfg.msp.use_vlm_predicate bool
        # cfg.msp.use_vlm_explain bool
        # cfg.msp.top_k int
        # cfg.msp.region_grid dict (bounds/resolution)
        self.msp_cfg = MSPNoBNNConfig(**cfg.msp)

        self.engine = MSPNoBNNEngine(combined_logpdf_fn=_combined_logpdf, cfg=self.msp_cfg)

        self.dataset_candidates = kwargs.get("candidate_targets", []) or []

        # Optional VLM client
        self.vlm = None
        if self.msp_cfg.use_vlm_predicate or self.msp_cfg.use_vlm_explain:
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError("GOOGLE_API_KEY missing but VLM is enabled in config.")
            self.vlm = GeminiVLMClient(GeminiVLMConfig(api_key=api_key))

        self._outputs_to_save: List[str] = [f"Question: {self._question}\n"]

        print("\n[PLANNER V9 INIT - NO BNN]")
        print(f"  - mode: {self.msp_cfg.mode}")
        print(f"  - use_vlm_predicate: {self.msp_cfg.use_vlm_predicate}")
        print(f"  - use_vlm_explain:   {self.msp_cfg.use_vlm_explain}")
        print(f"  - # dataset candidates: {len(self.dataset_candidates)}")

    @property
    def t(self) -> int:
        return self._t

    def _current_graph_objects(self) -> List[SceneObject]:
        out: List[SceneObject] = []
        oids = list(getattr(self.sg_sim, "object_node_ids", []) or [])
        nms = list(getattr(self.sg_sim, "object_node_names", []) or [])

        for oid, nm in zip(oids, nms):
            try:
                pos_norm = self.sg_sim.get_position_from_id(oid)
                if pos_norm is None:
                    continue
                pos_hab = np.asarray(pos_normal_to_habitat(np.array(pos_norm, np.float32)), np.float32)
                out.append(SceneObject(
                    obj_id=str(oid),
                    name=str(nm).lower(),
                    position=pos_hab,
                    size=np.array([0.5, 0.5, 0.5], np.float32),
                    source="graph",
                ))
            except Exception:
                continue

        # dataset candidates (optional)
        for c in self.dataset_candidates:
            try:
                pos = np.asarray(c.get("position", [0, 0, 0]), np.float32)
                size = np.asarray(c.get("size", [0.5, 0.5, 0.5]), np.float32)
                name = str(c.get("name", "obj")).lower()
                cid = str(c.get("id", name))
                out.append(SceneObject(
                    obj_id=cid,
                    name=name,
                    position=pos,
                    size=size,
                    source="dataset",
                ))
            except Exception:
                continue

        print(f"[PLANNER V9] Retrieved {len(out)} objects (graph + dataset).")
        return out

    def get_next_action(
        self,
        agent_yaw_rad: float = 0.0,
        agent_pos_hab: Optional[np.ndarray] = None,
    ):
        """
        Returns:
          (target_pose, target_id, is_confident, confidence, extra_dict)

        - For mode="which": target_id is the best object id; target_pose is its position (normalized space if sg_sim returns that).
        - For mode="where": target_id None; target_pose is the region centroid (hab coords); extra contains saved artifact paths.
        """
        agent_pos_hab = np.asarray(agent_pos_hab if agent_pos_hab is not None else [0, 0, 0], np.float32)

        objects = self._current_graph_objects()
        img_path = _safe_latest_image(self._out_path)

        # 1) Parse query frame
        frame = parse_query_frame(self._question)
        print(f"[PLANNER V9] Parsed frame: {frame}")

        # 2) Optional VLM predicate params
        pred_params: Optional[PredicateParams] = None
        if self.vlm and self.msp_cfg.use_vlm_predicate:
            pred_params = self.vlm.get_predicate_params(
                image_path=img_path,
                question=self._question,
                anchor_name=frame.anchor_class,
            )
        # else: engine will compute heuristic predicate params.

        # 3) Run mode
        if self.msp_cfg.mode == "which":
            rows, best_obj, summary = self.engine.run_which_mode(
                frame=frame,
                objects=objects,
                agent_pos=agent_pos_hab,
                agent_yaw=agent_yaw_rad,
                predicate_from_vlm=pred_params,
            )
            self._log_rows(rows, mode="which")

            # target id + pose (best object)
            target_id = best_obj.obj_id if best_obj else None
            target_pose = None
            if best_obj and best_obj.source == "graph":
                try:
                    target_pose = self.sg_sim.get_position_from_id(int(best_obj.obj_id))
                except Exception:
                    target_pose = None
            else:
                target_pose = best_obj.position.tolist() if best_obj else None

            is_conf, conf = self.engine.compute_confidence_from_rows(rows)

            # optional VLM explanation for top-k
            explanation = None
            if self.vlm and self.msp_cfg.use_vlm_explain:
                explanation = self.vlm.explain_which(
                    image_path=img_path,
                    question=self._question,
                    top_rows=rows[: self.msp_cfg.top_k],
                )

            extra = {
                "mode": "which",
                "best_object_id": target_id,
                "confidence": conf,
                "engine_summary": summary,
                "vlm_explanation": explanation,
            }

            self._write_step_log(agent_yaw_rad, rows, extra)
            self._t += 1
            return target_pose, target_id, is_conf, conf, extra

        # WHERE MODE
        region_cfg = RegionGridConfig(**(self.msp_cfg.region_grid or {}))
        region = compute_region_posterior(
            engine=self.engine,
            frame=frame,
            objects=objects,
            agent_pos=agent_pos_hab,
            agent_yaw=agent_yaw_rad,
            predicate_from_vlm=pred_params,
            grid_cfg=region_cfg,
        )
        summary = summarize_region(region, top_frac=self.msp_cfg.where_top_frac)

        artifact_paths = save_region_artifacts(
            out_dir=self._out_path,
            step=self._t,
            region=region,
            summary=summary,
        )

        explanation = None
        if self.vlm and self.msp_cfg.use_vlm_explain:
            explanation = self.vlm.explain_where(
                image_path=img_path,
                question=self._question,
                region_summary=summary,
                nearby_objects=summary.get("nearby_objects", []),
            )

        conf = float(summary.get("confidence", 0.0))
        is_conf = conf >= self.msp_cfg.confidence_threshold

        extra = {
            "mode": "where",
            "region_summary": summary,
            "artifacts": artifact_paths,
            "vlm_explanation": explanation,
        }

        self._write_step_log(agent_yaw_rad, None, extra)
        self._t += 1

        # Return centroid as "pose"
        centroid = summary.get("centroid_xyz", None)
        return (centroid, None, is_conf, conf, extra)

    def _log_rows(self, rows: List[Dict[str, Any]], mode: str):
        csv_path = self._out_path / "params_comparison_v9_nobnn.csv"
        data = []
        for r in rows:
            data.append({
                "step": self._t,
                "mode": mode,
                "candidate": r["candidate_id"],
                "name": r["name"],
                "logp": r["logp_total"],
                "logp_sem": r["logp_sem"],
                "logp_metric": r["logp_metric"],
                "logp_pred": r["logp_pred"],
                "theta": r["theta0"],
                "phi": r["phi0"],
                "kappa": r["kappa"],
                "source": r["source"],
            })
        df = pd.DataFrame(data)
        df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)

    def _write_step_log(self, agent_yaw_rad: float, rows, extra: Dict[str, Any]):
        top_str = ""
        if rows:
            top5 = rows[:5]
            top_str = "\n".join([f"  {r['name']} ({r['candidate_id']}): logp={r['logp_total']:.2f}"
                                 for r in top5])

        log_entry = (
            f"\n--- Step {self._t} ---\n"
            f"Agent Yaw (world): {agent_yaw_rad:.3f}\n"
            f"Mode: {extra.get('mode')}\n"
            f"Top Candidates:\n{top_str}\n"
            f"Extra: {str(extra)[:1500]}\n"
        )
        self._outputs_to_save.append(log_entry)
        try:
            with open(self._out_path / "llm_outputs_v9_nobnn.txt", "w") as f:
                f.write("\n".join(self._outputs_to_save))
        except Exception as e:
            print(f"[PLANNER V9] Error saving logs: {e}")
