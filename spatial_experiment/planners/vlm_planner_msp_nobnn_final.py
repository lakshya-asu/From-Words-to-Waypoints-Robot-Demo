#!/usr/bin/env python3
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from graph_eqa.envs.utils import pos_normal_to_habitat, pos_habitat_to_normal
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
    try:
        img = get_latest_image(Path(out_path))
        return str(img) if img else None
    except Exception:
        return None


class VLMPlannerMSP_NoBNN_Final:
    """
    Returns:
      (target_pose, target_id, is_confident, confidence, extra)

    extra["action_type"] ∈ {"goto_object","goto_frontier","lookaround","walkaround","answer"}
    """

    def __init__(self, cfg, sg_sim, question, gt=None, out_path: str = ".", **kwargs):
        self.sg_sim = sg_sim
        self._question = question
        self._out_path = Path(out_path)
        self._t = 0

        # IMPORTANT: primary anchor hint passed by runner (e.g., "sofa")
        self._anchor_object_id = kwargs.get("anchor_object_id", None)

        self.msp_cfg = MSPNoBNNConfig(**cfg.msp_nobnn)
        self.engine = MSPNoBNNEngine(combined_logpdf_fn=_combined_logpdf, cfg=self.msp_cfg)

        self.dataset_candidates = kwargs.get("candidate_targets", []) or []

        self.vlm = None
        if getattr(self.msp_cfg, "use_vlm_predicate", False) or getattr(self.msp_cfg, "use_vlm_explain", False):
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError("GOOGLE_API_KEY missing but VLM is enabled in config.")
            self.vlm = GeminiVLMClient(GeminiVLMConfig(api_key=api_key))

        # runner compatibility
        self._outputs_to_save: List[str] = [f"Question: {self._question}\n"]
        self.full_plan = ""

        # exploration bookkeeping
        self._anchor_fail_streak = 0
        self._lookaround_count = 0
        self._walkaround_count = 0

        print("\n[MSP No-BNN FINAL INIT]")
        print(f"  - mode: {getattr(self.msp_cfg, 'mode', 'which')}")
        if hasattr(self.msp_cfg, "semantic_mode"):
            print(f"  - semantic_mode: {self.msp_cfg.semantic_mode}")
        print(f"  - # dataset candidates: {len(self.dataset_candidates)}")
        print(f"  - anchor_object_id (primary_object): {self._anchor_object_id}")

    @property
    def t(self) -> int:
        return self._t

    # ---------------------- SG helpers --------------------------------------

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
                out.append(
                    SceneObject(
                        obj_id=str(oid),
                        name=str(nm).lower(),
                        position=pos_hab,
                        size=np.array([0.5, 0.5, 0.5], np.float32),
                        source="graph",
                    )
                )
            except Exception:
                continue

        for c in self.dataset_candidates:
            try:
                pos = np.asarray(c.get("position", [0, 0, 0]), np.float32)
                size = np.asarray(c.get("size", [0.5, 0.5, 0.5]), np.float32)
                name = str(c.get("name", "obj")).lower()
                cid = str(c.get("id", name))
                out.append(SceneObject(obj_id=cid, name=name, position=pos, size=size, source="dataset"))
            except Exception:
                continue

        print(f"[MSP Final] Retrieved {len(out)} objects (graph + dataset).")
        return out

    def _current_frontiers(self) -> List[Tuple[str, np.ndarray]]:
        fids = list(getattr(self.sg_sim, "frontier_node_ids", []) or [])
        out: List[Tuple[str, np.ndarray]] = []
        for fid in fids:
            try:
                p_norm = self.sg_sim.get_position_from_id(fid)
                if p_norm is None:
                    continue
                p_hab = np.asarray(pos_normal_to_habitat(np.array(p_norm, np.float32)), np.float32)
                out.append((str(fid), p_hab))
            except Exception:
                continue
        return out

    def _choose_best_frontier(self, frontiers: List[Tuple[str, np.ndarray]], anchor_pos_hab: np.ndarray) -> Optional[str]:
        if not frontiers:
            return None
        dists = [(fid, float(np.linalg.norm(p_hab - anchor_pos_hab))) for fid, p_hab in frontiers]
        dists.sort(key=lambda x: x[1])
        return dists[0][0]

    # ---------------------- Anchor override logic ----------------------------

    def _resolve_anchor_from_primary(
        self,
        objects: List[SceneObject],
        frame,
        primary_object: Optional[str],
    ) -> Optional[SceneObject]:
        """
        Use primary_object (e.g., "sofa") to force anchor class,
        then refine among multiple candidates using question phrase tokens
        (e.g., "2 seater sofa").
        """
        if not primary_object:
            return None
        p = str(primary_object).lower().strip()
        if not p:
            return None

        # candidates that match the primary object string
        prim_cands = [o for o in objects if p in (o.name or "").lower()]
        if not prim_cands:
            return None

        phrase = (frame.anchor_class or "").lower().strip()
        phrase_tokens = [t for t in re.findall(r"[a-z0-9_]+", phrase) if t]

        if not phrase_tokens:
            return prim_cands[0]

        best = prim_cands[0]
        best_overlap = -1
        for o in prim_cands:
            nm_tokens = set(re.findall(r"[a-z0-9_]+", (o.name or "").lower()))
            overlap = sum(1 for t in phrase_tokens if t in nm_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best = o

        return best

    # ---------------------- main call ---------------------------------------

    def get_next_action(
        self,
        agent_yaw_rad: float = 0.0,
        agent_pos_hab: Optional[np.ndarray] = None,
    ):
        agent_pos_hab = np.asarray(agent_pos_hab if agent_pos_hab is not None else [0, 0, 0], np.float32)

        objects = self._current_graph_objects()
        img_path = _safe_latest_image(self._out_path)
        frame = parse_query_frame(self._question)

        # ==========================================================
        # ANCHOR RESOLVE:
        # 1) If primary_object provided, force anchor selection.
        # 2) Else fallback to heuristic distribution + confidence.
        # ==========================================================
        anchor_obj = self._resolve_anchor_from_primary(objects, frame, self._anchor_object_id)

        if anchor_obj is not None:
            # Hard-force anchor => DO NOT do anchor uncertainty exploration
            anchor_pos = anchor_obj.position
            aconf = 1.0
            self._anchor_fail_streak = 0
        else:
            anchor_dist = self.engine.resolve_anchor_distribution(frame, objects, agent_pos_hab)
            if not anchor_dist:
                extra = {"action_type": "lookaround", "reason": "no objects yet", "num_yaws": 8}
                self._write_step_log(agent_yaw_rad, None, extra)
                self._t += 1
                return None, None, False, 0.0, extra

            anchor_obj = anchor_dist[0][0]
            anchor_pos = anchor_obj.position

            aconf = 1.0
            if hasattr(self.engine, "anchor_confidence"):
                try:
                    aconf = float(self.engine.anchor_confidence(anchor_dist))
                except Exception:
                    aconf = 1.0

            # only explore if primary_object not provided
            anchor_thresh = float(getattr(self.msp_cfg, "anchor_confidence_threshold", 0.65))
            if aconf < anchor_thresh:
                self._anchor_fail_streak += 1

                if self._lookaround_count < 2:
                    self._lookaround_count += 1
                    extra = {
                        "action_type": "lookaround",
                        "reason": "anchor_uncertain",
                        "anchor_candidate": {"id": anchor_obj.obj_id, "name": anchor_obj.name, "conf": aconf},
                        "num_yaws": 8,
                    }
                    self._write_step_log(agent_yaw_rad, None, extra)
                    self._t += 1
                    return None, None, False, float(aconf), extra

                self._walkaround_count += 1
                extra = {
                    "action_type": "walkaround",
                    "reason": "anchor_uncertain_after_lookaround",
                    "anchor_candidate": {"id": anchor_obj.obj_id, "name": anchor_obj.name, "conf": aconf},
                    "radius_m": 0.75,
                    "num_waypoints": 6,
                }
                self._write_step_log(agent_yaw_rad, None, extra)
                self._t += 1
                return None, None, False, float(aconf), extra

            self._anchor_fail_streak = 0

        # ---- optional VLM predicate params
        pred_params: Optional[PredicateParams] = None
        if self.vlm and getattr(self.msp_cfg, "use_vlm_predicate", False):
            pred_params = self.vlm.get_predicate_params(
                image_path=img_path,
                question=self._question,
                anchor_name=frame.anchor_class,
            )

        # ---- WHICH
        if self.msp_cfg.mode == "which":
            rows, best_obj, summary = self.engine.run_which_mode(
                frame=frame,
                objects=objects,
                agent_pos=agent_pos_hab,
                agent_yaw=agent_yaw_rad,
                predicate_from_vlm=pred_params,
            )
            is_conf, conf = self.engine.compute_confidence_from_rows(rows)

            if is_conf and best_obj is not None:
                extra = {
                    "action_type": "answer",
                    "mode": "which",
                    "best_object_id": best_obj.obj_id,
                    "confidence": float(conf),
                    "anchor": {"id": anchor_obj.obj_id, "name": anchor_obj.name, "pos": anchor_pos.tolist()},
                    "top_k": rows[: self.msp_cfg.top_k],
                }
                self._log_rows(rows, mode="which")
                self._write_step_log(agent_yaw_rad, rows, extra)
                self._t += 1
                return None, best_obj.obj_id, True, float(conf), extra

            # not confident -> lookaround once then frontier
            if conf < float(getattr(self.msp_cfg, "confidence_threshold", 0.85)) and self._lookaround_count < 3:
                self._lookaround_count += 1
                extra = {"action_type": "lookaround", "reason": "which_not_confident", "num_yaws": 8, "confidence": float(conf)}
                self._log_rows(rows, mode="which")
                self._write_step_log(agent_yaw_rad, rows, extra)
                self._t += 1
                return None, None, False, float(conf), extra

            frontiers = self._current_frontiers()
            best_frontier = self._choose_best_frontier(frontiers, anchor_pos)
            if best_frontier is not None:
                extra = {"action_type": "goto_frontier", "mode": "which", "reason": "not_confident", "confidence": float(conf), "frontier_id": best_frontier}
                self._log_rows(rows, mode="which")
                self._write_step_log(agent_yaw_rad, rows, extra)
                self._t += 1
                return self._nav_pose_for_frontier(best_frontier), best_frontier, False, float(conf), extra

            extra = {"action_type": "walkaround", "reason": "which_not_confident_no_frontiers", "radius_m": 0.75, "num_waypoints": 6}
            self._log_rows(rows, mode="which")
            self._write_step_log(agent_yaw_rad, rows, extra)
            self._t += 1
            return None, None, False, float(conf), extra

        # ---- WHERE
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
        artifact_paths = save_region_artifacts(out_dir=self._out_path, step=self._t, region=region, summary=summary)

        conf = float(summary.get("confidence", 0.0))
        is_conf = conf >= float(self.msp_cfg.confidence_threshold)

        if is_conf:
            extra = {
                "action_type": "answer",
                "mode": "where",
                "confidence": float(conf),
                "anchor": {"id": anchor_obj.obj_id, "name": anchor_obj.name, "pos": anchor_pos.tolist()},
                "region_summary": summary,
                "artifacts": artifact_paths,
            }
            self._write_step_log(agent_yaw_rad, None, extra)
            self._t += 1
            return summary.get("centroid_xyz", None), None, True, float(conf), extra

        # not confident -> lookaround then frontier then walkaround
        if self._lookaround_count < 3:
            self._lookaround_count += 1
            extra = {"action_type": "lookaround", "reason": "where_not_confident", "num_yaws": 8, "confidence": float(conf)}
            self._write_step_log(agent_yaw_rad, None, extra)
            self._t += 1
            return None, None, False, float(conf), extra

        frontiers = self._current_frontiers()
        best_frontier = self._choose_best_frontier(frontiers, anchor_pos)
        if best_frontier is not None:
            extra = {"action_type": "goto_frontier", "mode": "where", "reason": "region_not_confident", "confidence": float(conf), "frontier_id": best_frontier}
            self._write_step_log(agent_yaw_rad, None, extra)
            self._t += 1
            return self._nav_pose_for_frontier(best_frontier), best_frontier, False, float(conf), extra

        extra = {"action_type": "walkaround", "reason": "where_not_confident_no_frontiers", "radius_m": 0.75, "num_waypoints": 6, "confidence": float(conf)}
        self._write_step_log(agent_yaw_rad, None, extra)
        self._t += 1
        return None, None, False, float(conf), extra

    # ---------------------- pose helpers -------------------------------------

    def _nav_pose_for_object(self, obj: SceneObject):
        if obj.source == "graph":
            try:
                return self.sg_sim.get_position_from_id(int(obj.obj_id))
            except Exception:
                try:
                    return self.sg_sim.get_position_from_id(obj.obj_id)
                except Exception:
                    return None
        return obj.position.tolist()

    def _nav_pose_for_frontier(self, frontier_id: str):
        try:
            return self.sg_sim.get_position_from_id(int(frontier_id))
        except Exception:
            try:
                return self.sg_sim.get_position_from_id(frontier_id)
            except Exception:
                return None

    # ---------------------- logging -----------------------------------------

    def _log_rows(self, rows: List[Dict[str, Any]], mode: str):
        csv_path = self._out_path / "params_comparison_msp_nobnn_final.csv"
        data = []
        for r in rows:
            data.append(
                {
                    "step": self._t,
                    "mode": mode,
                    "candidate": r.get("candidate_id"),
                    "name": r.get("name"),
                    "logp": r.get("logp_total"),
                    "anchor_id": r.get("anchor_id"),
                    "anchor_name": r.get("anchor_name"),
                }
            )
        if not data:
            return
        df = pd.DataFrame(data)
        df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)

    def _write_step_log(self, agent_yaw_rad: float, rows: Optional[List[Dict[str, Any]]], extra: Dict[str, Any]):
        top_str = ""
        if rows:
            top5 = rows[:5]
            try:
                top_str = "\n".join(
                    [f"  {r.get('name')} ({r.get('candidate_id')}): logp={float(r.get('logp_total', 0)):.2f}" for r in top5]
                )
            except Exception:
                top_str = str(top5)

        log_entry = (
            f"\n--- Step {self._t} ---\n"
            f"Agent Yaw (world): {float(agent_yaw_rad):.3f}\n"
            f"Action: {extra.get('action_type')}\n"
            f"Mode: {extra.get('mode', '')}\n"
            f"Top Candidates:\n{top_str}\n"
            f"Extra: {str(extra)[:1500]}\n"
        )
        self._outputs_to_save.append(log_entry)
        self.full_plan = "\n".join(self._outputs_to_save)

        try:
            with open(self._out_path / "llm_outputs_msp_nobnn_final.txt", "w") as f:
                f.write(self.full_plan)
        except Exception:
            pass