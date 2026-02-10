# /home/artemis/project/graph_eqa_swagat/spatial_experiment/msp_nobnn/adapter.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

from .core import MSPNoBNNEngine, QueryFrame, SceneObject, PredicateParams
from .region import RegionGridConfig, compute_region_posterior, summarize_region, save_region_artifacts


@dataclass
class AdapterConfig:
    top_frac: float = 0.02
    grid_half_extent_x: float = 3.0
    grid_half_extent_y: float = 1.5
    grid_half_extent_z: float = 3.0
    grid_resolution: int = 28
    save_artifacts: bool = True


def _rank_objects_by_region_centroid(objects: List[SceneObject], centroid_xyz: np.ndarray) -> List[Dict[str, Any]]:
    """
    Simple WHICH-mode ranking proxy:
      score = -distance(object_center, region_centroid)
    (Baseline-friendly and deterministic)
    """
    ranked = []
    for o in objects:
        d = float(np.linalg.norm(o.position - centroid_xyz))
        ranked.append({
            "id": o.obj_id,
            "name": o.name,
            "score": -d,
            "dist_to_centroid": d,
            "pos": [float(o.position[0]), float(o.position[1]), float(o.position[2])],
        })
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


def compose_pdf_and_select(
    engine: MSPNoBNNEngine,
    frame: QueryFrame,
    objects: List[SceneObject],
    agent_pos: np.ndarray,
    agent_yaw: float,
    predicate_from_vlm: Optional[PredicateParams],
    out_dir: Optional[str],
    step: int,
    cfg: Optional[AdapterConfig] = None,
) -> Dict[str, Any]:
    """
    Returns a unified output dict used by the planner.

    Output:
      best_xyz_hab: np.ndarray shape (3,)  (we use centroid as best point)
      ranked_objects: list (for WHICH-mode)
      confidence_level, is_confident
      region_npz, heatmap_png
      debug: params, anchor info
    """
    cfg = cfg or AdapterConfig()

    grid_cfg = RegionGridConfig(
        half_extent_x=cfg.grid_half_extent_x,
        half_extent_y=cfg.grid_half_extent_y,
        half_extent_z=cfg.grid_half_extent_z,
        resolution=cfg.grid_resolution,
    )

    region = compute_region_posterior(
        engine=engine,
        frame=frame,
        objects=objects,
        agent_pos=agent_pos,
        agent_yaw=float(agent_yaw),
        predicate_from_vlm=predicate_from_vlm,
        grid_cfg=grid_cfg,
    )
    if "error" in region:
        return {
            "is_confident": False,
            "confidence_level": 0.0,
            "error": region["error"],
            "ranked_objects": [],
        }

    summary = summarize_region(region, top_frac=cfg.top_frac)
    if "error" in summary:
        return {
            "is_confident": False,
            "confidence_level": 0.0,
            "error": summary["error"],
            "ranked_objects": [],
            "debug": {"region": region},
        }

    centroid = np.asarray(summary["centroid_xyz"], dtype=np.float32)

    ranked = _rank_objects_by_region_centroid(objects, centroid)

    artifacts = {}
    if cfg.save_artifacts and out_dir is not None:
        artifacts = save_region_artifacts(out_dir=out_dir, step=step, region=region, summary=summary)

    conf = float(summary.get("confidence", 0.0))
    is_conf = bool(conf >= 0.8)  # tune if you want, but keep fixed for baseline

    return {
        "best_xyz_hab": centroid,
        "ranked_objects": ranked,
        "confidence_level": conf,
        "is_confident": is_conf,
        "region_npz": artifacts.get("npz"),
        "heatmap_png": artifacts.get("heatmap_png"),
        "debug": {
            "anchor": region.get("anchor"),
            "params": region.get("params"),
            "summary": summary,
        },
    }