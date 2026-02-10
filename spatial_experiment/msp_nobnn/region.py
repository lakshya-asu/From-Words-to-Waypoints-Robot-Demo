from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .core import MSPNoBNNEngine, QueryFrame, SceneObject, PredicateParams


@dataclass
class RegionGridConfig:
    # grid centered around anchor position
    half_extent_x: float = 3.0
    half_extent_y: float = 1.5
    half_extent_z: float = 3.0
    resolution: int = 28  # per axis

    # for visualization: take a y-slice around anchor_y
    slice_dy: float = 0.25


def compute_region_posterior(
    engine: MSPNoBNNEngine,
    frame: QueryFrame,
    objects: List[SceneObject],
    agent_pos: np.ndarray,
    agent_yaw: float,
    predicate_from_vlm: Optional[PredicateParams],
    grid_cfg: RegionGridConfig,
) -> Dict[str, Any]:
    """
    Compute logp on a dense 3D grid and return arrays + metadata.
    """
    anchor_dist = engine.resolve_anchor_distribution(frame, objects, agent_pos)
    if not anchor_dist:
        return {"error": "no objects"}
    anchor_obj = anchor_dist[0][0]
    anchor_pos = anchor_obj.position

    theta0, phi0, kappa = engine.build_predicate_params(
        frame=frame,
        anchor_pos=anchor_pos,
        agent_pos=agent_pos,
        agent_yaw=agent_yaw,
        predicate_from_vlm=predicate_from_vlm,
    )

    # Build grid
    rx = grid_cfg.resolution
    ry = grid_cfg.resolution
    rz = grid_cfg.resolution

    xs = np.linspace(anchor_pos[0] - grid_cfg.half_extent_x, anchor_pos[0] + grid_cfg.half_extent_x, rx).astype(np.float32)
    ys = np.linspace(anchor_pos[1] - grid_cfg.half_extent_y, anchor_pos[1] + grid_cfg.half_extent_y, ry).astype(np.float32)
    zs = np.linspace(anchor_pos[2] - grid_cfg.half_extent_z, anchor_pos[2] + grid_cfg.half_extent_z, rz).astype(np.float32)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    Xf = X.reshape(-1)
    Yf = Y.reshape(-1)
    Zf = Z.reshape(-1)

    # For region mode, we treat "semantic" center as the *region itself*, not a target object.
    # So we neutralize semantic term by setting sigma_s very large and mu at anchor (or agent).
    # (You can replace this later with a learned/heuristic spatial prior.)
    params = {
        "mu_x": float(anchor_pos[0]),
        "mu_y": float(anchor_pos[1]),
        "mu_z": float(anchor_pos[2]),
        "sigma_s": 1e6,  # semantic neutral
        "x0": float(anchor_pos[0]),
        "y0": float(anchor_pos[1]),
        "z0": float(anchor_pos[2]),
        "d0": float(frame.distance_m),
        "sigma_m": float(max(0.25, frame.sigma_m)),
        "theta0": float(theta0),
        "phi0": float(phi0),
        "kappa": float(kappa),
    }

    logp = engine.combined_logpdf(Xf, Yf, Zf, params).astype(np.float32)

    return {
        "anchor": {"id": anchor_obj.obj_id, "name": anchor_obj.name, "pos": anchor_pos.tolist()},
        "grid": {"xs": xs, "ys": ys, "zs": zs, "shape": (rx, ry, rz)},
        "logp": logp,
        "params": params,
    }


def summarize_region(region: Dict[str, Any], top_frac: float = 0.02) -> Dict[str, Any]:
    """
    Summarize top region by taking the top fraction of points (by logp),
    returning centroid + spread + confidence proxy.
    """
    if "error" in region:
        return {"error": region["error"]}

    xs = region["grid"]["xs"]
    ys = region["grid"]["ys"]
    zs = region["grid"]["zs"]
    rx, ry, rz = region["grid"]["shape"]

    logp = region["logp"]
    # convert flattened index -> xyz
    n = logp.shape[0]
    k = max(10, int(top_frac * n))

    idx = np.argpartition(logp, -k)[-k:]
    idx = idx[np.argsort(logp[idx])[::-1]]

    # map indices
    # index order: i * (ry*rz) + j*(rz) + k
    rr = ry * rz
    ii = idx // rr
    jj = (idx % rr) // rz
    kk = idx % rz

    pts = np.stack([xs[ii], ys[jj], zs[kk]], axis=1).astype(np.float32)

    centroid = pts.mean(axis=0)
    cov = np.cov(pts.T) if pts.shape[0] >= 3 else np.eye(3, dtype=np.float32)
    spread = np.sqrt(np.maximum(np.diag(cov), 1e-6)).astype(np.float32)

    # confidence proxy: how sharp the peak is vs rest
    # (logp_max - logp_median) mapped into [0,1]
    lp_max = float(logp[idx[0]])
    lp_med = float(np.median(logp))
    gap = max(0.0, lp_max - lp_med)
    conf = float(1.0 - np.exp(-gap / 5.0))

    return {
        "anchor": region["anchor"],
        "centroid_xyz": centroid.tolist(),
        "spread_xyz": spread.tolist(),
        "top_k_points_xyz": pts[: min(200, pts.shape[0])].tolist(),
        "confidence": conf,
        # filled later in save_region_artifacts:
        "nearby_objects": [],
    }


def save_region_artifacts(out_dir: os.PathLike, step: int, region: Dict[str, Any], summary: Dict[str, Any]) -> Dict[str, str]:
    """
    Saves:
      - npz: grid + logp + params + anchor
      - png: top-down heatmap slice (x-z plane at y near anchor)
    """
    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    npz_path = os.path.join(out_dir, f"region_step_{step:04d}.npz")
    png_path = os.path.join(out_dir, f"region_step_{step:04d}_heatmap.png")

    xs = region["grid"]["xs"]
    ys = region["grid"]["ys"]
    zs = region["grid"]["zs"]
    rx, ry, rz = region["grid"]["shape"]
    logp = region["logp"].reshape(rx, ry, rz)

    np.savez_compressed(
        npz_path,
        xs=xs, ys=ys, zs=zs,
        logp=logp,
        params=region["params"],
        anchor=region["anchor"],
        summary=summary,
    )

    # simple heatmap: take y slice closest to anchor y
    anchor_y = float(region["anchor"]["pos"][1])
    j = int(np.argmin(np.abs(ys - anchor_y)))

    heat = logp[:, j, :]  # (rx, rz)
    # normalize for visualization
    heat_viz = heat - np.max(heat)
    heat_viz = np.exp(heat_viz)  # convert to pseudo-prob

    plt.figure()
    plt.imshow(
        heat_viz.T,
        origin="lower",
        extent=[xs[0], xs[-1], zs[0], zs[-1]],
        aspect="auto",
    )
    plt.xlabel("x (hab)")
    plt.ylabel("z (hab)")
    plt.title(f"MSP Where-mode heatmap (y-slice @ {ys[j]:.2f}) step {step}")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    return {"npz": npz_path, "heatmap_png": png_path}
