#!/usr/bin/env python3
"""
Generate a spatial dataset (v2) for ONE scene/region using ONE reference class.

- Only uses objects from the specified HM3D region (strict membership).
- Reference is the first object whose category name contains --reference-substr (e.g., "couch").
- initial_pose: FORCED to a fixed spawn (x,y,z,yaw) for every question.
- For each question:
    * choose a predicate from {in front of, behind, to the right of, to the left of}
    * choose a distance in [--min-dist, --max-dist]
    * gather 3–5 candidate targets along the ray from the reference in that direction,
      using ONLY objects within the same region. The closest to the ideal ray target is the GT.

Output schema matches your v2 runner (no extra fields).
"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from tqdm import tqdm

import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis
from graph_eqa.envs.habitat_interface import HabitatInterface

DATASET_BASE_PATH = "/datasets/hm3d/train"

# world-frame unit directions (Y-up; -Z "forward")
SPATIAL_RELATIONSHIPS = {
    "in front of":     np.array([ 0.0, 0.0, -1.0], dtype=np.float32),
    "behind":          np.array([ 0.0, 0.0,  1.0], dtype=np.float32),
    "to the right of": np.array([ 1.0, 0.0,  0.0], dtype=np.float32),
    "to the left of":  np.array([-1.0, 0.0,  0.0], dtype=np.float32),
}

# ----------------- helpers -----------------
def call_maybe(x):
    return x() if callable(x) else x

def safe_center(aabb) -> np.ndarray:
    try:
        c = call_maybe(aabb.center)
    except Exception:
        c = (np.asarray(aabb.max, dtype=np.float32) + np.asarray(aabb.min, dtype=np.float32)) / 2.0
    return np.asarray(c, dtype=np.float32)

def safe_cat_name(cat) -> Optional[str]:
    if cat is None:
        return None
    try:
        n = call_maybe(cat.name)
    except Exception:
        try:
            n = cat.name()
        except Exception:
            n = None
    return None if n is None else str(n).lower()

def valid_xyz(v) -> bool:
    v = np.asarray(v, dtype=np.float32)
    return v.shape[-1] == 3 and np.all(np.isfinite(v))

def make_habitat_cfg():
    return SimpleNamespace(
        scene_type="hm3d",
        dataset_type="train",
        sim_gpu=0,
        inflation_radius=0.25,
        img_width=640,
        img_height=480,
        camera_height=1.5,
        camera_tilt_deg=-30,
        agent_z_offset=0.0,
        hfov=120,
        z_offset=0,
        use_semantic_data=True,
    )

def open_sim(scene_dir: str, scene_id: str):
    glb = os.path.join(DATASET_BASE_PATH, scene_dir, f"{scene_id}.basis.glb")
    if not os.path.exists(glb):
        raise FileNotFoundError(glb)
    hab_cfg = make_habitat_cfg()
    habitat_data = HabitatInterface(glb, cfg=hab_cfg, device="cpu")
    return habitat_data, habitat_data._sim

def region_by_id(scene, rid: str):
    for r in getattr(scene, "regions", []) or []:
        if str(getattr(r, "id", "")) == str(rid):
            return r
    return None

def region_objects(region) -> List[Any]:
    return list(getattr(region, "objects", []) or [])

# --- candidate gathering limited to region ---
def gather_candidates_from_region(region, ref_pos: np.ndarray, direction_vec: np.ndarray,
                                  desired_dist: float, k: int = 5) -> List[Dict[str, Any]]:
    """Collect up to k objects in the region roughly along ray(ref_pos + t*dir), near desired_dist."""
    ray_target = ref_pos + direction_vec * desired_dist
    cand: List[Dict[str, Any]] = []

    max_angle_deg = 35.0
    cos_th = float(np.cos(np.deg2rad(max_angle_deg)))
    min_t = 0.4 * desired_dist
    max_t = 1.8 * desired_dist
    dir_u = direction_vec / (np.linalg.norm(direction_vec) + 1e-8)

    for obj in region_objects(region):
        if not obj or obj.category is None:
            continue
        c = safe_center(obj.aabb)
        if not valid_xyz(c):
            continue
        v = c - ref_pos
        t = float(np.dot(v, dir_u))
        if t <= 0:
            continue
        if not (min_t <= t <= max_t):
            continue
        v_u = v / (np.linalg.norm(v) + 1e-8)
        if float(np.dot(v_u, dir_u)) < cos_th:
            continue
        d_err = float(np.linalg.norm(c - ray_target))
        cand.append({
            "id": obj.id,
            "name": safe_cat_name(obj.category) or "object",
            "position": c.astype(float).tolist(),
            "distance_to_target": d_err,
        })

    cand.sort(key=lambda x: x["distance_to_target"])
    seen, uniq = set(), []
    for it in cand:
        key = (it["id"], it["name"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(it)
        if len(uniq) >= k:
            break
    return uniq

# ----------------- core -----------------
def build_single_region_dataset(
    sim: "habitat_sim.Simulator",
    scene_dir: str,
    scene_id: str,
    region_id: str,
    reference_substr: str,
    num_questions: int,
    min_dist: float,
    max_dist: float,
    spawn_pos: Tuple[float, float, float],
    spawn_yaw: float,
) -> List[Dict[str, Any]]:

    scene = sim.semantic_scene
    region = region_by_id(scene, region_id)
    if region is None:
        raise RuntimeError(f"Region {region_id} not found.")

    # choose reference in region (first whose category name contains substring)
    ref_obj = None
    for o in region_objects(region):
        nm = safe_cat_name(o.category) or ""
        if reference_substr.lower() in nm:
            c = safe_center(o.aabb)
            if valid_xyz(c):
                ref_obj = (o, c, nm)
                break
    if ref_obj is None:
        raise RuntimeError(f"No reference object containing '{reference_substr}' found in region {region_id}.")

    ref, ref_pos, ref_name = ref_obj

    # initial pose: FIXED spawn (no snapping)
    spawn_pos = np.asarray(spawn_pos, dtype=np.float32)
    q = quat_from_angle_axis(float(spawn_yaw), np.array([0.0, 1.0, 0.0], dtype=np.float32))
    init_pose = {
        "position": spawn_pos.astype(float).tolist(),
        "rotation": [float(q.x), float(q.y), float(q.z), float(q.w)],
    }

    # make questions
    rng = np.random.default_rng(1234)
    rel_items = list(SPATIAL_RELATIONSHIPS.items())
    tasks: List[Dict[str, Any]] = []

    for _ in range(num_questions):
        # retry until we find candidates in the region
        for _attempt in range(60):
            rel_str, dir_vec = rel_items[rng.integers(0, len(rel_items))]
            dist = float(rng.uniform(min_dist, max_dist))
            cands = gather_candidates_from_region(region, ref_pos, dir_vec, dist, k=5)
            if not cands:
                cands = gather_candidates_from_region(region, ref_pos, dir_vec, max(0.2, dist*0.85), k=5)
            if cands:
                gt = dict(cands[0]); gt.pop("distance_to_target", None)
                tasks.append({
                    "scene_dir": scene_dir,
                    "scene_id": scene_id,
                    "scene": scene_dir,
                    "question": f"What object is approximately {round(dist,1)} meters {rel_str} the {ref_name}?",
                    "initial_pose": init_pose,
                    "reference_object": {
                        "id": ref.id,
                        "name": ref_name,
                        "position": ref_pos.astype(float).tolist(),
                    },
                    "reference_room": {
                        "id": region_id,
                        "name": safe_cat_name(getattr(region, "category", None)),
                        # keep center == spawn for consistency with your runner/planner
                        "center": spawn_pos.astype(float).tolist(),
                    },
                    "candidate_targets": cands,
                    "ground_truth_target": gt,
                })
                break

    return tasks

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--scene-id", required=True)
    ap.add_argument("--region-id", required=True)
    ap.add_argument("--reference-substr", default="couch", help="Substring match for reference class (lowercase)")
    ap.add_argument("--num-questions", type=int, default=10)
    ap.add_argument("--min-dist", type=float, default=0.2)
    ap.add_argument("--max-dist", type=float, default=2.0)
    # fixed spawn pose for all questions
    ap.add_argument("--spawn-x", type=float, required=True)
    ap.add_argument("--spawn-y", type=float, required=True)
    ap.add_argument("--spawn-z", type=float, required=True)
    ap.add_argument("--spawn-yaw", type=float, required=True, help="Yaw in radians, +CCW around +Y")
    ap.add_argument("--output-path", type=str, default="spatial_experiment/data/spatial_dataset_v2.json")
    args = ap.parse_args()

    habitat_data, sim = open_sim(args.scene_dir, args.scene_id)
    try:
        tasks = build_single_region_dataset(
            sim=sim,
            scene_dir=args.scene_dir,
            scene_id=args.scene_id,
            region_id=args.region_id,
            reference_substr=args.reference_substr,
            num_questions=args.num_questions,
            min_dist=args.min_dist,
            max_dist=args.max_dist,
            spawn_pos=(args.spawn_x, args.spawn_y, args.spawn_z),
            spawn_yaw=args.spawn_yaw,
        )
    finally:
        try:
            habitat_data._sim.close(destroy=True)
        except Exception:
            pass

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(tasks, f, indent=2)
    print(f"[saved] {len(tasks)} items → {out_path}")

if __name__ == "__main__":
    main()
