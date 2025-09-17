#!/usr/bin/env python3
import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Any
from types import SimpleNamespace

import numpy as np
from tqdm import tqdm

import habitat_sim  # only for quat helper
from graph_eqa.envs.habitat_interface import HabitatInterface

# --- Paths ---
SEM_LIST_PATH = "/datasets/hm3d/train/train-semantic-annots-files.json"
DATASET_BASE_PATH = "/datasets/hm3d/train"  # root of HM3D train split

# Reproducibility
random.seed(42)
np.random.seed(42)

# Categories (lowercased comparison)
REFERENCE_CATEGORIES = {
    c.lower() for c in [
        "sofa", "bed", "oven", "refrigerator", "dining table", "tv stand",
        "armchair", "desk", "toilet", "kitchen island",
    ]
}

# World-frame unit directions (Y-up; Z-forward = -Z for yaw below)
SPATIAL_RELATIONSHIPS = {
    "in front of":     np.array([ 0.0, 0.0, -1.0]),  # forward (-Z)
    "behind":          np.array([ 0.0, 0.0,  1.0]),  # back (+Z)
    "to the right of": np.array([ 1.0, 0.0,  0.0]),  # right (+X)
    "to the left of":  np.array([-1.0, 0.0,  0.0]),  # left  (-X)
}

# ---------- helpers to handle API differences ----------
def call_maybe(x):
    """Return x() if callable, else x."""
    return x() if callable(x) else x

def safe_center(aabb):
    """Return np.float32[3] center for AABB across habitat-sim API variants."""
    try:
        c = call_maybe(aabb.center)
    except Exception:
        # some builds expose aabb as tuple-like [min, max]; fallback to mid-point
        c = (np.asarray(aabb.max, dtype=np.float32) + np.asarray(aabb.min, dtype=np.float32)) / 2.0
    c = np.asarray(c, dtype=np.float32)
    return c

def safe_cat_name(cat):
    """Return lowercase category name across API variants (name or name())."""
    if cat is None:
        return None
    try:
        n = call_maybe(cat.name)
    except Exception:
        try:
            n = cat.name()  # just in case
        except Exception:
            n = None
    if n is None:
        return None
    return str(n).lower()

def valid_center(v) -> bool:
    v = np.asarray(v, dtype=np.float32)
    return v.shape[-1] == 3 and np.all(np.isfinite(v))

# ---------- dataset discovery ----------
def load_valid_scenes() -> list[tuple[str, str]]:
    """
    Returns a list of (scene_dir, scene_id) tuples for HM3D train scenes that have both
    <scene_id>.basis.glb and <scene_id>.semantic.glb present under DATASET_BASE_PATH/scene_dir/.
    """
    print(f"Loading semantic scene index from {SEM_LIST_PATH}...")
    try:
        with open(SEM_LIST_PATH) as f:
            all_files = json.load(f)
    except FileNotFoundError:
        print(f"FATAL ERROR: Semantic index file not found at {SEM_LIST_PATH}.")
        raise SystemExit(1)

    valid = []
    for p in all_files:
        if not p.endswith(".semantic.glb"):
            continue
        parts = Path(p).parts
        scene_dir = parts[-2]                        # e.g. "00006-HkseAnWCgqk"
        scene_id = Path(parts[-1]).stem.replace(".semantic", "")  # "HkseAnWCgqk"
        base = os.path.join(DATASET_BASE_PATH, scene_dir, scene_id)
        if os.path.exists(base + ".semantic.glb") and os.path.exists(base + ".basis.glb"):
            valid.append((scene_dir, scene_id))
    print(f"Found {len(valid)} scenes with valid semantics.")
    return valid

def make_habitat_cfg():
    """
    Mirror the YAML `habitat:` block used by your working EQA runner.
    Using SimpleNamespace keeps it lightweight and compatible with HabitatInterface.
    """
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

def open_sim_with_habitat_interface(scene_dir: str, scene_id: str):
    """
    Construct the simulator the SAME way as run_vlm_planner_eqa_habitat.py:
    - load <scene_id>.basis.glb via HabitatInterface (which also binds semantics)
    Returns (habitat_data, sim) or (None, None) on failure.
    """
    basis_glb = os.path.join(DATASET_BASE_PATH, scene_dir, f"{scene_id}.basis.glb")
    if not os.path.exists(basis_glb):
        print(f"[Skip] Missing basis GLB: {basis_glb}")
        return None, None
    habitat_data = None
    try:
        hab_cfg = make_habitat_cfg()
        habitat_data = HabitatInterface(basis_glb, cfg=hab_cfg, device="cpu")
        sim = habitat_data._sim
        if not sim.semantic_scene or not getattr(sim.semantic_scene, "objects", None):
            print(f"[Skip] No semantic objects in {scene_dir}/{scene_id}")
            habitat_data._sim.close(destroy=True)
            return None, None
        return habitat_data, sim
    except Exception as e:
        print(f"[Error] HabitatInterface failed for {scene_dir}/{scene_id}: {e}")
        try:
            if habitat_data is not None:
                habitat_data._sim.close(destroy=True)
        except Exception:
            pass
        return None, None

def try_get_start(sim, ref_pos):
    """
    Prefer a true navigable point via pathfinder; otherwise synthesize a start 2m away.
    """
    try:
        if hasattr(sim, "pathfinder") and sim.pathfinder.is_loaded:
            p = sim.pathfinder.get_random_navigable_point()
            if sim.pathfinder.is_navigable(p):
                return p
    except Exception:
        pass
    p = np.array(ref_pos, dtype=np.float32)
    p[0] += 2.0  # +X offset
    return p

# ---------- core generation ----------
def generate_spatial_tasks(
    sim: "habitat_sim.Simulator",
    scene_dir: str,
    scene_id: str,
    num_questions_per_scene: int
) -> List[Dict[str, Any]]:
    tasks = []
    scene = sim.semantic_scene
    if not scene or not scene.objects:
        print("Warning: Scene loaded without semantic objects. Skipping.")
        return []

    # Filter reference objects by category (lowercase compare)
    valid_ref_objects = []
    for obj in scene.objects:
        if not obj or obj.category is None:
            continue
        name_lc = safe_cat_name(obj.category)
        if name_lc is None or name_lc not in REFERENCE_CATEGORIES:
            continue
        center = safe_center(obj.aabb)
        if not valid_center(center):
            continue
        valid_ref_objects.append((obj, center, name_lc))

    if not valid_ref_objects:
        return []

    # Don’t oversample if the scene is sparse
    num_to_make = min(num_questions_per_scene, len(valid_ref_objects))

    for _ in range(num_to_make):
        ref_obj, ref_pos, ref_name_lc = random.choice(valid_ref_objects)
        relationship_str, direction_vec = random.choice(list(SPATIAL_RELATIONSHIPS.items()))
        distance = round(random.uniform(1.0, 3.0), 1)

        # World-frame target position
        target_pos = ref_pos + direction_vec * distance

        # Find nearest object (exclude ref)
        closest_obj, closest_center, min_dist = None, None, float("inf")
        for obj in scene.objects:
            if not obj or obj.id == ref_obj.id or obj.category is None:
                continue
            c = safe_center(obj.aabb)
            if not valid_center(c):
                continue
            d = np.linalg.norm(c - target_pos)
            if d < min_dist:
                min_dist, closest_obj, closest_center = d, obj, c
        if closest_obj is None:
            continue

        # Create start pose
        start_pos = try_get_start(sim, ref_pos)
        direction_to_ref = ref_pos - start_pos
        # yaw that faces -Z forward convention -> angle = atan2(x, -z)
        yaw = np.arctan2(direction_to_ref[0], -direction_to_ref[2])
        start_rotation = habitat_sim.utils.common.quat_from_angle_axis(
            yaw, np.array([0.0, 1.0, 0.0], dtype=np.float32)
        )

        # Natural-language question
        ref_human_name = safe_cat_name(ref_obj.category) or "object"
        question = f"What object is approximately {distance} meters {relationship_str} the {ref_human_name}?"

        # NOTE:
        # - Include BOTH scene_dir and scene_id for downstream path building.
        # - Also include "scene" = scene_dir for compatibility with older code that expected it.
        tasks.append({
            "scene_dir": scene_dir,                         # e.g., "00006-HkseAnWCgqk"  (folder)
            "scene_id": scene_id,                           # e.g., "HkseAnWCgqk"       (file stem)
            "scene": scene_dir,                             # compatibility with older code
            "question": question,
            "initial_pose": {
                "position": np.asarray(start_pos, dtype=float).tolist(),
                "rotation": [start_rotation.x, start_rotation.y, start_rotation.z, start_rotation.w],
            },
            "reference_object": {
                "id": ref_obj.id,
                "name": ref_human_name,
                "position": ref_pos.astype(float).tolist(),
            },
            "ground_truth_target": {
                "id": closest_obj.id,
                "name": safe_cat_name(closest_obj.category) or "object",
                "position": np.asarray(closest_center, dtype=float).tolist(),
            },
        })

    return tasks

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(description="Generate a spatial reasoning dataset for Habitat.")
    parser.add_argument("--output-path", type=str, default="spatial_experiment/data/spatial_dataset_v1.json")
    parser.add_argument("--num-scenes", type=int, default=10)
    parser.add_argument("--num-questions-per-scene", type=int, default=5)
    args = parser.parse_args()

    valid_scenes = load_valid_scenes()
    num_to_sample = min(args.num_scenes, len(valid_scenes))
    selected = random.sample(valid_scenes, num_to_sample)

    all_tasks = []
    print(f"Generating {args.num_questions_per_scene} questions for {len(selected)} scenes...")

    for scene_dir, scene_id in tqdm(selected, desc="Processing Scenes"):
        habitat_data, sim = open_sim_with_habitat_interface(scene_dir, scene_id)
        if sim is None:
            continue
        try:
            tasks_for_scene = generate_spatial_tasks(sim, scene_dir, scene_id, args.num_questions_per_scene)
            all_tasks.extend(tasks_for_scene)
        except Exception as e:
            print(f"[Error] generation failed for {scene_dir}/{scene_id}: {e}")
        finally:
            # Fully destroy between scenes to avoid asset config reload issues
            try:
                habitat_data._sim.close(destroy=True)
            except Exception:
                pass

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_tasks, f, indent=4)

    print(f"\nSuccessfully generated {len(all_tasks)} tasks.")
    print(f"Dataset saved to: {args.output_path}")

if __name__ == "__main__":
    main()
