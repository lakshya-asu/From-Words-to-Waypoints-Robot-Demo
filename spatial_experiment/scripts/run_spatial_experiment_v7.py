#!/usr/bin/env python3
"""
v7 runner:
- Imports VLMPlannerEQAGeminiSpatialV7 (VLM-based Predicate).
- Passes 'agent_yaw_rad' to the planner for Affordance-Aware reasoning.
- Fixes SceneGraphSim initialization to prevent 'NoneType' errors.
"""

import json
from pathlib import Path
from datetime import datetime
import os
import click
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

import habitat_sim
from habitat_sim.utils.common import quat_from_coeffs

import hydra_python

# --- V7 IMPORT ---
from spatial_experiment.planners.vlm_planner_gemini_spatial_v7 import (
    VLMPlannerEQAGeminiSpatialV7,
)

from graph_eqa.envs.habitat import run
from graph_eqa.envs.habitat_interface import HabitatInterface
from graph_eqa.logging.rr_logger import RRLogger
from graph_eqa.logging.utils import log_experiment_status, should_skip_experiment
from graph_eqa.occupancy_mapping.geom import get_cam_intr, get_scene_bnds
from graph_eqa.occupancy_mapping.tsdf import TSDFPlanner
from graph_eqa.scene_graph.scene_graph_sim import SceneGraphSim
from graph_eqa.utils.data_utils import get_traj_len_from_poses
from graph_eqa.utils.hydra_utils import initialize_hydra_pipeline
from graph_eqa.envs.utils import pos_habitat_to_normal, pos_normal_to_habitat

if not OmegaConf.has_resolver("env"):
    OmegaConf.register_new_resolver(
        "env",
        lambda key, default=None: os.environ.get(str(key), default),
    )

# --- SG warm-up helpers ---
def _warmup_like_eqa(habitat_data: HabitatInterface, pipeline, out_dir, rr_logger, tsdf_planner, sg_sim):
    agent = habitat_data._sim.get_agent(0)
    init_pts_hab = np.array(agent.get_state().position, dtype=np.float32)
    init_angle = float(habitat_data.get_heading_angle())
    tilt = float(getattr(habitat_data.cfg, "camera_tilt_deg", -30.0))
    poses = habitat_data.get_init_poses_eqa(init_pts_hab, init_angle, tilt)
    try:
        run(
            pipeline, habitat_data, poses, output_path=out_dir,
            rr_logger=rr_logger, tsdf_planner=tsdf_planner, sg_sim=sg_sim,
            save_image=True, segmenter=None,
        )
    except Exception as e:
        click.secho(f"[warmup] non-fatal render error: {e}", fg="yellow")

def _ensure_scenegraph_initialized(habitat_data: HabitatInterface, pipeline, out_dir, rr_logger, tsdf_planner, sg_sim):
    need = (
        not hasattr(sg_sim, "filtered_netx_graph")
        or sg_sim.filtered_netx_graph is None
        or not hasattr(sg_sim, "curr_agent_id")
    )
    if not need:
        return
    click.secho("[init] SG not ready after warm-up; forcing one more tick…", fg="yellow")
    agent = habitat_data._sim.get_agent(0)
    init_pts_hab = np.array(agent.get_state().position, dtype=np.float32)
    init_angle = float(habitat_data.get_heading_angle())
    tilt = float(getattr(habitat_data.cfg, "camera_tilt_deg", -30.0))
    poses = habitat_data.get_init_poses_eqa(init_pts_hab, init_angle, tilt)
    try:
        run(
            pipeline, habitat_data,
            poses[:2] if len(poses) > 2 else poses,
            output_path=out_dir, rr_logger=rr_logger,
            tsdf_planner=tsdf_planner, sg_sim=sg_sim,
            save_image=True, segmenter=None,
        )
    except Exception as e:
        click.secho(f"[init] non-fatal render error: {e}", fg="yellow")


def _teleport_agent(habitat_data: HabitatInterface, init_pos, init_rot):
    st = habitat_sim.AgentState()
    st.position = np.asarray(init_pos, dtype=np.float32)
    st.rotation = quat_from_coeffs(np.asarray(init_rot, dtype=np.float32))
    habitat_data._sim.get_agent(0).set_state(st)


def _shortest_path_len_pf(habitat_data: HabitatInterface, start_pos_hab, goal_pos_hab) -> float:
    pf = habitat_data.pathfinder
    sp = habitat_sim.nav.ShortestPath()
    sp.requested_start = np.array(start_pos_hab, dtype=np.float32)
    sp.requested_end = np.array(goal_pos_hab, dtype=np.float32)
    if not pf.find_path(sp) or len(sp.points) < 2:
        return 0.0
    pts = np.asarray(sp.points, dtype=np.float32)
    segs = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    return float(segs.sum())


def main(cfg):
    with open(cfg.data.spatial_data_path, "r") as f:
        tasks_data = json.load(f)

    base_output_root = Path(__file__).resolve().parent.parent / cfg.output_path
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = base_output_root / run_stamp
    output_path.mkdir(parents=True, exist_ok=True)

    results_filename = output_path / f"{cfg.results_filename}.json"
    device = f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu"

    for task_index, task_data in enumerate(tqdm(tasks_data, desc="Running Tasks (V7 Gemini)")):
        scene_dir = task_data.get("scene_dir", task_data.get("scene"))
        if scene_dir is None:
            click.secho(f"[skip] Task {task_index}: missing scene info.", fg="yellow")
            continue
        scene_id = task_data.get("scene_id", scene_dir.split("-", 1)[-1])
        scene_glb_path = f"{cfg.data.scene_data_path}/{scene_dir}/{scene_id}.basis.glb"

        question = str(task_data["question"])
        gt = task_data["ground_truth_target"]
        candidates = task_data.get("candidate_targets", [])
        reference_object = task_data.get("reference_object", {})
        reference_room = task_data.get("reference_room", {})
        experiment_id = f'{task_index}_{scene_dir}_{gt["name"]}'

        if should_skip_experiment(experiment_id, filename=results_filename):
            click.secho(f"Skipping finished task: {experiment_id}", fg="yellow")
            continue

        task_output_path: Path = hydra_python.resolve_output_path(output_path / experiment_id)
        habitat_data = None
        pipeline = None
        try:
            click.secho(f"[v7] Executing task: {experiment_id}", fg="green")
            click.secho(f"Question: {question}", fg="cyan")

            # --- Habitat + pipeline ---
            habitat_data = HabitatInterface(scene_glb_path, cfg=cfg.habitat, device=device)
            pipeline = initialize_hydra_pipeline(cfg.hydra, habitat_data, task_output_path)
            rr_logger = RRLogger(task_output_path)

            # Spawn from dataset pose
            init_pos = np.array(task_data["initial_pose"]["position"], dtype=np.float32)
            init_rot = np.array(task_data["initial_pose"]["rotation"], dtype=np.float32)
            _teleport_agent(habitat_data, init_pos, init_rot)
            click.secho("[spawn] dataset initial_pose", fg="cyan")

            # TSDF + SG
            floor_height = float(init_pos[1])
            tsdf_bnds, _ = get_scene_bnds(habitat_data.pathfinder, floor_height)
            cam_intr = get_cam_intr(cfg.habitat.hfov, cfg.habitat.img_height, cfg.habitat.img_width)
            tsdf_planner = TSDFPlanner(
                cfg=cfg.frontier_mapping,
                vol_bnds=tsdf_bnds,
                cam_intr=cam_intr,
                floor_height_offset=0,
                pts_init=init_pos,
                rr_logger=rr_logger,
            )

            # --- FIXED: Added enrich_object_labels=" " to fix the 'NoneType' crash ---
            sg_sim = SceneGraphSim(
                cfg, task_output_path, pipeline, rr_logger,
                device=device, clean_ques_ans=question, 
                enrich_object_labels=" " 
            )

            _warmup_like_eqa(habitat_data, pipeline, task_output_path, rr_logger, tsdf_planner, sg_sim)
            _ensure_scenegraph_initialized(habitat_data, pipeline, task_output_path, rr_logger, tsdf_planner, sg_sim)

            # Re-teleport back to dataset pose
            _teleport_agent(habitat_data, init_pos, init_rot)

            if (
                not hasattr(sg_sim, "filtered_netx_graph") or sg_sim.filtered_netx_graph is None
                or not hasattr(sg_sim, "curr_agent_id")
            ):
                click.secho("[skip] SceneGraph not initialized; skipping task.", fg="yellow")
                log_experiment_status(
                    experiment_id, False,
                    metrics={"error": "scene_graph_not_initialized"},
                    filename=results_filename
                )
                continue

            # --- V7 PLANNER INITIALIZATION ---
            vlm_planner = VLMPlannerEQAGeminiSpatialV7(
                cfg.vlm, sg_sim, question, gt, task_output_path,
                reference_object=reference_object, reference_room=reference_room,
                candidate_targets=candidates,
            )

            num_steps = 20
            traj_length = 0.0
            target_declaration = {}
            succ = False

            for step_count in range(num_steps):
                # --- CHANGE FOR V7: PASS HEADING ---
                current_heading = float(habitat_data.get_heading_angle())
                
                target_pose, target_id, is_confident, confidence_level, target_declaration = (
                    vlm_planner.get_next_action(agent_yaw_rad=current_heading)
                )
                # -----------------------------------

                if is_confident or (step_count == num_steps - 1) or (cfg.vlm.early_exit and step_count >= 0):
                    click.secho(f"Termination condition. confident={is_confident}", fg="blue")
                    break

                if target_pose is None:
                    click.secho("Planner did not provide a valid navigation target.", fg="red")
                    continue

                agent = habitat_data._sim.get_agent(0)
                current_pos_hab = np.array(agent.get_state().position, dtype=np.float32)

                target_hab = pos_normal_to_habitat(np.array(target_pose, dtype=np.float32))
                target_hab[1] = current_pos_hab[1]

                spath = habitat_sim.nav.ShortestPath()
                spath.requested_start = current_pos_hab
                spath.requested_end = target_hab
                found = habitat_data.pathfinder.find_path(spath)
                if not found or not spath.points:
                    click.secho("Cannot find navigable path. Continuing..", fg="red")
                    continue

                desired_path_norm = pos_habitat_to_normal(np.array(spath.points)[:-1])
                try:
                    rr_logger.log_traj_data(desired_path_norm)
                    rr_logger.log_target_poses(target_pose)
                except Exception:
                    pass

                poses = habitat_data.get_trajectory_from_path_habitat_frame(
                    target_pose, desired_path_norm, habitat_data.get_heading_angle(), cfg.habitat.camera_tilt_deg
                )
                if poses is None:
                    click.secho("Cannot build trajectory from path. Continuing..", fg="red")
                    continue

                click.secho(f"Executing trajectory at step {step_count}", fg="yellow")
                run(
                    pipeline, habitat_data, poses, output_path=task_output_path,
                    rr_logger=rr_logger, tsdf_planner=tsdf_planner, sg_sim=sg_sim,
                    save_image=cfg.vlm.use_image, segmenter=None,
                )
                traj_length += get_traj_len_from_poses(poses)

            # ---- Metrics ----
            declared_id = target_declaration.get("declared_target_object_id", "N/A")
            gt_id = gt["id"]
            cand_ids = {c["id"] for c in candidates} if candidates else set()
            succ_primary = (str(declared_id) == str(gt_id))
            succ_any_candidate = (declared_id in cand_ids)
            succ = succ_primary or succ_any_candidate

            msg = "SUCCESS (GT)" if succ_primary else ("SUCCESS (candidate)" if succ_any_candidate else "FAILURE")
            click.secho(f"{msg}. chosen={declared_id} expected={gt_id}", fg=("green" if succ else "red"))

            agent_final_pos = habitat_data._sim.get_agent(0).get_state().position
            gt_pos_hab = np.array(gt["position"], dtype=np.float32)
            loc_err = float(np.linalg.norm(agent_final_pos - gt_pos_hab))

            shortest_path_len = _shortest_path_len_pf(habitat_data, init_pos, gt_pos_hab)
            spl = (shortest_path_len / max(traj_length, shortest_path_len)) if shortest_path_len > 0 else 0.0
            if not succ_primary:
                spl = 0.0

            metrics = {
                "success": succ,
                "success_gt": succ_primary,
                "success_any_candidate": succ_any_candidate,
                "localization_error": loc_err,
                "spl": float(spl),
                "vlm_steps": vlm_planner.t,
                "total_steps": step_count + 1,
                "confidence_level": float(target_declaration.get("confidence_level", 0.0)),
                "traj_length": float(traj_length),
                "shortest_path_len": float(shortest_path_len),
                "declared_id": declared_id,
                "ground_truth_id": gt_id,
                "num_candidates": len(candidates),
            }
            log_experiment_status(experiment_id, succ, metrics=metrics, filename=results_filename)

        finally:
            try:
                if habitat_data is not None and hasattr(habitat_data, "_sim"):
                    habitat_data._sim.close(destroy=True)
            except Exception:
                pass
            try:
                if pipeline is not None:
                    pipeline.save()
            except Exception:
                pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cf", "--cfg_file", help="cfg file name (without .yaml)",
        # Default to v7 config
        default="spatial_vqa_v7", type=str, required=False,
    )
    args = parser.parse_args()
    config_path = Path(__file__).resolve().parent.parent / "cfg" / f"{args.cfg_file}.yaml"
    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    main(cfg)