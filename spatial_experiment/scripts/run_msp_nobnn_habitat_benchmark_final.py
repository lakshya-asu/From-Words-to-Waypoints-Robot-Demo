# /home/artemis/project/graph_eqa_swagat/spatial_experiment/scripts/run_msp_nobnn_habitat_benchmark_final.py

from tqdm import tqdm
from omegaconf import OmegaConf
import click
import os, time
from pathlib import Path
import numpy as np
import torch
import csv
import json

from graph_eqa.logging.utils import should_skip_experiment, log_experiment_status
from graph_eqa.envs.utils import pos_habitat_to_normal, pos_normal_to_habitat
from graph_eqa.occupancy_mapping.geom import get_scene_bnds, get_cam_intr
from graph_eqa.envs.habitat import run
from graph_eqa.logging.rr_logger import RRLogger
from graph_eqa.occupancy_mapping.tsdf import TSDFPlanner
from graph_eqa.utils.data_utils import get_traj_len_from_poses
from graph_eqa.utils.hydra_utils import initialize_hydra_pipeline

from graph_eqa.scene_graph.scene_graph_sim import SceneGraphSim
from graph_eqa.envs.habitat_interface import HabitatInterface

import habitat_sim
import hydra_python

from spatial_experiment.planners.vlm_planner_msp_nobnn_final import VLMPlannerMSP_NoBNN_Final


SEM_LIST = "/datasets/hm3d/train/train-semantic-annots-files.json"
with open(SEM_LIST) as f:
    _semantic_ok = set()
    for p in json.load(f):
        base = os.path.basename(p).split(".")[0]
        _semantic_ok.add(base)


def scene_has_semantics(scene_id: str) -> bool:
    return scene_id in _semantic_ok


def load_init_poses_csv(init_pose_path: str):
    out = {}
    with open(init_pose_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scene_floor = row["scene_floor"]
            x = float(row["init_x"])
            y = float(row["init_y"])
            z = float(row["init_z"])
            ang = float(row["init_angle"])
            out[scene_floor] = {"init_pts": np.array([x, y, z], dtype=np.float32), "init_angle": ang}
    return out


def load_questions_msp_csv(qpath: str):
    data = []
    with open(qpath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def is_msp_csv(qpath: str) -> bool:
    with open(qpath, "r") as f:
        header = f.readline().strip().split(",")
    return "msp_question" in header


# -----------------------------
# FIXED: lookaround + walkaround
# -----------------------------
def run_lookaround(
    pipeline,
    habitat_data,
    rr_logger,
    tsdf_planner,
    sg_sim,
    output_path,
    segmenter=None,
    save_image: bool = True,
    camera_tilt_deg: float = 0.0,
    num_yaws: int = 8,
):
    """
    Rotate in place and collect observations.

    IMPORTANT:
    - Pose format MUST match HabitatInterface.get_init_poses_eqa() output.
    - get_init_poses_eqa() in this repo uses positional args (no init_pts=... keywords).
    """
    click.secho(f"[explore] lookaround num_yaws={num_yaws}", fg="yellow")

    agent = habitat_data._sim.get_agent(0)
    st = agent.get_state()
    pos = st.position  # HAB coords (x,y,z)

    yaw0_rad = float(habitat_data.get_heading_angle())
    yaw0_deg = float(np.degrees(yaw0_rad))

    ny = max(1, int(num_yaws))
    for k in range(ny):
        yaw_deg = yaw0_deg + (360.0 * k / float(ny))

        # ✅ positional args only (matches baseline)
        poses = habitat_data.get_init_poses_eqa(
            np.array([pos[0], pos[1], pos[2]], dtype=np.float32),
            float(yaw_deg),
            float(camera_tilt_deg),
        )

        run(
            pipeline,
            habitat_data,
            poses,
            output_path=output_path,
            rr_logger=rr_logger,
            tsdf_planner=tsdf_planner,
            sg_sim=sg_sim,
            save_image=save_image,
            segmenter=segmenter,
        )


def run_walkaround(
    pipeline,
    habitat_data,
    rr_logger,
    tsdf_planner,
    sg_sim,
    output_path,
    segmenter=None,
    save_image: bool = True,
    camera_tilt_deg: float = 0.0,
    radius_m: float = 0.75,
    num_waypoints: int = 6,
    do_lookaround_each: bool = True,
    lookaround_yaws: int = 6,
):
    """
    Walk a small loop around current position to get parallax + new objects.
    Uses nav stack, then (optionally) a lookaround at each waypoint.

    Returns: total trajectory length executed (meters).
    """
    click.secho(f"[explore] walkaround radius_m={radius_m} num_waypoints={num_waypoints}", fg="yellow")

    agent = habitat_data._sim.get_agent(0)
    st = agent.get_state()
    center = np.array([st.position[0], st.position[1], st.position[2]], dtype=np.float32)
    center_h = float(center[1])

    r = float(radius_m)
    k = max(3, int(num_waypoints))
    angles = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)

    traj_len_total = 0.0

    for a in angles:
        wp = center.copy()
        wp[0] += r * float(np.cos(a))
        wp[2] += r * float(np.sin(a))
        wp[1] = center_h

        # pathfinder expects HAB coords
        path = habitat_sim.nav.ShortestPath()
        path.requested_start = st.position
        path.requested_end = wp

        found = habitat_data.pathfinder.find_path(path)
        if (not found) or (path.points is None) or (len(path.points) < 2):
            click.secho("[walkaround] no path to waypoint; skipping", fg="red")
            continue

        desired_path_normal = pos_habitat_to_normal(np.array(path.points)[:-1])
        rr_logger.log_traj_data(desired_path_normal)

        current_heading = habitat_data.get_heading_angle()

        # IMPORTANT: get_trajectory_from_path_habitat_frame expects target_pose in NORMAL coords
        wp_normal = pos_habitat_to_normal(np.array([wp], dtype=np.float32))[0]

        poses = habitat_data.get_trajectory_from_path_habitat_frame(
            wp_normal,
            desired_path_normal,
            current_heading,
            float(camera_tilt_deg),
        )
        if poses is None:
            click.secho("[walkaround] could not generate trajectory; skipping", fg="red")
            continue

        run(
            pipeline,
            habitat_data,
            poses,
            output_path=output_path,
            rr_logger=rr_logger,
            tsdf_planner=tsdf_planner,
            sg_sim=sg_sim,
            save_image=save_image,
            segmenter=segmenter,
        )
        traj_len_total += float(get_traj_len_from_poses(poses))

        # refresh state after moving
        agent = habitat_data._sim.get_agent(0)
        st = agent.get_state()
        center = np.array([st.position[0], st.position[1], st.position[2]], dtype=np.float32)
        center[1] = center_h

        if do_lookaround_each:
            run_lookaround(
                pipeline=pipeline,
                habitat_data=habitat_data,
                rr_logger=rr_logger,
                tsdf_planner=tsdf_planner,
                sg_sim=sg_sim,
                output_path=output_path,
                segmenter=segmenter,
                save_image=save_image,
                camera_tilt_deg=float(camera_tilt_deg),
                num_yaws=int(lookaround_yaws),
            )

    return traj_len_total


def main(cfg):
    qpath = cfg.data.question_data_path
    if not is_msp_csv(qpath):
        raise RuntimeError("This runner is MSP-only. Provide questions_msp_*.csv with msp_question column.")

    click.secho("[mode] MSP No-BNN FINAL runner", fg="cyan")
    questions_data = load_questions_msp_csv(qpath)
    init_pose_data = load_init_poses_csv(cfg.data.init_pose_data_path)

    output_path = Path(__file__).resolve().parent.parent / cfg.output_path
    os.makedirs(str(output_path), exist_ok=True)
    results_filename = output_path / f"{cfg.results_filename}.json"
    device = f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu"

    if not cfg.data.use_semantic_data:
        from hydra_python.detection.detic_segmenter import DeticSegmenter
        segmenter = DeticSegmenter(cfg)
    else:
        segmenter = None

    for question_ind in tqdm(range(len(questions_data))):
        q = questions_data[question_ind]
        scene = q["scene"]
        floor = str(q["floor"])
        scene_floor = f"{scene}_{floor}"
        scene_id = scene[6:]

        msp_question = q["msp_question"]
        primary_object = q.get("primary_object", None)
        experiment_id = f"{question_ind}_{scene}_{floor}"

        if should_skip_experiment(experiment_id, filename=results_filename):
            click.secho(f"Skipping==Index: {question_ind} Scene: {scene} Floor: {floor}=======", fg="yellow")
            continue
        click.secho(f"Executing=========Index: {question_ind} Scene: {scene} Floor: {floor}=======", fg="green")

        raw_question_path = output_path / experiment_id
        if raw_question_path.exists():
            click.secho(f"[resume] Reusing existing folder: {raw_question_path}", fg="yellow")
            question_path = raw_question_path
        else:
            question_path = hydra_python.resolve_output_path(raw_question_path)

        scene_name = f"{cfg.data.scene_data_path}/{scene}/{scene_id}.basis.glb"
        if cfg.data.use_semantic_data and not scene_has_semantics(scene_id):
            click.secho(f"[skip] {scene} has no semantics; skipping.", fg="yellow")
            continue

        habitat_data = HabitatInterface(scene_name, cfg=cfg.habitat, device=device)
        pipeline = initialize_hydra_pipeline(cfg.hydra, habitat_data, question_path)
        rr_logger = RRLogger(question_path)

        init_pts = init_pose_data[scene_floor]["init_pts"]
        init_angle = init_pose_data[scene_floor]["init_angle"]

        pts_normal = pos_habitat_to_normal(init_pts)
        floor_height = pts_normal[-1]
        tsdf_bnds, scene_size = get_scene_bnds(habitat_data.pathfinder, floor_height)
        cam_intr = get_cam_intr(cfg.habitat.hfov, cfg.habitat.img_height, cfg.habitat.img_width)

        tsdf_planner = TSDFPlanner(
            cfg=cfg.frontier_mapping,
            vol_bnds=tsdf_bnds,
            cam_intr=cam_intr,
            floor_height_offset=0,
            pts_init=pts_normal,
            rr_logger=rr_logger,
        )

        sg_sim = SceneGraphSim(
            cfg,
            question_path,
            pipeline,
            rr_logger,
            device=device,
            clean_ques_ans=" ",
            enrich_object_labels=" ",
        )

        # Initial sweep
        poses0 = habitat_data.get_init_poses_eqa(init_pts, init_angle, cfg.habitat.camera_tilt_deg)
        run(
            pipeline,
            habitat_data,
            poses0,
            output_path=question_path,
            rr_logger=rr_logger,
            tsdf_planner=tsdf_planner,
            sg_sim=sg_sim,
            save_image=cfg.vlm.use_image,
            segmenter=segmenter,
        )

        vlm_question = f"{msp_question} (Anchor object id: {primary_object})" if primary_object else msp_question

        vlm_planner = VLMPlannerMSP_NoBNN_Final(
            cfg.vlm,
            sg_sim,
            vlm_question,
            gt=None,
            out_path=str(question_path),
            anchor_object_id=primary_object,
        )
        # --- DEBUG: Aggressive Probe ---
        click.secho("\n=== [DEBUG] Probing SceneGraphSim Internals ===", fg="magenta", bold=True)
        
        # 1. Let's see all attributes available in sg_sim
        attrs = [a for a in dir(sg_sim) if not a.startswith('__')]
        click.secho(f"Available attributes: {attrs}", fg="cyan")

        # 2. Try to find the node storage (often called 'nodes', 'graph', or '_objects')
        storage_names = ['nodes', '_nodes', 'graph', 'objects', '_objects', 'node_map']
        for name in storage_names:
            if hasattr(sg_sim, name):
                data = getattr(sg_sim, name)
                click.secho(f"Found storage '{name}' of type {type(data)}", fg="green")
                
                # If it's a dict, let's look at one entry
                if isinstance(data, dict) and len(data) > 0:
                    first_key = list(data.keys())[0]
                    node_obj = data[first_key]
                    click.secho(f"Sample Node ({first_key}) attributes: {dir(node_obj)}", fg="yellow")
                    
                    # Look for semantic/habitat IDs
                    for info in ['id', 'habitat_id', 'semantic_id', 'class_id', 'metadata']:
                        if hasattr(node_obj, info):
                            click.secho(f"  - Node.{info}: {getattr(node_obj, info)}", fg="white")

        click.secho("===============================================\n", fg="magenta", bold=True)

        num_steps = int(getattr(cfg.planner, "num_steps", 4))
        succ = False
        planning_steps = 0
        traj_length = 0.0
        final_pred = None
        confidence_level = 0.0  # keep defined even if we never set it in loop

        for cnt_step in range(num_steps):
            start = time.time()

            agent = habitat_data._sim.get_agent(0)
            st = agent.get_state()
            current_pos = st.position
            agent_pos_hab = np.array([current_pos[0], current_pos[1], current_pos[2]], dtype=np.float32)
            agent_yaw_rad = float(habitat_data.get_heading_angle())

            target_pose, target_id, is_confident, confidence_level, extra = vlm_planner.get_next_action(
                agent_yaw_rad=agent_yaw_rad,
                agent_pos_hab=agent_pos_hab,
            )

            click.secho(f"Planner time step {cnt_step} (vlm_step={planning_steps}) = {time.time()-start:.3f}s", fg="green")

            if is_confident or (confidence_level > 0.9):
                final_pred = extra
                succ = True
                rr_logger.log_text_data(vlm_planner.full_plan + "\n" + f"FINAL: {final_pred}")
                break

            action_type = (extra or {}).get("action_type", "goto_frontier")

            if action_type == "lookaround":
                run_lookaround(
                    pipeline=pipeline,
                    habitat_data=habitat_data,
                    rr_logger=rr_logger,
                    tsdf_planner=tsdf_planner,
                    sg_sim=sg_sim,
                    output_path=question_path,
                    camera_tilt_deg=float(cfg.habitat.camera_tilt_deg),
                    segmenter=segmenter,
                    save_image=bool(cfg.vlm.use_image),
                    num_yaws=int((extra or {}).get("num_yaws", 8)),
                )
                rr_logger.log_text_data(vlm_planner.full_plan)
                planning_steps += 1
                continue

            if action_type == "walkaround":
                traj_length += float(
                    run_walkaround(
                        pipeline=pipeline,
                        habitat_data=habitat_data,
                        rr_logger=rr_logger,
                        tsdf_planner=tsdf_planner,
                        sg_sim=sg_sim,
                        output_path=question_path,
                        camera_tilt_deg=float(cfg.habitat.camera_tilt_deg),
                        segmenter=segmenter,
                        save_image=bool(cfg.vlm.use_image),
                        radius_m=float((extra or {}).get("radius_m", 0.75)),
                        num_waypoints=int((extra or {}).get("num_waypoints", 6)),
                        do_lookaround_each=True,
                        lookaround_yaws=6,
                    )
                )
                rr_logger.log_text_data(vlm_planner.full_plan)
                planning_steps += 1
                continue

            # ------------------
            # Existing: NAVIGATE to target_pose
            # ------------------
            if target_pose is not None:
                current_heading = habitat_data.get_heading_angle()
                agent = habitat_data._sim.get_agent(0)
                st = agent.get_state()
                current_pos = st.position

                frontier_habitat = pos_normal_to_habitat(target_pose)
                frontier_habitat[1] = current_pos[1]

                path = habitat_sim.nav.ShortestPath()
                path.requested_start = current_pos
                path.requested_end = frontier_habitat

                found_path = habitat_data.pathfinder.find_path(path)
                if not found_path:
                    click.secho(f"Cannot find navigable path at step {cnt_step}. Continuing..", fg="red")
                    continue

                desired_path = pos_habitat_to_normal(np.array(path.points)[:-1])
                rr_logger.log_traj_data(desired_path)
                rr_logger.log_target_poses(target_pose)

                poses = habitat_data.get_trajectory_from_path_habitat_frame(
                    target_pose, desired_path, current_heading, cfg.habitat.camera_tilt_deg
                )
                if poses is None:
                    click.secho(f"Cannot generate trajectory from path at step {cnt_step}. Continuing..", fg="red")
                    continue

                click.secho(f"Executing trajectory at step {cnt_step} (vlm_step={planning_steps})", fg="yellow")
                run(
                    pipeline,
                    habitat_data,
                    poses,
                    output_path=question_path,
                    rr_logger=rr_logger,
                    tsdf_planner=tsdf_planner,
                    sg_sim=sg_sim,
                    save_image=cfg.vlm.use_image,
                    segmenter=segmenter,
                )
                traj_length += get_traj_len_from_poses(poses)
                rr_logger.log_text_data(vlm_planner.full_plan)
                planning_steps += 1
            else:
                click.secho(f"Planner returned no target_pose at step {cnt_step}. Continuing...", fg="red")

        metrics = {
            "vlm_steps": planning_steps,
            "overall_steps": cnt_step,
            "is_confident": bool(final_pred is not None),
            "confidence_level": float(confidence_level) if final_pred is not None else 0.0,
            "traj_length": float(traj_length),
            "final_pred": final_pred,
            "mode": "msp_nobnn",
            "msp_mode": str(cfg.vlm.msp_nobnn.mode),
        }

        log_experiment_status(experiment_id, succ, metrics=metrics, filename=results_filename)

        habitat_data._sim.close(destroy=True)
        pipeline.save()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file name", default="", type=str, required=True)
    args = parser.parse_args()

    config_path = Path(__file__).resolve().parent.parent / "cfg" / f"{args.cfg_file}.yaml"
    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    main(cfg)