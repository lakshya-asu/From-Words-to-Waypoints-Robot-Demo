# /home/artemis/project/graph_eqa_swagat/spatial_experiment/scripts/run_msp_benchmark.py
# (Your runner, corrected for: planner.num_steps missing + anchor disambiguation inputs)

from tqdm import tqdm
from omegaconf import OmegaConf
import click
import os
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

# --- Smart MSP Planner ---
from spatial_experiment.planners.vlm_planner_msp import VLMPlannerMSP_Smart


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
            out[scene_floor] = {
                "init_pts": np.array([float(row["init_x"]), float(row["init_y"]), float(row["init_z"])], dtype=np.float32),
                "init_angle": float(row["init_angle"]),
            }
    return out

def load_questions_msp_csv(qpath: str):
    data = []
    with open(qpath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def run_lookaround(pipeline, habitat_data, rr_logger, tsdf_planner, sg_sim, output_path, segmenter=None, save_image=True, num_yaws=8):
    agent = habitat_data._sim.get_agent(0)
    pos = agent.get_state().position
    yaw0_deg = np.degrees(float(habitat_data.get_heading_angle()))
    for k in range(num_yaws):
        yaw_deg = yaw0_deg + (360.0 * k / num_yaws)
        poses = habitat_data.get_init_poses_eqa(np.array(pos), float(yaw_deg), 0.0)
        run(pipeline, habitat_data, poses, output_path, rr_logger, tsdf_planner, sg_sim, save_image, segmenter)

def _get_num_steps(cfg) -> int:
    """
    Accept:
      - cfg.planner.num_steps / max_steps / steps
      - cfg.vlm.num_steps / max_steps / steps   (common in your setup)
    Default 30.
    """
    for root in ["planner", "vlm"]:
        node = getattr(cfg, root, None)
        if node is None:
            continue
        for k in ["num_steps", "max_steps", "steps"]:
            try:
                v = node.get(k) if hasattr(node, "get") else getattr(node, k, None)
                if v is not None:
                    return int(v)
            except Exception:
                pass
    return 30

def main(cfg):
    qpath = cfg.data.question_data_path
    click.secho("[mode] MSP SMART VLM runner", fg="cyan")

    questions_data = load_questions_msp_csv(qpath)
    init_pose_data = load_init_poses_csv(cfg.data.init_pose_data_path)

    output_path = Path(__file__).resolve().parent.parent / cfg.output_path
    os.makedirs(str(output_path), exist_ok=True)
    results_filename = output_path / f"{cfg.results_filename}.json"
    device = f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu"

    # Segmenter init (Detic or GT)
    segmenter = None if cfg.data.use_semantic_data else hydra_python.detection.detic_segmenter.DeticSegmenter(cfg)

    num_steps = _get_num_steps(cfg)
    click.secho(f"[runner] num_steps={num_steps}", fg="yellow")

    for question_ind in tqdm(range(len(questions_data))):
        q = questions_data[question_ind]
        scene, floor = q["scene"], str(q["floor"])
        scene_id = scene[6:]
        experiment_id = f"{question_ind}_{scene}_{floor}"

        if should_skip_experiment(experiment_id, filename=results_filename):
            continue

        question_path = hydra_python.resolve_output_path(output_path / experiment_id)
        if cfg.data.use_semantic_data and not scene_has_semantics(scene_id):
            continue

        habitat_data = HabitatInterface(
            f"{cfg.data.scene_data_path}/{scene}/{scene_id}.basis.glb",
            cfg=cfg.habitat,
            device=device,
        )
        pipeline = initialize_hydra_pipeline(cfg.hydra, habitat_data, question_path)
        rr_logger = RRLogger(question_path)

        # Init TSDF
        init_pts = init_pose_data[f"{scene}_{floor}"]["init_pts"]
        tsdf_bnds, _ = get_scene_bnds(habitat_data.pathfinder, pos_habitat_to_normal(init_pts)[-1])
        cam_intr = get_cam_intr(cfg.habitat.hfov, cfg.habitat.img_height, cfg.habitat.img_width)

        tsdf_planner = TSDFPlanner(
            cfg.frontier_mapping, tsdf_bnds, cam_intr, 0, pos_habitat_to_normal(init_pts), rr_logger
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

        poses = habitat_data.get_init_poses_eqa(
            init_pts, init_pose_data[f"{scene}_{floor}"]["init_angle"], 0.0
        )
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

        # --- Anchor disambiguation inputs from CSV ---
        # Your CSV fields:
        #   anchor_label, anchor_center_x/y/z
        anchor_label = q.get("anchor_label", None) or q.get("primary_object", None)
        anchor_center_hab = None
        try:
            if q.get("anchor_center_x", None) is not None:
                anchor_center_hab = np.array(
                    [float(q["anchor_center_x"]), float(q["anchor_center_y"]), float(q["anchor_center_z"])],
                    dtype=np.float32,
                )
        except Exception:
            anchor_center_hab = None

        vlm_planner = VLMPlannerMSP_Smart(
        cfg.vlm,
        sg_sim,
        q["msp_question"],
        out_path=str(question_path),
        anchor_label=anchor_label,
        anchor_center_hab=anchor_center_hab,
        anchor_front_yaw_world=float(q["ann_yaw_rad"]) if q.get("ann_yaw_rad", None) not in [None, ""] else None,
    )

        succ = False
        traj_length = 0.0
        final_pred = None

        for step in range(num_steps):
            agent_st = habitat_data._sim.get_agent(0).get_state()

            target_pose, target_id, is_conf, conf, extra = vlm_planner.get_next_action(
                agent_yaw_rad=float(habitat_data.get_heading_angle()),
                agent_pos_hab=np.array(agent_st.position),
            )

            # "answer" is decided inside the planner; if it claims confident, stop.
            if is_conf or (conf > 0.9 and extra.get("action_type") == "answer"):
                final_pred = extra
                succ = True
                rr_logger.log_text_data(f"FINAL ANSWER: {final_pred}")
                break

            action_type = extra.get("action_type", "goto_frontier")

            if action_type == "lookaround":
                run_lookaround(pipeline, habitat_data, rr_logger, tsdf_planner, sg_sim, question_path, segmenter)
                continue

            if target_pose is not None:
                path = habitat_sim.nav.ShortestPath()
                path.requested_start = agent_st.position

                # sg_sim.get_position_from_id returns NORMAL frame
                # convert to habitat frame for pathfinding:
                end_hab = pos_normal_to_habitat(target_pose) if len(target_pose) == 3 else target_pose
                end_hab = np.asarray(end_hab, dtype=np.float32)
                end_hab[1] = agent_st.position[1]  # keep height

                path.requested_end = end_hab

                if habitat_data.pathfinder.find_path(path):
                    desired_path = pos_habitat_to_normal(np.array(path.points)[:-1])
                    rr_logger.log_traj_data(desired_path)

                    poses = habitat_data.get_trajectory_from_path_habitat_frame(
                        pos_habitat_to_normal([path.requested_end])[0],
                        desired_path,
                        habitat_data.get_heading_angle(),
                        0.0,
                    )

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
                else:
                    click.secho(f"Pathfinding failed to {action_type} {target_id}", fg="red")

        log_experiment_status(
            experiment_id,
            succ,
            metrics={
                "traj_length": float(traj_length),
                "final_pred": final_pred,
                "mode": "msp_smart",
            },
            filename=results_filename,
        )

        habitat_data._sim.close(destroy=True)
        pipeline.save()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(Path(__file__).resolve().parent.parent / "cfg" / f"{args.cfg_file}.yaml")
    OmegaConf.resolve(cfg)
    main(cfg)