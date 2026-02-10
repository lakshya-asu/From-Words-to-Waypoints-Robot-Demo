#!/usr/bin/env python3
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

from spatial_experiment.planners.vlm_planner_gemini_spatial_v6 import VLMPlannerEQAGeminiSpatialV6
from graph_eqa.envs.habitat import run
from graph_eqa.envs.habitat_interface import HabitatInterface
from graph_eqa.logging.rr_logger import RRLogger
from graph_eqa.logging.utils import log_experiment_status, should_skip_experiment
from graph_eqa.occupancy_mapping.geom import get_cam_intr, get_scene_bnds
from graph_eqa.occupancy_mapping.tsdf import TSDFPlanner
from graph_eqa.scene_graph.scene_graph_sim import SceneGraphSim
from graph_eqa.utils.hydra_utils import initialize_hydra_pipeline
from graph_eqa.envs.utils import pos_habitat_to_normal, pos_normal_to_habitat

if not OmegaConf.has_resolver("env"):
    OmegaConf.register_new_resolver("env", lambda key, default=None: os.environ.get(str(key), default))

def _teleport_agent(habitat_data, init_pos, init_rot):
    st = habitat_sim.AgentState()
    st.position = np.asarray(init_pos, dtype=np.float32)
    st.rotation = quat_from_coeffs(np.asarray(init_rot, dtype=np.float32))
    habitat_data._sim.get_agent(0).set_state(st)

def main(cfg):
    with open(cfg.data.spatial_data_path, "r") as f: tasks_data = json.load(f)
    base_output_root = Path(__file__).resolve().parent.parent / cfg.output_path
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = base_output_root / run_stamp
    output_path.mkdir(parents=True, exist_ok=True)
    results_filename = output_path / f"{cfg.results_filename}_v6.json"
    device = f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu"

    for task_index, task_data in enumerate(tqdm(tasks_data, desc="Running Tasks (V6 BNN)")):
        scene_dir = task_data.get("scene_dir")
        if not scene_dir: continue
        scene_glb_path = f"{cfg.data.scene_data_path}/{scene_dir}/{scene_dir.split('-')[-1]}.basis.glb"
        gt = task_data["ground_truth_target"]
        experiment_id = f'{task_index}_{scene_dir}_{gt["name"]}'
        
        if should_skip_experiment(experiment_id, filename=results_filename): continue
        task_output_path = hydra_python.resolve_output_path(output_path / experiment_id)
        
        habitat_data = HabitatInterface(scene_glb_path, cfg=cfg.habitat, device=device)
        pipeline = initialize_hydra_pipeline(cfg.hydra, habitat_data, task_output_path)
        rr_logger = RRLogger(task_output_path)
        
        init_pos = np.array(task_data["initial_pose"]["position"], dtype=np.float32)
        init_rot = np.array(task_data["initial_pose"]["rotation"], dtype=np.float32)
        _teleport_agent(habitat_data, init_pos, init_rot)
        
        floor_height = float(init_pos[1])
        tsdf_bnds, _ = get_scene_bnds(habitat_data.pathfinder, floor_height)
        cam_intr = get_cam_intr(cfg.habitat.hfov, cfg.habitat.img_height, cfg.habitat.img_width)
        tsdf_planner = TSDFPlanner(cfg.frontier_mapping, tsdf_bnds, cam_intr, 0, init_pos, rr_logger)
        
        # --- FIXED SceneGraphSim call (added enrich_object_labels=" ") ---
        sg_sim = SceneGraphSim(cfg, task_output_path, pipeline, rr_logger, device, str(task_data["question"]), enrich_object_labels=" ")
        
        run(pipeline, habitat_data, habitat_data.get_init_poses_eqa(init_pos, float(habitat_data.get_heading_angle()), -30.0), output_path=task_output_path, rr_logger=rr_logger, tsdf_planner=tsdf_planner, sg_sim=sg_sim, save_image=True)
        _teleport_agent(habitat_data, init_pos, init_rot)

        vlm_planner = VLMPlannerEQAGeminiSpatialV6(
            cfg.vlm, sg_sim, str(task_data["question"]), gt, task_output_path,
            reference_object=task_data.get("reference_object"), candidate_targets=task_data.get("candidate_targets")
        )

        for step_count in range(20):
            heading = float(habitat_data.get_heading_angle())
            target_pose, target_id, is_conf, conf, decl = vlm_planner.get_next_action(agent_yaw_rad=heading)
            
            if is_conf: break
            if target_pose is None: continue
            
            current_pos = np.array(habitat_data._sim.get_agent(0).get_state().position, dtype=np.float32)
            target_hab = pos_normal_to_habitat(np.array(target_pose, dtype=np.float32))
            target_hab[1] = current_pos[1]
            spath = habitat_sim.nav.ShortestPath()
            spath.requested_start = current_pos
            spath.requested_end = target_hab
            if habitat_data.pathfinder.find_path(spath):
                poses = habitat_data.get_trajectory_from_path_habitat_frame(target_pose, pos_habitat_to_normal(np.array(spath.points)[:-1]), heading, -30.0)
                run(pipeline, habitat_data, poses, output_path=task_output_path, rr_logger=rr_logger, tsdf_planner=tsdf_planner, sg_sim=sg_sim, save_image=True)

        habitat_data._sim.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", default="spatial_vqa_v6", type=str)
    args = parser.parse_args()
    cfg = OmegaConf.load(Path(__file__).parent.parent / "cfg" / f"{args.cfg_file}.yaml")
    OmegaConf.resolve(cfg)
    main(cfg)