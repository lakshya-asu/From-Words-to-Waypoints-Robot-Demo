from tqdm import tqdm
from omegaconf import OmegaConf
import click
import os
from pathlib import Path
import sys
import torch 

import numpy as np
import hydra_python as hydra
from hydra_python.run import run_eqa
from hydra_python._plugins import habitat

from Graph_EQA.logging.rr_logger import RRLogger
from Graph_EQA.occupancy_mapping.tsdf import TSDFPlanner
from Graph_EQA.occupancy_mapping.utils import *
from Graph_EQA.occupancy_mapping.geom import *
from Graph_EQA.occupancy_mapping.utils import pos_habitat_to_normal

from hydra_python.utils import load_eqa_data, initialize_hydra_pipeline, get_instruction_from_eqa_data
from hydra_python.detection.detic_segmenter import DeticSegmenter

def main(cfg):
    exploration_type = "object_node" # 'frontiers

    questions_data, init_pose_data = load_eqa_data(cfg.data)
    
    output_path = cfg.output_path
    os.makedirs(cfg.output_path, exist_ok=True)
    output_path = Path(cfg.output_path)

    device = f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu"

    eqa_enrich_labels = OmegaConf.load(cfg.data.eqa_dataset_enrich_labels)

    if not cfg.data.use_semantic_data:
        segmenter = DeticSegmenter(cfg)
    else:
        segmenter = None

    for question_ind in tqdm(range(len(questions_data))):
        if question_ind in np.arange(2):
            continue
        question_data = questions_data[question_ind]
        

        # Planner reset with the new quesion
        question_path = hydra.resolve_output_path(output_path / f'{question_ind}_{question_data["scene"]}')
        scene_name = f'{cfg.data.scene_data_path}/{question_data["scene"]}/{question_data["scene"][6:]}.basis.glb'
        vlm_question, clean_ques_ans, choices, vlm_pred_candidates = get_instruction_from_eqa_data(question_data)
        
        habitat_data = habitat.HabitatInterface(
            scene_name, 
            cfg=cfg.habitat,
            device=device,)
        pipeline = initialize_hydra_pipeline(cfg.hydra, habitat_data, question_path)

        rr_logger = RRLogger(question_path)

        click.secho(f'\n========\nIndex: {question_ind} Scene: {question_data["scene"]} Floor: {question_data["floor"]}',fg="green",)

        # Extract initial pose
        scene_floor = question_data["scene"] + "_" + question_data["floor"]
        answer = question_data["answer"]
        init_pts = init_pose_data[scene_floor]["init_pts"]
        init_angle = init_pose_data[scene_floor]["init_angle"]
        # init_pts[0] = -0.154
        # init_pts[2] = 0.69

        # Setup TSDF planner
        pts_normal = pos_habitat_to_normal(init_pts)
        floor_height = pts_normal[-1]
        tsdf_bnds, scene_size = get_scene_bnds(habitat_data.pathfinder, floor_height)
        cam_intr = get_cam_intr(cfg.habitat.hfov, cfg.habitat.img_height, cfg.habitat.img_width)
        
        # Initialize TSDF
        tsdf_planner = TSDFPlanner(
            cfg=cfg.frontier_mapping,
            vol_bnds=tsdf_bnds,
            cam_intr=cam_intr,
            floor_height_offset=0,
            pts_init=pts_normal,
            rr_logger=rr_logger,
        )

        if f'{question_ind}_{question_data["scene"]}' in eqa_enrich_labels:
            enrich_labels = eqa_enrich_labels[f'{question_ind}_{question_data["scene"]}']['labels']
        else:
            enrich_labels = ' '
        sg_sim = hydra.SceneGraphSim(
            cfg, 
            question_path, 
            pipeline, 
            rr_logger, 
            device=device, 
            clean_ques_ans=clean_ques_ans,
            enrich_object_labels=enrich_labels)

        # Get poses for hydra at init view
        poses = habitat_data.get_init_poses_eqa(init_pts, init_angle, cfg.habitat.camera_tilt_deg)
        # Get scene graph for init view
        run_eqa(
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

        # LOG NAVMESH
        graph_nodes = np.array([habitat_data.G.nodes[n]["pos"] for n in habitat_data.G]).squeeze()
        positions_navmesh = np.array([pos_habitat_to_normal(p) for p in graph_nodes])
        rr_logger.log_navmesh_data(positions_navmesh)
        
        click.secho(f"Question:\n{vlm_question} \n Answer: {answer}",fg="green",)

        num_steps = 7
        for i in range(num_steps):
            current_heading = habitat_data.get_heading_angle()
            if 'frontier' in exploration_type:
                _, target_pose = tsdf_planner.sample_frontier()
            if 'object_node' in exploration_type:
                obj_idx = random.randint(0,len(sg_sim.object_node_ids))
                object_id = sg_sim.object_node_ids[obj_idx]
                target_pose = sg_sim.get_position_from_id(object_id)
                # desired_path = tsdf_planner.path_to_frontier(target_pose)
                click.secho(f'Sampled object_id:{object_id}, object name: {sg_sim.object_node_names[obj_idx]}', fg='yellow')
            
            # # Create a path object
            agent = habitat_data._sim.get_agent(0)  # Assuming agent ID 0
            current_pos = agent.get_state().position
            frontier_habitat = pos_normal_to_habitat(target_pose)
            frontier_habitat[1] = current_pos[1]
            path = habitat_sim.nav.ShortestPath()
            path.requested_start = current_pos
            path.requested_end = frontier_habitat
            # Compute the shortest path
            found_path = habitat_data.pathfinder.find_path(path)
            if found_path:
                desired_path = pos_habitat_to_normal(np.array(path.points)[:-1])
                rr_logger.log_traj_data(desired_path)
                rr_logger.log_target_poses(target_pose)
            else:
                click.secho(f"Cannot find navigable path: {i}",fg="red",)
                continue

            poses = habitat_data.get_trajectory_from_path_habitat_frame(target_pose, desired_path, current_heading, cfg.habitat.camera_tilt_deg)
            click.secho(f"Executing trajectory: {i}",fg="yellow",)
            run_eqa(
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
            # bb = hydra.get_bb_from_sem(habitat_data)
            import ipdb; ipdb.set_trace()
        pipeline.save()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file name", default="", type=str, required=True)
    args = parser.parse_args()

    config_path = Path(__file__).resolve().parent / 'commands' / 'cfg' / f'{args.cfg_file}.yaml'
    cfg = OmegaConf.load(config_path)

    OmegaConf.resolve(cfg)
    main(cfg)