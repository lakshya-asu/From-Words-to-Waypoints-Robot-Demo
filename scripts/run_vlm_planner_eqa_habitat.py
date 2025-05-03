from tqdm import tqdm
from omegaconf import OmegaConf
import click
import os, time
from pathlib import Path
import numpy as np
import torch

from graph_eqa.logging.utils import should_skip_experiment, log_experiment_status
from graph_eqa.envs.utils import pos_habitat_to_normal, pos_normal_to_habitat
from graph_eqa.occupancy_mapping.geom import get_scene_bnds, get_cam_intr
from graph_eqa.envs.habitat import run
from graph_eqa.logging.rr_logger import RRLogger
from graph_eqa.occupancy_mapping.tsdf import TSDFPlanner
from graph_eqa.utils.data_utils import load_eqa_data, get_instruction_from_eqa_data, get_traj_len_from_poses
from graph_eqa.utils.hydra_utils import initialize_hydra_pipeline


from graph_eqa.scene_graph.scene_graph_sim import SceneGraphSim
from graph_eqa.planners import VLMPlannerEQAGemini, VLMPlannerEQAGPT, VLMPlannerEQAClaude, VLMPlannerEQALlama4
from graph_eqa.envs.habitat_interface import HabitatInterface

import habitat_sim
import hydra_python

def main(cfg):
    questions_data, init_pose_data = load_eqa_data(cfg.data)

    output_path = Path(__file__).resolve().parent.parent / cfg.output_path
    os.makedirs(str(output_path), exist_ok=True)
    results_filename = output_path / f'{cfg.results_filename}.json'
    device = f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu"

    eqa_enrich_labels = OmegaConf.load(Path(__file__).resolve().parent.parent / cfg.data.explore_eqa_dataset_enrich_labels)

    if not cfg.data.use_semantic_data:
        from hydra_python.detection.detic_segmenter import DeticSegmenter
        segmenter = DeticSegmenter(cfg)
    else:
        segmenter = None

    successes = 0
    for question_ind in tqdm(range(len(questions_data))):

        question_data = questions_data[question_ind]
        scene_floor = question_data["scene"] + "_" + question_data["floor"]
        answer = question_data["answer"]
        experiment_id = f'{question_ind}_{question_data["scene"]}_{question_data["floor"]}'



        if should_skip_experiment(experiment_id, filename=results_filename):
            click.secho(f'Skipping==Index: {question_ind} Scene: {question_data["scene"]} Floor: {question_data["floor"]}=======',fg="yellow",)
            continue
        else:
            click.secho(f'Executing=========Index: {question_ind} Scene: {question_data["scene"]} Floor: {question_data["floor"]}=======',fg="green",)

        # Planner reset with the new quesion
        question_path = hydra_python.resolve_output_path(output_path / experiment_id)
        scene_name = f'{cfg.data.scene_data_path}/{question_data["scene"]}/{question_data["scene"][6:]}.basis.glb'
        
        vlm_question, clean_ques_ans, choices, vlm_pred_candidates = get_instruction_from_eqa_data(question_data)
        habitat_data = HabitatInterface(
            scene_name, 
            cfg=cfg.habitat,
            device=device,)
        pipeline = initialize_hydra_pipeline(cfg.hydra, habitat_data, question_path)
        
        rr_logger = RRLogger(question_path)

        # Extract initial pose
        init_pts = init_pose_data[scene_floor]["init_pts"]
        init_angle = init_pose_data[scene_floor]["init_angle"]

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
            label = eqa_enrich_labels[f'{question_ind}_{question_data["scene"]}']['labels']
        else:
            label = ' '

        sg_sim = SceneGraphSim(
            cfg, 
            question_path, 
            pipeline, 
            rr_logger, 
            device=device, 
            clean_ques_ans=clean_ques_ans,
            # enrich_object_labels=eqa_enrich_labels[f'{question_ind}_{question_data["scene"]}']['labels'])
            enrich_object_labels=label)

        # Get poses for hydra at init view
        poses = habitat_data.get_init_poses_eqa(init_pts, init_angle, cfg.habitat.camera_tilt_deg)
        # Get scene graph for init view
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

        if 'gpt' in cfg.vlm.name.lower():
            vlm_planner = VLMPlannerEQAGPT(
                cfg.vlm,
                sg_sim,
                vlm_question, vlm_pred_candidates, choices, answer, 
                question_path)
        elif 'gemini' in cfg.vlm.name.lower():
            vlm_planner = VLMPlannerEQAGemini(
                cfg.vlm,
                sg_sim,
                vlm_question, vlm_pred_candidates, choices, answer, 
                question_path)
        elif 'claude' in cfg.vlm.name.lower():
            vlm_planner = VLMPlannerEQAClaude(
                cfg.vlm,
                sg_sim,
                vlm_question, vlm_pred_candidates, choices, answer, 
                question_path)
        elif 'llama' in cfg.vlm.name.lower():
            vlm_planner = VLMPlannerEQALlama4(
                cfg.vlm,
                sg_sim,
                vlm_question, vlm_pred_candidates, choices, answer, 
                question_path)
        else:
            raise NotImplementedError('VLM planner not implemented.')
        
        click.secho(f'Index:{question_ind} Scene: {question_data["scene"]} Floor: {question_data["floor"]}',fg="green",)
        click.secho(f"Question:\n{vlm_planner._question} \n Answer: {answer}",fg="green",)

        num_steps = 20
        succ = False
        planning_steps = 0
        traj_length = 0.
        for cnt_step in range(num_steps):
            start = time.time()
            target_pose, target_id, is_confident, confidence_level, answer_output = vlm_planner.get_next_action()
            click.secho(f"VLM planning time for overall step {cnt_step} and vlm step {planning_steps} is {time.time()-start}",fg="green",)
            
            # TODO: Vary this from 0.5 to 0.9
            if is_confident or (confidence_level>0.9):

                succ = (answer == answer_output)
                if succ:
                    successes += 1
                    result = f"Success at vlm step{planning_steps} for {question_ind}:{scene_floor}"
                    click.secho(result,fg="blue",)
                    click.secho(f"VLM Planner answer: {answer_output}, Correct answer: {answer}",fg="blue",)
                else:
                    result = f"Failure at vlm step {planning_steps} for {question_ind}:{scene_floor}"
                    click.secho(result,fg="red",)
                    click.secho(f"VLM Planner answer: {answer_output}, Correct answer: {answer}",fg="red",)
                rr_logger.log_text_data(vlm_planner.full_plan + "\n" + result)
                break

            if target_pose is not None:

                # desired_path = tsdf_planner.sample_frontier()
                current_heading = habitat_data.get_heading_angle()
                # desired_path = tsdf_planner.path_to_frontier(target_pose) # not being used anymore

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
                    click.secho(f"Cannot find navigable path at {cnt_step}. Continuing..",fg="red",)
                    continue

                poses = habitat_data.get_trajectory_from_path_habitat_frame(target_pose, desired_path, current_heading, cfg.habitat.camera_tilt_deg)
                if poses is not None:
                    click.secho(f"Executing trajectory at overall step {cnt_step} and vlm step {planning_steps}",fg="yellow",)
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

                    ## If trajectory successfully executed
                    rr_logger.log_text_data(vlm_planner.full_plan)
                    planning_steps+=1
                else:
                    click.secho(f"Cannot find trajectory from navigable path at {cnt_step}. Continuing..",fg="red",)
                    continue
            else:
                click.secho(f"VLM planner failed at overall step {cnt_step}. Continuing...",fg="red",)
        
        metrics = {
            'vlm_steps': planning_steps,
            'overall_steps': cnt_step,
            'is_confident': is_confident,
            'confidence_level': confidence_level,
            'traj_length': traj_length
        }
        log_experiment_status(experiment_id, succ, metrics=metrics, filename=results_filename)
        habitat_data._sim.close(destroy=True)
        pipeline.save()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file name", default="", type=str, required=True)
    args = parser.parse_args()

    config_path = Path(__file__).resolve().parent.parent / 'cfg' / f'{args.cfg_file}.yaml'
    cfg = OmegaConf.load(config_path)

    OmegaConf.resolve(cfg)
    main(cfg)