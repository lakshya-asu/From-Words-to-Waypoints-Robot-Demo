import sys
import os
import time
from pathlib import Path
import click

# Force hydra/omegaconf to resolve properly
import hydra_python as hydra

# Core spatial_experiment modules
from stretch.core.parameters import Parameters
from spatial_experiment.real_world.robotis_hydra_agent import RobotisHydraAgent
from spatial_experiment.planners.multi_agent_fat_planner import MultiAgentFatPlanner
from graph_eqa.scene_graph.scene_graph_sim import SceneGraphSim

# New ROS 2 Client
from spatial_experiment.real_world.ai_worker_client import AIWorkerRobotClient
import rclpy

class MockPipeline:
    """Mock pipeline for offline/dynamic scene graphs required by SceneGraphSim."""
    def __init__(self, ds_graph=None):
        self.graph = ds_graph


class RealWorldCfg:
    """Mock configuration mirroring the YAML structure for offline execution."""
    class scene_graph_sim:
        enrich_rooms = False
        save_image = False
        include_regions = True
        no_scene_graph = False
        enrich_frontiers = True
        enrich_provider = "qwen"
        class key_frame_selection:
            choose_final_image = False
            use_clip_for_images = False
            use_siglip_for_images = False
    class vlm:
        # VLM Planner config matches the structure expected by MultiAgentFatPlanner
        pass


@click.command()
@click.option("--question", default="Where is the object?", help="Current query.")
@click.option("--scene_graph_path", default="spatial_experiment/zed2i/backend/dsg.json", help="Path to offline dsg.json.")
@click.option("--max_steps", default=10, type=int, help="Maximum number of planner steps to execute.")
@click.option("--vlm_provider", default="qwen", help="Which VLM provider to use (qwen, openai, claude, gemini).")
def run_ai_worker_offline_main(question, scene_graph_path, max_steps, vlm_provider):
    """
    Main entrypoint for running the spatial_experiment AI Brain on the physical ROBOTIS AI Worker
    in an Ablation setup (using a pre-recorded offline scene graph).
    """
    click.secho(f"--- Starting spatial_experiment for ROBOTIS AI Worker (OFFLINE ABLATION) ---", fg="green", bold=True)
    click.secho(f"Question/Task: {question}", fg="yellow")
    click.secho(f"VLM Provider: {vlm_provider}", fg="yellow")
    click.secho(f"Scene Graph Path: {scene_graph_path}", fg="yellow")
    
    # 1. Load the Offline Graph
    try:
        click.secho(f"Loading prerecorded graph from {scene_graph_path}...", fg="cyan")
        ds_graph = hydra.DynamicSceneGraph.load(scene_graph_path)
        pipeline_mock = MockPipeline(ds_graph)
    except Exception as e:
        click.secho(f"Failed to load Hydra Scene Graph from {scene_graph_path}: {e}", fg="red")
        sys.exit(1)
        
    out_path = Path("outputs/run_ai_worker_offline")
    out_path.mkdir(parents=True, exist_ok=True)
    
    cfg = RealWorldCfg()

    # 2. Initialize Scene Graph Manager with mock pipeline
    click.secho("Initializing SceneGraphSim from offline graph...", fg="cyan")
    sg_sim = SceneGraphSim(
        cfg=cfg, 
        output_path=out_path, 
        pipeline=pipeline_mock, 
        device="cuda" if hydra.torch.cuda.is_available() else "cpu",
        clean_ques_ans=question,
        enrich_object_labels="object"
    )
    
    # Rebuild internal networkx representation and relationships
    sg_sim._build_sg_from_hydra_graph()

    # 3. Initialize ROS 2 hardware connection
    rclpy.init()
    robot_client = AIWorkerRobotClient()
    
    if not robot_client.start():
        click.secho("Failed to start AIWorkerRobotClient. Aborting.", fg="red")
        robot_client.stop()
        sys.exit(1)

    # 4. Setup Agent Parameters
    parameters = Parameters(
        encoder="clip",
        voxel_size=0.05,
        use_scene_graph=True,
        trajectory_pos_err_threshold=0.2,
        trajectory_rot_err_threshold=0.75,
        tts_engine="dummy",
        hydra_update_freq=5.0,
        task={"command": question},
        agent={"sweep_head_on_update": False, "use_realtime_updates": False},
        motion_planner={
            "step_size": 0.1, "rotation_step_size": 0.1,
            "frontier": {
                "dilate_frontier_size": 2, "dilate_obstacle_size": 2,
                "default_expand_frontier_size": 5, "min_dist": 0.5, "step_dist": 0.5,
                "min_points_for_clustering": 5, "num_clusters": 5, "cluster_threshold": 0.5,
                "contract_traversible_size": 1
            },
            "goals": {"manipulation_radius": 0.5},
            "shortcut_plans": False, "simplify_plans": False,
        },
        data={"initial_state": {"head": [0.0, 0.0]}}
    )
    
    # 5. Initialize Core Agent (Mapping, Perception, Motion Control)
    click.secho("Initializing RobotisHydraAgent...", fg="cyan")
    agent = RobotisHydraAgent(
        robot=robot_client,
        parameters=parameters,
        sg_sim=sg_sim,
        output_path=out_path,
        enable_realtime_updates=False # MUST BE FALSE for offline graph setup so we don't overwrite map
    )
    
    agent.start(can_move=True, verbose=True)

    # 6. Initialize Multi-Agent Planner
    click.secho(f"Initializing Multi-Agent Planner...", fg="cyan")
    vlm_planner = MultiAgentFatPlanner(
        cfg=cfg.vlm,
        sg_sim=sg_sim,
        question=question,
        out_path=str(out_path),
        agent_providers={
            "orchestrator": vlm_provider,
            "grounding": vlm_provider,
            "spatial": vlm_provider,
            "logical": vlm_provider,
            "qa": vlm_provider,
            "verifier": vlm_provider
        }
    )

    # 7. Run the execution loop
    try:
        click.secho("Starting execution loop...", fg="magenta", bold=True)
        agent.run_eqa_vlm_planner(
            vlm_planner=vlm_planner,
            sg_sim=sg_sim,
            manual_wait=False,
            max_planning_steps=max_steps,
            go_home_at_end=False
        )
        click.secho("Execution completed successfully.", fg="green")
        
    except KeyboardInterrupt:
        click.secho("\nExecution interrupted by user.", fg="yellow")
    except Exception as e:
        click.secho(f"\nExecution failed: {e}", fg="red")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        click.secho("Shutting down...", fg="cyan")
        agent.stop()
        robot_client.stop()

if __name__ == "__main__":
    run_ai_worker_offline_main()
