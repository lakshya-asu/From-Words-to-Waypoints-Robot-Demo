import sys
import os
import time
from pathlib import Path
import click

# Force hydra/omegaconf to resolve properly
import hydra_python as hydra
# from omegaconf import OmegaConf

# Core spatial_experiment modules
from stretch.core.parameters import Parameters
from spatial_experiment.real_world.robotis_hydra_agent import RobotisHydraAgent
from spatial_experiment.planners.multi_agent_fat_planner import MultiAgentFatPlanner
from graph_eqa.scene_graph.scene_graph_sim import SceneGraphSim

# New ROS 2 Client
from spatial_experiment.real_world.ai_worker_client import AIWorkerRobotClient
import rclpy

class MockPipeline:
    """Mock pipeline for offline/dynamic scene graphs if required by SceneGraphSim."""
    def __init__(self, ds_graph=None):
        self.graph = ds_graph


class RealWorldCfg:
    """Mock configuration mirroring the YAML structure for real-world execution."""
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
@click.option("--question", default="Explore the environment.", help="Initial command or question for the robot.")
@click.option("--max_steps", default=10, type=int, help="Maximum number of planner steps to execute.")
@click.option("--vlm_provider", default="qwen", help="Which VLM provider to use (qwen, openai, claude, gemini).")
def run_ai_worker_agent_main(question, max_steps, vlm_provider):
    """
    Main entrypoint for running the spatial_experiment AI Brain on the physical ROBOTIS AI Worker.
    Uses ROS 2 (rclpy) for hardware communication.
    """
    click.secho(f"--- Starting spatial_experiment for ROBOTIS AI Worker ---", fg="green", bold=True)
    click.secho(f"Question/Task: {question}", fg="yellow")
    click.secho(f"VLM Provider: {vlm_provider}", fg="yellow")
    
    # 1. Initialize ROS 2
    rclpy.init()
    robot_client = AIWorkerRobotClient()
    
    if not robot_client.start():
        click.secho("Failed to start AIWorkerRobotClient. Aborting.", fg="red")
        robot_client.stop()
        sys.exit(1)

    # 2. Setup Agent Parameters
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

    out_path = Path("outputs/run_ai_worker")
    out_path.mkdir(parents=True, exist_ok=True)
    
    cfg = RealWorldCfg()

    # 3. Initialize Scene Graph Manager
    click.secho("Initializing SceneGraphSim...", fg="cyan")
    pipeline_mock = MockPipeline()
    sg_sim = SceneGraphSim(
        cfg=cfg, 
        output_path=out_path, 
        pipeline=pipeline_mock, 
        device="cuda" if hydra.torch.cuda.is_available() else "cpu",
        clean_ques_ans=question,
        enrich_object_labels="object"
    )
    
    # 4. Initialize Core Agent (Mapping, Perception, Motion Control)
    click.secho("Initializing RobotisHydraAgent...", fg="cyan")
    agent = RobotisHydraAgent(
        robot=robot_client,
        parameters=parameters,
        sg_sim=sg_sim,
        output_path=out_path,
        enable_realtime_updates=False # Change to True if multi-threading map updates is desired
    )
    
    agent.start(can_move=True, verbose=True)

    # 5. Initialize Multi-Agent Planner
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

    # 6. Run the loop
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
    run_ai_worker_agent_main()
