"""Commands for running with the habitat simulator."""

from spark_dsg.open3d_visualization import DsgVisualizer
import hydra_python as hydra
import pathlib
import click
import yaml

import numpy as np
import random
import matplotlib.pyplot as plt
import habitat_sim
import pickle
from itertools import chain
from Graph_EQA.logging.rr_logger import RRLogger


def _get_trajectory(data, prev_dsg, seed, use_full_scene=False):
    if prev_dsg:
        prev_path = pathlib.Path(prev_dsg).resolve()
        poses = hydra.Trajectory.from_scene_graph(str(prev_path))
    elif use_full_scene:
        poses = data.get_full_trajectory(seed=seed)
        if poses is None:
            click.secho(
                "Failed to find trajectory for single room! Defaulting to random",
                fg="yellow",
            )
            poses = data.get_random_trajectory(seed=seed)
    else:
        click.secho(
                "Finding random trajectory",
                fg="yellow",
            )
        poses = data.get_random_trajectory(seed=seed, target_length_m=0.1)
        

    click.secho(f"Trajectory is {poses.get_path_length()} meters long", fg="green")
    return poses

def _plot_sg_trajs(navmesh_nodes, poses, target_poses, pipeline, output_path, suffix, view='XY'):

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(navmesh_nodes[:,0], navmesh_nodes[:,1], navmesh_nodes[:,2], c='r', marker='o', s=1, label='NavMesh Vertices')

    # Plotting trajectories
    # pos_traj = np.array([v[1] for v in poses])
    pos_traj = np.array(poses)
    ax.plot(pos_traj[:,0], pos_traj[:,1], pos_traj[:,2], color='b', linewidth=2, label='Path')
    ax.scatter(pos_traj[0,0], pos_traj[0,1], pos_traj[0,2],  c='b', marker='D', s=15, label='Start')
    ax.scatter(np.array(target_poses)[:,0], np.array(target_poses)[:,1], np.array(target_poses)[:,2],  c='red', marker='D', s=30, label='Target pose')
    
    place_node_positions = []
    active_frontier_place_node_positions = []
    object_node_positions = []
    for node in pipeline.graph.nodes:
        z = node.attributes.position[2]
        if 'p' in node.id.category.lower():
            # print(f"layer: {node.layer}. Category: {node.id.category.lower()}{node.id.category_id}. Active Frontier: {node.attributes.active_frontier}")
            # if z<1.0 and z>0.25:
            place_node_positions.append(node.attributes.position)
        if 'f' in node.id.category.lower():
            # print(f"layer: {node.layer}. Category: {node.id.category.lower()}{node.id.category_id}. Active Frontier: {node.attributes.active_frontier}")
            # if z<1.0 and z>0.25:
            active_frontier_place_node_positions.append(node.attributes.position)
        if 'o' in node.id.category.lower():
            object_node_positions.append(node.attributes.position)

    place_node_positions = np.array(place_node_positions)
    active_frontier_place_node_positions = np.array(active_frontier_place_node_positions)
    object_node_positions = np.array(object_node_positions)

    agent_node_positions = []
    for layer in pipeline.graph.dynamic_layers:
        for node in layer.nodes:
            # print(f"layer: {node.layer}. Category: {node.id.category.lower()} {node.id.category_id}")
            if 'a' in node.id.category.lower():
                agent_node_positions.append(node.attributes.position)
    agent_node_positions = np.array(agent_node_positions)

    ax.scatter(place_node_positions[:,0], place_node_positions[:,1], place_node_positions[:,2],  c='green', marker='o', s=6, label='Place nodes')
    ax.scatter(active_frontier_place_node_positions[:,0], active_frontier_place_node_positions[:,1], active_frontier_place_node_positions[:,2],  c='black', marker='D', s=10, label='Frontier nodes')
    ax.scatter(agent_node_positions[:,0], agent_node_positions[:,1], agent_node_positions[:,2],  c='blue', marker='o', s=12, label='Agent nodes')
    ax.scatter(object_node_positions[:,0], object_node_positions[:,1], object_node_positions[:,2],  c='purple', marker='s', s=14, label='Object nodes')

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')

    if view == 'XY':
        ax.view_init(elev=90, azim=-90)
        filename = output_path / f"hm3d_planner_{suffix}_XY.png"
    if view == 'XZ':
        ax.view_init(elev=0, azim=-90)
        filename = output_path / f"hm3d_planner_{suffix}_XZ.png"

    plt.legend()
    plt.title("NavMesh Vertices Visualization")
    plt.savefig(filename)

    with open(output_path / f"hm3d_planner_plt_{suffix}.pkl", 'wb') as f:
        pickle.dump(fig, f)

    plt.close()


@click.group(name="habitat")
def cli():
    """Entry points for running on habitat."""
    pass


@cli.command(name="run")
@click.argument("scene_path", type=click.Path(exists=True))
@click.argument("scene_type", type=str, default='mp3d')
@click.option("-o", "--output-path", default=None)
@click.option("-l", "--label-space", default="ade20k_mp3d")
@click.option("-p", "--prev-dsg", default=None, help="dsg containing trajectory")
@click.option("-s", "--seed", default=None, help="random seed")
@click.option("-v", "--visualize", default=False, help="start visualizer", is_flag=True)
@click.option("-g", "--glog-level", default=0, help="minimum glog level")
@click.option("-f", "--use-full-scene", is_flag=True, help="use-full-scene")
@click.option("-y", "--force", is_flag=True, help="overwrite previous output")
@click.option("--verbosity", default=0, help="glog verbosity")
@click.option("--show-images", default=False, help="show semantics", is_flag=True)
@click.option("--show-config", default=False, help="show hydra config", is_flag=True)
@click.option("--show-progress", default=False, help="show progress bar", is_flag=True)
@click.option("--config-verbosity", default=0, help="glog verbosity to print configs")
def run(
    scene_path,
    scene_type,
    output_path,
    label_space,
    prev_dsg,
    seed,
    visualize,
    glog_level,
    use_full_scene,
    force,
    verbosity,
    show_images,
    show_config,
    show_progress,
    config_verbosity,
):
    """Run Hydra against a habitat scene."""
    from hydra_python._plugins import habitat

    
    if 'hm3d' in scene_type:
        label_space = 'hm3d'

    hydra.set_glog_level(glog_level, verbosity)
    output_path = hydra.resolve_output_path(output_path, force=force)


    data = habitat.HabitatInterface(scene_path, scene_type=scene_type, inflation_radius=0.25)
    poses = _get_trajectory(data, prev_dsg, seed, use_full_scene=use_full_scene)

    configs = hydra.load_configs("habitat", labelspace_name=label_space)
    if not configs:
        click.secho(
            f"Invalid config: dataset 'habitat' and label space '{label_space}'",
            fg="red",
        )
        return

    pipeline_config = hydra.PipelineConfig(configs)
    pipeline_config.enable_reconstruction = True
    pipeline_config.label_names = {i: x for i, x in enumerate(data.colormap.names)}
    data.colormap.fill_label_space(pipeline_config.label_space)

    if output_path:
        pipeline_config.logs.log_dir = str(output_path)

    pipeline = hydra.HydraPipeline(
        pipeline_config, robot_id=0, config_verbosity=config_verbosity
    )
    pipeline.init(configs, hydra.create_camera(data.camera_info))
    if output_path:
        glog_dir = output_path / "logs"
        if not glog_dir.exists():
            glog_dir.mkdir()
        # TODO(nathan) make pathlib bindings
        hydra.set_glog_dir(str(glog_dir))

    rr_logger = RRLogger(output_path)
    if visualize:
        visualizer = DsgVisualizer(start_remote=False)
        visualizer.update_graph(pipeline.graph)
    else:
        visualizer = None

    try:
        hydra.run(
            pipeline,
            data,
            poses,
            visualizer=visualizer,
            show_images=show_images,
            show_progress=show_progress,
            output_path=output_path,
            suffix='test',
            rr_logger=rr_logger,
        )
    finally:
        pipeline.save()
        if visualizer is not None:
            visualizer.stop()


@cli.command(name="reconstruction")
@click.argument("scene_path", type=click.Path(exists=True))
@click.option("-o", "--output-path", default=None)
@click.option("-l", "--label-space", default="ade20k_mp3d")
@click.option("-p", "--prev-dsg", default=None, help="dsg containing trajectory")
@click.option("-s", "--seed", default=None, help="random seed")
@click.option("-g", "--glog-level", default=0, help="minimum glog level")
@click.option("-f", "--use-full-scene", is_flag=True, help="use-full-scene")
@click.option("-y", "--force", is_flag=True, help="overwrite previous output")
@click.option("--verbosity", default=0, help="glog verbosity")
@click.option("--show-images", default=False, help="show semantics", is_flag=True)
@click.option("--show-progress", default=False, help="show progress bar", is_flag=True)
@click.option("--voxel-size", default=0.1, type=float)
def reconstruction(
    scene_path,
    output_path,
    label_space,
    prev_dsg,
    seed,
    glog_level,
    use_full_scene,
    force,
    verbosity,
    show_images,
    show_progress,
    voxel_size,
):
    """Reconstruct a mesh and TSDF of a habitat scene."""
    from hydra_python._plugins import habitat

    hydra.set_glog_level(glog_level, verbosity)
    output_path = hydra.resolve_output_path(output_path, force=force)
    data = habitat.HabitatInterface(scene_path)
    poses = _get_trajectory(data, prev_dsg, seed, use_full_scene=use_full_scene)

    configs = hydra.load_reconstruction_configs(
        "habitat",
        data.camera_info,
        labelspace_name=label_space,
        voxel_size=voxel_size,
    )
    if not configs:
        click.secho(
            f"Invalid config: dataset 'habitat' and label space '{label_space}'",
            fg="red",
        )
        return

    pipeline_config = hydra.PipelineConfig(configs)
    pipeline_config.label_names = {i: x for i, x in enumerate(data.colormap.names)}
    data.colormap.fill_label_space(pipeline_config.label_space)
    if output_path:
        pipeline_config.logs.log_dir = str(output_path)

    pipeline = hydra.HydraReconstruction(pipeline_config, configs)

    try:
        hydra.run(
            pipeline,
            data,
            poses,
            show_images=show_images,
            show_progress=show_progress,
            step_callback=None,
            output_path=output_path,
            suffix='',
        )
    finally:
        pipeline.save()



@cli.command(name="record")
@click.argument("scene_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("-p", "--prev-dsg", default=None, help="dsg containing trajectory")
@click.option("-s", "--seed", default=None, help="random seed")
@click.option("-f", "--use-full-scene", is_flag=True, help="use-full-scene")
@click.option("-y", "--force", is_flag=True, help="overwrite previous output")
@click.option("--show-images", default=False, help="show semantics", is_flag=True)
@click.option("--show-progress", default=False, help="show progress bar", is_flag=True)
def record(
    scene_path,
    output_path,
    prev_dsg,
    seed,
    use_full_scene,
    force,
    show_images,
    show_progress,
):
    """Create a dataset of inputs from MP3D for Hydra."""
    from hydra_python._plugins import habitat

    output_path = hydra.resolve_output_path(output_path, force=force)
    data = habitat.HabitatInterface(scene_path)
    with (output_path / "camera_info.yaml").open("w") as fout:
        fout.write(yaml.dump(data.camera_info))

    poses = _get_trajectory(data, prev_dsg, seed, use_full_scene=use_full_scene)

    with hydra.DatasetLogger(output_path) as recorder:
        hydra.run(
            recorder,
            data,
            poses,
            show_images=show_images,
            show_progress=show_progress,
            step_callback=None,
            output_path=output_path,
            suffix='',
        )

@cli.command(name="camera-info")
@click.argument("scene_path", type=click.Path(exists=True))
def camera_info(scene_path):
    """Create a dataset of inputs from MP3D for Hydra."""
    from hydra_python._plugins import habitat

    data = habitat.HabitatInterface(scene_path)
    print(data.camera_info)


@cli.command(name="run_sg_planner")
@click.argument("scene_path", type=click.Path(exists=True))
@click.argument("scene_type", type=str, default='mp3d')
@click.option("-o", "--output-path", default=None)
@click.option("-l", "--label-space", default="ade20k_mp3d")
@click.option("-p", "--prev-dsg", default=None, help="dsg containing trajectory")
@click.option("-s", "--seed", default=None, help="random seed")
@click.option("-v", "--visualize", default=False, help="start visualizer", is_flag=True)
@click.option("-g", "--glog-level", default=0, help="minimum glog level")
@click.option("-f", "--use-full-scene", is_flag=True, help="use-full-scene")
@click.option("-y", "--force", is_flag=True, help="overwrite previous output")
@click.option("--verbosity", default=0, help="glog verbosity")
@click.option("--show-images", default=False, help="show semantics", is_flag=True)
@click.option("--show-config", default=False, help="show hydra config", is_flag=True)
@click.option("--show-progress", default=False, help="show progress bar", is_flag=True)
@click.option("--config-verbosity", default=0, help="glog verbosity to print configs")
def run_sg_sim(
    scene_path,
    scene_type,
    output_path,
    label_space,
    prev_dsg,
    seed,
    visualize,
    glog_level,
    use_full_scene,
    force,
    verbosity,
    show_images,
    show_config,
    show_progress,
    config_verbosity,
):
    """Run Hydra against a habitat scene."""
    from hydra_python._plugins import habitat

    if 'hm3d' in scene_type:
        label_space = 'hm3d'

    hydra.set_glog_level(glog_level, verbosity)
    output_path = hydra.resolve_output_path(output_path, force=force)
    data = habitat.HabitatInterface(scene_path, scene_type=scene_type)

    configs = hydra.load_configs("habitat", labelspace_name=label_space)
    if not configs:
        click.secho(
            f"Invalid config: dataset 'habitat' and label space '{label_space}'",
            fg="red",
        )
        return

    pipeline_config = hydra.PipelineConfig(configs)
    pipeline_config.enable_reconstruction = True
    pipeline_config.label_names = {i: x for i, x in enumerate(data.colormap.names)}
    data.colormap.fill_label_space(pipeline_config.label_space)

    if output_path:
        pipeline_config.logs.log_dir = str(output_path)

    pipeline = hydra.HydraPipeline(
        pipeline_config, robot_id=0, config_verbosity=config_verbosity
    )
    pipeline.init(configs, hydra.create_camera(data.camera_info))
    if output_path:
        glog_dir = output_path / "logs"
        if not glog_dir.exists():
            glog_dir.mkdir()
        # TODO(nathan) make pathlib bindings
        hydra.set_glog_dir(str(glog_dir))

    rr_logger = RRLogger(output_path)
    if visualize:
        visualizer = DsgVisualizer(start_remote=False)
        visualizer.update_graph(pipeline.graph)
    else:
        visualizer = None

    # FOR PLOTTING THE NAVMESH
    graph_nodes = [data.G.nodes[n]["pos"] for n in data.G]
    positions_camera = np.array([hydra._plugins.habitat._camera_point_from_habitat(p, z_offset=data.z_offset) for p in graph_nodes])

    agent_state = data.get_state()
    click.secho(f"Agent state: {agent_state}",fg="yellow",)

    # Find intial exploration traj
    click.secho("Finding rotate in place trajectory",fg="yellow",)
    poses = data.get_rotate_in_place_trajectory(seed=seed)

    # click.secho("Finding random trajectory",fg="yellow",)
    # poses = data.get_random_trajectory(seed=seed, target_length_m=0.1)

    suffix = 't_0'
    hydra.run(
        pipeline,
        data,
        poses,
        visualizer=visualizer,
        show_images=show_images,
        show_progress=show_progress,
        output_path=output_path,
        suffix=suffix,
        rr_logger=rr_logger,
    )

    pos_traj = np.array([v[1] for v in poses])

    target_poses = []
    target_poses.append(pos_traj[-1])
    poses_to_plot = [v[1] for v in poses]
    _plot_sg_trajs(positions_camera, poses_to_plot, target_poses, pipeline, output_path, suffix, view='XY')
    _plot_sg_trajs(positions_camera, poses_to_plot, target_poses, pipeline, output_path, suffix, view='XZ')

    agent_state = data.get_state()
    click.secho(f"Agent state: {agent_state}",fg="yellow",)

    sg_sim = hydra.SceneGraphSim(output_path, pipeline)
    # sg_sim.test_sg()
    
    # Randomly sample a frontier node and goto it
    active_frontier_place_node_positions = []
    for node in pipeline.graph.nodes:
        if 'f' in node.id.category.lower():
            active_frontier_place_node_positions.append(node.attributes.position)
    
    target_pose = random.choice(active_frontier_place_node_positions).copy()
    target_pose[2] = agent_state[2]
    target_poses.append(target_pose)
    
    poses = data.get_trajectory_to_pose(agent_state, target_pose)
    suffix = 't_1'
    hydra.run(
        pipeline,
        data,
        poses,
        visualizer=visualizer,
        show_images=show_images,
        show_progress=show_progress,
        output_path=output_path,
        suffix=suffix,
    )

    sg_sim.update()
    # sg_sim.test_sg()
    poses_to_plot = [v[1] for v in poses]
    _plot_sg_trajs(positions_camera, poses_to_plot, target_poses, pipeline, output_path, suffix, view='XY')
    _plot_sg_trajs(positions_camera, poses_to_plot, target_poses, pipeline, output_path, suffix, view='XZ')


    pipeline.save()
    if visualizer is not None:
        visualizer.stop()

@cli.command(name="run_sg_planner")
@click.argument("scene_path", type=click.Path(exists=True))
@click.argument("scene_type", type=str, default='mp3d')
@click.option("-o", "--output-path", default=None)
@click.option("-l", "--label-space", default="ade20k_mp3d")
@click.option("-p", "--prev-dsg", default=None, help="dsg containing trajectory")
@click.option("-s", "--seed", default=None, help="random seed")
@click.option("-v", "--visualize", default=False, help="start visualizer", is_flag=True)
@click.option("-g", "--glog-level", default=0, help="minimum glog level")
@click.option("-f", "--use-full-scene", is_flag=True, help="use-full-scene")
@click.option("-y", "--force", is_flag=True, help="overwrite previous output")
@click.option("--verbosity", default=0, help="glog verbosity")
@click.option("--show-images", default=False, help="show semantics", is_flag=True)
@click.option("--show-config", default=False, help="show hydra config", is_flag=True)
@click.option("--show-progress", default=False, help="show progress bar", is_flag=True)
@click.option("--config-verbosity", default=0, help="glog verbosity to print configs")
def run_vlm_planner(
    scene_path,
    scene_type,
    output_path,
    label_space,
    prev_dsg,
    seed,
    visualize,
    glog_level,
    use_full_scene,
    force,
    verbosity,
    show_images,
    show_config,
    show_progress,
    config_verbosity,
):
    """Run Hydra against a habitat scene."""
    from hydra_python._plugins import habitat

    if 'hm3d' in scene_type:
        label_space = 'hm3d'

    hydra.set_glog_level(glog_level, verbosity)
    output_path = hydra.resolve_output_path(output_path, force=force)
    data = habitat.HabitatInterface(scene_path, scene_type=scene_type)

    configs = hydra.load_configs("habitat", labelspace_name=label_space)
    if not configs:
        click.secho(
            f"Invalid config: dataset 'habitat' and label space '{label_space}'",
            fg="red",
        )
        return

    pipeline_config = hydra.PipelineConfig(configs)
    pipeline_config.enable_reconstruction = True
    pipeline_config.label_names = {i: x for i, x in enumerate(data.colormap.names)}
    data.colormap.fill_label_space(pipeline_config.label_space)

    if output_path:
        pipeline_config.logs.log_dir = str(output_path)

    pipeline = hydra.HydraPipeline(
        pipeline_config, robot_id=0, config_verbosity=config_verbosity
    )
    pipeline.init(configs, hydra.create_camera(data.camera_info))
    if output_path:
        glog_dir = output_path / "logs"
        if not glog_dir.exists():
            glog_dir.mkdir()
        # TODO(nathan) make pathlib bindings
        hydra.set_glog_dir(str(glog_dir))

    rr_logger = RRLogger(output_path)
    if visualize:
        visualizer = DsgVisualizer(start_remote=False)
        visualizer.update_graph(pipeline.graph)
    else:
        visualizer = None

    # FOR PLOTTING THE NAVMESH
    graph_nodes = [data.G.nodes[n]["pos"] for n in data.G]
    positions_camera = np.array([hydra._plugins.habitat._camera_point_from_habitat(p, z_offset=data.z_offset) for p in graph_nodes])

    agent_state = data.get_state()
    click.secho(f"Agent state: {agent_state}",fg="yellow",)

    # Find intial exploration traj
    click.secho("Finding rotate in place trajectory",fg="yellow",)
    poses = data.get_rotate_in_place_trajectory(seed=seed)

    suffix = 't_0'
    hydra.run(
        pipeline,
        data,
        poses,
        visualizer=visualizer,
        show_images=show_images,
        show_progress=show_progress,
        output_path=output_path,
        suffix=suffix,
        rr_logger=rr_logger,
    )

    pos_traj = np.array([v[1] for v in poses])

    target_poses = []
    target_poses.append(pos_traj[-1])
    poses_to_plot = [v[1] for v in poses]
    orientations_to_plot = [v[2] for v in poses]
    
    instr = 'Go to the kitchen.'
    planner = hydra.VLMPLanner(instr, output_path, pipeline)
    t=1
    while not planner.done:
        agent_state = data.get_state()

        click.secho(f"Planning at step: {planner.t}",fg="blue",)
        target_pose = planner.get_next_action()
        if target_pose is not None:
            target_pose[2] = agent_state[2] # TODO(saumya): filter frontier nodes to only include nodes in agent place
            poses = data.get_trajectory_to_pose(agent_state, target_pose)

            suffix = f't_{planner.t}'
            if poses is not None:
                poses_to_plot.extend([v[1] for v in poses])
                orientations_to_plot.extend([v[2] for v in poses])
                target_poses.append(target_pose)
                rr_logger.log_traj_data(poses_to_plot)
                for i in range(len(poses_to_plot)):
                    rr_logger.log_agent_tf(poses_to_plot[i], orientations_to_plot[i])
                rr_logger.log_target_poses(target_poses)
                rr_logger.log_text_data(planner.full_plan)

                click.secho(f"Executing trajectory: {planner.t}",fg="yellow",)
                hydra.run(
                    pipeline,
                    data,
                    poses,
                    visualizer=visualizer,
                    show_images=show_images,
                    show_progress=show_progress,
                    output_path=output_path,
                    suffix=suffix,
                    rr_logger=rr_logger,
                )
                
                _plot_sg_trajs(positions_camera, poses_to_plot, target_poses, pipeline, output_path, suffix, view='XY')
                _plot_sg_trajs(positions_camera, poses_to_plot, target_poses, pipeline, output_path, suffix, view='XZ')
                t+=1

    pipeline.save()
    if visualizer is not None:
        visualizer.stop()