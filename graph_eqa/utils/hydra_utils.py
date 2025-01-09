import click
import hydra_python as hydra
from pathlib import Path
from omegaconf import OmegaConf

def initialize_hydra_pipeline(cfg, habitat_data, output_path):
    hydra.set_glog_level(cfg.glog_level, cfg.verbosity)
    configs = hydra.load_configs("habitat", labelspace_name=cfg.label_space)
    if not configs:
        click.secho(
            f"Invalid config: dataset 'habitat' and label space '{cfg.label_space}'",
            fg="red",
        )
        return
    pipeline_config = hydra.PipelineConfig(configs)
    pipeline_config.enable_reconstruction = True

    if habitat_data.cfg.use_semantic_data:
        pipeline_config.label_names = {i: x for i, x in enumerate(habitat_data.colormap.names)}
        habitat_data.colormap.fill_label_space(pipeline_config.label_space) # TODO: check
    else:
        config_path = Path(__file__).resolve().parent.parent.parent.parent / 'config/label_spaces/hm3d_label_space.yaml'
        hm3d_labelspace = OmegaConf.load(config_path)
        names = [d.name for d in hm3d_labelspace.label_names]
        colormap = hydra.SegmentationColormap.from_names(names=names)
        pipeline_config.label_names = {i: x for i, x in enumerate(colormap.names)}
        colormap.fill_label_space(pipeline_config.label_space) # TODO: check
    
    if output_path:
        pipeline_config.logs.log_dir = str(output_path)
    pipeline = hydra.HydraPipeline(
        pipeline_config, robot_id=0, config_verbosity=cfg.config_verbosity, freeze_global_info=False)
    pipeline.init(configs, hydra.create_camera(habitat_data.camera_info))

    if output_path:
        glog_dir = output_path / "logs"
        if not glog_dir.exists():
            glog_dir.mkdir()
        hydra.set_glog_dir(str(glog_dir))
    
    return pipeline