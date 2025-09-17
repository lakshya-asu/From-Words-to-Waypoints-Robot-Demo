#!/usr/bin/env python3
import os, json, sys
from pathlib import Path
from collections import Counter

import numpy as np
from omegaconf import OmegaConf
import click

from graph_eqa.envs.habitat_interface import HabitatInterface

def _scene_path_from_id(cfg, scene_id: str) -> str:
    """
    Expected layout:
      /datasets/hm3d/train/00006-HkseAnWCgqk/HkseAnWCgqk.basis.glb
    Where scene_id = '00006-HkseAnWCgqk'
    """
    folder = scene_id                     # '00006-HkseAnWCgqk'
    fname = scene_id.split("-", 1)[1]     # 'HkseAnWCgqk'
    return f"{cfg.data.scene_data_path}/{folder}/{fname}.basis.glb"

@click.command()
@click.option("--cfg-file", "-cf", required=True, help="cfg yaml name (without .yaml)")
@click.option("--scene", required=True, help='Scene id like "3_00006-HkseAnWCgqk"')
def main(cfg_file, scene):
    # Load config (same style as your runner)
    config_path = Path(__file__).resolve().parent.parent / "cfg" / f"{cfg_file}.yaml"
    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)

    scene_path = _scene_path_from_id(cfg, scene)
    device = "cuda:0"

    # Spin up Habitat (your project wrapper)
    habitat = HabitatInterface(scene_path, cfg=cfg.habitat, device=device)

    # Try semantic scene (requires semantic mesh to be available/auto-loaded by HabitatInterface)
    sem_scene = getattr(habitat._sim, "semantic_scene", None)

    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "scene_inspect" / f"inspect_{scene}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if sem_scene is None or len(getattr(sem_scene, "objects", [])) == 0:
        click.secho("No semantic objects found via Habitat. "
                    "Make sure cfg.data.use_semantic_data=True and the scene has a .semantic.glb.", fg="yellow")
        # Write an empty inventory so downstream steps won’t crash
        (out_dir / "objects.json").write_text(json.dumps({"objects": [], "by_label": {}}, indent=2))
        sys.exit(0)

    rows = []
    label_counts = Counter()

    def _maybe_call(x):
        return x() if callable(x) else x

    for obj in sem_scene.objects:
        # label
        cat = getattr(obj, "category", None)
        name_attr = getattr(cat, "name", None) if cat is not None else None
        label = _maybe_call(name_attr) if name_attr is not None else None
        label = (label or "unknown").lower()

        # center
        aabb = getattr(obj, "aabb", None)
        center_attr = getattr(aabb, "center", None) if aabb is not None else None
        center_val = _maybe_call(center_attr) if center_attr is not None else None

        if center_val is None:
            center = [float("nan")] * 3
        else:
            # center_val might be a tuple, numpy vec, or Magnum vec
            try:
                center = list(center_val)  # works for tuple/np/magnum
            except TypeError:
                # last-resort attempt
                center = [float(center_val[0]), float(center_val[1]), float(center_val[2])]

        oid = getattr(obj, "id", None)
        rows.append({"id": oid, "label": label, "center_habitat": center})
        label_counts[label] += 1

    data = {
        "scene": scene,
        "objects": rows,
        "by_label": dict(label_counts.most_common())
    }
    (out_dir / "objects.json").write_text(json.dumps(data, indent=2))

    click.secho(f"Found {len(rows)} objects. Top labels:", fg="green")
    for lab, cnt in label_counts.most_common(25):
        click.echo(f"  {lab}: {cnt}")

    click.secho(f"Wrote: {out_dir/'objects.json'}", fg="cyan")

if __name__ == "__main__":
    main()
