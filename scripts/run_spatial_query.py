#!/usr/bin/env python3
# Metric–semantic–predicate runner (e.g., "2 meters right of oven")
# Scene slug example: "00006-HkseAnWCgqk"

import os, re, json, math, sys
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import click
from omegaconf import OmegaConf
import habitat_sim  # ShortestPath

from graph_eqa.envs.habitat_interface import HabitatInterface
from graph_eqa.utils.hydra_utils import initialize_hydra_pipeline
from graph_eqa.envs.habitat import run as run_hab
from graph_eqa.envs.utils import pos_habitat_to_normal

# ----------------- helpers -----------------

def _scene_path_from_slug(cfg, scene_slug: str) -> str:
    # Dataset layout:
    #   /datasets/hm3d/train/00006-HkseAnWCgqk/HkseAnWCgqk.basis.glb
    folder = scene_slug
    fname = scene_slug.split("-", 1)[1]
    return f"{cfg.data.scene_data_path}/{folder}/{fname}.basis.glb"

def _maybe_call(x):
    return x() if callable(x) else x

def _vec3_to_np(v) -> np.ndarray:
    # Coerce Magnum/iterable to np.float32[3]
    return np.array([float(v[0]), float(v[1]), float(v[2])], dtype=np.float32)

def _points_to_np(points) -> np.ndarray:
    # Convert a sequence of vec3 -> (N,3) float32
    return np.stack([_vec3_to_np(p) for p in points], axis=0).astype(np.float32)

def _plan(pathfinder, start_np: np.ndarray, end_np: np.ndarray):
    sp = habitat_sim.nav.ShortestPath()
    sp.requested_start = start_np
    sp.requested_end = end_np
    ok = pathfinder.find_path(sp)
    return ok, sp

def _find_navigable_anchor_near(pathfinder, center_np: np.ndarray):
    """Return ANY navigable point near the object center (no reachability requirement)."""
    def snap_ok(pt):
        s = _vec3_to_np(pathfinder.snap_point(pt))
        try:
            return s if pathfinder.is_navigable(s) else None
        except Exception:
            ok, _ = _plan(pathfinder, s, s)
            return s if ok else None

    s = snap_ok(center_np)
    if s is not None:
        return s

    radii = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    angles = np.linspace(0, 2*np.pi, 36, endpoint=False)
    for r in radii:
        for th in angles:
            cand = center_np + np.array([math.cos(th)*r, 0.0, math.sin(th)*r], dtype=np.float32)
            s = snap_ok(cand)
            if s is not None:
                return s
    return None

def _find_navigable_near(pathfinder, center_np: np.ndarray, radii=(0.0, 0.25, 0.5, 1.0, 1.5), k=24):
    """Snap given point to navmesh; if unusable, search a ring around it."""
    def snap_ok(pt):
        s = _vec3_to_np(pathfinder.snap_point(pt))
        try:
            return s if pathfinder.is_navigable(s) else None
        except Exception:
            ok, _ = _plan(pathfinder, s, s)
            return s if ok else None

    p = snap_ok(center_np)
    if p is not None:
        return p
    angles = np.linspace(0, 2*np.pi, k, endpoint=False)
    for r in radii:
        for th in angles:
            cand = center_np + np.array([math.cos(th)*r, 0.0, math.sin(th)*r], dtype=np.float32)
            p = snap_ok(cand)
            if p is not None:
                return p
    return None

def _get_semantic_objects(habitat: HabitatInterface) -> List[Dict[str, Any]]:
    sem_scene = getattr(habitat._sim, "semantic_scene", None)
    objs: List[Dict[str, Any]] = []
    if sem_scene is None or len(getattr(sem_scene, "objects", [])) == 0:
        return objs
    for obj in sem_scene.objects:
        cat = getattr(obj, "category", None)
        name_attr = getattr(cat, "name", None) if cat is not None else None
        label = _maybe_call(name_attr) if name_attr is not None else None
        label = (label or "unknown").lower()
        aabb = getattr(obj, "aabb", None)
        center_attr = getattr(aabb, "center", None) if aabb is not None else None
        center_val = _maybe_call(center_attr) if center_attr is not None else None
        if center_val is None:
            continue
        center_np = _vec3_to_np(center_val)
        objs.append({
            "id": getattr(obj, "id", None),
            "label": label,
            "center_habitat": center_np
        })
    return objs

def _basis_from_heading(yaw_rad: float):
    # Agent-frame directions on XZ (Y up)
    front = np.array([math.sin(yaw_rad), 0.0, math.cos(yaw_rad)], dtype=np.float32)
    right = np.array([math.cos(yaw_rad), 0.0, -math.sin(yaw_rad)], dtype=np.float32)
    return {"front": front, "back": -front, "right": right, "left": -right}

_Q_DIRS = {
    "left": "left", "to the left of": "left",
    "right": "right", "to the right of": "right",
    "front": "front", "in front of": "front", "ahead of": "front",
    "back": "back", "behind": "back",
}

def parse_query(q: str) -> Tuple[float, str, str]:
    q = q.strip().lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(m|meter|meters|metre|metres)?", q)
    if not m:
        raise ValueError("No numeric distance found.")
    dist = float(m.group(1))
    direction = None
    for k in sorted(_Q_DIRS.keys(), key=lambda s: -len(s)):
        if k in q:
            direction = _Q_DIRS[k]
            break
    if direction is None:
        raise ValueError("No direction (left/right/front/back) found.")
    idx = q.rfind(" of ")
    obj = q[idx+4:].strip() if idx != -1 else q.split()[-1]
    for art in ("the ", "a ", "an "):
        if obj.startswith(art):
            obj = obj[len(art):]
            break
    return dist, direction, obj.strip()

def _set_agent_yaw_to_face(agent, pos_xyz: np.ndarray, target_xyz: np.ndarray):
    """Rotate agent so its forward faces from pos -> target on XZ plane."""
    from habitat_sim.utils.common import quat_from_angle_axis
    # forward mapping in this repo is [sin(yaw), 0, cos(yaw)]
    v = np.array([target_xyz[0] - pos_xyz[0], 0.0, target_xyz[2] - pos_xyz[2]], dtype=np.float32)
    if np.linalg.norm(v) < 1e-6:
        return
    yaw = math.atan2(v[0], v[2])  # matches front=[sin(yaw),0,cos(yaw)]
    q = quat_from_angle_axis(yaw, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    st = agent.get_state()
    st.rotation = q
    agent.set_state(st)

def _resample_polyline(points: np.ndarray, step: float = 0.15) -> np.ndarray:
    """Upsample N×3 polyline at roughly uniform spacing (XZ plane), preserving endpoints."""
    if points.shape[0] <= 1:
        return points.copy()
    out = [points[0]]
    for i in range(1, len(points)):
        p0, p1 = points[i-1], points[i]
        seg = p1 - p0
        seg_xz = np.array([seg[0], 0.0, seg[2]], dtype=np.float32)
        seg_len = float(np.linalg.norm(seg_xz))
        if seg_len < 1e-6:
            continue
        dir_xz = seg_xz / (seg_len + 1e-8)
        d = step
        while d < seg_len - 1e-6:
            newp = p0 + np.array([dir_xz[0]*d, 0.0, dir_xz[2]*d], dtype=np.float32)
            newp[1] = p0[1]  # keep Y
            out.append(newp)
            d += step
        out.append(p1)
    return np.vstack(out).astype(np.float32)

# ----------------- CLI -----------------

@click.command()
@click.option("--cfg-file", "-cf", required=True, help="cfg yaml name (without .yaml)")
@click.option("--scene", required=True, help='Scene slug, e.g. "00006-HkseAnWCgqk"')
@click.option("--query", "-q", required=True, help='e.g., "2 meters right of oven"')
@click.option("--out-root", default="outputs/spatial_queries", help="Output root folder")
@click.option("--execute", is_flag=True, help="Follow path & render frames (direct Habitat capture)")
@click.option("--log", is_flag=True, help="Log markers/paths to Rerun")
@click.option("--teleport", is_flag=True, help="Teleport agent to the oven/stove anchor before offset")
@click.option("--backoff", type=float, default=0.0,
              help="Meters to start away from the object (opposite requested direction) after teleport.")
@click.option("--hard-exit", is_flag=True, help="os._exit(0) to avoid native segfaults at teardown")
@click.option("--face-object", is_flag=True, default=True,
              help="Rotate the agent to face the target object at each frame.")
@click.option("--reverse-approach", is_flag=True, default=False,
              help="Traverse the waypoints in reverse order (for inverted approach).")

def main(cfg_file, scene, query, out_root, execute, log, teleport, backoff, face_object, reverse_approach, hard_exit):
    config_path = Path(__file__).resolve().parent.parent / "cfg" / f"{cfg_file}.yaml"
    cfg = OmegaConf.load(config_path); OmegaConf.resolve(cfg)

    scene_path = _scene_path_from_slug(cfg, scene)
    device = "cuda:0"
    habitat = None
    rr = None

    try:
        habitat = HabitatInterface(scene_path, cfg=cfg.habitat, device=device)

        out_dir = Path(__file__).resolve().parent.parent / out_root / f"spatial_{scene}"
        out_dir.mkdir(parents=True, exist_ok=True)

        objs = _get_semantic_objects(habitat)
        if not objs:
            raise SystemExit("No semantic objects available (.semantic.glb missing?)")

        # Agent pose & heading
        agent = habitat._sim.get_agent(0)
        st = agent.get_state()
        agent_pos = _vec3_to_np(st.position)
        try:
            heading = habitat.get_heading_angle()
        except Exception:
            heading = 0.0
        basis = _basis_from_heading(heading)

        # Parse query
        dist_m, direction, label = parse_query(query)

        # Collect object candidates by alias
        aliases = [label]
        if label == "oven":
            aliases += ["stove", "range", "cooktop", "kitchen appliance", "appliance"]
        cands = [o for o in objs if any(a in o["label"] for a in aliases)]
        if not cands:
            raise SystemExit(f'No objects found for {aliases}')
        cands.sort(key=lambda o: np.linalg.norm(o["center_habitat"] - agent_pos))

        # ---------- Stage A: pick object and an anchor near it ----------
        anchor = None
        if teleport:
            # Do NOT require reachability from current start; just find any navigable point near object
            for o in cands:
                obj_anchor = _find_navigable_anchor_near(habitat.pathfinder, o["center_habitat"])
                if obj_anchor is not None:
                    anchor = (o, obj_anchor)
                    break
        else:
            # If not teleporting, require reachability from current start
            for o in cands:
                obj_snap = _vec3_to_np(habitat.pathfinder.snap_point(o["center_habitat"]))
                ok, sp = _plan(habitat.pathfinder, agent_pos, obj_snap)
                if ok:
                    anchor = (o, obj_snap)
                    break
            if anchor is None:
                for o in cands:
                    base = o["center_habitat"]
                    radii = [0.25, 0.5, 1.0, 1.5]
                    angles = np.linspace(0, 2*np.pi, 24, endpoint=False)
                    done = False
                    for r in radii:
                        for th in angles:
                            cand = base + np.array([math.cos(th)*r, 0.0, math.sin(th)*r], dtype=np.float32)
                            cand = _vec3_to_np(habitat.pathfinder.snap_point(cand))
                            ok, sp = _plan(habitat.pathfinder, agent_pos, cand)
                            if ok:
                                anchor = (o, cand); done = True; break
                        if done: break
                    if done: break

        if anchor is None:
            result = {
                "scene": scene, "query": query,
                "parsed": {"distance_m": float(dist_m), "direction": direction, "object_label": label},
                "agent_pos_habitat": agent_pos.astype(float).tolist(),
                "status": "no_navigable_anchor_found_near_object"
            }
            (out_dir / "result.json").write_text(json.dumps(result, indent=2))
            click.secho("No navigable anchor found near oven/stove.", fg="red")
            return

        target_obj, obj_anchor = anchor

        # Teleport near object if requested
        if teleport:
            agent_state = agent.get_state()
            agent_state.position = obj_anchor
            agent.set_state(agent_state)
            st = agent.get_state()
            agent_pos = _vec3_to_np(st.position)

            # Optional: back off for longer approach
            if backoff and backoff > 0.0:
                dir_vec_for_backoff = basis[direction]  # agent-frame direction
                desired_start = obj_anchor - dir_vec_for_backoff * float(backoff)
                start_nav = _find_navigable_near(habitat.pathfinder, desired_start)
                if start_nav is not None:
                    agent_state = agent.get_state()
                    agent_state.position = start_nav.astype(np.float32)
                    agent.set_state(agent_state)
                    st = agent.get_state()
                    agent_pos = _vec3_to_np(st.position)
                else:
                    click.secho(f"Backoff {backoff}m requested but no navigable start found near desired point; continuing without backoff.", fg="yellow")

        # ---------- Stage B: apply offset & robust search for a reachable goal ----------
        # Build the desired offset (agent-frame)
        dir_vec = basis[direction]
        desired = obj_anchor + dir_vec * dist_m

        # plan helper with Y aligned to agent height
        def plan_to(end_pt_world: np.ndarray):
            end_pt_adj = end_pt_world.copy()
            end_pt_adj[1] = agent_pos[1]  # keep same height as agent
            end_pt_adj = _vec3_to_np(habitat.pathfinder.snap_point(end_pt_adj))
            return _plan(habitat.pathfinder, agent_pos, end_pt_adj), end_pt_adj

        # 1) Try exact desired point
        (ok_goal, sp_goal), goal_snap = plan_to(desired)

        # 2) If not reachable, sample a dense fan around the object and pick best candidate
        if not ok_goal:
            radii = [0.35, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]
            thetas = np.linspace(0, 2*np.pi, 16, endpoint=False)
            best = None  # (score, cand_goal_snap, sp)
            for r in radii:
                for th in thetas:
                    cand = obj_anchor + np.array([math.cos(th)*r, 0.0, math.sin(th)*r], dtype=np.float32)
                    (ok2, sp2), cand_snap = plan_to(cand)
                    if not ok2:
                        continue
                    cand_dir = cand - obj_anchor
                    cand_len = np.linalg.norm(cand_dir + 1e-6)
                    cand_dir_n = cand_dir / (cand_len + 1e-8)
                    desired_dir_n = dir_vec / (np.linalg.norm(dir_vec) + 1e-8)
                    align = float(np.clip(np.dot(cand_dir_n, desired_dir_n), -1.0, 1.0))  # [-1, 1]
                    dist_penalty = abs(r - dist_m)
                    score = align - 0.3 * dist_penalty
                    if (best is None) or (score > best[0]):
                        best = (score, cand_snap, sp2)
            if best is not None:
                _, goal_snap, sp_goal = best
                ok_goal = True

        found_path = ok_goal
        path_points_np = _points_to_np(sp_goal.points) if found_path else np.zeros((0,3), dtype=np.float32)

        # Densify for smoother video
        if found_path and path_points_np.shape[0] > 1:
            path_points_np = _resample_polyline(path_points_np, step=0.15)

        if reverse_approach:
            path_points_np = path_points_np[::-1].copy()

        # distance from object center to final goal
        goal_dist_m = float(np.linalg.norm(goal_snap - target_obj["center_habitat"]))

        # ---------- Result ----------
        result = {
            "scene": scene,
            "query": query,
            "parsed": {"distance_m": float(dist_m), "direction": direction, "object_label": label},
            "agent_pos_habitat": agent_pos.astype(float).tolist(),
            "object": {
                "id": target_obj["id"],
                "label": target_obj["label"],
                "center_habitat": target_obj["center_habitat"].astype(float).tolist(),
                "anchor_nav": obj_anchor.astype(float).tolist()
            },
            "goal_habitat": goal_snap.astype(float).tolist(),
            "goal_distance_from_object_m": goal_dist_m,
            "path_found": bool(found_path),
            "path_num_points": int(path_points_np.shape[0]),
            "teleported": bool(teleport),
            "backoff_m": float(backoff or 0.0),
            "face_object": bool(face_object),
            "reverse_approach": bool(reverse_approach),
        }
        (out_dir / "result.json").write_text(json.dumps(result, indent=2))
        click.secho(f"Goal (habitat): {result['goal_habitat']}", fg="green")
        click.secho(f"Path found: {bool(found_path)}  points: {result['path_num_points']}", fg="cyan")
        click.secho(f"Wrote: {out_dir/'result.json'}", fg="cyan")

        # ---------- Optional Rerun logging ----------
        if log:
            try:
                from graph_eqa.logging.rr_logger import RRLogger
                rr = RRLogger(out_dir)
                rr.log_target_poses(goal_snap.astype(np.float32))
                if found_path and path_points_np.shape[0] > 0:
                    rr.log_traj_data(path_points_np[:-1] if path_points_np.shape[0] > 1 else path_points_np)
                rr.log_text_data(json.dumps(result, indent=2))
            except Exception as e:
                click.secho(f"Rerun log warning: {e}", fg="yellow")

        # ---------- Execute & render (direct Habitat capture) ----------
        if found_path and path_points_np.shape[0] > 0 and execute:
            agent = habitat._sim.get_agent(0)

            # pick an RGB/color sensor
            sensor_keys = list(habitat._sim._sensors.keys())
            rgb_key = next((k for k in sensor_keys if "color" in k.lower() or "rgb" in k.lower()), None)
            if rgb_key is None:
                click.secho("No RGB/color sensor found; cannot capture frames.", fg="red")
            else:
                import imageio.v2 as imageio
                frames_saved = 0
                st = agent.get_state()
                y_level = st.position[1]

                # 0) capture the current view (before moving)
                if face_object:
                    _set_agent_yaw_to_face(agent, st.position, target_obj["center_habitat"])
                obs = habitat._sim.get_sensor_observations()
                rgb = obs.get(rgb_key, next(iter(obs.values())) if isinstance(obs, dict) and obs else None)
                if rgb is not None:
                    arr = np.asarray(rgb)
                    if arr.ndim == 3 and arr.shape[2] >= 3:
                        arr = arr[:, :, :3]
                    imageio.imwrite((out_dir / f"frame_{frames_saved:05d}.png").as_posix(), arr)
                    frames_saved += 1

                # 1) step through ALL waypoints (include last)
                for p in path_points_np:
                    st.position = np.array([float(p[0]), float(y_level), float(p[2])], dtype=np.float32)
                    agent.set_state(st)
                    if face_object:
                        _set_agent_yaw_to_face(agent, st.position, target_obj["center_habitat"])
                    obs = habitat._sim.get_sensor_observations()
                    rgb = obs.get(rgb_key, next(iter(obs.values())) if isinstance(obs, dict) and obs else None)
                    if rgb is not None:
                        arr = np.asarray(rgb)
                        if arr.ndim == 3 and arr.shape[2] >= 3:
                            arr = arr[:, :, :3]
                        imageio.imwrite((out_dir / f"frame_{frames_saved:05d}.png").as_posix(), arr)
                        frames_saved += 1

                click.secho(f"Saved {frames_saved} frames (frame_00000.png ...).", fg="green")

    finally:
        # Hard close to reduce native teardown crashes
        try:
            if habitat is not None and getattr(habitat, "_sim", None) is not None:
                habitat._sim.close(destroy=True)
        except Exception:
            pass
        if hard_exit:
            os._exit(0)

if __name__ == "__main__":
    main()
