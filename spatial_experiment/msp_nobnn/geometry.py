from __future__ import annotations
import math
import numpy as np
from typing import Tuple

def wrap_angle(a: float) -> float:
    two_pi = 2.0 * math.pi
    return (a % two_pi + two_pi) % two_pi

def camera_theta_to_world(vlm_theta: float, agent_yaw: float) -> float:
    # camera: 0 right, pi/2 forward, pi left, 3pi/2 back
    theta_forward = vlm_theta - (math.pi / 2.0)
    return wrap_angle(agent_yaw + theta_forward)

def circular_blend(theta_a: float, theta_b: float, w: float) -> float:
    """
    Blend angles robustly on the circle using unit vectors.
    w in [0,1] is weight on theta_b.
    """
    w = float(max(0.0, min(1.0, w)))
    va = np.array([math.cos(theta_a), math.sin(theta_a)], np.float32)
    vb = np.array([math.cos(theta_b), math.sin(theta_b)], np.float32)
    v = (1.0 - w) * va + w * vb
    if float(np.linalg.norm(v)) < 1e-9:
        return wrap_angle(theta_a)
    return wrap_angle(float(math.atan2(v[1], v[0])))

def estimate_object_front_yaw(reference_hab: np.ndarray, agent_pos_hab: np.ndarray) -> float:
    ref = np.asarray(reference_hab, dtype=np.float32)
    agent = np.asarray(agent_pos_hab, dtype=np.float32)
    dx = agent[0] - ref[0]
    dz = agent[2] - ref[2]
    yaw_obj_to_agent = math.atan2(dz, dx)
    return wrap_angle(yaw_obj_to_agent + math.pi)

def predicate_offset_from_text(q: str) -> float:
    q = (q or "").lower()
    if "left" in q:
        return +math.pi / 2.0
    if "right" in q:
        return -math.pi / 2.0
    if "behind" in q or "back of" in q or "backside" in q:
        return math.pi
    return 0.0
