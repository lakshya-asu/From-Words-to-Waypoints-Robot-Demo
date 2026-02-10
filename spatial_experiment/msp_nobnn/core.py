from __future__ import annotations
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np

from .geometry import (
    wrap_angle,
    camera_theta_to_world,
    estimate_object_front_yaw,
    predicate_offset_from_text,
    circular_blend,
)

# ---------------------- Data structures --------------------------------------

@dataclass
class SceneObject:
    obj_id: str
    name: str
    position: np.ndarray  # (x,y,z) habitat coords
    size: np.ndarray      # (w,d,h) approx
    source: str           # "graph" or "dataset"

@dataclass
class QueryFrame:
    distance_m: float
    sigma_m: float
    predicate: str
    anchor_class: str
    anchor_constraints: List[str]
    target_class: Optional[str] = None
    anchor_hint: Optional[str] = None

@dataclass
class PredicateParams:
    theta_cam: float
    phi_cam: float
    kappa: float
    confidence: float = 0.5
    reasoning: str = ""

@dataclass
class MSPNoBNNConfig:
    # outputs
    mode: str = "which"  # "which" or "where"
    top_k: int = 5
    confidence_threshold: float = 0.80

    # VLM knobs (optional)
    use_vlm_predicate: bool = False
    use_vlm_explain: bool = False

    # predicate fusion
    fusion_weight: float = 0.5
    default_kappa: float = 8.0
    default_phi: float = math.pi / 2.0

    # IMPORTANT: the config keys you added in yaml
    semantic_mode: str = "neutral"   # "neutral" or "candidate"
    sigma_s_scale: float = 0.50
    sigma_m_scale: float = 0.30

    # where-mode grid defaults
    region_grid: Optional[Dict[str, Any]] = None
    where_top_frac: float = 0.02


# ---------------------- Query parsing ----------------------------------------

def _parse_distance(q: str) -> float:
    ql = (q or "").lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*meters?", ql)
    return float(m.group(1)) if m else 0.0

def _parse_anchor_class(q: str) -> str:
    """
    Parse multi-word anchor noun phrase.
    Examples:
      "behind the exercise machine" -> "exercise machine"
      "in front of the 2 seater sofa" -> "2 seater sofa"
    """
    ql = (q or "").lower()

    # capture after: "<predicate> of (the)? <noun phrase>"
    # stop at punctuation or end.
    m = re.search(
        r"(?:left|right|behind|in\s+front|front)\s+of\s+(?:the\s+)?(.+?)(?:\?|\.|,|;|$)",
        ql,
    )
    if m:
        phrase = m.group(1).strip()
        # remove parentheses info like "(anchor object id: ...)"
        phrase = re.sub(r"\(.*?\)", "", phrase).strip()
        # normalize multiple spaces
        phrase = re.sub(r"\s+", " ", phrase).strip()
        return phrase

    # fallback: last 2-3 tokens (better than last token)
    toks = re.findall(r"[a-zA-Z0-9_]+", ql)
    if not toks:
        return "object"
    return " ".join(toks[-3:]).strip()

def _parse_anchor_hint(question: str) -> Optional[str]:
    ql = (question or "").lower()
    m = re.search(r"\(\s*anchor\s+object\s+id\s*:\s*([^)]+)\)", ql)
    if not m:
        return None
    hint = m.group(1).strip()
    hint = re.sub(r"\s+", " ", hint)
    return hint or None

def _parse_predicate(q: str) -> str:
    ql = (q or "").lower()
    if "left" in ql:
        return "left_of"
    if "right" in ql:
        return "right_of"
    if "behind" in ql or "back of" in ql:
        return "behind"
    if "in front" in ql or "front of" in ql:
        return "in_front_of"
    if "above" in ql or "on top" in ql:
        return "above"
    if "below" in ql or "under" in ql:
        return "below"
    return "near"

def _parse_constraints(q: str) -> List[str]:
    ql = (q or "").lower()
    cons: List[str] = []
    if "closest to me" in ql or "closest to agent" in ql or "closest to the agent" in ql:
        cons.append("closest_to_agent")
    if "close to wall" in ql or "near the wall" in ql or "next to the wall" in ql or "against the wall" in ql:
        cons.append("near_wall")
    if "close to table" in ql or "near the table" in ql or "next to the table" in ql:
        cons.append("near_table")
    if "across the table" in ql or "across table" in ql:
        cons.append("across_table")
    return cons

def _sigma_from_text(question: str, base: float) -> float:
    ql = (question or "").lower()
    if "exactly" in ql or "precisely" in ql:
        return 0.5 * base
    if "around" in ql or "roughly" in ql or "about" in ql:
        return 1.3 * base
    return base

def parse_query_frame(question: str) -> QueryFrame:
    d0 = _parse_distance(question)
    pred = _parse_predicate(question)
    anchor = _parse_anchor_class(question)
    constraints = _parse_constraints(question)

    sigma_m = _sigma_from_text(question, base=0.6)

    anchor_hint = _parse_anchor_hint(question)

    return QueryFrame(
        distance_m=float(d0),
        sigma_m=float(sigma_m),
        predicate=pred,
        anchor_class=anchor,
        anchor_constraints=constraints,
        anchor_hint=anchor_hint,
    )


# ---------------------- Engine (no BNN) --------------------------------------

class MSPNoBNNEngine:
    def __init__(self, combined_logpdf_fn: Callable, cfg: MSPNoBNNConfig):
        self.combined_logpdf = combined_logpdf_fn
        self.cfg = cfg

    # --------- Anchor selection ---------------------------------------------

    def resolve_anchor_distribution(
        self,
        frame: QueryFrame,
        objects: List[SceneObject],
        agent_pos: np.ndarray,
    ) -> List[Tuple["SceneObject", float]]:
        """
        Anchor resolution policy (hint-first):
        1) If frame.anchor_hint exists (e.g. 'sofa'), strongly prioritize objects whose name matches hint.
        2) If multiple match, use token overlap with anchor_class phrase (e.g. '2 seater sofa').
        3) Tie-break with distance to agent (prefer closer for quicker disambiguation).
        No hard structural filtering.
        """
        hint = (frame.anchor_hint or "").lower().strip()
        phrase = (frame.anchor_class or "").lower().strip()

        hint_tokens = re.findall(r"[a-z0-9_]+", hint) if hint else []
        phrase_tokens = re.findall(r"[a-z0-9_]+", phrase) if phrase else []

        scored: List[Tuple["SceneObject", float]] = []

        for o in objects:
            nm = (o.name or "").lower()
            nm_tokens = set(re.findall(r"[a-z0-9_]+", nm))

            # --- scoring ---
            s = 0.0

            # (A) Strong hint match (primary_object)
            # If hint exists, this dominates.
            if hint_tokens:
                # count overlaps with hint tokens
                hint_overlap = sum(1 for t in hint_tokens if t in nm_tokens)
                if hint_overlap > 0:
                    s += 10.0 + 2.0 * hint_overlap  # strong boost
                else:
                    s -= 10.0  # push non-hint objects down hard

            # (B) Phrase match ("2 seater sofa") as secondary disambiguator
            if phrase_tokens:
                phrase_overlap = sum(1 for t in phrase_tokens if t in nm_tokens)
                s += 1.0 * phrase_overlap

            # (C) tie-breaker: prefer closer to agent (small weight)
            dist = float(np.linalg.norm(o.position - agent_pos))
            s += -0.05 * dist

            # (D) existing constraint hooks (optional)
            if "closest_to_agent" in frame.anchor_constraints:
                s += -dist

            if "near_wall" in frame.anchor_constraints:
                s += -self._distance_to_wall_heuristic(o.position, objects)

            if "near_table" in frame.anchor_constraints or "across_table" in frame.anchor_constraints:
                table = self._best_named_object(objects, "table")
                if table is not None:
                    d = float(np.linalg.norm(o.position - table.position))
                    if "near_table" in frame.anchor_constraints:
                        s += -d
                    if "across_table" in frame.anchor_constraints:
                        v_obj = o.position - table.position
                        v_agent = agent_pos - table.position
                        if float(np.linalg.norm(v_obj)) > 1e-6 and float(np.linalg.norm(v_agent)) > 1e-6:
                            v_obj = v_obj / (np.linalg.norm(v_obj) + 1e-9)
                            v_agent = v_agent / (np.linalg.norm(v_agent) + 1e-9)
                            s += -float(np.dot(v_obj, v_agent))

            scored.append((o, float(s)))

        scored.sort(key=lambda t: t[1], reverse=True)

        # Optional: if hint was provided but NOTHING matched (all got penalized),
        # fall back to phrase-based matching (so exploration can still recover).
        if hint_tokens:
            best_score = scored[0][1] if scored else -1e9
            if best_score < -5.0:
                # remove hint penalty by re-scoring without hint constraint
                frame2 = QueryFrame(
                    distance_m=frame.distance_m,
                    sigma_m=frame.sigma_m,
                    predicate=frame.predicate,
                    anchor_class=frame.anchor_class,
                    anchor_constraints=frame.anchor_constraints,
                    target_class=frame.target_class,
                    anchor_hint=None,
                )
                return self.resolve_anchor_distribution(frame2, objects, agent_pos)

        return scored


    def anchor_confidence(self, anchor_dist: List[Tuple["SceneObject", float]]) -> float:
        """
        Confidence from score gap between best and 2nd best.
        If all equal => low confidence.
        """
        if not anchor_dist:
            return 0.0
        if len(anchor_dist) == 1:
            return 1.0
        gap = float(anchor_dist[0][1] - anchor_dist[1][1])
        # map to [0,1]
        conf = float(1.0 - np.exp(-max(0.0, gap)))
        return conf

    def _best_named_object(self, objects: List[SceneObject], name_substr: str) -> Optional[SceneObject]:
        name_substr = (name_substr or "").lower()
        for o in objects:
            if name_substr in (o.name or "").lower():
                return o
        return None

    def _distance_to_wall_heuristic(self, p: np.ndarray, objects: List[SceneObject]) -> float:
        walls = [o for o in objects if "wall" in (o.name or "").lower()]
        if walls:
            return float(min(np.linalg.norm(o.position - p) for o in walls))
        pts = np.array([o.position for o in objects], np.float32)
        if len(pts) < 3:
            return 10.0
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        dx = min(abs(p[0] - mins[0]), abs(maxs[0] - p[0]))
        dz = min(abs(p[2] - mins[2]), abs(maxs[2] - p[2]))
        return float(min(dx, dz))

    # --------- Kernel parameters --------------------------------------------

    def _size_scale(self, obj: SceneObject) -> float:
        w, d, h = [float(x) for x in obj.size[:3]]
        return max(w, d, h, 1e-3)

    def build_metric_semantic_params(
        self,
        frame: QueryFrame,
        anchor_pos: np.ndarray,
        candidate_obj: SceneObject,
    ) -> Dict[str, float]:
        """
        IMPORTANT:
        - For ranking candidates by MSP at their centers, the semantic term can be neutralized
          (default) so ranking is driven by metric+predicate relative to anchor.
        - If semantic_mode="candidate", semantic term is centered at candidate position
          (at candidate center logp_sem becomes 0, but sigma_s can still affect if you later
          evaluate elsewhere).
        """
        max_dim = self._size_scale(candidate_obj)

        sigma_m = float(max(0.25, self.cfg.sigma_m_scale * max_dim))
        sigma_m = _sigma_from_text("", sigma_m)  # keep as-is, text already used in frame.sigma_m
        # allow query sigma to widen/sharpen slightly
        sigma_m = float(max(0.25, sigma_m * float(max(0.5, min(2.0, frame.sigma_m)))))

        if (self.cfg.semantic_mode or "neutral") == "candidate":
            sigma_s = float(max(1e-4, self.cfg.sigma_s_scale * max_dim))
            mu_x, mu_y, mu_z = [float(x) for x in candidate_obj.position]
        else:
            # neutral semantic term
            sigma_s = 1e6
            mu_x, mu_y, mu_z = [float(x) for x in anchor_pos]

        return {
            "mu_x": float(mu_x),
            "mu_y": float(mu_y),
            "mu_z": float(mu_z),
            "sigma_s": float(sigma_s),
            "x0": float(anchor_pos[0]),
            "y0": float(anchor_pos[1]),
            "z0": float(anchor_pos[2]),
            "d0": float(frame.distance_m),
            "sigma_m": float(sigma_m),
        }

    def build_predicate_params(
        self,
        frame: QueryFrame,
        anchor_pos: np.ndarray,
        agent_pos: np.ndarray,
        agent_yaw: float,
        predicate_from_vlm: Optional[PredicateParams],
    ) -> Tuple[float, float, float]:
        yaw_front_world = estimate_object_front_yaw(anchor_pos, agent_pos)
        offset = predicate_offset_from_text(frame.predicate.replace("_", " "))
        theta_prior_world = wrap_angle(yaw_front_world + offset)
        phi_prior = float(self.cfg.default_phi)
        kappa_prior = float(self.cfg.default_kappa)

        if predicate_from_vlm is None:
            return theta_prior_world, phi_prior, kappa_prior

        theta_vlm_world = camera_theta_to_world(predicate_from_vlm.theta_cam, agent_yaw)
        theta0 = circular_blend(theta_prior_world, theta_vlm_world, float(self.cfg.fusion_weight))
        phi0 = float(predicate_from_vlm.phi_cam)
        kappa = float(predicate_from_vlm.kappa)
        return float(theta0), float(phi0), float(kappa)

    # --------- Mode: WHICH ---------------------------------------------------

    def run_which_mode(
        self,
        frame: QueryFrame,
        objects: List[SceneObject],
        agent_pos: np.ndarray,
        agent_yaw: float,
        predicate_from_vlm: Optional[PredicateParams] = None,
    ) -> Tuple[List[Dict[str, Any]], Optional[SceneObject], Dict[str, Any]]:

        anchor_dist = self.resolve_anchor_distribution(frame, objects, agent_pos)
        if not anchor_dist:
            return [], None, {"error": "no objects"}

        anchor_obj = anchor_dist[0][0]
        anchor_pos = anchor_obj.position

        theta0, phi0, kappa = self.build_predicate_params(
            frame=frame,
            anchor_pos=anchor_pos,
            agent_pos=agent_pos,
            agent_yaw=agent_yaw,
            predicate_from_vlm=predicate_from_vlm,
        )

        rows: List[Dict[str, Any]] = []
        for o in objects:
            if o.obj_id == anchor_obj.obj_id:
                continue

            base = self.build_metric_semantic_params(frame, anchor_pos, o)
            params = {**base, "theta0": theta0, "phi0": phi0, "kappa": kappa}

            x = np.array([o.position[0]], np.float32)
            y = np.array([o.position[1]], np.float32)
            z = np.array([o.position[2]], np.float32)

            logp_total = float(self.combined_logpdf(x, y, z, params)[0])

            rows.append({
                "candidate_id": o.obj_id,
                "name": o.name,
                "source": o.source,
                "pos": o.position,
                "anchor_id": anchor_obj.obj_id,
                "anchor_name": anchor_obj.name,
                "theta0": float(theta0),
                "phi0": float(phi0),
                "kappa": float(kappa),
                "logp_total": float(logp_total),
            })

        rows.sort(key=lambda r: r["logp_total"], reverse=True)
        best = None
        if rows:
            best_id = rows[0]["candidate_id"]
            best = next((o for o in objects if o.obj_id == best_id), None)

        summary = {
            "anchor_chosen": {"id": anchor_obj.obj_id, "name": anchor_obj.name, "pos": anchor_pos.tolist()},
            "theta0": float(theta0),
            "phi0": float(phi0),
            "kappa": float(kappa),
            "top_ids": [r["candidate_id"] for r in rows[: min(5, len(rows))]],
        }
        return rows, best, summary

    def compute_confidence_from_rows(self, rows: List[Dict[str, Any]]) -> Tuple[bool, float]:
        if not rows:
            return False, 0.0
        if len(rows) == 1:
            return True, 1.0
        gap = float(rows[0]["logp_total"] - rows[1]["logp_total"])
        conf = float(1.0 - math.exp(-max(0.0, gap) / 2.0))
        return conf >= float(self.cfg.confidence_threshold), conf