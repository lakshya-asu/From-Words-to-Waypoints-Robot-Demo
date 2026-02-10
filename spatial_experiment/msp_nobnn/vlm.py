from __future__ import annotations
import base64
import json
import mimetypes
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import google.generativeai as genai


@dataclass
class GeminiVLMConfig:
    api_key: str
    model_predicate: str = "models/gemini-2.5-pro"     # safer default for most installs
    model_explain: str = "models/gemini-2.0-flash"     # fast narration


@dataclass
class PredicateParams:
    theta_cam: float
    phi_cam: float
    kappa: float
    confidence: float = 0.5
    reasoning: str = ""


def _b64encode_image(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None


def _guess_mime(path: str) -> str:
    return mimetypes.guess_type(path)[0] or "image/png"


def _extract_json(text: str) -> Dict[str, Any]:
    """
    Gemini sometimes wraps JSON in markdown fences; robustly extract first JSON object.
    """
    if not text:
        return {}
    text = text.strip()
    # fast path
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    # strip markdown fences
    if "```" in text:
        parts = text.split("```")
        # look for a fenced block that contains '{'
        for p in parts:
            p2 = p.strip()
            if p2.startswith("{") and p2.endswith("}"):
                return json.loads(p2)
            # sometimes starts with json\n{...}
            if "{" in p2 and "}" in p2:
                i = p2.find("{")
                j = p2.rfind("}")
                if i >= 0 and j > i:
                    return json.loads(p2[i : j + 1])

    # generic: find first {...}
    i = text.find("{")
    j = text.rfind("}")
    if i >= 0 and j > i:
        return json.loads(text[i : j + 1])

    return {}


class GeminiVLMClient:
    def __init__(self, cfg: GeminiVLMConfig):
        self.cfg = cfg
        genai.configure(api_key=cfg.api_key)
        self.model_pred = genai.GenerativeModel(model_name=cfg.model_predicate)
        self.model_exp = genai.GenerativeModel(model_name=cfg.model_explain)

    def _image_payload(self, image_path: Optional[str]) -> Optional[Dict[str, Any]]:
        if not image_path or not os.path.exists(image_path):
            return None
        b64 = _b64encode_image(image_path)
        if not b64:
            return None
        return {
            "mime_type": _guess_mime(image_path),
            "data": base64.b64decode(b64),
        }

    def get_predicate_params(self, image_path: Optional[str], question: str, anchor_name: str) -> PredicateParams:
        """
        Returns camera-frame theta/phi/kappa (theta: 0 right, pi/2 forward, pi left, 3pi/2 back).
        We force JSON via instructions and response_mime_type.
        """
        prompt = f"""
You are a spatial predicate estimator.
Return ONLY valid JSON (no markdown, no extra text).

Task:
Given the image and query, output the direction from the REFERENCE OBJECT ({anchor_name}) to the TARGET LOCATION.

Camera-frame conventions:
- theta_cam: 0 rad = image right, 1.57 = forward into image, 3.14 = image left, 4.71 = toward camera
- phi_cam: 1.57 = level, 0 = above, 3.14 = below
- kappa: concentration, higher = sharper belief
- confidence: in [0,1]

Return JSON exactly with keys:
{{
  "reasoning": "...",
  "theta_cam": number,
  "phi_cam": number,
  "kappa": number,
  "confidence": number
}}

Query: {question}
Reference object: {anchor_name}
""".strip()

        parts: List[Any] = []
        img = self._image_payload(image_path)
        if img is not None:
            parts.append(img)
        parts.append(prompt)

        resp = self.model_pred.generate_content(
            parts,
            generation_config={
                "temperature": 0.2,
                "response_mime_type": "application/json",
            },
        )

        data = _extract_json(getattr(resp, "text", "") or "")
        # safe defaults
        theta_cam = float(data.get("theta_cam", 1.57))
        phi_cam = float(data.get("phi_cam", 1.57))
        kappa = float(data.get("kappa", 8.0))
        conf = float(data.get("confidence", 0.5))
        reasoning = str(data.get("reasoning", ""))

        conf = float(max(0.0, min(1.0, conf)))
        return PredicateParams(theta_cam=theta_cam, phi_cam=phi_cam, kappa=kappa, confidence=conf, reasoning=reasoning)

    def explain_where(
        self,
        image_path: Optional[str],
        question: str,
        region_summary: Dict[str, Any],
        nearby_objects: List[Dict[str, Any]],
    ) -> str:
        prompt = f"""
Return ONLY valid JSON (no markdown, no extra text).

You are describing a computed spatial probability region in a 3D indoor scene.

User question: {question}

Computed region summary (hab coords):
centroid_xyz: {region_summary.get("centroid_xyz")}
spread_xyz:   {region_summary.get("spread_xyz")}
confidence:   {region_summary.get("confidence")}

Nearby objects (use these names as landmarks if helpful):
{nearby_objects}

Output JSON:
{{
  "description": "1-3 sentence concrete description",
  "grounding_relations": ["relation1", "relation2"]
}}
""".strip()

        parts: List[Any] = []
        img = self._image_payload(image_path)
        if img is not None:
            parts.append(img)
        parts.append(prompt)

        resp = self.model_exp.generate_content(
            parts,
            generation_config={
                "temperature": 0.3,
                "response_mime_type": "application/json",
            },
        )
        data = _extract_json(getattr(resp, "text", "") or "")
        return str(data.get("description", "")).strip()

    def explain_which(
        self,
        image_path: Optional[str],
        question: str,
        top_rows: List[Dict[str, Any]],
    ) -> str:
        prompt = f"""
Return ONLY valid JSON (no markdown, no extra text).

User question: {question}

We scored objects by a spatial probability model and got top candidates:
{[{ "id": r.get("candidate_id"), "name": r.get("name"), "logp": r.get("logp_total") } for r in top_rows]}

Output JSON:
{{
  "explanation": "1-3 sentence explanation of which object is most likely and why"
}}
""".strip()

        parts: List[Any] = []
        img = self._image_payload(image_path)
        if img is not None:
            parts.append(img)
        parts.append(prompt)

        resp = self.model_exp.generate_content(
            parts,
            generation_config={
                "temperature": 0.3,
                "response_mime_type": "application/json",
            },
        )
        data = _extract_json(getattr(resp, "text", "") or "")
        return str(data.get("explanation", "")).strip()