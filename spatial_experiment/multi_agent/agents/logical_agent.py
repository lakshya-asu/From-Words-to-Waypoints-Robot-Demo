import os
import json
import base64
import mimetypes
import google.generativeai as genai
from typing import Dict, Any

from ..blackboard import Blackboard

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

class LogicalAgent:
    def __init__(self, model_name="models/gemini-2.5-pro"):
        self.model = genai.GenerativeModel(model_name=model_name)
        self.schema = genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "reasoning": genai.protos.Schema(type=genai.protos.Type.STRING, description="Logical deduction steps to answer the QA question based on the visual evidence and semantic scene graph state."),
                "answer": genai.protos.Schema(type=genai.protos.Type.STRING, description="The final, definitive answer to the user's question. If multiple choice, clearly state the chosen option."),
                "confidence": genai.protos.Schema(type=genai.protos.Type.NUMBER, description="Confidence score between 0.0 and 1.0")
            },
            required=["reasoning", "answer", "confidence"],
        )

    def process(self, blackboard: Blackboard) -> Dict[str, Any]:
        has_image = blackboard.current_image_path and os.path.exists(blackboard.current_image_path)

        sys_prompt = """
        SYSTEM: You are a Logical Reasoner for a robotic spatial reasoning pipeline.
        YOUR GOAL: Answer a complex logical or multiple-choice query posed by the user based on the environment state and visual clues.
        
        CRITICAL RULES:
        1. Base your answer STRICTLY on the visual clues provided in the image (if available) and the listed semantic state of the room.
        2. If the user provided multiple choices (e.g., A, B, C), your 'answer' must explicitly select one.
        3. Do not output coordinate navigation instructions. Focus on directly answering the factual or logical question.
        4. If you lack enough information to be certain, state your best logical guess but lower your confidence score.
        5. Check GLOBAL FAILURE HISTORY. If your exact previous answer was rejected, deduce an alternative answer.
        """
        
        prompt = f"""
        {sys_prompt}
        
        Question: {blackboard.question}
        
        Agent Exact Position: {blackboard.agent_pose_hab}
        Agent Yaw (rad): {blackboard.agent_yaw_rad}
        
        Scene Graph Candidates (with exact positions):
        {json.dumps([{{'id': o['id'], 'name': o.get('name', ''), 'position': o.get('position')}} for o in blackboard.available_objects], indent=2)}
        
        Current Environment Semantic State:
        {blackboard.semantic_state}
        
        GLOBAL FAILURE HISTORY:
        {blackboard.global_history}
        """
        
        parts = [{"text": prompt}]
        if has_image:
            mime = mimetypes.guess_type(blackboard.current_image_path)[0] or "image/png"
            parts.append({"inline_data": {"mime_type": mime, "data": encode_image(blackboard.current_image_path)}})

        messages = [{"role": "user", "parts": parts}]
        
        try:
            resp = self.model.generate_content(
                messages,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json", 
                    temperature=0.2, 
                    response_schema=self.schema
                )
            )
            d = json.loads(resp.text)
            
            out = {
                "ok": True,
                "answer": d["answer"],
                "confidence": float(d["confidence"]),
                "reasoning": d["reasoning"]
            }
            blackboard.append_event("Logical", "AnswerQuestion", out, "PASS")
            return out
        except Exception as e:
            blackboard.append_event("Logical", "Error", str(e), "FAIL")
            return {"ok": False, "error": str(e)}
