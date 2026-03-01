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

class QaAgent:
    def __init__(self, model_name="models/gemini-3-flash-preview"):
        self.model = genai.GenerativeModel(model_name=model_name)

    def process(self, blackboard: Blackboard) -> Dict[str, Any]:
        has_image = blackboard.current_image_path and os.path.exists(blackboard.current_image_path)
        
        frontier_ids = [str(f.get("id")) for f in blackboard.available_frontiers]
        if not frontier_ids:
            frontier_ids = ["NONE"]
            
        object_ids = [str(o.get("id")) for o in blackboard.available_objects]
        if not object_ids:
            object_ids = ["NONE"]

        all_ids = list(set(object_ids + frontier_ids + ["NONE"]))
        
        is_mcq = bool(getattr(blackboard, "choices", None))
        
        properties = {
            "reasoning": genai.protos.Schema(
                type=genai.protos.Type.STRING, 
                description="Break down the query to identify the required target object, map the answer choices to symbols, and deduce the logical next step."
            ),
            "action_type": genai.protos.Schema(
                type=genai.protos.Type.STRING,
                enum=["goto_object", "goto_frontier", "lookaround", "answer"],
                description="The action to take. 'goto_object' to go to a known object, 'goto_frontier' to explore for missing context, 'lookaround' to spin, 'answer' if the final answer is definitively known now."
            ),
            "chosen_id": genai.protos.Schema(
                type=genai.protos.Type.STRING, 
                description="The Node ID of the object or frontier to navigate to. Use 'NONE' if action_type is lookaround or answer.",
                enum=all_ids
            ),
            "confidence": genai.protos.Schema(
                type=genai.protos.Type.NUMBER,
                description="Confidence score between 0.0 and 1.0 of the chosen action or answer."
            )
        }
        
        if is_mcq:
            properties["answer"] = genai.protos.Schema(
                type=genai.protos.Type.STRING,
                enum=["A", "B", "C", "D", "NONE"],
                description="Pick exactly ONE option symbol (A, B, C, or D) from the choices provided when action_type is 'answer'. Never pick an option that does not exist. Use 'NONE' if no answer yet."
            )
        else:
            properties["answer"] = genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="The text answer to the question if action_type is 'answer'. Otherwise leave empty."
            )

        schema = genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties=properties,
            required=["reasoning", "action_type", "chosen_id", "confidence", "answer"]
        )

        sys_prompt = """
        SYSTEM: You are an End-to-End QA and Navigation Agent.
        YOUR GOAL: You are tasked with answering the user's explicit question based on visual input and the topological scene graph.
        
        When faced with a question (especially a multiple choice one):
        1. Parse the query to figure out what object or area is being referred to.
        2. Break down the answer choices into variables/symbols (A, B, C...).
        3. If you do not see the required object in the environment to confidently answer the question, output `action_type="goto_frontier"` and choose the most logical frontier ID to explore.
        4. If you see the object in the Scene Graph but need a better visual angle, output `action_type="goto_object"`.
        5. If you have enough visual and semantic context to answer the query definitively, output `action_type="answer"`, and provide EXACTLY the option symbol (A, B, C, or D) in the `answer` field. Never guess options that are not provided. Use 'NONE' for answer if not ready.
        6. Check the GLOBAL FAILURE HISTORY to avoid repeating mistakes or looping between the same frontiers.
        """
        
        prompt = f"""
        {sys_prompt}
        
        Current Question: {blackboard.question}
        Mode: {blackboard.mode}
        {"Choices: " + json.dumps(blackboard.choices) if getattr(blackboard, "choices", None) else ""}
        
        Agent Exact Position: {blackboard.agent_pose_hab}
        Agent Yaw (rad): {blackboard.agent_yaw_rad}
        
        Scene Graph Candidates (with exact positions):
        {json.dumps([{'id': o.get('id', ''), 'name': o.get('name', ''), 'position': o.get('position')} for o in blackboard.available_objects if isinstance(o, dict)], indent=2)}
        
        Available Frontiers:
        {json.dumps([{'id': f.get('id', ''), 'position': f.get('position')} for f in blackboard.available_frontiers if isinstance(f, dict)], indent=2)}
        
        Environment Scene Graph (Topological Layout):
        {blackboard.scene_graph_str}
        
        Current Environment Semantic State (Agent Room Node):
        {blackboard.agent_semantic_state}
        
        GLOBAL FAILURE HISTORY (VERIFIER FEEDBACK):
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
                    temperature=0.1,
                    response_schema=schema,
                ),
            )
            parsed = json.loads(resp.text)
            
            out = {
                "ok": True,
                "action_type": parsed.get("action_type", "lookaround"),
                "chosen_id": parsed.get("chosen_id", "NONE"),
                "answer": parsed.get("answer", ""),
                "confidence": float(parsed.get("confidence", 0.0)),
                "reasoning": parsed.get("reasoning", "")
            }
            blackboard.append_event("QA", out["action_type"], out, "PASS")
            return out
        except Exception as e:
            error_msg = f"Failed to run QA agent: {e}"
            blackboard.append_event("QA", "Error", error_msg, "FAIL")
            return {"ok": False, "error": error_msg}
