import os
import json
from typing import Dict, Any, Literal
from pydantic import BaseModel, Field

from openai import OpenAI
from spatial_experiment.multi_agent.blackboard import Blackboard

class QaOutput(BaseModel):
    reasoning: str = Field(description="Break down the query to identify the required target object, map the answer choices to symbols, and deduce the logical next step.")
    action_type: Literal["goto_object", "goto_frontier", "lookaround", "answer"] = Field(description="The action to take. 'goto_object' to go to a known object, 'goto_frontier' to explore for missing context, 'lookaround' to spin, 'answer' if the final answer is definitively known now.")
    chosen_id: str = Field(description="The Node ID of the object or frontier to navigate to. Use 'NONE' if action_type is lookaround or answer.")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0 of the chosen action or answer.")
    answer: str = Field(description="If action_type is 'answer', provide EXACTLY the option symbol (e.g. A, B, C) from the choices provided. Otherwise leave empty.")

class OpenAIQaAgent:
    def __init__(self, model_name="gpt-4o"):
        if "OPENAI_API_KEY" not in os.environ:
            raise RuntimeError("OPENAI_API_KEY must be set in the environment.")
        self.model_name = model_name
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def process(self, blackboard: Blackboard) -> Dict[str, Any]:
        frontier_ids = [str(f.get("id")) for f in blackboard.available_frontiers]
        if not frontier_ids:
            frontier_ids = ["NONE"]
            
        object_ids = [str(o.get("id")) for o in blackboard.available_objects]
        if not object_ids:
            object_ids = ["NONE"]

        sys_prompt = """
        SYSTEM: You are an End-to-End QA and Navigation Agent.
        YOUR GOAL: You are tasked with answering the user's explicit question based on visual input and the topological scene graph.
        
        When faced with a question (especially a multiple choice one):
        1. Parse the query to figure out what object or area is being referred to.
        2. Break down the answer choices into variables/symbols (A, B, C...).
        3. If you do not see the required object in the environment to confidently answer the question, output `action_type="goto_frontier"` and choose the most logical frontier ID to explore.
        4. If you see the object in the Scene Graph but need a better visual angle, output `action_type="goto_object"`.
        5. If you have enough visual and semantic context to answer the query definitively, output `action_type="answer"`, and provide EXACTLY the option symbol (e.g., A, B, C) in the `answer` field.
        6. Check the GLOBAL FAILURE HISTORY to avoid repeating mistakes or looping between the same frontiers.
        """
        
        prompt = f"""
        Current Question: {blackboard.question}
        Mode: {blackboard.mode}
        {"Choices: " + json.dumps(blackboard.choices) if getattr(blackboard, "choices", None) else ""}
        
        Agent Exact Position: {blackboard.agent_pose_hab}
        Agent Yaw (rad): {blackboard.agent_yaw_rad}
        
        Scene Graph Candidates (with exact positions):
        {json.dumps([{{'id': o['id'], 'name': o.get('name', ''), 'position': o.get('position')}} for o in blackboard.available_objects], indent=2)}
        
        Available Frontiers:
        {json.dumps([{{'id': f['id'], 'position': f.get('position')}} for f in blackboard.available_frontiers], indent=2)}
        
        Environment Scene Graph (Topological Layout):
        {blackboard.scene_graph_str}
        
        Current Environment Semantic State (Agent Room Node):
        {blackboard.agent_semantic_state}
        
        GLOBAL FAILURE HISTORY (VERIFIER FEEDBACK):
        {blackboard.global_history}
        """
        
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ]
        
        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=messages,
                response_format=QaOutput,
                temperature=0.1
            )
            parsed = completion.choices[0].message.parsed.model_dump()
            
            out = {
                "ok": True,
                "action_type": parsed.get("action_type", "lookaround"),
                "chosen_id": parsed.get("chosen_id", "NONE"),
                "answer": parsed.get("answer", ""),
                "confidence": float(parsed.get("confidence", 0.0)),
                "reasoning": parsed.get("reasoning", "")
            }
            
            blackboard.append_event("QA(OpenAI)", out["action_type"], out, "PASS")
            return out
        except Exception as e:
            error_msg = f"Failed to infer MCQ QA (OpenAI): {e}"
            blackboard.append_event("QA(OpenAI)", "Error", error_msg, "FAIL")
            return {"ok": False, "error": error_msg}
