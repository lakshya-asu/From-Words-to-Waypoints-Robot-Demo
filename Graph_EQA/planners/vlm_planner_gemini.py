import json
from enum import Enum
from typing import List, Tuple, Literal, Any, Union, Optional, Annotated
import time
import hydra_python as hydra
import base64

import google.generativeai as genai
import os
import mimetypes
from hydra_python.utils import get_instruction_from_eqa_data, get_latest_image
from pydantic import BaseModel

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Choose a Gemini model.
gemini_model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def create_planner_response(frontier_node_list, room_node_list, region_node_list, object_node_list, Answer_options):
    
    frontier_step = genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            'explanation_frontier': genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="Explain reasoning for choosing this frontier to explore by referencing list of objects (<id> and <name>) connected to that frontier node via a link (refer to scene graph)."
            ),
            'frontier_id': genai.protos.Schema(
                type=genai.protos.Type.STRING,
                enum=[member.value for member in frontier_node_list]
            ),
        },
        required=['explanation_frontier', 'frontier_id']
    )

    object_step = genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            'explanation_room': genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="Explain very briefly reasoning for selecting this room."
            ),
            'room_id': genai.protos.Schema(
                type=genai.protos.Type.STRING,
                enum=[member.name for member in room_node_list],
                description="Choose the room which contains the object you want to goto."
            ),
            'room_name': genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="Refer to the the scene graph to output the room_name corresponding to the selected room_id"         
            ),
            'explanation_obj': genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="Explain very briefly reasoning for selecting this object in the selected room."
            ),
            # 'explanation_region': genai.protos.Schema(
            #     type=genai.protos.Type.STRING,
            #     description="Explain very briefly reasoning for selecting this region."
            # ),
            
            # 'region_id': genai.protos.Schema(
            #     type=genai.protos.Type.STRING,
            #     enum=[member.name for member in region_node_list],
            #     description="Only select from region nodes connected to the room node (in the room)."
            # ),
            'object_id': genai.protos.Schema(
                type=genai.protos.Type.STRING,
                enum=[member.name for member in object_node_list],
                description="Only select from objects within the room chosen."
            ),
            'object_name': genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="Refer to the the scene graph to output the object_name corresponding to the selected object_id" 
            )
        },
        # required=['explanation_room', 'explanation_region', 'explanation_obj', 'room_id', 'region_id', 'object_id']
        required=['explanation_room', 'explanation_obj', 'room_id', 'room_name', 'object_id', 'object_name']

    )

    answer = genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            'explanation_ans': genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="Select the correct answer from the options."
            ),
            'answer': genai.protos.Schema(
                type=genai.protos.Type.STRING,
                enum=[member.name for member in Answer_options]
            ),
            'value': genai.protos.Schema(
                type=genai.protos.Type.STRING,
                enum=[member.value for member in Answer_options]
            ),
            'explanation_conf': genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="Explain the reasoning behind the confidence level of your answer."
            ),
            'confidence_level': genai.protos.Schema(
                type=genai.protos.Type.NUMBER,
                description="Rate your level of confidence. Provide a value between 0 and 1; 0 for not confident at all and 1 for absolutely certain that you can answer the question. This value represents your confidence in answering the question correctly, and not confidence pertaining to choosing the next actions."
            ),
            'is_confident': genai.protos.Schema(
                type=genai.protos.Type.BOOLEAN,
                description="Choose TRUE, if you are very confident about answering the question correctly.  Very IMPORTANT: Only answer TRUE when you have a visual confirmation (from the image) as well as from the scene graph that your answer is correct. Choose TRUE, if you have explored enough and are certain about answering the question correctly and no further exploration will help you answer the question better. Choose 'FALSE', if you are uncertain of the answer. Do not be overconfident. Clarification: This is not your confidence in choosing the next action, but your confidence in answering the question correctly."
            )
        },
        required=['explanation_ans', 'answer', 'value', 'explanation_conf', 'confidence_level', 'is_confident']
    )

    image_description = genai.protos.Schema(
        type = genai.protos.Type.STRING,
        description="Describe the CURRENT IMAGE. Pay special attention to features that can help answer the question or select future actions."
    )

    scene_graph_description = genai.protos.Schema(
        type = genai.protos.Type.STRING,
        description="Describe the SCENE GRAPH. Pay special attention to features that can help answer the question or select future actions."
    )

    question_type = genai.protos.Schema(
        type=genai.protos.Type.STRING,
        enum=["Identification", "Counting", "Existence", "State", "Location"],
        description="Use this to describe the type of question you are being asked."
    )

    step = genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            'Goto_frontier_node_step': frontier_step,  # Ensure these are Schema objects
            'Goto_object_node_step': object_step,
            'answer': answer,
        },
        description="Choose only one of 'Goto_frontier_node_step', 'Goto_object_node_step', or 'answer'."
    )

    steps = genai.protos.Schema(
        type = genai.protos.Type.ARRAY,
        items = step,
        min_items = 1)

    response_schema = genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties = {
            'steps': steps,
            'image_description': image_description,
            'scene_graph_description': scene_graph_description,
            'question_type': question_type
        },
        required=['steps', 'image_description', 'scene_graph_description', 'question_type']
    )

    return response_schema

class VLMPLannerEQAGemini:
    def __init__(self, cfg, sg_sim, question, pred_candidates, choices, answer, output_path):
        
        self._question, self.choices, self.vlm_pred_candidates = question, choices, pred_candidates
        self._answer = answer
        self._output_path = output_path
        self._vlm_type = cfg.name
        self._use_image = cfg.use_image

        self._example_plan = '' #TODO(saumya)
        self._history = ''
        self.full_plan = ''
        self._t = 0
        self._add_history = cfg.add_history

        self._outputs_to_save = [f'Question: {self._question}. \n Answer: {self._answer} \n']
        self.sg_sim = sg_sim
        self.temp = cfg.temp

    @property
    def t(self):
        return self._t
    
    def get_actions(self): 
        object_node_list = Enum('object_node_list', {id: name for id, name in zip(self.sg_sim.object_node_ids, self.sg_sim.object_node_names)}, type=str)
        
        if len(self.sg_sim.frontier_node_ids)> 0:
            frontier_node_list = Enum('frontier_node_list', {ac: ac for ac in self.sg_sim.frontier_node_ids}, type=str)
        else:
            frontier_node_list = Enum('frontier_node_list', {'frontier_0': 'Do not choose this option. No more frontiers left.'}, type=str)
        room_node_list = Enum('room_node_list', {id: name for id, name in zip(self.sg_sim.room_node_ids, self.sg_sim.room_node_names)}, type=str)
        region_node_list = Enum('region_node_list', {ac: ac for ac in self.sg_sim.region_node_ids}, type=str)
        Answer_options = Enum('Answer_options', {token: choice for token, choice in zip(self.vlm_pred_candidates, self.choices)}, type=str)
        return frontier_node_list, room_node_list, region_node_list, object_node_list, Answer_options
    
    # @property
    # def agent_role_prompt(self):
    #     scene_graph_desc = "A scene graph represents an indoor environment in a hierarchical tree structure consisting of nodes and edges/links. There are six types of nodes: building, rooms, visited areas, frontiers, objects, and agent in the environemnt. \n \
    #         The tree structure is as follows: At the highest level 5 is a 'building' node. \n \
    #         At level 4 are room nodes. There are links connecting the building node to each room node. \n \
    #         At the lower level 3, are region and frontier nodes. 'region' node represent region of room that is already explored. Frontier nodes represent areas that are at the boundary of visited and unexplored areas. There are links from room nodes to corresponding region and frontier nodes depicted which room they are located in. \n \
    #         At the lowest level 2 are object nodes and agent nodes. There is an edge from region node to each object node depicting which visited area of which room the object is located in. \
    #         There are also links between frontier nodes and objects nodes, depicting the objects in the vicinity of a frontier node. \n \
    #         Finally the agent node is where you are located in the environment. There is an edge between a region node and the agent node, depicting which visited area of which room the agent is located in."
    #     current_state_des = "'CURRENT STATE' will give you the exact location of the agent in the scene graph by giving you the agent node id, location, room_id and room name. Additionally, you will also be given the current view of the agent as an image. "
        
    #     prompt = f'''You are an excellent hierarchical graph planning agent. 
    #         Your goal is to navigate an unseen environment to confidently answer a multiple-choice question about the environment.
    #         As you explore the environment, your sensors are building a scene graph representation (in json format) and you have access to that scene graph.  
    #         {scene_graph_desc}. {current_state_des} 
            
    #         You also have to choose the next action, one which will enable you to answer the question better. 
    #         Goto_object_node_step: Navigates near a certain object in the scene graph. Choose this action to get a good view of the region aroung this object, if you think going near this object will help you answer the question better.
    #         Important to note, the scene contains incomplete information about the environment (objects may be missing, relationships might be unclear), so it is useful to go near relevant objects to get a better view to answer the question. 
    #         Use a scene graph as an imperfect guide to lead you to relevant regions to inspect.
    #         Choose the object in a hierarchical manner by first reasoning about which room you should goto to best answer the question, and then choose the specific object. \n
    #         Goto_frontier_node_step: First check if going to any object node is useful (Goto_object_node_step). If you think that using action "Goto_object_node_step" is not useful, in other words, if you think that none of the object nodes in the current scene graph will provide any useful information to answer the question better, then choose a frontier node to goto. This will expand the scene graph and give you access to unexplored new areas.
    #         This action will navigate you to a frontier (unexplored) region of the environment and will provide you information about new objects/rooms not yet in the scene graph. 
    #         Choose this frontier based on the objects connected this frontier, in other words, Goto the frontier near which you see objects that are useful for answering the question or seem useful as a good exploration direction. Explain reasoning for choosing this frontier, by listing the list of objects (<id> and <name>) connected to this frontier node via a link (refer to scene graph) \n \
            
    #         Given the current state information, if you confident, try to answer the question. Explain the reasoning for selecting the answer.
    #         Finally, report whether you are confident in answering the question. 
    #         Explain the reasoning behind the confidence level of your answer. Rate your level of confidence. 
    #         Provide a value between 0 and 1; 0 for not confident at all and 1 for absolutely certain.
    #         Do not use just commensense knowledge to decide confidence. 
    #         Choose TRUE, if you have explored enough and are certain about answering the question correctly and no further exploration will help you answer the question better. If you have only a partial view of something that is useful to answer the question or it is too far away in the image, take an action to go near that object rather than answering the question with any confidence.
    #         Choose 'FALSE', if you are uncertain of the answer and should explore more to ground your answer in the current envioronment. 
    #         Clarification: This is not your confidence in choosing the next action, but your confidence in answering the question correctly.
    #         If you are unable to answer the question with high confidence, and need more information to answer the question, then you can take two kinds of steps in the environment: Goto_object_node_step or Goto_frontier_node_step 
            
    #         While choosing either of the above actions, play close attention to 'HISTORY' especially the previous object nodes you have been to. 
    #         Avoid going to the same object nodes again and again.
    #         Describe the CURRENT IMAGE. Pay special attention to features that can help answer the question or select future actions.
    #         Describe the SCENE GRAPH. Pay special attention to features that can help answer the question or select future actions.

    #         QUESTION TYPES: You will be asked questions concerning five distinct categories: 1) Identification, 2) Counting, 3) Existence, 4) State, and 5) Location.
    #         You should first determine what type of question you are being asked.
    #         If you are asked a question regarding Identification, use both scene graph and image information to confirm you have the relevant pieces of information in the question to answer it correctly.
    #         If you are asked a question regarding Counting, take multiple exploratory actions to ensure you can correctly count the objects you are being asked about in the question. Additionally, for Counting questions, you should make sure the number of objects you are counting are all present in the scene graph before answering.
    #         If you are asked a question regarding Existence, be careful to sufficiently explore the environment you are in before concluding the existence of an object. If the question involves a type of room and there are multiple rooms of that type, e.g., bedrooms, explore each bedroom first before answering.
    #         If you are asked a question regarding State, always use both scene graph and image information to confirm and validate among the multiple choice answers before answering. If the object for which you are asked to determine its state is not in the scene graph, explore more.
    #         If you are asked a question regarding Location, take multiple exploratory actions to ensure you gather enough information from both the scene graph and image information to be confident in your answer.

    #         For each of the above question types, you should explore all areas relevant to the answer choices provided to you. Do not choose to answer the question without sufficiently exploring each of the areas relevant to the answer choices.
    #         '''
    #     return prompt

    @property
    def agent_role_prompt(self):
        scene_graph_desc = "A scene graph represents an indoor environment in a hierarchical tree structure consisting of nodes and edges/links. There are six types of nodes: building, rooms, visited areas, frontiers, objects, and agent in the environemnt. \n \
            The tree structure is as follows: At the highest level 5 is a 'building' node. \n \
            At level 4 are room nodes. There are links connecting the building node to each room node. \n \
            At the lower level 3, are region and frontier nodes. 'region' node represent region of room that is already explored. Frontier nodes represent areas that are at the boundary of visited and unexplored areas. There are links from room nodes to corresponding region and frontier nodes depicted which room they are located in. \n \
            At the lowest level 2 are object nodes and agent nodes. There is an edge from region node to each object node depicting which visited area of which room the object is located in. \
            There are also links between frontier nodes and objects nodes, depicting the objects in the vicinity of a frontier node. \n \
            Finally the agent node is where you are located in the environment. There is an edge between a region node and the agent node, depicting which visited area of which room the agent is located in."
        current_state_des = "'CURRENT STATE' will give you the exact location of the agent in the scene graph by giving you the agent node id, location, room_id and room name. Additionally, you will also be given the current view of the agent as an image. "
        
        prompt = f'''You are an excellent hierarchical graph planning agent. 
            Your goal is to navigate an unseen environment to confidently answer a multiple-choice question about the environment.
            As you explore the environment, your sensors are building a scene graph representation (in json format) and you have access to that scene graph.  
            {scene_graph_desc}. {current_state_des} 
            
            Nodes in the scene graph will give you information about the 'buildings', 'rooms', 'frontier' nodes and 'objects' in the scene.
            Edges in the scene graph tell you about connected components in the scenes: For example, Edge from a room node to object node will tell you which objects are in which room.
            Frontier nodes represent areas that are at the boundary of visited and unexplored empty areas. Edges from frontiers to objects denote which objects are close to that frontier node. Use this information to choose the next frontier to explore.
            You are required to report whether using the scene graph and your current state and image, you are able to answer the question 'CORRECTLY' with very high Confidence. If you think that exploring the scene more or going nearer to relevant objects will give you more information to better answer the question, you should do that.
            You are also required to provide a brief description of the current image 'image_description' you are given and explain if that image has any useful features that can help answer the question.
            You also have to choose the next action, one which will enable you to answer the question better. You can choose between two action types: Goto_frontier_node_step and Goto_object_node_step.
            Goto_frontier_node_step: Navigates to a frontier (unexplored) node and will provide you with a new observation/image and the scene graph will be augmented/updated. Use this to explore unseen areas to discover new areas/rooms/objects. Provide explanation for why you are choosing a specific frontier (e.g. relevant objects near it)
            Goto_object_node_step: Navigates to a certain seen object. This can be used if you think going nearer to that object or to the area around that object will help you answer the quesion better, since you will be given an image of that area in the next step. 
            This action will not provide much new information in the scene graph but will take you nearer to a seen location if you want to reexamine it.
            In summary, Use Goto_frontier_node_step to explore new areas, use Goto_object_node_step to revisit explored areas to get a better view, answer the question and mention if you are confident or not.

            You will also have access to your own HISTORY, which will describe where you have been, what actions you took, and the question type you are answering.
            '''
        return prompt

    def get_current_state_prompt(self, scene_graph, agent_state):
        # TODO(saumya): Include history
        prompt = f"At t = {self.t}: \n \
            CURRENT AGENT STATE: {agent_state}. \n \
            SCENE GRAPH: {scene_graph}. \n "

        if self._add_history:
            prompt += f"HISTORY: {self._history}"

        return prompt

    def update_history(self, agent_state, step, question_type):
        if (step['step_type'] == 'Goto_object_node_step'):
            action = f"Goto object_id: {step['choice']} object name: {step['value']}"
        elif step['step_type'] == 'Goto_frontier_node_step':
            action = f"Goto frontier_id: {step['choice']}"
        else:
            action = f"Answer: {step['choice']}: {step['value']}.  Confident: {step['is_confident']}, Confidence level:{step['confidence_level']}"
        
        last_step = f'''
            [Agent state(t={self.t}): {agent_state}, 
            Action(t={self.t}): {action},
            Question Type: {question_type} 
        '''
        self._history += last_step
    
    def get_gemini_output(self, current_state_prompt):
        # TODO(blake):
        messages=[
            {"role": "model", "parts": [{"text": f"AGENT ROLE: {self.agent_role_prompt}"}]},
            {"role": "model", "parts": [{"text": f"QUESTION: {self._question}"}]},
            {"role": "user", "parts": [{"text": f"CURRENT STATE: {current_state_prompt}."}]},
        ]
        
        if self._use_image:
            image_path = get_latest_image(self._output_path)
            base64_image = encode_image(image_path) 
            mime_type = mimetypes.guess_type(image_path)[0]
            messages.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": "CURRENT IMAGE: This image represents the current view of the agent. Use this as additional information to answer the question."
                        },
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": base64_image
                            }
                        }
                    ]
                }
            )

        frontier_node_list, room_node_list, region_node_list, object_node_list, Answer_options = self.get_actions()

        succ=False
        while not succ:
            try:
                start = time.time()
                response = gemini_model.generate_content(
                    messages,
                    generation_config=genai.GenerationConfig(
                    response_mime_type="application/json", 
                    response_schema=create_planner_response(
                        frontier_node_list, 
                        room_node_list, 
                        region_node_list, 
                        object_node_list, 
                        Answer_options)
                        )
                )

                print(f"Time taken for planning next step: {time.time()-start}s")
                if (True): # If the model refuses to respond, you will get a refusal message
                    succ=True
            except Exception as e:
                print(f"An error occurred: {e}. Sleeping for 45s")
                time.sleep(45)
        
        
        json_response = response.text
        response_dict = json.loads(json_response)
        step_out = response_dict["steps"][0]
        sg_desc = response_dict["scene_graph_description"]
        if self._use_image:
            img_desc = response_dict["image_description"]
        else:
            img_desc = ' '

        question_type = response_dict["question_type"]

        if step_out:
            step = {}
            step_type = list(step_out.keys())[0]
            step['step_type'] = step_type
            if (step_type == 'Goto_object_node_step'):
                step['choice'] = step_out[step_type]['object_id']
                step['value']= step_out[step_type]['object_name']
                step['explanation'] = step_out[step_type]['explanation_obj']
                step['room'] = step_out[step_type]['room_name']
                step['explanation_room'] = step_out[step_type]['explanation_room']
            elif (step_type == 'Goto_frontier_node_step'):
                step['choice'] = step_out[step_type]['frontier_id']
                step['explanation'] = step_out[step_type]['explanation_frontier']
            else:
                step['choice'] = step_out[step_type]['answer']
                step['value'] = step_out[step_type]['value']
                step['explanation'] = step_out[step_type]['explanation_ans']
                step['is_confident'] = step_out[step_type]['is_confident']
                step['confidence_level'] = step_out[step_type]['confidence_level']
        else:
            step = None
         
        return step, img_desc, sg_desc, question_type
    
        # import ipdb; ipdb.set_trace()
        # class GeminiStep:
        #     self.step_type = ""
        #     self.choice = ""
        #     self.value = ""
        #     self.explanation = ""
        #     self.room = "No room"
        #     self.explanation_room = "No room explanation"
        #     self.is_confident = False
        #     self.confidence_level = 0.0

        # class Answer:
        #     self.answer = ""
        #     self.is_confident = False
        #     self.explanation_ans = ""
        #     self.confidence_level = 0.0
        #     self.value = ""

        # answer = Answer()
        # step = GeminiStep()

        # step.is_confident = False
        # step.confidence_level = 0.0
        # step.room = "No room selected"
        # step.explanation_room = "No room explanation"
        # step.explanation = "No obj explanation"

        # sg_desc = response_dict["scene_graph_description"]
        
        # if self._use_image:
        #     img_desc = response_dict["image_description"]
        # else:
        #     img_desc = ' '

        # if step_out:
        #     step.step_type = next(iter(step_out))
        # else:
        #     step.step_type = 'Done'
        #     step.choice = 'Done_step'
        #     step.value = 'Done'
        #     step.explanation = ' '
        #     return step, img_desc, sg_desc

        # step = {}
        # if (step.step_type == 'Goto_object_node_step'):
        #     step.choice = step_out[step.step_type]['object_id']
        #     step.value = step_out[step.step_type]['object_name']
        #     step.explanation = step_out[step.step_type]['explanation_obj']
        #     step.room = step_out[step.step_type]['room_name']
        #     step.explanation_room = step_out[step.step_type]['explanation_room']
        # elif (step.step_type == 'Goto_frontier_node_step'):
        #     step.choice = step_out[step.step_type]['frontier_id']
        #     step.value = step_out[step.step_type]['frontier_id']
        #     step.explanation = step_out[step.step_type]['explanation_frontier']
        # else:
        #     step.choice = step_out[step.step_type]['answer']
        #     step.value = step_out[step.step_type]['value']
        #     step.explanation = step_out[step.step_type]['explanation_ans']
        #     step.is_confident = step_out[step.step_type]['is_confident']
        #     step.confidence_level = step_out[step.step_type]['confidence_level']

        # return step, img_desc, sg_desc
    

    def get_next_action(self):        
        agent_state = self.sg_sim.get_current_semantic_state_str()
        current_state_prompt = self.get_current_state_prompt(self.sg_sim.scene_graph_str, agent_state)

        sg_desc=''
        step, img_desc, sg_desc, question_type = self.get_gemini_output(current_state_prompt)

        print(f'At t={self._t}: \n Step: {step}, Question type: {question_type}')
        # Saving outputs to file
        self._outputs_to_save.append(f'At t={self._t}: \n \
                                        Agent state: {agent_state} \n \
                                        VLM step: {step} \n \
                                        Image desc: {img_desc} \n \
                                        Scene graph desc: {sg_desc} \n \
                                        Question type: {question_type} \n')
        self.full_plan = ' '.join(self._outputs_to_save)
        with open(self._output_path / "llm_outputs.json", "w") as text_file:
            text_file.write(self.full_plan)

        if step is None or step['choice'] == 'Do not choose this option. No more frontiers left.':
            return None, None, False, 0., " "
        

        if self._add_history:
            self.update_history(agent_state, step, question_type)

        self._t += 1

        if step['step_type'] == 'answer':
            return None, None, step['is_confident'], step['confidence_level'], step['choice']
        else:
            target_pose = self.sg_sim.get_position_from_id(step['choice'])
            target_id = step['choice']
            return target_pose, target_id, False, 0, " "