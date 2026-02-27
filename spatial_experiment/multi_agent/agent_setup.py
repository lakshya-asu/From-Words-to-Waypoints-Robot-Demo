from spatial_experiment.multi_agent.agents.orchestrator_agent import OrchestratorAgent
from spatial_experiment.multi_agent.agents.grounding_agent import GroundingAgent
from spatial_experiment.multi_agent.agents.spatial_agent import SpatialAgent
from spatial_experiment.multi_agent.agents.verifier_agent import VerifierAgent
from spatial_experiment.multi_agent.agents.logical_agent import LogicalAgent
from spatial_experiment.multi_agent.agents.qa_agent import QaAgent

class AgentFactory:
    @staticmethod
    def create_agent(role: str, provider: str = "gemini", **kwargs):
        role = role.lower()
        provider = provider.lower()
        
        if role == "orchestrator":
            if provider == "openai":
                from spatial_experiment.multi_agent.agents.openai_orchestrator_agent import OpenAIOrchestratorAgent
                return OpenAIOrchestratorAgent(**kwargs)
            return OrchestratorAgent(**kwargs)
            
        elif role == "logical":
            if provider == "openai":
                from spatial_experiment.multi_agent.agents.openai_logical_agent import OpenAILogicalAgent
                return OpenAILogicalAgent(**kwargs)
            return LogicalAgent(**kwargs)
            
        elif role == "grounding":
            return GroundingAgent(**kwargs)
            
        elif role == "spatial":
            return SpatialAgent(**kwargs)
            
        elif role == "verifier":
            return VerifierAgent(**kwargs)
            
        elif role == "qa":
            if provider == "openai":
                from spatial_experiment.multi_agent.agents.openai_qa_agent import OpenAIQaAgent
                return OpenAIQaAgent(**kwargs)
            return QaAgent(**kwargs)
            
        else:
            raise ValueError(f"Unknown agent role: {role}")
