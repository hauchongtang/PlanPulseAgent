"""
Supervisor Agent - Orchestrates multiple specialized agents using LangGraph's supervisor pattern.
This agent analyzes incoming tasks and delegates them to the most appropriate specialized agent.
"""

from typing import Dict, List, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from app.api.settings import get_api_key
from app.agents.notion_agent import NotionAgent
from app.agents.math_agent import MathAgent


class SupervisorState(BaseModel):
    """State management for the supervisor workflow."""
    messages: List[Dict[str, Any]] = []
    task: str = ""
    selected_agent: Optional[str] = None
    agent_response: Optional[Dict[str, Any]] = None
    final_response: str = ""
    reasoning: str = ""
    confidence_scores: Dict[str, float] = {}


class SupervisorAgent:
    """
    Supervisor agent that coordinates multiple specialized agents.
    Uses LangGraph's state management to orchestrate agent selection and task delegation.
    """
    
    def __init__(self):
        """Initialize the supervisor with specialized agents."""
        self.name = "supervisor_agent"
        self.description = (
            "I am a supervisor agent that coordinates between specialized agents. "
            "I analyze your request and delegate it to the most appropriate agent."
        )
        
        # Initialize specialized agents
        self.agents = {
            "notion_agent": NotionAgent(),
            "math_agent": MathAgent()
        }
        
        self.supervisor_model = self._create_supervisor_model()
        self.workflow = self._create_workflow()
    
    def _create_supervisor_model(self) -> ChatGoogleGenerativeAI:
        """Create the supervisor's reasoning model."""
        api_key = get_api_key()
        if not api_key:
            raise ValueError("Google API key is not configured")
        
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2,  # Low temperature for consistent reasoning
            max_retries=2,
            google_api_key=api_key
        )
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for agent supervision."""
        workflow = StateGraph(SupervisorState)
        
        # Add nodes
        workflow.add_node("analyze_task", self._analyze_task)
        workflow.add_node("select_agent", self._select_agent)
        workflow.add_node("execute_task", self._execute_task)
        workflow.add_node("finalize_response", self._finalize_response)
        
        # Add edges
        workflow.set_entry_point("analyze_task")
        workflow.add_edge("analyze_task", "select_agent")
        workflow.add_edge("select_agent", "execute_task")
        workflow.add_edge("execute_task", "finalize_response")
        workflow.add_edge("finalize_response", END)
        
        # Compile with memory
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    def _analyze_task(self, state: SupervisorState) -> SupervisorState:
        """Analyze the incoming task and gather confidence scores from agents."""
        task = state.task
        confidence_scores = {}
        
        # Get confidence scores from each agent
        for agent_name, agent in self.agents.items():
            confidence_scores[agent_name] = agent.can_handle(task)
        
        state.confidence_scores = confidence_scores
        
        # Add reasoning about the task
        reasoning_prompt = f"""
        Analyze this task: "{task}"
        
        Agent confidence scores:
        {chr(10).join([f"- {name}: {score:.2f}" for name, score in confidence_scores.items()])}
        
        Provide a brief analysis of what type of task this is and which agent seems most suitable.
        """
        
        try:
            response = self.supervisor_model.invoke([HumanMessage(content=reasoning_prompt)])
            state.reasoning = response.content
        except Exception as e:
            state.reasoning = f"Analysis error: {str(e)}"
        
        return state
    
    def _select_agent(self, state: SupervisorState) -> SupervisorState:
        """Select the most appropriate agent based on confidence scores."""
        if not state.confidence_scores:
            state.selected_agent = "notion_agent"  # Default fallback
            return state
        
        # Select agent with highest confidence score
        selected_agent = max(state.confidence_scores.items(), key=lambda x: x[1])[0]
        
        # If confidence is too low across all agents, use a general approach
        max_confidence = max(state.confidence_scores.values())
        if max_confidence < 0.3:
            # Use the agent with slightly higher confidence but add a note
            state.selected_agent = selected_agent
            state.reasoning += f"\n\nNote: Low confidence across all agents (max: {max_confidence:.2f}). Using {selected_agent} as best option."
        else:
            state.selected_agent = selected_agent
        
        return state
    
    def _execute_task(self, state: SupervisorState) -> SupervisorState:
        """Execute the task using the selected agent."""
        if not state.selected_agent or state.selected_agent not in self.agents:
            state.agent_response = {
                "success": False,
                "response": "No suitable agent found",
                "error": "Agent selection failed"
            }
            return state
        
        selected_agent = self.agents[state.selected_agent]
        
        try:
            response = selected_agent.process_task(state.task)
            state.agent_response = response
        except Exception as e:
            state.agent_response = {
                "success": False,
                "response": f"Agent execution failed: {str(e)}",
                "error": str(e),
                "agent": state.selected_agent
            }
        
        return state
    
    def _finalize_response(self, state: SupervisorState) -> SupervisorState:
        """Finalize the response with supervisor context."""
        if not state.agent_response:
            state.final_response = "No response generated"
            return state
        
        agent_response = state.agent_response.get("response", "")
        selected_agent = state.selected_agent
        confidence = state.confidence_scores.get(selected_agent, 0.0)
        
        # Create a comprehensive response
        if state.agent_response.get("success", False):
            state.final_response = f"{agent_response}"
        else:
            state.final_response = f"I encountered an issue: {agent_response}"
        
        return state
    
    def process_message(self, message: str) -> Dict[str, Any]:
        """
        Process a message through the supervisor workflow.
        
        Args:
            message: The user's input message
            
        Returns:
            Dict containing the supervisor's orchestrated response
        """
        try:
            # Create initial state
            initial_state = SupervisorState(
                task=message,
                messages=[{"role": "user", "content": message}]
            )
            
            # Execute workflow
            config = {"configurable": {"thread_id": "default"}}
            final_state = self.workflow.invoke(initial_state, config=config)
            
            return {
                "success": True,
                "response": final_state.final_response,
                "supervisor_metadata": {
                    "selected_agent": final_state.selected_agent,
                    "confidence_scores": final_state.confidence_scores,
                    "reasoning": final_state.reasoning,
                    "agent_response": final_state.agent_response
                },
                "workflow_type": "supervisor"
            }
            
        except Exception as e:
            return {
                "success": False,
                "response": f"Supervisor workflow failed: {str(e)}",
                "error": str(e),
                "workflow_type": "supervisor"
            }
    
    def get_agent_capabilities(self) -> Dict[str, List[str]]:
        """Get capabilities of all managed agents."""
        capabilities = {}
        for agent_name, agent in self.agents.items():
            capabilities[agent_name] = agent.get_capabilities()
        return capabilities
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on supervisor and all agents."""
        try:
            api_key = get_api_key()
            agent_status = {}
            
            for agent_name, agent in self.agents.items():
                try:
                    # Simple capability check
                    capabilities = agent.get_capabilities()
                    agent_status[agent_name] = {
                        "status": "healthy",
                        "capabilities_count": len(capabilities)
                    }
                except Exception as e:
                    agent_status[agent_name] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }
            
            return {
                "status": "healthy",
                "supervisor_model": "gemini-2.5-flash",
                "api_key_configured": bool(api_key),
                "agents": agent_status,
                "workflow_compiled": self.workflow is not None
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
