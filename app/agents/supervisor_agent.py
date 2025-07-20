"""
Supervisor Agent - Orchestrates multiple specialized agents using LangGraph's supervisor pattern.
This agent analyzes incoming tasks and delegates them to the most appropriate specialized agent.
"""

from typing import Dict, List, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
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
        """Initialize the supervisor with specialized agents and memory management."""
        self.name = "supervisor_agent"
        self.description = (
            "I am a supervisor agent that coordinates between specialized agents. "
            "I analyze your request and delegate it to the most appropriate agent."
        )
        
        # Memory management configuration
        self.max_messages_per_user = 10  # Maximum messages before summarization
        
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
        from typing import Annotated
        
        # Define the state schema for LangGraph - using simple dict for messages
        class WorkflowState(BaseModel):
            messages: List[Dict[str, Any]] = []
            task: str = ""
            selected_agent: Optional[str] = None
            agent_response: Optional[Dict[str, Any]] = None
            final_response: str = ""
            reasoning: str = ""
            confidence_scores: Dict[str, float] = {}
        
        workflow = StateGraph(WorkflowState)
        
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
    
    def _manage_conversation_memory(self, messages: List[Dict[str, Any]], user_id: str) -> List[Dict[str, Any]]:
        """
        Manage conversation memory with 10-message limit and summarization.

        Args:
            messages: Current conversation messages
            user_id: User identifier for logging/debugging
            
        Returns:
            Managed message list with maximum 10 messages
        """
        # If we're within the limit, return as-is
        if len(messages) <= self.max_messages_per_user:
            return messages
        
        # We need to summarize - take all but the last message (current user message)
        messages_to_summarize = messages[:-1]  # All except the current message
        current_message = messages[-1]  # The current user message
        
        # Create summary of the conversation
        summary = self._summarize_conversation(messages_to_summarize, user_id)
        
        # Create a summary message
        summary_message = {
            "role": "system", 
            "content": f"Previous conversation summary: {summary}",
            "type": "summary"
        }
        
        # Return: [summary_message, current_user_message]
        return [summary_message, current_message]
    
    def _summarize_conversation(self, messages: List[Dict[str, Any]], user_id: str) -> str:
        """
        Summarize a conversation using the supervisor model.
        
        Args:
            messages: Messages to summarize
            user_id: User identifier
            
        Returns:
            Conversation summary
        """
        try:
            # Create conversation text for summarization
            conversation_text = "\n".join([
                f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
                for msg in messages
                if msg.get('content') and msg.get('type') != 'summary'  # Skip previous summaries
            ])
            
            if not conversation_text.strip():
                return "No significant conversation history to summarize."
            
            summary_prompt = f"""
            Summarize this conversation concisely, focusing on:
            1. Key topics discussed
            2. Important calculations, results, or answers provided
            3. User preferences or context that should be remembered
            4. Any ongoing tasks or requests
            
            Conversation:
            {conversation_text}
            
            Provide a concise summary (max 150 words):
            """
            
            response = self.supervisor_model.invoke(summary_prompt)
            return response.content[:400]  # Limit summary length
            
        except Exception as e:
            return f"Conversation included topics discussed previously (summary unavailable: {str(e)})"
    
    def _analyze_task(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the incoming task and gather confidence scores from agents."""
        # Handle both dict and object state formats
        if hasattr(state, 'task'):
            task = state.task
        else:
            task = state.get("task", "")
            
        confidence_scores = {}
        
        # Get confidence scores from each agent
        for agent_name, agent in self.agents.items():
            confidence_scores[agent_name] = agent.can_handle(task)
        
        # Add reasoning about the task
        reasoning_prompt = f"""
        Analyze this task: "{task}"
        
        Agent confidence scores:
        {chr(10).join([f"- {name}: {score:.2f}" for name, score in confidence_scores.items()])}
        
        Provide a brief analysis of what type of task this is and which agent seems most suitable.
        """
        
        reasoning = ""
        try:
            # Use string prompt directly instead of HumanMessage for simple invoke
            response = self.supervisor_model.invoke(reasoning_prompt)
            reasoning = response.content
        except Exception as e:
            reasoning = f"Analysis error: {str(e)}"
        
        # Return updated state
        return {
            "confidence_scores": confidence_scores,
            "reasoning": reasoning
        }
    
    def _select_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Select the most appropriate agent based on confidence scores."""
        # Handle both dict and object state formats
        if hasattr(state, 'confidence_scores'):
            confidence_scores = state.confidence_scores
        else:
            confidence_scores = state.get("confidence_scores", {})
        
        if not confidence_scores:
            return {"selected_agent": "notion_agent"}  # Default fallback
        
        # Select agent with highest confidence score
        selected_agent = max(confidence_scores.items(), key=lambda x: x[1])[0]
        
        # If confidence is too low across all agents, use a general approach
        max_confidence = max(confidence_scores.values())
        reasoning_update = ""
        if max_confidence < 0.3:
            # Use the agent with slightly higher confidence but add a note
            reasoning_update = f"\n\nNote: Low confidence across all agents (max: {max_confidence:.2f}). Using {selected_agent} as best option."
        
        result = {"selected_agent": selected_agent}
        if reasoning_update:
            if hasattr(state, 'reasoning'):
                current_reasoning = state.reasoning
            else:
                current_reasoning = state.get("reasoning", "")
            result["reasoning"] = current_reasoning + reasoning_update
        
        return result
    
    def _execute_task(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the task using the selected agent with conversation context."""
        # Handle both dict and object state formats
        if hasattr(state, 'selected_agent'):
            selected_agent_name = state.selected_agent
            task = state.task
            messages = getattr(state, 'messages', [])
        else:
            selected_agent_name = state.get("selected_agent")
            task = state.get("task", "")
            messages = state.get("messages", [])
        
        if not selected_agent_name or selected_agent_name not in self.agents:
            return {
                "agent_response": {
                    "success": False,
                    "response": "No suitable agent found",
                    "error": "Agent selection failed"
                }
            }
        
        selected_agent = self.agents[selected_agent_name]
        
        try:
            # Build context from conversation history for the agent
            if len(messages) > 1:  # More than just the current message
                # Filter out the current message and any system summaries for context
                context_messages = [
                    msg for msg in messages[:-1] 
                    if msg.get('role') != 'system' or msg.get('type') != 'summary'
                ]
                
                # Include summary if present
                summary_messages = [
                    msg for msg in messages 
                    if msg.get('role') == 'system' and msg.get('type') == 'summary'
                ]
                
                context_parts = []
                
                # Add summary context if available
                if summary_messages:
                    latest_summary = summary_messages[-1]  # Get the most recent summary
                    context_parts.append(f"Context: {latest_summary.get('content', '')}")
                
                # Add recent conversation context (last 2 exchanges)
                if context_messages:
                    recent_context = context_messages[-2:]  # Last 2 non-summary messages
                    context_text = "\n".join([
                        f"Recent: {msg.get('content', str(msg))}" 
                        for msg in recent_context
                    ])
                    context_parts.append(context_text)
                
                if context_parts:
                    enhanced_task = f"{chr(10).join(context_parts)}\n\nCurrent request: {task}"
                else:
                    enhanced_task = task
            else:
                enhanced_task = task
            
            response = selected_agent.process_task(enhanced_task)
            return {"agent_response": response}
        except Exception as e:
            return {
                "agent_response": {
                    "success": False,
                    "response": f"Agent execution failed: {str(e)}",
                    "error": str(e),
                    "agent": selected_agent_name
                }
            }
    
    def _finalize_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize the response with supervisor context and update conversation history."""
        # Handle both dict and object state formats
        if hasattr(state, 'agent_response'):
            agent_response = state.agent_response
            messages = getattr(state, 'messages', [])
        else:
            agent_response = state.get("agent_response", {})
            messages = state.get("messages", [])
        
        if not agent_response:
            return {"final_response": "No response generated"}
        
        agent_response_text = agent_response.get("response", "")
        
        # Create a comprehensive response
        if agent_response.get("success", False):
            final_response = f"{agent_response_text}"
        else:
            final_response = f"I encountered an issue: {agent_response_text}"
        
        # Add assistant response to conversation history
        assistant_message = {"role": "assistant", "content": final_response}
        updated_messages = messages + [assistant_message]
        
        return {
            "final_response": final_response,
            "messages": updated_messages
        }
    
    def process_message(self, message: str, user_id: str = None) -> Dict[str, Any]:
        """
        Process a message through the supervisor workflow with user-specific memory.
        
        Args:
            message: The user's input message
            user_id: Unique identifier for the user (for memory isolation)
            
        Returns:
            Dict containing the supervisor's orchestrated response
        """
        try:
            # Use user-specific thread ID for memory isolation
            thread_id = f"user_{user_id}" if user_id else "default"
            config = {"configurable": {"thread_id": thread_id}}
            
            # Get current conversation state (if any) from memory
            try:
                current_state = self.workflow.get_state(config)
                existing_messages = current_state.values.get("messages", []) if current_state.values else []
            except:
                existing_messages = []
            
            # Create initial state with conversation history and memory management
            new_message = {"role": "user", "content": message}
            all_messages = existing_messages + [new_message]
            
            # Apply memory management (summarize if over 5 messages)
            managed_messages = self._manage_conversation_memory(all_messages, user_id or "default")
            
            initial_state = {
                "task": message,
                "messages": managed_messages,
                "selected_agent": None,
                "agent_response": None,
                "final_response": "",
                "reasoning": "",
                "confidence_scores": {}
            }
            
            # Execute workflow with user-specific thread ID for memory isolation
            final_state = self.workflow.invoke(initial_state, config=config)
            
            # Extract results from final state - handle both dict and object formats
            if hasattr(final_state, 'final_response'):
                response = final_state.final_response
                selected_agent = final_state.selected_agent
                confidence_scores = final_state.confidence_scores
                reasoning = final_state.reasoning
                agent_response = final_state.agent_response
            else:
                response = final_state.get("final_response", "No response generated")
                selected_agent = final_state.get("selected_agent")
                confidence_scores = final_state.get("confidence_scores", {})
                reasoning = final_state.get("reasoning", "")
                agent_response = final_state.get("agent_response", {})
            
            return {
                "success": True,
                "response": response,
                "supervisor_metadata": {
                    "selected_agent": selected_agent,
                    "confidence_scores": confidence_scores,
                    "reasoning": reasoning,
                    "agent_response": agent_response
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
