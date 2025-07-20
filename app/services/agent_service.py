"""
Agent orchestration service using supervisor pattern with specialized agents.
Provides separation of concerns by delegating tasks to appropriate specialized agents.
"""

from typing import Dict, List, Any
from app.agents.supervisor_agent import SupervisorAgent


class AgentService:
    """Service class for managing the supervisor agent and specialized sub-agents."""
    
    def __init__(self):
        """Initialize the agent service with supervisor pattern."""
        self._supervisor = None
    
    def _get_supervisor(self) -> SupervisorAgent:
        """Get or create the supervisor agent (lazy initialization)."""
        if self._supervisor is None:
            self._supervisor = SupervisorAgent()
        return self._supervisor
    
    def process_message(self, message: str, user_id: str = None) -> Dict[str, Any]:
        """
        Process a user message through the supervisor agent workflow with user-specific memory.
        
        Args:
            message: The user's input message
            user_id: Unique identifier for the user (for memory isolation)
            
        Returns:
            Dict containing the orchestrated response and metadata
        """
        try:
            supervisor = self._get_supervisor()
            result = supervisor.process_message(message, user_id=user_id)
            
            # Enhance the response with service-level metadata
            result["service_metadata"] = {
                "service_type": "supervisor_orchestrated",
                "available_agents": list(supervisor.agents.keys()),
                "workflow_version": "1.0"
            }
            
            return result
                
        except Exception as e:
            return {
                "success": False,
                "response": f"Agent service failed: {str(e)}",
                "error": str(e),
                "service_metadata": {
                    "service_type": "supervisor_orchestrated",
                    "error_location": "service_level"
                }
            }
    
    def get_available_agents(self) -> List[str]:
        """Get a list of available agent names."""
        try:
            supervisor = self._get_supervisor()
            return list(supervisor.agents.keys())
        except Exception:
            return []
    
    def get_agent_capabilities(self) -> Dict[str, List[str]]:
        """Get capabilities of all managed agents."""
        try:
            supervisor = self._get_supervisor()
            return supervisor.get_agent_capabilities()
        except Exception as e:
            return {"error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the agent service and all sub-agents."""
        try:
            supervisor = self._get_supervisor()
            health_status = supervisor.health_check()
            
            # Add service-level health information
            health_status["service_info"] = {
                "pattern": "supervisor",
                "agents_managed": len(supervisor.agents),
                "workflow_enabled": True
            }
            
            return health_status
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "service_info": {
                    "pattern": "supervisor",
                    "initialization_failed": True
                }
            }


# Global instance (singleton pattern)
agent_service = AgentService()
