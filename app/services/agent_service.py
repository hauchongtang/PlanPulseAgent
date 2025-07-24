"""
Agent orchestration service using supervisor pattern with specialized agents.
Provides separation of concerns by delegating tasks to appropriate specialized agents.
"""

from datetime import datetime
from typing import Dict, List, Any

import pytz
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
            
            # Get current Singapore time for timestamp context
            singapore_tz = pytz.timezone('Asia/Singapore')
            singapore_time = datetime.now(singapore_tz)
            timestamp = singapore_time.strftime("%Y-%m-%d %H:%M:%S %Z")
            
            # Create timestamp context that can be used by agents
            timestamp_context = {
                'singapore_time': timestamp,
                'singapore_date': singapore_time.strftime("%Y-%m-%d"),
                'singapore_datetime': singapore_time.strftime("%Y-%m-%d %H:%M:%S"),
                'singapore_iso': singapore_time.isoformat(),
                'year': singapore_time.year,
                'month': singapore_time.month,
                'day': singapore_time.day,
                'hour': singapore_time.hour,
                'minute': singapore_time.minute,
                'weekday': singapore_time.strftime("%A"),
                'note': 'Current Singapore time for reference'
            }
            
            # Process the message with timestamp context
            result = supervisor.process_message(message, user_id=user_id, timestamp_context=timestamp_context)
            
            # Add timestamp context to the result for client reference
            result['timestamp_context'] = timestamp_context
            
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


def get_agent_service() -> AgentService:
    """
    Get a scoped instance of AgentService.
    
    This creates a new instance each time it's called, ensuring fresh state
    and avoiding singleton-related issues with code changes and state persistence.
    
    Returns:
        AgentService: A fresh instance of the agent service
    """
    return AgentService()
