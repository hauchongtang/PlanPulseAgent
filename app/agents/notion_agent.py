"""
Notion Agent - Specialized agent for handling Notion-related tasks.
This agent focuses on calendar events, database queries, and Notion integrations.
"""

from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import BaseTool

from app.api.settings import get_api_key
from app.tools.notion.getevents import get_events, get_events_formatted
from app.tools.notion.createevent import create_event, get_create_event_options
from app.tools.calendar.add import add


class NotionAgent:
    """
    Specialized agent for Notion and calendar operations.
    Handles tasks related to:
    - Retrieving events from Notion databases (JSON and formatted)
    - Creating new events in Notion
    - Calendar operations
    - Date-based queries
    - Schedule management
    """
    
    def __init__(self):
        """Initialize the Notion agent with calendar and Notion tools."""
        self.name = "notion_agent"
        self.description = (
            "I specialize in Notion database operations and calendar management. "
            "I can retrieve events, create new events, query schedules, and handle date-related tasks. "
            "I support both JSON and formatted text outputs for events."
        )
        self.tools = [get_events, get_events_formatted, create_event, get_create_event_options]
        self._agent = None
    
    def _create_model(self) -> ChatGoogleGenerativeAI:
        """Create a specialized model for Notion operations."""
        api_key = get_api_key()
        if not api_key:
            raise ValueError("Google API key is not configured")
        
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,  # Lower temperature for more consistent calendar operations
            max_retries=2,
            google_api_key=api_key,
            convert_system_message_to_human=True
        )
    
    def _get_agent(self):
        """Get or create the Notion agent executor."""
        if self._agent is None:
            model = self._create_model()
            self._agent = create_react_agent(model, self.tools)
        return self._agent
    
    def process_task(self, task: str) -> Dict[str, Any]:
        """
        Process a Notion-related task.
        
        Args:
            task: The task description
            
        Returns:
            Dict containing the agent's response and metadata
        """
        try:
            agent = self._get_agent()
            
            # Add system instructions to the task
            system_instruction = (
                "You are a specialized Notion assistant. "
                "You excel at retrieving events, managing schedules, and working with dates. "
                "Always provide clear, structured responses about events and dates. "
                "When working with dates, be precise and consider time zones if relevant."
                "Please ensure that the data displayed is beautifully formatted and easy to read on Telegram."
            )
            
            enhanced_task = f"{system_instruction}\n\nUser request: {task}"
            input_message = {"role": "user", "content": enhanced_task}
            response = agent.invoke({"messages": [input_message]})
            
            messages = response.get("messages", [])
            if messages:
                last_message = messages[-1]
                return {
                    "success": True,
                    "response": last_message.content if hasattr(last_message, 'content') else str(last_message),
                    "agent": self.name,
                    "tools_available": [tool.name for tool in self.tools],
                    "message_count": len(messages)
                }
            else:
                return {
                    "success": False,
                    "response": "No response generated",
                    "agent": self.name,
                    "error": "Empty response"
                }
                
        except Exception as e:
            return {
                "success": False,
                "response": f"Notion agent error: {str(e)}",
                "agent": self.name,
                "error": str(e)
            }
    
    def can_handle(self, task: str) -> float:
        """
        Determine if this agent can handle the given task.
        
        Args:
            task: The task description
            
        Returns:
            Confidence score (0.0 to 1.0) indicating ability to handle the task
        """
        task_lower = task.lower()
        
        # High confidence keywords
        high_confidence_keywords = [
            'notion', 'calendar', 'event', 'schedule', 'appointment',
            'meeting', 'date', 'database', 'query', 'retrieve', "free", "availability",
        ]
        
        # Medium confidence keywords
        medium_confidence_keywords = [
            'today', 'tomorrow', 'yesterday', 'next week', 'last week',
            'month', 'year', 'time', 'when', 'what\'s on'
        ]
        
        high_matches = sum(1 for keyword in high_confidence_keywords if keyword in task_lower)
        medium_matches = sum(1 for keyword in medium_confidence_keywords if keyword in task_lower)
        
        # Calculate confidence score
        if high_matches >= 2:
            return 0.9
        elif high_matches >= 1:
            return 0.8
        elif medium_matches >= 2:
            return 0.6
        elif medium_matches >= 1:
            return 0.4
        else:
            return 0.1
    
    def get_capabilities(self) -> List[str]:
        """Get a list of this agent's capabilities."""
        return [
            "Retrieve events from Notion databases",
            "Create new events in Notion",
            "Query calendar schedules",
            "Handle date-based operations",
            "Manage appointments and meetings",
            "Process date and time-related queries"
            "Handles queries related to whether user is free on a specific date"
        ]
