"""
Math Agent - Specialized agent for handling mathematical calculations and operations.
This agent focuses on calculations, unit conversions, and mathematical problem-solving.
"""

from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import BaseTool

from app.api.settings import get_api_key
from app.tools.math.calculations import calculate, convert_units, percentage_calculator


class MathAgent:
    """
    Specialized agent for mathematical operations and calculations.
    Handles tasks related to:
    - Mathematical calculations and expressions
    - Unit conversions
    - Percentage calculations
    - Statistical operations
    - Mathematical problem-solving
    """
    
    def __init__(self):
        """Initialize the Math agent with calculation tools."""
        self.name = "math_agent"
        self.description = (
            "I specialize in mathematical calculations, unit conversions, and solving math problems. "
            "I can handle arithmetic, algebra, trigonometry, and various mathematical operations."
        )
        self.tools = [calculate, convert_units, percentage_calculator]
        self._agent = None
    
    def _create_model(self) -> ChatGoogleGenerativeAI:
        """Create a specialized model for mathematical operations."""
        api_key = get_api_key()
        if not api_key:
            raise ValueError("Google API key is not configured")
        
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.1,  # Very low temperature for precise calculations
            max_retries=2,
            google_api_key=api_key,
            convert_system_message_to_human=True
        )
    
    def _get_agent(self):
        """Get or create the Math agent executor."""
        if self._agent is None:
            model = self._create_model()
            self._agent = create_react_agent(model, self.tools)
        return self._agent
    
    def process_task(self, task: str) -> Dict[str, Any]:
        """
        Process a math-related task.
        
        Args:
            task: The task description
            
        Returns:
            Dict containing the agent's response and metadata
        """
        try:
            agent = self._get_agent()
            
            # Add system instructions to the task
            system_instruction = (
                "You are a specialized mathematics assistant. "
                "You excel at calculations, unit conversions, and solving mathematical problems. "
                "Always double-check your calculations and provide step-by-step explanations when helpful. "
                "Be precise with numbers and units. When using tools, explain what you're calculating."
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
                "response": f"Math agent error: {str(e)}",
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
            'calculate', 'math', 'equation', 'formula', 'solve',
            'convert', 'percentage', 'percent', 'arithmetic',
            'algebra', 'trigonometry', 'sum', 'multiply', 'divide'
        ]
        
        # Medium confidence keywords
        medium_confidence_keywords = [
            'number', 'numbers', 'add', 'subtract', 'plus', 'minus',
            'times', 'divided', 'square', 'root', 'power', 'units',
            'kg', 'lbs', 'miles', 'km', 'celsius', 'fahrenheit'
        ]
        
        # Mathematical symbols and patterns
        math_patterns = ['+', '-', '*', '/', '=', '%', '^', '(', ')', 'sqrt', 'sin', 'cos']
        
        high_matches = sum(1 for keyword in high_confidence_keywords if keyword in task_lower)
        medium_matches = sum(1 for keyword in medium_confidence_keywords if keyword in task_lower)
        pattern_matches = sum(1 for pattern in math_patterns if pattern in task_lower)
        
        # Calculate confidence score
        if high_matches >= 2 or pattern_matches >= 2:
            return 0.9
        elif high_matches >= 1 or pattern_matches >= 1:
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
            "Perform mathematical calculations",
            "Evaluate mathematical expressions",
            "Convert between units (weight, length, temperature)",
            "Calculate percentages and percentage changes",
            "Solve arithmetic and algebraic problems",
            "Handle trigonometric functions",
            "Process mathematical formulas"
        ]
