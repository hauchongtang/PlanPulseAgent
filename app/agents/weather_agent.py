"""
Weather Agent - Specialized agent for handling weather-related tasks.
This agent focuses on weather forecasts, current conditions, and weather planning.
"""

import json
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from app.api.settings import get_api_key
from app.tools.weather.get_weather_by_date import get_weather_by_date, get_uv


class WeatherAgent:
    """
    Specialized agent for weather operations and forecasts.
    Handles tasks related to:
    - Current weather conditions
    - Weather forecasts
    - Weather planning and advice
    - Date-specific weather queries
    """
    
    def __init__(self):
        """Initialize the Weather agent with weather tools."""
        self.name = "weather_agent"
        self.description = (
            "I specialize in weather information and forecasts. "
            "I can provide current weather conditions, forecasts for specific dates, "
            "and weather-related planning advice for Singapore."
        )
        self.tools = [get_weather_by_date, get_uv]
        self._agent = None
        
        # Define weather-related keywords for routing
        self.weather_keywords = [
            "weather", "forecast", "temperature", "rain", "sunny", "cloudy", 
            "storm", "wind", "humidity", "precipitation", "climate", "hot", "cold",
            "raining", "drizzle", "thunderstorm", "haze", "clear", "overcast",
            "weather today", "weather tomorrow", "weather forecast", "will it rain",
            "temperature today", "how hot", "how cold", "weather conditions", "uv", "uv index"
        ]
        
        # Time-related keywords that often pair with weather
        self.time_weather_keywords = [
            "today", "tomorrow", "this week", "next week", "this weekend",
            "tonight", "morning", "afternoon", "evening", "later"
        ]
        
        # Weather planning keywords
        self.planning_keywords = [
            "umbrella", "jacket", "outdoor", "picnic", "beach", "sports",
            "event", "outing", "trip", "vacation", "holiday"
        ]
    
    def _create_model(self) -> ChatGoogleGenerativeAI:
        """Create a specialized model for weather operations."""
        api_key = get_api_key()
        if not api_key:
            raise ValueError("Google API key is not configured")
        
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.2,  # Low temperature for consistent weather information
            max_retries=2,
            google_api_key=api_key,
            convert_system_message_to_human=True
        )
    
    def _get_agent(self):
        """Get or create the Weather agent executor."""
        if self._agent is None:
            model = self._create_model()
            self._agent = create_react_agent(model, self.tools)
        return self._agent
    
    def can_handle(self, task: str) -> float:
        """
        Determine if this agent can handle the given task.
        
        Args:
            task: The task description
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        task_lower = task.lower()
        confidence = 0.1  # Base confidence
        
        # Early filter: if task contains obvious non-weather keywords, reduce confidence
        non_weather_indicators = [
            "calculate", "math", "equation", "calendar", "event", "meeting",
            "schedule appointment", "what time is it", "current time", "clock",
            "transport", "bus", "mrt", "directions", "email", "message",
            "notion", "database"
        ]
        
        if any(indicator in task_lower for indicator in non_weather_indicators):
            # Still check for weather keywords, but with reduced confidence
            has_weather = any(keyword in task_lower for keyword in self.weather_keywords)
            if not has_weather:
                return 0.05  # Very low confidence for non-weather queries
        
        # Very high confidence (0.95+) for explicit weather queries
        high_confidence_phrases = [
            "weather forecast", "weather today", "weather tomorrow", 
            "will it rain", "temperature today", "weather conditions",
            "how hot", "how cold", "weather like", "forecast for"
        ]
        
        if any(phrase in task_lower for phrase in high_confidence_phrases):
            confidence = max(confidence, 0.95)
        
        # High confidence (0.9) for weather-specific words
        direct_weather_words = [
            "weather", "forecast", "rain", "temperature", "sunny", "cloudy",
            "storm", "thunderstorm", "drizzle", "haze", "humidity" "raining"
        ]
        
        weather_word_matches = sum(1 for word in direct_weather_words if word in task_lower)
        if weather_word_matches >= 2:
            confidence = max(confidence, 0.9)
        elif weather_word_matches == 1:
            confidence = max(confidence, 0.8)
        
        # Medium-high confidence for time + weather combinations
        time_matches = sum(1 for keyword in self.time_weather_keywords if keyword in task_lower)
        if time_matches > 0 and any(keyword in task_lower for keyword in self.weather_keywords):
            confidence = max(confidence, 0.7)
        
        # Medium confidence for weather planning queries
        planning_matches = sum(1 for keyword in self.planning_keywords if keyword in task_lower)
        if planning_matches > 0:
            confidence = max(confidence, 0.6)
        
        # Check for general weather keywords
        keyword_matches = sum(1 for keyword in self.weather_keywords if keyword in task_lower)
        if keyword_matches >= 3:
            confidence = max(confidence, 0.8)
        elif keyword_matches >= 2:
            confidence = max(confidence, 0.7)
        elif keyword_matches == 1:
            confidence = max(confidence, 0.5)
        
        # Lower confidence for very generic queries without weather context
        generic_phrases = ["what", "how", "when", "where", "today", "tomorrow"]
        if (len(task_lower.split()) <= 3 and 
            any(phrase in task_lower for phrase in generic_phrases) and
            not any(keyword in task_lower for keyword in self.weather_keywords)):
            confidence = max(confidence, 0.1)  # Keep at base level
        
        return round(confidence, 2)
    
    def get_confidence_explanation(self, task: str) -> Dict[str, Any]:
        """
        Get detailed explanation of confidence scoring for debugging.
        
        Args:
            task: The task description
            
        Returns:
            Dict containing confidence details and reasoning
        """
        task_lower = task.lower()
        explanations = []
        confidence = 0.1
        
        # Check high confidence phrases
        high_confidence_phrases = [
            "weather forecast", "weather today", "weather tomorrow", 
            "will it rain", "temperature today", "weather conditions",
            "how hot", "how cold", "weather like", "forecast for"
        ]
        high_matches = [phrase for phrase in high_confidence_phrases if phrase in task_lower]
        if high_matches:
            confidence = max(confidence, 0.95)
            explanations.append(f"High confidence weather phrases (0.95): {high_matches}")
        
        # Check direct weather words
        direct_weather_words = [
            "weather", "forecast", "rain", "temperature", "sunny", "cloudy",
            "storm", "thunderstorm", "drizzle", "haze", "humidity"
        ]
        weather_matches = [word for word in direct_weather_words if word in task_lower]
        if len(weather_matches) >= 2:
            confidence = max(confidence, 0.9)
            explanations.append(f"Multiple weather words (0.9): {weather_matches}")
        elif len(weather_matches) == 1:
            confidence = max(confidence, 0.8)
            explanations.append(f"Single weather word (0.8): {weather_matches}")
        
        # Check time + weather combinations
        time_matches = [kw for kw in self.time_weather_keywords if kw in task_lower]
        if time_matches and any(keyword in task_lower for keyword in self.weather_keywords):
            confidence = max(confidence, 0.7)
            explanations.append(f"Time + weather combination (0.7): {time_matches}")
        
        # Check planning keywords
        planning_matches = [kw for kw in self.planning_keywords if kw in task_lower]
        if planning_matches:
            confidence = max(confidence, 0.6)
            explanations.append(f"Weather planning keywords (0.6): {planning_matches}")
        
        # Check general weather keywords
        all_weather_matches = [kw for kw in self.weather_keywords if kw in task_lower]
        if len(all_weather_matches) >= 3:
            confidence = max(confidence, 0.8)
            explanations.append(f"Multiple weather keywords (0.8): {all_weather_matches}")
        elif len(all_weather_matches) >= 2:
            confidence = max(confidence, 0.7)
            explanations.append(f"Two weather keywords (0.7): {all_weather_matches}")
        elif len(all_weather_matches) == 1:
            confidence = max(confidence, 0.5)
            explanations.append(f"One weather keyword (0.5): {all_weather_matches}")
        
        return {
            "confidence": round(confidence, 2),
            "explanations": explanations,
            "task": task,
            "agent": self.name
        }
    
    def process_task(self, task: str) -> Dict[str, Any]:
        """
        Process a weather-related task.
        
        Args:
            task: The task to process
            
        Returns:
            Dict containing the response and metadata
        """
        try:
            agent = self._get_agent()
            
            # Add system instructions to the task
            system_instruction = (
                "You are a specialized weather assistant for Singapore. "
                "You excel at providing weather forecasts, current conditions, and weather planning advice. "
                "Always provide helpful, accurate weather information using the available tools. "
                
                "IMPORTANT DATE HANDLING: "
                "- When users ask about 'today', 'current weather', or don't specify a date, use today's date from the SYSTEM CONTEXT. "
                "- When users ask about 'tomorrow', calculate the next day from today's date. "
                "- When users ask about specific dates, use those dates. "
                "- When calling the get_weather_by_date tool, use the date in YYYY-MM-DD format. "
                "- Look for [SYSTEM CONTEXT] in the message for current date information. "
                
                "Provide practical advice based on weather conditions (e.g., umbrella recommendations for rain)."

                "If the user asks about UV index, use the get_uv tool to provide accurate UV information."

                "If the user asks about weather planning, provide advice based on current and forecasted conditions."

                "If the user asks whether it will rain (or other conditions), provide the likelihood of rain based on current conditions of get_weather_by_date tool."
            )
            
            enhanced_task = f"{system_instruction}\n\nUser request: {task}"
            input_message = {"role": "user", "content": enhanced_task}
            response = agent.invoke({"messages": [input_message]})
            
            # Extract the final message content
            final_message = response.get("messages", [])[-1]
            content = final_message.content if hasattr(final_message, 'content') else str(final_message)
            
            return {
                "success": True,
                "response": content,
                "agent": self.name,
                "task_type": "weather_query"
            }
            
        except Exception as e:
            return {
                "success": False,
                "response": f"I encountered an issue processing your weather request: {str(e)}",
                "error": str(e),
                "agent": self.name
            }
    
    def get_capabilities(self) -> List[str]:
        """Get list of weather agent capabilities."""
        return [
            "Get current weather conditions",
            "Provide weather forecasts for specific dates",
            "Weather planning and advice",
            "Temperature and precipitation information",
            "Weather condition descriptions",
            "Singapore-specific weather data"
        ]


# Available tools for external use
WEATHER_TOOLS = [
    get_weather_by_date
]
