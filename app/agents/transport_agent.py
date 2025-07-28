import json
from typing import List, Dict, Any
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from app.api.settings import get_api_key
from app.tools.transport.transport_search import find_bus_stops_near_location, get_transport_summary
from app.tools.transport.google_places import search_places, get_place_coordinates
from app.tools.transport.lta_datamall import get_nearby_bus_stops, get_bus_arrival_timing


class TransportAgent:
    """
    Transport agent specializing in Singapore public transport information.
    Handles bus stop searches, location queries, and real-time arrival data.
    """
    
    def __init__(self):
        """Initialize the transport agent."""
        self.name = "transport_agent"
        self.description = (
            "I specialize in Singapore public transport information. "
            "I can find bus stops near locations, provide real-time arrival timings, "
            "and help with public transport planning using Google Places and LTA DataMall APIs."
        )
        self.tools = [
            get_transport_summary,
            get_bus_arrival_timing
        ]
        self._agent = None
        
        # Define transport-related keywords for routing (more specific)
        self.transport_keywords = [
            "bus", "transport", "mrt", "train", "station", "stop", "arrival", "timing",
            "how to get", "directions", "travel", "commute", "public transport",
            "bus stop", "nearby", "location", "bus stop code", "arrival time",
            "transit", "transportation", "subway", "rail", "busstop", "bus service",
        ]
        
        # Keywords for bus arrival queries (more specific)
        self.arrival_keywords = [
            "arrival time", "arrival times", "when is the next bus", "bus timing",
            "next bus", "bus arrival", "arrival at", "timing at", 
            "bus schedule", "bus timetable", "real time", "live timing",
            "when does the bus arrive", "bus departure"
        ]
        
        # Singapore-specific location keywords for higher confidence
        self.singapore_locations = [
            "singapore", "sg", "orchard", "marina bay", "changi", "jurong",
            "tampines", "woodlands", "yishun", "toa payoh", "ang mo kio",
            "bedok", "pasir ris", "hougang", "punggol", "sengkang",
            "bukit timah", "clementi", "queenstown", "bishan", "serangoon"
        ]
        
        # Transport-specific phrases that indicate high confidence
        self.transport_phrases = [
            "public transport", "bus service", "bus route", "bus number",
            "bus stop code", "lta", "sbs transit", "smrt", "go-ahead"
        ]
    
    def _create_model(self) -> ChatGoogleGenerativeAI:
        """Create a specialized model for transport operations."""
        api_key = get_api_key()
        if not api_key:
            raise ValueError("Google API key is not configured")
        
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.2,  # Low temperature for consistent transport information
            max_retries=2,
            google_api_key=api_key,
            convert_system_message_to_human=True
        )
    
    def _get_agent(self):
        """Get or create the Transport agent executor."""
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
        
        # Early filter: if task contains obvious non-transport keywords, reduce confidence
        non_transport_indicators = [
            "calculate", "math", "equation", "calendar", "event", "meeting",
            "schedule appointment", "what time is it", "current time", "clock",
            "weather", "temperature", "email", "message"
        ]
        
        if any(indicator in task_lower for indicator in non_transport_indicators):
            # Still check for transport keywords, but with reduced confidence
            has_transport = any(keyword in task_lower for keyword in self.transport_keywords)
            if not has_transport:
                return 0.05  # Very low confidence for non-transport queries
        
        # Very high confidence (0.95+) for bus arrival timing queries
        if any(keyword in task_lower for keyword in self.arrival_keywords):
            confidence = max(confidence, 0.95)
        
        # Very high confidence for Singapore transport-specific phrases
        if any(phrase in task_lower for phrase in self.transport_phrases):
            confidence = max(confidence, 0.95)
        
        # High confidence (0.9) for explicit transport queries
        high_confidence_phrases = [
            "bus stop", "bus arrival", "bus timing", "public transport",
            "how to get to", "directions to", "travel to", "commute to",
            "transport near", "bus near", "mrt near", "train to", "subway",
            "transit", "transportation"
        ]
        
        for phrase in high_confidence_phrases:
            if phrase in task_lower:
                confidence = max(confidence, 0.9)
        
        # Boost confidence for Singapore-specific locations
        singapore_boost = 0.0
        if any(location in task_lower for location in self.singapore_locations):
            singapore_boost = 0.2
        
        # Medium-high confidence (0.7-0.8) for transport-related keywords
        keyword_matches = sum(1 for keyword in self.transport_keywords if keyword in task_lower)
        if keyword_matches >= 3:
            confidence = max(confidence, 0.8)
        elif keyword_matches >= 2:
            confidence = max(confidence, 0.7)
        elif keyword_matches == 1:
            confidence = max(confidence, 0.5)
        
        # Apply Singapore location boost
        confidence = min(1.0, confidence + singapore_boost)
        
        # Medium confidence (0.6) for numbered routes/services
        import re
        if re.search(r'\b(?:service|bus|route)\s*\d+\b', task_lower):
            confidence = max(confidence, 0.6)
        
        # Bus stop codes (4-5 digits) indicate transport query
        if re.search(r'\b\d{4,5}\b', task_lower):
            confidence = max(confidence, 0.8)
        
        # Medium confidence for location queries that might need transport
        location_indicators = ["near", "at", "around", "close to", "vicinity", "nearby"]
        if any(indicator in task_lower for indicator in location_indicators):
            # Only boost if there are also transport-related words
            has_transport_context = any(keyword in task_lower for keyword in self.transport_keywords[:8])  # Core transport words
            if has_transport_context:
                confidence = max(confidence, 0.4)
            else:
                confidence = max(confidence, 0.2)  # Lower boost for pure location queries
        
        # Lower confidence for very generic queries without transport context
        generic_phrases = ["where", "what", "how", "when", "find", "search"]
        if (len(task_lower.split()) <= 3 and 
            any(phrase in task_lower for phrase in generic_phrases) and
            not any(keyword in task_lower for keyword in self.transport_keywords[:8])):
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
        
        # Check arrival keywords
        arrival_matches = [kw for kw in self.arrival_keywords if kw in task_lower]
        if arrival_matches:
            confidence = max(confidence, 0.95)
            explanations.append(f"Bus arrival keywords found: {arrival_matches}")
        
        # Check transport phrases
        transport_matches = [phrase for phrase in self.transport_phrases if phrase in task_lower]
        if transport_matches:
            confidence = max(confidence, 0.95)
            explanations.append(f"Singapore transport phrases: {transport_matches}")
        
        # Check high confidence phrases
        high_confidence_phrases = [
            "bus stop", "bus arrival", "bus timing", "public transport",
            "how to get to", "directions to", "travel to", "commute to",
            "transport near", "bus near", "mrt near", "train to", "subway",
            "transit", "transportation"
        ]
        high_matches = [phrase for phrase in high_confidence_phrases if phrase in task_lower]
        if high_matches:
            confidence = max(confidence, 0.9)
            explanations.append(f"High confidence transport phrases: {high_matches}")
        
        # Check Singapore locations
        sg_matches = [loc for loc in self.singapore_locations if loc in task_lower]
        singapore_boost = 0.2 if sg_matches else 0.0
        if sg_matches:
            explanations.append(f"Singapore locations boost (+0.2): {sg_matches}")
        
        # Check keyword matches
        keyword_matches = [kw for kw in self.transport_keywords if kw in task_lower]
        if len(keyword_matches) >= 3:
            confidence = max(confidence, 0.8)
            explanations.append(f"Multiple transport keywords (0.8): {keyword_matches}")
        elif len(keyword_matches) >= 2:
            confidence = max(confidence, 0.7)
            explanations.append(f"Two transport keywords (0.7): {keyword_matches}")
        elif len(keyword_matches) == 1:
            confidence = max(confidence, 0.5)
            explanations.append(f"One transport keyword (0.5): {keyword_matches}")
        
        # Apply Singapore boost
        confidence = min(1.0, confidence + singapore_boost)
        
        # Check for numbered routes
        import re
        if re.search(r'\b(?:service|bus|route)\s*\d+\b', task_lower):
            confidence = max(confidence, 0.6)
            explanations.append("Bus service number detected (0.6)")
        
        # Check for bus stop codes
        if re.search(r'\b\d{4,5}\b', task_lower):
            confidence = max(confidence, 0.8)
            explanations.append("Bus stop code detected (0.8)")
        
        # Check location indicators
        location_indicators = ["near", "at", "around", "close to", "vicinity", "nearby"]
        location_matches = [ind for ind in location_indicators if ind in task_lower]
        if location_matches:
            confidence = max(confidence, 0.4)
            explanations.append(f"Location indicators (0.4): {location_matches}")
        
        return {
            "confidence": round(confidence, 2),
            "explanations": explanations,
            "task": task,
            "agent": self.name
        }
    
    def process_task(self, task: str) -> Dict[str, Any]:
        """
        Process a transport-related task.
        
        Args:
            task: The task to process
            
        Returns:
            Dict containing the response and metadata
        """
        try:
            agent = self._get_agent()
            
            # Add system instructions to the task
            system_instruction = (
                "You are a specialized Singapore public transport assistant. "
                "You excel at finding bus stops, providing real-time arrival times, and helping with transport planning. "
                "Always provide helpful, accurate information about Singapore public transport. "
                "Use the available tools to get real-time data when needed. "
                "Be friendly and provide practical travel advice."

                "**Search for bus stops near locations**: "
                "- If the user asks about bus arrival times at a location, use the search_places tool to get nearby coordinates."
                "- Then, use the get_transport_summary tool to find bus stops within 500 meters of those coordinates."
                "- Format the response with bus stop details including distance and services available as well as arrival times. Nearby bus stops should be sorted by distance."

                "Example Query:"
                "- What are the busstops near Pioneer Mall ?"
                "- What are the bus near Jurong East MRT ?"

                "If all else fails:"
                "Respond with: Please upload your desired location directly to the chat."
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
                "task_type": "transport_query"
            }
            
        except Exception as e:
            return {
                "success": False,
                "response": f"I encountered an issue processing your transport request: {str(e)}",
                "error": str(e),
                "agent": self.name
            }
    
    def _handle_bus_query(self, task: str) -> str:
        """Handle bus stop and timing related queries."""
        # Extract location from the query
        location = self._extract_location_from_query(task)
        if location:
            return get_transport_summary.invoke({"location_query": location})
        else:
            return "Please specify a location to find nearby bus stops. For example: 'bus stops near Orchard Road'"
    
    def _handle_arrival_query(self, task: str) -> str:
        """Handle bus arrival timing queries for specific bus stops."""
        # Try to extract bus stop code from the query
        bus_stop_code = self._extract_bus_stop_code(task)
        
        if bus_stop_code:
            try:
                from app.tools.transport.lta_datamall import get_bus_arrival_timing
                arrival_result = get_bus_arrival_timing.invoke({"bus_stop_code": bus_stop_code})
                arrival_data = json.loads(arrival_result)
                
                if "error" in arrival_data:
                    return f"❌ Could not get arrival times for bus stop {bus_stop_code}: {arrival_data.get('error')}"
                
                return self._format_arrival_times(arrival_data, bus_stop_code)
                
            except Exception as e:
                return f"❌ Error getting arrival times: {str(e)}"
        else:
            # Try to extract location and provide general guidance
            location = self._extract_location_from_query(task)
            if location:
                return f"I can help you find arrival times! First, let me find bus stops near {location}:\n\n" + get_transport_summary.invoke({"location_query": location})
            else:
                return """🚌 **Bus Arrival Times**

To get real-time arrival times, I need either:
• A bus stop code (e.g. "arrival times at 09047")
• A location (e.g. "arrival times near Orchard Road")

**Example queries:**
• "Bus arrival times at 09047"
• "When is the next bus at bus stop 11519"
• "Arrival times near Marina Bay Sands"
"""
    
    def _handle_directions_query(self, task: str) -> str:
        """Handle directions and travel queries."""
        location = self._extract_location_from_query(task)
        if location:
            summary = get_transport_summary.invoke({"location_query": location})
            return f"🚌 **Public Transport Options for {location}:**\n\n{summary}\n\n💡 **Tip:** Use these bus stops to plan your journey. Check arrival times in real-time for the best travel experience!"
        else:
            return "Please specify your destination. For example: 'How to get to Marina Bay Sands'"
    
    def _handle_location_query(self, task: str) -> str:
        """Handle location-based transport queries."""
        location = self._extract_location_from_query(task)
        if location:
            return get_transport_summary.invoke({"location_query": location})
        else:
            return "I can help you find transport options near any location in Singapore. Please specify the location you're interested in."
    
    def _handle_general_query(self, task: str) -> str:
        """Handle general transport queries."""
        # Try to extract a location anyway
        location = self._extract_location_from_query(task)
        if location:
            return get_transport_summary.invoke({"location_query": location})
        else:
            return """🚌 **Singapore Transport Assistant**

I can help you with:
• Finding bus stops near any location
• Real-time bus arrival timings
• Public transport directions
• Travel planning assistance

**Examples:**
• "Bus stops near Queensway Shopping Centre"
• "How to get to Marina Bay Sands"
• "Transport options around Orchard Road"

Just tell me where you want to go or what location you need transport information for!"""
    
    def _extract_location_from_query(self, query: str) -> str:
        """
        Extract location from a natural language query.
        
        Args:
            query: The user's query
            
        Returns:
            Extracted location string or empty string if not found
        """
        query_lower = query.lower()
        
        # Remove common transport-related phrases to isolate location
        remove_phrases = [
            "bus stops near", "bus stop near", "bus near", "transport near",
            "how to get to", "directions to", "travel to", "commute to",
            "bus stops around", "transport around", "near", "around",
            "bus timing at", "bus arrival at", "at", "in", "to"
        ]
        
        location = query
        for phrase in remove_phrases:
            location = location.lower().replace(phrase, "").strip()
        
        # Clean up the location string
        location = location.replace("?", "").replace(".", "").strip()
        
        # If we have something left that looks like a location, return it
        if len(location) > 2 and not location.isdigit():
            return location
        
        # Fallback: try to find location after common prepositions
        prepositions = ["near", "at", "around", "to", "in"]
        for prep in prepositions:
            if prep in query_lower:
                parts = query_lower.split(prep, 1)
                if len(parts) > 1:
                    potential_location = parts[1].strip().replace("?", "").replace(".", "")
                    if len(potential_location) > 2:
                        return potential_location
        
        return ""
    
    def _extract_bus_stop_code(self, query: str) -> str:
        """
        Extract bus stop code from a query.
        
        Args:
            query: The user's query
            
        Returns:
            Bus stop code if found, empty string otherwise
        """
        import re
        
        # Look for patterns like "09047", "11519", etc. (4-5 digit codes)
        code_patterns = [
            r'\b(\d{4,5})\b',  # 4-5 digit numbers
            r'code\s*[:\-]?\s*(\d{4,5})',  # "code: 09047" or "code 09047"
            r'stop\s*[:\-]?\s*(\d{4,5})',  # "stop: 09047" or "stop 09047"
        ]
        
        for pattern in code_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return ""
    
    def _format_arrival_times(self, arrival_data: dict, bus_stop_code: str) -> str:
        """
        Format bus arrival times into a user-friendly response.
        
        Args:
            arrival_data: Raw arrival data from LTA DataMall
            bus_stop_code: The bus stop code
            
        Returns:
            Formatted arrival times string
        """
        services = arrival_data.get("services", [])
        
        if not services:
            return f"🚌 **Bus Stop {bus_stop_code}**\n\nNo bus services currently available at this stop."
        
        response = f"🚌 **Bus Stop {bus_stop_code} - Live Arrival Times**\n\n"
        
        for service in services:
            service_no = service.get("service_no", "")
            operator = service.get("operator", "")
            
            response += f"**Service {service_no}** ({operator})\n"
            
            # Format next 3 buses
            buses = [
                ("Next bus", service.get("next_bus", {})),
                ("Following bus", service.get("next_bus_2", {})),
                ("After that", service.get("next_bus_3", {}))
            ]
            
            bus_times = []
            for label, bus_info in buses:
                if bus_info and bus_info.get("estimated_arrival"):
                    try:
                        from datetime import datetime
                        arrival_time = datetime.fromisoformat(bus_info.get("estimated_arrival").replace('Z', '+00:00'))
                        now = datetime.now(arrival_time.tzinfo)
                        minutes_away = int((arrival_time - now).total_seconds() / 60)
                        
                        if minutes_away <= 0:
                            time_display = "Arriving"
                        elif minutes_away == 1:
                            time_display = "1 min"
                        else:
                            time_display = f"{minutes_away} mins"
                        
                        # Add load info if available
                        load = bus_info.get("load", "")
                        if load:
                            load_emoji = {"SEA": "🟢", "SDA": "🟡", "LSD": "🔴"}.get(load, "")
                            time_display += f" {load_emoji}"
                        
                        bus_times.append(time_display)
                    except:
                        continue
            
            if bus_times:
                response += f"  🕐 {' • '.join(bus_times)}\n"
            else:
                response += f"  🕐 No timing available\n"
            
            response += "\n"
        
        response += "🟢 Seats Available • 🟡 Standing Available • 🔴 Limited Standing\n"
        response += f"\n💡 **Tip:** Ask about \"bus stops near [location]\" to find other nearby stops"
        
        return response
    
    def get_capabilities(self) -> List[str]:
        """Get list of transport agent capabilities."""
        return [
            "Find bus stops near locations",
            "Get real-time bus arrival timings",
            "Provide public transport directions",
            "Search Singapore locations using Google Places",
            "Access LTA DataMall for live transport data",
            "Calculate walking distances to bus stops",
            "Transport planning assistance"
        ]


# Main transport functions for external use
def search_bus_stops_near_location(location_query: str) -> str:
    """
    Search for bus stops near a specified location.
    This is the main function for transport queries.
    
    Args:
        location_query: Text description of the location (e.g., "Queensway Shopping Centre")
        
    Returns:
        Human-readable string with transport information
    """
    return get_transport_summary.invoke({"location_query": location_query})


def get_detailed_transport_info(location_query: str) -> str:
    """
    Get detailed transport information including coordinates and full bus data.
    
    Args:
        location_query: Text description of the location
        
    Returns:
        JSON string with detailed transport information
    """
    return find_bus_stops_near_location.invoke({"location_query": location_query})


# Legacy function for backward compatibility
def text_search(query: str) -> str:
    """
    Legacy function - now uses the new transport search system.
    
    Args:
        query: The search query string.
        
    Returns:
        JSON string with the search results.
    """
    try:
        # Use the new transport search for better results
        return find_bus_stops_near_location.invoke({"location_query": query})
    except Exception as e:
        return json.dumps({
            "error": f"Transport search failed: {str(e)}",
            "query": query
        })


# Available tools for external use
TRANSPORT_TOOLS = [
    find_bus_stops_near_location,
    get_transport_summary,
    search_places,
    get_place_coordinates,
    get_nearby_bus_stops,
    get_bus_arrival_timing
]