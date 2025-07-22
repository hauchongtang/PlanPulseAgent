import json
from typing import List, Dict, Any
from langchain_core.tools import tool

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
        
        # Define transport-related keywords for routing
        self.transport_keywords = [
            "bus", "transport", "mrt", "train", "station", "stop", "arrival", "timing",
            "how to get", "directions", "travel", "commute", "public transport",
            "bus stop", "near", "nearby", "location", "where is", "find",
            "bus stop code", "arrival time", "when is the next bus"
        ]
        
        # Keywords for bus arrival queries
        self.arrival_keywords = [
            "arrival time", "arrival times", "when is the next bus", "bus timing",
            "next bus", "bus arrival", "arrival at", "timing at"
        ]
    
    def can_handle(self, task: str) -> float:
        """
        Determine if this agent can handle the given task.
        
        Args:
            task: The task description
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        task_lower = task.lower()
        
        # Very high confidence for bus arrival timing queries
        if any(keyword in task_lower for keyword in self.arrival_keywords):
            return 0.95
        
        # High confidence for explicit transport queries
        high_confidence_phrases = [
            "bus stop", "bus arrival", "bus timing", "public transport",
            "how to get to", "directions to", "travel to", "commute to",
            "transport near", "bus near", "mrt near", "train to"
        ]
        
        for phrase in high_confidence_phrases:
            if phrase in task_lower:
                return 0.9
        
        # Medium confidence for transport-related keywords
        keyword_matches = sum(1 for keyword in self.transport_keywords if keyword in task_lower)
        if keyword_matches >= 2:
            return 0.7
        elif keyword_matches == 1:
            return 0.5
        
        # Low confidence for location queries that might need transport
        location_indicators = ["near", "at", "around", "close to", "vicinity"]
        if any(indicator in task_lower for indicator in location_indicators):
            return 0.3
        
        return 0.1
    
    def process_task(self, task: str) -> Dict[str, Any]:
        """
        Process a transport-related task.
        
        Args:
            task: The task to process
            
        Returns:
            Dict containing the response and metadata
        """
        try:
            task_lower = task.lower()
            
            # Check if this is a bus arrival timing query (follow-up)
            if any(keyword in task_lower for keyword in self.arrival_keywords):
                response = self._handle_arrival_query(task)
            # Determine the type of transport query
            elif any(phrase in task_lower for phrase in ["bus stop", "bus near", "bus timing"]):
                # This is a bus stop/timing query
                response = self._handle_bus_query(task)
            elif any(phrase in task_lower for phrase in ["how to get", "directions", "travel to", "commute"]):
                # This is a directions query
                response = self._handle_directions_query(task)
            elif "near" in task_lower or "around" in task_lower:
                # This is a location-based query
                response = self._handle_location_query(task)
            else:
                # General transport query
                response = self._handle_general_query(task)
            
            return {
                "success": True,
                "response": response,
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