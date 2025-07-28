import json
from langchain_core.tools import tool

from app.tools.transport.google_places import get_place_coordinates
from app.tools.transport.lta_datamall import get_bus_stops_with_arrivals


@tool("find_bus_stops_near_location")
def find_bus_stops_near_location(location_query: str, max_distance: int = 500) -> str:
    """
    Find bus stops near a specified location using Google Places API and LTA DataMall.
    This is the main transport tool that combines place search with bus information.
    
    Args:
        location_query: Text description of the location (e.g., "Queensway Shopping Centre")
        max_distance: Maximum distance in meters to search for bus stops (default: 500m)
        
    Returns:
        JSON string with location details and nearby bus stops with arrival timings
    """
    try:
        # Step 1: Get coordinates for the location using Google Places
        coordinates_result = get_place_coordinates.invoke({"text_query": location_query})
        coordinates_data = json.loads(coordinates_result)
        
        if "error" in coordinates_data:
            return json.dumps({
                "error": f"Could not find location: {coordinates_data.get('error')}",
                "location_query": location_query
            })
        
        latitude = coordinates_data.get("latitude")
        longitude = coordinates_data.get("longitude")
        location_name = coordinates_data.get("name")
        location_address = coordinates_data.get("address")
        
        if not latitude or not longitude:
            return json.dumps({
                "error": "No coordinates found for the specified location",
                "location_query": location_query
            })
        
        # Step 2: Get nearby bus stops with arrival timings using LTA DataMall
        bus_stops_result = get_bus_stops_with_arrivals.invoke({
            "latitude": latitude, 
            "longitude": longitude, 
            "max_distance": max_distance
        })
        bus_stops_data = json.loads(bus_stops_result)
        if "error" in bus_stops_data:
            print("find_bus_stops_near_location -> Error in fetching busstop data")
            return json.dumps({
                "error": f"Could not fetch bus stops: {bus_stops_data.get('error')}",
                "location_query": location_query,
                "location_found": {
                    "name": location_name,
                    "address": location_address,
                    "latitude": latitude,
                    "longitude": longitude
                }
            })
        
        # Step 3: Combine results into a comprehensive response
        result = {
            "location_query": location_query,
            "location_found": {
                "name": location_name,
                "address": location_address,
                "latitude": latitude,
                "longitude": longitude
            },
            "search_parameters": {
                "max_distance_meters": max_distance
            },
            "bus_stops": bus_stops_data.get("bus_stops_with_arrivals", []),
            "summary": {
                "bus_stops_found": len(bus_stops_data.get("bus_stops_with_arrivals", [])),
                "search_successful": True
            }
        }
        
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({
            "error": f"Unexpected error in transport search: {str(e)}",
            "location_query": location_query
        })


@tool("get_transport_summary")
def get_transport_summary(location_query: str) -> str:
    """
    Get a human-readable summary of bus stops and services near a location.
    Perfect for agent responses to users.
    
    Args:
        location_query: Text description of the location
        
    Returns:
        Formatted string with transport information
    """
    try:
        # Get the detailed bus stop information
        detailed_result = find_bus_stops_near_location.invoke({"location_query": location_query})
        data = json.loads(detailed_result)
        
        if "error" in data:
            return f"❌ {data.get('error')}"
        
        location_info = data.get("location_found", {})
        bus_stops = data.get("bus_stops", [])
        
        if not bus_stops:
            return f"📍 **{location_info.get('name', location_query)}**\n{location_info.get('address', '')}\n\n🚌 No bus stops found within 500 meters."
        
        # Format the response
        response = f"📍 **{location_info.get('name', location_query)}**\n"
        response += f"{location_info.get('address', '')}\n\n"
        response += f"🚌 **{len(bus_stops)} bus stop(s) found nearby:**\n\n"
        
        for i, stop in enumerate(bus_stops[:5], 1):  # Limit to top 5 stops
            response += f"**{i}. {stop.get('description', 'Bus Stop')}**\n"
            response += f"📍 {stop.get('road_name', '')} (Code: {stop.get('bus_stop_code', 'N/A')})\n"
            response += f"📏 {stop.get('distance_meters', 0)}m away\n"
            
            # Add bus services and arrival times if available
            arrival_info = stop.get("arrival_info", {})
            services = arrival_info.get("services", [])
            
            if services:
                response += f"🚌 **{len(services)} Services Available:**\n"
                
                # Show first few services with arrival times
                for j, service in enumerate(services[:3], 1):
                    service_no = service.get("service_no", "")
                    next_bus = service.get("next_bus", {})
                    estimated_arrival = next_bus.get("estimated_arrival", "")
                    
                    if estimated_arrival:
                        # Parse arrival time for better display
                        try:
                            from datetime import datetime
                            arrival_time = datetime.fromisoformat(estimated_arrival.replace('Z', '+00:00'))
                            now = datetime.now(arrival_time.tzinfo)
                            minutes_away = int((arrival_time - now).total_seconds() / 60)
                            
                            if minutes_away <= 0:
                                time_display = "Arriving"
                            elif minutes_away == 1:
                                time_display = "1 min"
                            else:
                                time_display = f"{minutes_away} mins"
                        except:
                            time_display = "N/A"
                    else:
                        time_display = "N/A"
                    
                    response += f"  • **{service_no}**: {time_display}\n"
                
                # Show remaining service numbers if any
                if len(services) > 3:
                    remaining_services = [s.get("service_no", "") for s in services[3:]]
                    response += f"  • Other services: {', '.join(filter(None, remaining_services))}\n"
                
                response += f"\n💡 **Tip:** Ask \"bus arrival times at {stop.get('bus_stop_code')}\" for detailed timings\n"
            else:
                response += "🚌 No active services\n"
            
            response += "\n"
        
        if len(bus_stops) > 5:
            response += f"... and {len(bus_stops) - 5} more bus stops\n"
        
        return response
        
    except Exception as e:
        return f"❌ Error getting transport summary: {str(e)}"
