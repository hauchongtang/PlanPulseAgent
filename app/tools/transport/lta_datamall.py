import json
import requests
import math
from langchain_core.tools import tool
from typing import List, Dict

from app.api.settings import get_lta_datamall_key


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on earth in meters.
    """
    R = 6371000  # Earth's radius in meters
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = (math.sin(delta_lat / 2) * math.sin(delta_lat / 2) + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * 
         math.sin(delta_lon / 2) * math.sin(delta_lon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

def get_nearby_bus_stops_fn(latitude: float, longitude: float, max_distance: float = 0.5, limit: int = 10):
    try:
        api_key = get_lta_datamall_key()
        if not api_key:
            return json.dumps({"error": "LTA DataMall API key not configured"})
        
        page = 1
        pageSize = 500
        bus_stops = []
        url = "https://datamall2.mytransport.sg/ltaodataservice/BusStops"
        headers = {
            "AccountKey": api_key,
            "Accept": "application/json"
        }

        while (True):
            response = requests.get(f"{url}?$skip={page*pageSize}", headers=headers)
            response.raise_for_status()
            data = response.json()
            stops = data.get("value", [])
            if (stops is None or len(stops) == 0):
                break
            bus_stops.extend(stops)
            page += 1
        
        
        # Calculate distances and filter nearby stops
        nearby_stops = []
        max_distance_meters = max_distance * 1000  # Convert km to meters for comparison
        for stop in bus_stops:
            stop_lat = float(stop.get("Latitude", 0))
            stop_lon = float(stop.get("Longitude", 0))
            
            distance_meters = calculate_distance(latitude, longitude, stop_lat, stop_lon)
            distance_km = distance_meters / 1000  # Convert to km for storage
            if distance_meters <= max_distance_meters:
                stop_info = {
                    "bus_stop_code": stop.get("BusStopCode"),
                    "road_name": stop.get("RoadName"),
                    "description": stop.get("Description"),
                    "latitude": stop_lat,
                    "longitude": stop_lon,
                    "distance_km": round(distance_km, 3)
                }
                nearby_stops.append(stop_info)
        
        # Sort by distance and limit results
        nearby_stops.sort(key=lambda x: x["distance_km"])
        nearby_stops = nearby_stops[:limit]
        
        result = {
            "search_location": {
                "latitude": latitude,
                "longitude": longitude
            },
            "max_distance_km": max_distance,
            "bus_stops_found": len(nearby_stops),
            "bus_stops": nearby_stops
        }
        
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({
            "error": f"Error fetching bus stops: {str(e)}",
            "search_location": {"latitude": latitude, "longitude": longitude}
        })

@tool("get_nearby_bus_stops")
def get_nearby_bus_stops(latitude: float, longitude: float, max_distance: float = 0.5, limit: int = 10) -> str:
    """
    Get nearby bus stops using LTA DataMall API.
    
    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        max_distance: Maximum distance in kilometers (default: 0.5km)
        limit: Maximum number of bus stops to return (default: 10)
        
    Returns:
        JSON string with nearby bus stops information
    """
    return get_nearby_bus_stops_fn(latitude, longitude, max_distance, limit)


def get_bus_arrival_timing_fn(bus_stop_code: str) -> str:
    try:
        api_key = get_lta_datamall_key()
        if not api_key:
            return json.dumps({"error": "LTA DataMall API key not configured"})
        
        url = f"https://datamall2.mytransport.sg/ltaodataservice/v3/BusArrival"
        headers = {
            "AccountKey": api_key,
            "Accept": "application/json"
        }
        params = {
            "BusStopCode": bus_stop_code
        }
        
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()

        # Format the arrival information according to LTA DataMall v3 structure
        services = data.get("Services", [])
        formatted_services = []
        
        for service in services:
            service_info = {
                "service_no": service.get("ServiceNo"),
                "operator": service.get("Operator"),
                "next_bus": {
                    "estimated_arrival": service.get("NextBus", {}).get("EstimatedArrival"),
                    "latitude": service.get("NextBus", {}).get("Latitude"),
                    "longitude": service.get("NextBus", {}).get("Longitude"),
                    "visit_number": service.get("NextBus", {}).get("VisitNumber"),
                    "load": service.get("NextBus", {}).get("Load"),
                    "feature": service.get("NextBus", {}).get("Feature"),
                    "type": service.get("NextBus", {}).get("Type")
                },
                "next_bus_2": {
                    "estimated_arrival": service.get("NextBus2", {}).get("EstimatedArrival"),
                    "latitude": service.get("NextBus2", {}).get("Latitude"),
                    "longitude": service.get("NextBus2", {}).get("Longitude"),
                    "visit_number": service.get("NextBus2", {}).get("VisitNumber"),
                    "load": service.get("NextBus2", {}).get("Load"),
                    "feature": service.get("NextBus2", {}).get("Feature"),
                    "type": service.get("NextBus2", {}).get("Type")
                },
                "next_bus_3": {
                    "estimated_arrival": service.get("NextBus3", {}).get("EstimatedArrival"),
                    "latitude": service.get("NextBus3", {}).get("Latitude"),
                    "longitude": service.get("NextBus3", {}).get("Longitude"),
                    "visit_number": service.get("NextBus3", {}).get("VisitNumber"),
                    "load": service.get("NextBus3", {}).get("Load"),
                    "feature": service.get("NextBus3", {}).get("Feature"),
                    "type": service.get("NextBus3", {}).get("Type")
                }
            }
            formatted_services.append(service_info)
        
        result = {
            "bus_stop_code": bus_stop_code,
            "services_count": len(formatted_services),
            "services": formatted_services
        }
        
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({
            "error": f"Error fetching bus arrivals: {str(e)}",
            "bus_stop_code": bus_stop_code
        })

@tool("get_bus_arrival_timing")
def get_bus_arrival_timing(bus_stop_code: str) -> str:
    """
    Get real-time bus arrival information for a specific bus stop.
    
    Args:
        bus_stop_code: The bus stop code to get arrival timings for
        
    Returns:
        JSON string with bus arrival timings
    """
    return get_bus_arrival_timing_fn(bus_stop_code)


@tool("get_bus_stops_with_arrivals")
def get_bus_stops_with_arrivals(latitude: float, longitude: float, max_distance: float = 0.5, limit: int = 5) -> str:
    """
    Get nearby bus stops with their arrival timings.
    This combines both location search and arrival timing in one call.
    
    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate  
        max_distance: Maximum distance in kilometers (default: 0.5km)
        limit: Maximum number of bus stops to return (default: 5)
        
    Returns:
        JSON string with nearby bus stops and their arrival timings
    """
    try:
        # First get nearby bus stops
        stops_result = get_nearby_bus_stops_fn(latitude, longitude, max_distance, limit)
        stops_data = json.loads(stops_result)
        
        if "error" in stops_data:
            return stops_result
        
        bus_stops = stops_data.get("bus_stops", [])
        
        # Get arrival timings for each bus stop
        stops_with_arrivals = []
        for stop in bus_stops:
            bus_stop_code = stop.get("bus_stop_code")
            
            # Get arrival timings
            arrivals_result = get_bus_arrival_timing_fn(bus_stop_code)
            arrivals_data = json.loads(arrivals_result)
            
            stop_with_arrivals = {
                **stop,  # Include all stop information
                "arrival_info": arrivals_data if "error" not in arrivals_data else {"error": arrivals_data.get("error")}
            }
            stops_with_arrivals.append(stop_with_arrivals)
        
        result = {
            "search_location": {
                "latitude": latitude,
                "longitude": longitude
            },
            "max_distance_km": max_distance,
            "bus_stops_with_arrivals": stops_with_arrivals
        }
        
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({
            "error": f"Error fetching bus stops with arrivals: {str(e)}",
            "search_location": {"latitude": latitude, "longitude": longitude}
        })
