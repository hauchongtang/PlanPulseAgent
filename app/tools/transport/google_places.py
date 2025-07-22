import json
from langchain_core.tools import tool
from typing import Optional
from google.maps import places_v1

from app.api.settings import get_google_places_key


@tool("search_places")
def search_places(text_query: str, location_bias: Optional[str] = None) -> str:
    """
    Search for places using Google Places API with google-maps-places package.
    Uses free tier parameters to minimize costs.
    
    Args:
        text_query: The text query to search for (e.g., "Queensway Shopping Centre")
        location_bias: Optional location bias in format "lat,lng" to prioritize nearby results
        
    Returns:
        JSON string with place details including coordinates
    """
    try:
        api_key = get_google_places_key()
        if not api_key:
            return json.dumps({"error": "Google API key not configured"})
        
        # Create the Places client
        client = places_v1.PlacesClient(
            client_options={"api_key": api_key}
        )
        
        # Create the search request
        request = places_v1.SearchTextRequest(
            text_query=text_query,
            max_result_count=5,  # Limit results to minimize costs
        )
        
        # Add location bias if provided
        if location_bias:
            try:
                lat, lng = map(float, location_bias.split(','))
                request.location_bias = places_v1.SearchTextRequest.LocationBias(
                    circle=places_v1.Circle(
                        center=places_v1.LatLng(latitude=lat, longitude=lng),
                        radius=10000  # 10km radius
                    )
                )
            except ValueError:
                pass  # Invalid location bias format, continue without it
        
        # Set field mask to only get the fields we need (free tier optimization)
        field_mask = "places.id,places.displayName,places.formattedAddress,places.location,places.types"
        
        # Make the request
        response = client.search_text(
            request=request,
            metadata=[("x-goog-fieldmask", field_mask)]
        )
        
        # Format the response
        formatted_results = []
        for place in response.places:
            formatted_place = {
                "id": place.id,
                "name": place.display_name.text if place.display_name else "",
                "address": place.formatted_address,
                "latitude": place.location.latitude if place.location else None,
                "longitude": place.location.longitude if place.location else None,
                "types": list(place.types) if place.types else []
            }
            formatted_results.append(formatted_place)
        
        result = {
            "query": text_query,
            "places_found": len(formatted_results),
            "places": formatted_results
        }
        
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({
            "error": f"Places API error: {str(e)}",
            "query": text_query
        })


@tool("get_place_coordinates")
def get_place_coordinates(text_query: str) -> str:
    """
    Get coordinates for a specific place using Google Places API.
    Returns only the first result with coordinates.
    
    Args:
        text_query: The place to search for
        
    Returns:
        JSON string with coordinates or error message
    """
    try:
        result = search_places.invoke({"text_query": text_query})
        data = json.loads(result)
        
        if "error" in data:
            return result
        
        places = data.get("places", [])
        if not places:
            return json.dumps({
                "error": "No places found",
                "query": text_query
            })
        
        first_place = places[0]
        if not first_place.get("latitude") or not first_place.get("longitude"):
            return json.dumps({
                "error": "No coordinates found for this place",
                "query": text_query
            })
        
        return json.dumps({
            "query": text_query,
            "name": first_place.get("name"),
            "address": first_place.get("address"),
            "latitude": first_place.get("latitude"),
            "longitude": first_place.get("longitude")
        }, indent=2, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({
            "error": f"Error getting coordinates: {str(e)}",
            "query": text_query
        })
