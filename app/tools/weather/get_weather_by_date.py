import json
import requests
from datetime import datetime
from langchain_core.tools import tool

from app.api.settings import get_weather_url


def filter_weather_by_location(weather_data: dict, location: str) -> dict:
    """
    Filter weather data by location (area).
    
    Args:
        weather_data: The full weather response from Singapore API
        location: The location to filter for (case-insensitive partial match)
        
    Returns:
        Filtered weather data or the original data if location not found
    """
    try:
        # Check if we have the expected structure
        if not weather_data.get('data') or not weather_data['data'].get('items'):
            return weather_data
        
        location_lower = location.lower()
        filtered_items = []
        
        for item in weather_data['data']['items']:
            if 'forecasts' in item:
                # Filter forecasts for matching areas
                matching_forecasts = []
                for forecast in item['forecasts']:
                    area_name = forecast.get('area', '').lower()
                    if location_lower in area_name or area_name in location_lower:
                        matching_forecasts.append(forecast)
                
                # If we found matching forecasts, create a filtered item
                if matching_forecasts:
                    filtered_item = item.copy()
                    filtered_item['forecasts'] = matching_forecasts
                    filtered_items.append(filtered_item)
        
        # Create filtered response
        if filtered_items:
            filtered_data = weather_data.copy()
            filtered_data['data'] = weather_data['data'].copy()
            filtered_data['data']['items'] = filtered_items
            
            # Also filter area_metadata if it exists
            if 'area_metadata' in weather_data['data']:
                matching_metadata = []
                for metadata in weather_data['data']['area_metadata']:
                    area_name = metadata.get('name', '').lower()
                    if location_lower in area_name or area_name in location_lower:
                        matching_metadata.append(metadata)
                filtered_data['data']['area_metadata'] = matching_metadata
            
            # Add a note about the filtering
            filtered_data['location_filter'] = {
                'requested_location': location,
                'areas_found': len(filtered_items[0]['forecasts']) if filtered_items else 0,
                'note': f'Weather data filtered for areas matching "{location}"'
            }
            
            return filtered_data
        else:
            # Location not found, return original data with a note
            result = weather_data.copy()
            result['location_filter'] = {
                'requested_location': location,
                'areas_found': 0,
                'note': f'No areas found matching "{location}". Showing all areas.',
                'available_areas': [forecast.get('area') for item in weather_data['data']['items'] 
                                  for forecast in item.get('forecasts', [])][:10]  # Show first 10 areas
            }
            return result
            
    except Exception as e:
        # If filtering fails, return original data with error note
        result = weather_data.copy()
        result['location_filter'] = {
            'error': f'Failed to filter by location: {str(e)}',
            'note': 'Returning full weather data'
        }
        return result


@tool("get_weather_by_date")
def get_weather_by_date(date: str, location: str = None) -> str:
    """
    Get weather information for a specific date and optionally filter by location.
    
    Args:
        date: The date to get the weather for (format: YYYY-MM-DD)
        location: Optional location to filter weather data (e.g., "Orchard", "Tampines", "City")
        
    Returns:
        JSON string with weather data or error message
    """
    base_url = get_weather_url()
    if not base_url:
        return json.dumps({
            "error": "Weather service URL is not configured"
        })
    
    try:
        # Parse the input date and add current time
        try:
            parsed_date = datetime.strptime(date, '%Y-%m-%d')
            current_time = datetime.now().time()
            # Combine the input date with current time
            combined_datetime = datetime.combine(parsed_date.date(), current_time)
            formatted_datetime = f"{combined_datetime.strftime('%Y-%m-%d')}T{combined_datetime.strftime('%H:%M:%S')}"
        except ValueError:
            return json.dumps({
                "error": f"Invalid date format. Expected YYYY-MM-DD, got: {date}"
            })
        
        # Construct the API URL
        api_url = f"{base_url}/real-time/api/two-hr-forecast?date={formatted_datetime}"
        
        # Make the HTTP request
        response = requests.get(api_url)
        response.raise_for_status()  # Equivalent to EnsureSuccessStatusCode()
        
        # Parse the response
        weather_data = response.json()
        
        # If location is specified, filter the data for that location
        if location:
            filtered_data = filter_weather_by_location(weather_data, location)
            return json.dumps(filtered_data)
        
        # Return the full JSON response if no location specified
        return json.dumps(weather_data)
        
    except requests.exceptions.RequestException as e:
        return json.dumps({
            "error": f"Failed to fetch weather data: {str(e)}"
        })
    except json.JSONDecodeError as e:
        return json.dumps({
            "error": f"Failed to parse weather data: {str(e)}"
        })
    except Exception as e:
        return json.dumps({
            "error": f"Unexpected error: {str(e)}"
        })

@tool("get_uv")
def get_uv() -> str:
    """
    Get UV index for the current date.
    
    Returns:
        JSON string with UV index or error message
    """
    base_url = get_weather_url()
    if not base_url:
        return json.dumps({
            "error": "Weather service URL is not configured"
        })
    
    try:
        # Construct the API URL for UV index
        api_url = f"{base_url}/real-time/api/uv"
        
        # Make the HTTP request
        response = requests.get(api_url)
        response.raise_for_status()  # Equivalent to EnsureSuccessStatusCode()
        
        # Parse the response
        uv_data = response.json()
        
        return json.dumps(uv_data)
        
    except requests.exceptions.RequestException as e:
        return json.dumps({
            "error": f"Failed to fetch UV index data: {str(e)}"
        })
    except json.JSONDecodeError as e:
        return json.dumps({
            "error": f"Failed to parse UV index data: {str(e)}"
        })
    except Exception as e:
        return json.dumps({
            "error": f"Unexpected error: {str(e)}"
        })