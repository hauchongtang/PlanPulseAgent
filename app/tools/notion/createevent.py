import datetime
import json
from typing import Optional, List
from langchain_core.tools import tool
from notion_client import Client

from app.api.settings import get_notion_token, get_notion_db_id

notion_key = get_notion_token()
notion = Client(auth=notion_key)

# Hardcoded status options (updated to match your Notion database)
AVAILABLE_STATUSES = [
    "In progress",
    "Rescheduled",
    "KIV",
    "Cancelled",
    "Not started",
    "90 % Done",
    "Done"
]

# Hardcoded tag options (updated to match your Notion database)
AVAILABLE_TAGS = [
    "重要",
    "想让你知道",
    "旅行! ✈️",
    "琬淯出去",
    "把拖❤️",
    "皓聪出去"
]

@tool("create_event")
def create_event(
    title: str,
    date: str,
    status: str = "Not started",
    location: str = "",
    tags: str = "",
    notes: str = "",
    persons: str = ""
) -> str:
    """
    Create a new event in the Notion database.
    
    Args:
        title: Event title/name (required)
        date: Event date in YYYY-MM-DD format, or YYYY-MM-DD to YYYY-MM-DD for date range (required)
        status: Event status - options: "In progress", "To-do", "Rescheduled", "KIV", "Cancelled", "Not started", "90 % Done", "Done" (default: "Not started")
        location: Event location/venue (optional)
        tags: Comma-separated tags - options: 重要, 想让你知道, 旅行! ✈️, 玩清出去, 把蛋❤️, 皓聪出去 (optional)
        notes: Additional notes or description (optional)
        persons: Comma-separated list of person names/emails (optional, will be added as text since we can't auto-create Notion users)
        
    Returns:
        JSON string with creation result and event details
    """
    try:
        # Validate and parse date
        date_property = _parse_date(date)
        if not date_property:
            return json.dumps({
                "success": False,
                "error": "Invalid date format. Use YYYY-MM-DD or YYYY-MM-DD to YYYY-MM-DD for date ranges",
                "provided_date": date
            }, indent=2)
        
        # Validate status
        if status not in AVAILABLE_STATUSES:
            return json.dumps({
                "success": False,
                "error": f"Invalid status. Available options: {', '.join(AVAILABLE_STATUSES)}",
                "provided_status": status
            }, indent=2)
        
        # Parse and validate tags
        tag_list = []
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",")]
            invalid_tags = [tag for tag in tag_list if tag not in AVAILABLE_TAGS]
            if invalid_tags:
                return json.dumps({
                    "success": False,
                    "error": f"Invalid tags: {', '.join(invalid_tags)}. Available tags: {', '.join(AVAILABLE_TAGS)}",
                    "provided_tags": tags
                }, indent=2)
        
        # Build the properties for the new page
        properties = {
            "Name": {
                "title": [
                    {
                        "text": {
                            "content": title
                        }
                    }
                ]
            },
            "Date": date_property,
            "Status": {
                "status": {
                    "name": status
                }
            }
        }
        
        # Add location if provided
        if location:
            properties["Where"] = {
                "rich_text": [
                    {
                        "text": {
                            "content": location
                        }
                    }
                ]
            }
        
        # Add tags if provided
        if tag_list:
            properties["Tags"] = {
                "multi_select": [
                    {"name": tag} for tag in tag_list
                ]
            }
        
        # Add notes if provided
        if notes:
            properties["Mini Reminder Description"] = {
                "rich_text": [
                    {
                        "text": {
                            "content": notes
                        }
                    }
                ]
            }
        
        # Add persons as text (since we can't auto-create Notion users)
        if persons:
            # For now, we'll add persons to the notes field or create a simple text representation
            # You might want to create a separate "People (Text)" property in your Notion database
            person_text = f"People: {persons}"
            if notes:
                properties["Mini Reminder Description"]["rich_text"][0]["text"]["content"] += f"\n{person_text}"
            else:
                properties["Mini Reminder Description"] = {
                    "rich_text": [
                        {
                            "text": {
                                "content": person_text
                            }
                        }
                    ]
                }
        
        # Create the page in Notion
        response = notion.pages.create(
            parent={"database_id": get_notion_db_id()},
            properties=properties
        )
        
        # Return success response with event details
        result = {
            "success": True,
            "message": "Event created successfully",
            "event": {
                "id": response["id"],
                "title": title,
                "date": date,
                "status": status,
                "location": location,
                "tags": tag_list,
                "notes": notes,
                "persons": persons,
                "notion_url": response["url"],
                "created_time": response["created_time"]
            },
            "available_options": {
                "statuses": AVAILABLE_STATUSES,
                "tags": AVAILABLE_TAGS
            }
        }
        
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": f"Failed to create event: {str(e)}",
            "event_data": {
                "title": title,
                "date": date,
                "status": status,
                "location": location,
                "tags": tags,
                "notes": notes,
                "persons": persons
            }
        }
        return json.dumps(error_result, indent=2)


def _parse_date(date_str: str) -> Optional[dict]:
    """
    Parse date string into Notion date property format.
    Supports single dates (YYYY-MM-DD) and date ranges (YYYY-MM-DD to YYYY-MM-DD).
    
    Args:
        date_str: Date string to parse
        
    Returns:
        Notion date property dict or None if invalid
    """
    try:
        # Check if it's a date range
        if " to " in date_str:
            start_str, end_str = date_str.split(" to ", 1)
            start_date = datetime.datetime.strptime(start_str.strip(), "%Y-%m-%d").date()
            end_date = datetime.datetime.strptime(end_str.strip(), "%Y-%m-%d").date()
            
            return {
                "date": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                }
            }
        else:
            # Single date
            single_date = datetime.datetime.strptime(date_str.strip(), "%Y-%m-%d").date()
            return {
                "date": {
                    "start": single_date.isoformat()
                }
            }
    except ValueError:
        return None


@tool("get_create_event_options")
def get_create_event_options() -> str:
    """
    Get available options for creating events (statuses, tags, etc.).
    
    Returns:
        JSON string with all available options for event creation
    """
    options = {
        "available_statuses": AVAILABLE_STATUSES,
        "available_tags": AVAILABLE_TAGS,
        "date_format_examples": [
            "2025-07-21 (single date)",
            "2025-07-21 to 2025-07-23 (date range)"
        ],
        "usage_example": {
            "title": "Team Meeting",
            "date": "2025-07-21",
            "status": "Confirmed",
            "location": "Conference Room A",
            "tags": "work, meeting, important",
            "notes": "Quarterly review and planning session",
            "persons": "John Doe, Jane Smith, Bob Wilson"
        }
    }
    
    return json.dumps(options, indent=2, ensure_ascii=False)
