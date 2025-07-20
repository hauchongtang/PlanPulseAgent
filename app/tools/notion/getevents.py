import datetime
import json
from langchain_core.tools import tool
from notion_client import Client

from app.api.settings import get_notion_token, get_notion_db_id

notion_key = get_notion_token()
notion = Client(auth=notion_key)

def extract_text(rich_text_field):
  arr = rich_text_field if isinstance(rich_text_field, list) else []
  return "".join([i.get("plain_text", "") for i in arr])

def get_people(people_field):
  return ", ".join([p.get("name", "") for p in (people_field or [])])

def get_tags(tags_field):
  return ", ".join([t.get("name", "") for t in (tags_field or [])])

def get_date(date_field):
  if not date_field:
    return ""
  start = date_field.get("start")
  end = date_field.get("end")
  if start and end and end != start:
    return f"{start} — {end}"
  return start or ""

@tool("get_events")
def get_events(date_from: datetime.datetime, date_to: datetime.datetime) -> str:
  """
  Get all pages from Notion database between two dates.
  Returns JSON string with structured event data.
  """
  try:
    pages = notion.databases.query(
      **{
          "database_id": get_notion_db_id(),
          "filter": {
              "and": [
                  {
                      "property": "Date",
                      "date": {
                          "on_or_after": date_from.strftime("%Y-%m-%d"),
                      }
                  },
                  {
                      "property": "Date",
                      "date": {
                          "on_or_before": date_to.strftime("%Y-%m-%d"),
                      }
                  }
              ]
          },
          "sorts": [
              {
                  "property": "Date",
                  "direction": "ascending"
              }
          ]
      }
    )["results"]
    
    # Structure the events in a clean JSON format
    events = []
    for page in pages:
      props = page.get("properties", {})
      
      event = {
        "id": page.get("id", ""),
        "title": extract_text(props.get("Name", {}).get("title", [])),
        "date": get_date(props.get("Date", {}).get("date", {})),
        "status": props.get("Status", {}).get("status", {}).get("name", ""),
        "persons": get_people(props.get("Person", {}).get("people", [])),
        "location": extract_text(props.get("Where", {}).get("rich_text", [])),
        "tags": get_tags(props.get("Tags", {}).get("multi_select", [])),
        "notes": extract_text(props.get("Mini Reminder Description", {}).get("rich_text", [])),
        "notion_url": page.get("url", ""),
        "created_time": page.get("created_time", ""),
        "last_edited_time": page.get("last_edited_time", "")
      }
      events.append(event)
    
    # Return structured JSON that's also LLM-friendly
    result = {
      "query_period": {
        "from": date_from.strftime("%Y-%m-%d"),
        "to": date_to.strftime("%Y-%m-%d")
      },
      "total_events": len(events),
      "events": events
    }
    
    return json.dumps(result, indent=2, ensure_ascii=False)

  except Exception as e:
    print(e)
    error_result = {
      "error": True,
      "message": f"Error occurred with Notion Agent: {str(e)}",
      "query_period": {
        "from": date_from.strftime("%Y-%m-%d") if date_from else None,
        "to": date_to.strftime("%Y-%m-%d") if date_to else None
      },
      "total_events": 0,
      "events": []
    }
    return json.dumps(error_result, indent=2)


@tool("get_events_formatted")
def get_events_formatted(date_from: datetime.datetime, date_to: datetime.datetime) -> str:
  """
  Get all pages from Notion database between two dates.
  Returns human-readable formatted text (original format).
  """
  result_str = ""
  try:
    pages = notion.databases.query(
      **{
          "database_id": get_notion_db_id(),
          "filter": {
              "and": [
                  {
                      "property": "Date",
                      "date": {
                          "on_or_after": date_from.strftime("%Y-%m-%d"),
                      }
                  },
                  {
                      "property": "Date",
                      "date": {
                          "on_or_before": date_to.strftime("%Y-%m-%d"),
                      }
                  }
              ]
          },
          "sorts": [
              {
                  "property": "Date",
                  "direction": "ascending"
              }
          ]
      }
    )["results"]
    
    if not pages:
      return f"No events found between {date_from.strftime('%Y-%m-%d')} and {date_to.strftime('%Y-%m-%d')}"
    
    result_str += f"## Events from {date_from.strftime('%Y-%m-%d')} to {date_to.strftime('%Y-%m-%d')}\n\n"
    
    for page in pages:
      props = page.get("properties", {})
      title = extract_text(props.get("Name", {}).get("title", []))
      date = get_date(props.get("Date", {}).get("date", {}))
      status = props.get("Status", {}).get("status", {}).get("name", "")
      persons = get_people(props.get("Person", {}).get("people", []))
      where = extract_text(props.get("Where", {}).get("rich_text", []))
      tags = get_tags(props.get("Tags", {}).get("multi_select", []))
      notes = extract_text(props.get("Mini Reminder Description", {}).get("rich_text", []))
      url = page.get("url", "")

      block = (
        f"#### {title}\n"
        f"- **Date:** {date}\n"
        f"- **Status:** {status}\n"
        f"- **Person(s):** {persons}\n"
        f"- **Where:** {where}\n"
        f"- **Tags:** {tags}\n"
        f"- **Notes:** {notes}\n"
        f"- **Notion URL:** {url}\n\n"
      )

      result_str += block

  except Exception as e:
    print(e)
    return f"Error has occurred with Notion Agent -> {e}"
  return result_str