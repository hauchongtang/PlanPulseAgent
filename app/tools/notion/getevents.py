import datetime
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
    return f"Error has occured with Notion Agent -> {e}"
  print(result_str)
  return result_str