from notion_client import Client
import json
from langchain_core.tools import tool

from app.api.settings import get_notion_token, get_notion_summary_page_id

notion_key = get_notion_token()
notion = Client(auth=notion_key)

@tool("get_summary_page")
def get_summary_page() -> str:
    page_id = get_notion_summary_page_id()
    response = notion.pages.retrieve(page_id)
    return json.dumps(response, indent=2, ensure_ascii=False)