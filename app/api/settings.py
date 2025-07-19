import os
from dotenv import load_dotenv

load_dotenv()
userdata = os.environ
def get_api_key():
  return userdata.get("GOOGLE_API_KEY")

def get_notion_token():
  return userdata.get('NOTION_TOKEN')

def get_notion_db_id():
  return userdata.get('NOTION_DATABASE_ID')