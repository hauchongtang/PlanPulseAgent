import os
from dotenv import load_dotenv

load_dotenv()
userdata = os.environ
def get_api_key():
  return userdata.get("GOOGLE_API_KEY")

def get_google_places_key():
  return userdata.get("GOOGLE_PLACES_API_KEY")

def get_notion_token():
  return userdata.get('NOTION_TOKEN')

def get_notion_db_id():
  return userdata.get('NOTION_DATABASE_ID')

def get_notion_summary_page_id():
  return userdata.get('NOTION_SUMMARY_PAGE_ID')

def get_telegram_secret_key():
  return userdata.get('PLANPULSE_SECRET_KEY')

def get_lta_datamall_key():
  return userdata.get('LTA_DATAMALL_API_KEY')

def get_weather_url():
  return userdata.get('WEATHER_API_URL')