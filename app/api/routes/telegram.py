from fastapi import APIRouter
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from app.api.models import TelegramBase
from app.tools.calendar.add import add
from app.tools.notion.getevents import get_events

from app.api.settings import get_api_key

router = APIRouter(prefix="/telegram", tags=["telegram"])

@router.post("/",response_model=None)
async def pass_telegram_message_to_orchestrator(chat_id: str, chat_message: str):
  try:
    api_key = get_api_key()
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=1.0,
        max_retries=2,
        google_api_key=api_key
      )
    tools = [get_events, add]

    agent_executor = create_react_agent(model, tools)
    input_message = {"role": "user", "content": chat_message}
    response = agent_executor.invoke({"messages": [input_message]})
    for message in response["messages"]:
      message.pretty_print()
    return {
        "chat_id": chat_id,
      "chat_message": chat_message,
        "response": message
      }
  except Exception as e:
    return {
        "chat_id": chat_id,
        "chat_message": chat_message,
        "response": {"error -> ": str(e)}
      }
    