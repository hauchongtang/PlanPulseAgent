from typing import Any


class TelegramBase():
  chat_id: str
  chat_message: str
  response: dict[str | Any]