from langchain_core.tools import tool

@tool("add")
def add(x: int, y: int):
  """
  Adds x and y
  """
  return x + y