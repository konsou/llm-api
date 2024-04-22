from typing import TypedDict


# This is pure guesswork based on responses
class ToolCall(TypedDict):
    function: str
    parameters: dict[str, str]
