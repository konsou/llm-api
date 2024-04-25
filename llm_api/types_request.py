from typing import Literal, NotRequired, TypedDict


class FunctionDescription(TypedDict):
    description: NotRequired[str]
    name: str
    parameters: object  # JSON Schema object


class Tool(TypedDict):
    type: Literal["function"]
    function: FunctionDescription


ToolChoice = Literal["none", "auto"] | TypedDict(
    "ToolChoice", {"type": Literal["function"], "function": dict[Literal["name"], str]}
)

MessageRole = Literal["assistant", "user", "system", "tool"]


class Message(TypedDict):
    role: MessageRole
    content: str
    name: NotRequired[str]  # Added, not part of official OpenRouter API
    tools: NotRequired[list[Tool]]
    tool_choice: NotRequired[ToolChoice]
