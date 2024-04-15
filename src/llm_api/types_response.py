from typing import List, Optional, TypedDict, Union, Literal


class NonChatChoice(TypedDict):
    finish_reason: Optional[str]
    text: str


class FunctionCall(TypedDict):
    name: str
    arguments: str  # JSON format arguments


class ToolCall(TypedDict):
    id: str
    type: Literal["function"]
    function: FunctionCall


class Message(TypedDict):
    content: Optional[str]
    role: str
    tool_calls: Optional[List[ToolCall]]
    function_call: Optional[FunctionCall]


class NonStreamingChoice(TypedDict):
    finish_reason: Optional[str]
    message: Message


class Delta(TypedDict):
    content: Optional[str]
    role: Optional[str]
    tool_calls: Optional[List[ToolCall]]
    function_call: Optional[FunctionCall]


class StreamingChoice(TypedDict):
    finish_reason: Optional[str]
    delta: Delta


class Error(TypedDict):
    code: int
    message: str


class Usage(TypedDict):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    total_cost: float


class ChatCompletion(TypedDict):
    id: str
    choices: List[NonStreamingChoice | StreamingChoice | NonChatChoice | Error]
    created: int
    model: str
    object: str
    usage: Optional[Usage]


class Response(TypedDict):
    id: str
    choices: list[NonStreamingChoice | StreamingChoice | NonChatChoice | Error]
    created: int  # unix timestamp
    model: str
    object: Literal["chat.completion"] | Literal["chat.completion.chunk"]
    usage: Optional[Usage]
