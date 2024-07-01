from llm_api import types_request


def mock_tool_factory() -> list[types_request.Tool]:
    tools = [
        types_request.Tool(
            type="function",
            function=types_request.FunctionDescription(
                description="test function",
                name="test",
                parameters={
                    "type": "object",
                    "properties": {"test_argument": {"type": "string"}},
                    "required": ["test_argument"],
                },
            ),
        )
    ]
    return tools
