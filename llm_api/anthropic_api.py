import json
import logging
import time
from typing import Literal, NamedTuple

import anthropic
from anthropic.types import TextBlock, ToolUseBlock, ContentBlock

from llm_api import LlmApi, types_request
from llm_api.abc import ResponseAndUsage, Usage

logger = logging.getLogger(__name__)

PRICING_PER_MTOK = {
    "claude-3-5-sonnet-20240620": {
        "input": 3,
        "output": 15,
    },
}


class ToolUseResult(NamedTuple):
    tool_used: bool


class AnthropicApi(LlmApi):
    def __init__(
        self,
        model: str,
        api_key_name: str = "ANTHROPIC_API_KEY",
        timeout: int = 5,
    ):
        super().__init__(model=model, api_key_name=api_key_name, timeout=timeout)
        self._client = anthropic.Anthropic(
            api_key=self.api_key,
        )

    def _response_from_messages_implementation(
        self,
        messages: list[types_request.Message],
        tools: list[types_request.Tool] | None = None,
        tool_choice: Literal["auto", "required", "none"] = "auto",
        tag: str | None = None,
        response_format: Literal["json"] = None,
    ) -> ResponseAndUsage:

        # TODO: Anthropic doesn't seem to support system messages or the `name` field
        # - Convert system messages to user messages and prepend "System:"
        # - Prepend "Name:" if `name` is included

        try_number = 1
        max_tries = 3
        retry_delay = 30

        converted_tools = self._convert_tools_to_anthropic_format(tools)
        converted_tool_choice = self._convert_tool_choice_to_anthropic_format(
            tool_choice
        )

        while True:
            try:
                completion_kwargs = {
                    "messages": messages,
                    "model": self.model,
                    "max_tokens": 4096,
                }
                if tools is not None:
                    # Anthropic doesn't use `response_format` - instead, JSON is returned if a tool is used
                    completion_kwargs["tools"] = converted_tools
                    completion_kwargs["tool_choice"] = converted_tool_choice

                response_message: anthropic.types.Message = (
                    self._client.messages.create(**completion_kwargs)
                )

                break
            except (
                anthropic.RateLimitError,
                anthropic.BadRequestError,
                anthropic.APITimeoutError,
            ) as e:
                logger.warning(f"{e.message} - retrying in {retry_delay} seconds...")
                if try_number >= max_tries:
                    raise e
                time.sleep(retry_delay)
                retry_delay *= 2
                try_number += 1

        usage = self.parse_usage(response_message, tag=tag)

        parsed_response_content: str = self._parse_content(response_message.content)
        return ResponseAndUsage(parsed_response_content, usage)

    def _parse_content(self, content: list[ContentBlock]) -> str:
        """
        If there's only one piece of content AND it's text -> return it as a plain string

        In any other case the content contains multiple messages (either text or tool calls)
        Return a JSON list of the contents as a string in that case."""
        if len(content) == 1 and isinstance(content[0], TextBlock):
            return content[0].text

        content_string_list = [str(o.__dict__) for o in content]
        return json.dumps(content_string_list, ensure_ascii=False, indent=2)

    def _convert_tools_to_anthropic_format(
        self, tools: list[types_request.Tool] | None
    ) -> list[anthropic.types.ToolParam] | None:
        if tools is None:
            return None

        converted_tools: list[anthropic.types.ToolParam] = []
        for tool in tools:
            converted_tools.append(
                anthropic.types.ToolParam(
                    input_schema=tool["function"]["parameters"],  # type: ignore
                    name=tool["function"]["name"],
                    description=tool["function"]["description"],
                )
            )
        return converted_tools

    def _convert_tool_choice_to_anthropic_format(
        self, tool_choice: Literal["auto", "required", "none"]
    ) -> anthropic.types.message_create_params.ToolChoice:
        """
        NOTE 1: Current way of implementation doesn't support the Anthropic way to force the use of a predefined tool
        NOTE 2: "none" is not supported - just don't supply tools if you don't want to use them

        See https://docs.anthropic.com/en/docs/build-with-claude/tool-use#forcing-tool-use
        """
        if tool_choice == "required":
            return {"type": "any"}
        # Return "auto" for both "auto" and "none"
        return {"type": "auto"}

    def parse_usage(
        self, message: anthropic.types.Message, tag: str | None = None
    ) -> Usage:
        input_tokens = message.usage.input_tokens
        output_tokens = message.usage.output_tokens
        input_cost = self._input_token_cost(input_tokens)
        output_cost = self._output_token_cost(output_tokens)

        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=input_cost + output_cost,
            tag=tag,
        )

    def _input_token_cost(self, tokens: int) -> float:
        return self._token_cost(tokens, "input")

    def _output_token_cost(self, tokens: int) -> float:
        return self._token_cost(tokens, "output")

    def _token_cost(
        self, tokens: int, input_output: Literal["input", "output"]
    ) -> float:
        price_per_mtok = PRICING_PER_MTOK.get(self.model, {}).get(input_output, None)

        if price_per_mtok is None:
            logger.warning(f"No pricing info available for {self.model}")
            return 0

        return tokens / 1_000_000 * price_per_mtok


if __name__ == "__main__":
    api = AnthropicApi(model="claude-3-5-sonnet-20240620")
    response = api.response_from_messages(
        messages=[
            types_request.Message(
                role="user", content="What's the stock price for Microsoft?"
            )
        ],
        tools=[
            types_request.Tool(
                type="function",
                function=types_request.FunctionDescription(
                    description="Retrieves the current stock price for a given ticker symbol. The ticker symbol must be a valid symbol for a publicly traded company on a major US stock exchange like NYSE or NASDAQ. The tool will return the latest trade price in USD. It should be used when the user asks about the current or most recent price of a specific stock. It will not provide any other information about the stock or company.",
                    name="get_stock_price",
                    parameters={
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "The stock ticker symbol, e.g. AAPL for Apple Inc.",
                            }
                        },
                        "required": ["ticker"],
                    },
                ),
            )
        ],
        tool_choice="auto",
    )
    print(response)
