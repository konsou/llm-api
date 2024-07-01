import logging
import time
from typing import Literal

import anthropic
from anthropic.types import TextBlock, ToolUseBlock

from llm_api import LlmApi, types_request
from llm_api.abc import ResponseAndUsage, Usage

logger = logging.getLogger(__name__)

PRICING_PER_MTOK = {
    "claude-3-5-sonnet-20240620": {
        "input": 3,
        "output": 15,
    },
}


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

        try_number = 1
        max_tries = 3
        retry_delay = 30
        while True:
            try:
                completion_kwargs = {
                    "messages": messages,
                    "model": self.model,
                    "max_tokens": 4096,
                }
                if tools is not None:
                    # Anthropic doesn't use `response_format` - instead, JSON is returned if a tool is used
                    completion_kwargs["tools"] = tools
                    completion_kwargs["tool_choice"] = (tool_choice,)

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

        response_content: TextBlock | ToolUseBlock = response_message.content[0]
        parsed_response_content: str = self._parse_response_content(response_content)

        return ResponseAndUsage(parsed_response_content, usage)

    def _parse_response_content(
        self, response_content: TextBlock | ToolUseBlock
    ) -> str:
        if isinstance(response_content, TextBlock):
            return response_content.text
        else:
            raise NotImplementedError(
                f"Unsupported response content type: {type(response_content)}"
            )

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
    response = api.response_from_prompt("test test")
    print(response)
