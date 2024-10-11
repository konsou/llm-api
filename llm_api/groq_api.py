import logging
import time
from typing import Literal

import groq
from groq import Groq
from groq.types.chat import ChatCompletion

from llm_api import LlmApi, types_request
from llm_api.abc import ResponseAndUsage, Usage


logger = logging.getLogger(__name__)


class GroqApi(LlmApi):
    def __init__(
        self,
        model: str,
        api_key_name: str = "GROQ_API_KEY",
        timeout: int = 5,
    ):
        super().__init__(model=model, api_key_name=api_key_name, timeout=timeout)
        self._client = Groq(
            api_key=self.api_key,
        )

    @property
    def requires_alternating_roles(self) -> bool:
        raise NotImplementedError

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
                groq_kwargs = {
                    "messages": messages,
                    "model": self.model,
                    "tool_choice": tool_choice,
                }
                if tools is not None:
                    groq_kwargs["tools"] = tools
                if response_format == "json":
                    groq_kwargs["response_format"] = {"type": "json_object"}

                # See https://console.groq.com/docs/text-chat
                chat_completion = self._client.chat.completions.create(**groq_kwargs)

                break
            except (
                groq.RateLimitError,
                groq.BadRequestError,
                groq.APITimeoutError,
            ) as e:
                logger.warning(f"{e.message} - retrying in {retry_delay} seconds...")
                if try_number >= max_tries:
                    raise e
                time.sleep(retry_delay)
                retry_delay *= 2
                try_number += 1

        usage = self.parse_usage(chat_completion, tag=tag)
        return ResponseAndUsage(chat_completion.choices[0].message.content, usage)

    def parse_usage(self, completion: ChatCompletion, tag: str | None = None) -> Usage:
        return Usage(
            input_tokens=completion.usage.prompt_tokens,
            output_tokens=completion.usage.completion_tokens,
            cost=0,  # Groq is free (2024-04-24)
            tag=tag,
        )


if __name__ == "__main__":
    api = GroqApi(model="mixtral-8x7b-32768")
    response = api.response_from_prompt("tes test")
    print(response)
