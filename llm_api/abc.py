import os
from abc import ABC, abstractmethod
from typing import NamedTuple

import colorama
from dotenv import load_dotenv

from . import types_request


class ConfigurationError(Exception):
    pass


class ResponseAndUsage(NamedTuple):
    response: str
    input_tokens: int
    output_tokens: int
    cost: float | None = None


class LlmApi(ABC):
    def __init__(
        self,
        model: str,
        api_key_name: str,
        timeout: int = 5,
    ):
        self.model = model
        self.api_key_name = api_key_name
        self.timeout = timeout
        self.total_cost: float = 0

        load_dotenv()
        self.api_key = os.getenv(api_key_name)
        if not self.api_key:
            raise ConfigurationError(f"{api_key_name} env var is not set")

    @abstractmethod
    def _response_from_messages_implementation(
        self,
        messages: list[types_request.Message],
        tools: list[types_request.Tool] | None = None,
        tag: str | None = None,
    ) -> ResponseAndUsage:
        pass

    def response_from_messages(
        self,
        messages: list[types_request.Message],
        tools: list[types_request.Tool] | None = None,
        tag: str | None = None,
    ) -> str:
        response: ResponseAndUsage = self._response_from_messages_implementation(
            messages=messages, tools=tools, tag=tag
        )
        self.handle_usage(
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost=response.cost,
            tag=tag,
        )
        return response.response

    def response_from_prompt(
        self,
        prompt: str,
        tag: str | None = None,
    ) -> str:
        return self.response_from_messages(
            [{"role": "user", "content": prompt}], tag=tag
        )

    def handle_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        cost: float | None = None,
        tag: str | None = None,
    ) -> None:
        if cost is not None:
            self.total_cost += cost

        print_yellow(
            f"{f'[{tag}] ' if tag else ''}"
            f"token usage: input {input_tokens} tokens, "
            f"output {output_tokens} tokens, "
            f"total {input_tokens + output_tokens} tokens, "
            f"cost {cost}, "
            f"total cost {self.total_cost}",
        )


def print_yellow(text: str):
    print(f"{colorama.Fore.YELLOW}{text}{colorama.Style.RESET_ALL}")
