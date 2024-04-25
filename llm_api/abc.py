import os
from abc import ABC, abstractmethod
from typing import NamedTuple

from dotenv import load_dotenv

from . import types_request
from .text import print_yellow, print_cost


class ConfigurationError(Exception):
    pass


class Usage(NamedTuple):
    input_tokens: int
    output_tokens: int
    cost: float | None = None
    tag: str | None = None


class ResponseAndUsage(NamedTuple):
    response: str
    usage: Usage


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
        response: str
        usage: Usage
        response, usage = self._response_from_messages_implementation(
            messages=messages, tools=tools, tag=tag
        )
        self.handle_usage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost=usage.cost,
            tag=usage.tag,
        )
        return response

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

        print_cost(
            f"{f'[{tag}] ' if tag else ''}"
            f"[{self.model}] "
            f"token usage: input {input_tokens} tokens, "
            f"output {output_tokens} tokens, "
            f"cost {cost}, "
            f"total cost {self.total_cost}",
        )
