from typing import Literal

import openai.types.chat
from openai import OpenAI

import llm_api.abc
from llm_api import LlmApi, types_request
from llm_api.abc import ResponseAndUsage


class OllamaApi(LlmApi):
    def __init__(
        self,
        model: str,
        api_key_name: str = "OLLAMA_API_KEY",  # needs to be set but can be anything
        timeout: int = 5,
        base_url: str | None = None,
    ):
        if base_url is None:
            raise llm_api.abc.ConfigurationError("base_url needs to be set")

        super().__init__(model=model, api_key_name=api_key_name, timeout=timeout)

        self._client = OpenAI(
            base_url=base_url,
            api_key=self.api_key,  # required, but unused
        )

    def _response_from_messages_implementation(
        self,
        messages: list[types_request.Message],
        tools: list[types_request.Tool] | None = None,
        tool_choice: Literal["auto", "required", "none"] = "auto",
        tag: str | None = None,
        response_format: Literal["json"] | None = None,
    ) -> ResponseAndUsage:
        completion: openai.types.chat.ChatCompletion = (
            self._client.chat.completions.create(
                model=self.model,
                messages=messages,  # TODO: type conversion?
            )
        )
        return ResponseAndUsage(
            response=completion.choices[0].message.content,
            usage=self.parse_usage(completion),
        )

    def parse_usage(
        self,
        completion: openai.types.chat.ChatCompletion | None = None,
        tag: str | None = None,
    ) -> llm_api.abc.Usage:
        if completion is None:
            return llm_api.abc.Usage(0, 0, 0, tag=tag)
        return llm_api.abc.Usage(
            input_tokens=completion.usage.prompt_tokens,
            output_tokens=completion.usage.completion_tokens,
            cost=0,  # Local OLLAMA is free
            tag=tag,
        )


if __name__ == "__main__":
    api = OllamaApi(model="llama3:8b", base_url="http://192.168.13.184:11434/v1")
    response = api.response_from_prompt("tes test")
    print(response)
