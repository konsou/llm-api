import time

import groq
from groq import Groq
from groq.types.chat import ChatCompletion

from llm_api import LlmApi, types_request
from llm_api.abc import ResponseAndUsage, Usage
from llm_api.text import print_warning


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

    def _response_from_messages_implementation(
        self,
        messages: list[types_request.Message],
        tools: list[types_request.Tool] | None = None,
        tag: str | None = None,
    ) -> ResponseAndUsage:
        if tools is not None:
            raise NotImplementedError("Tools not implemented yet")

        try_number = 1
        max_tries = 3
        retry_delay = 30
        while True:
            try:
                # See https://console.groq.com/docs/text-chat
                chat_completion = self._client.chat.completions.create(
                    messages=messages,
                    model=self.model,
                )
                break
            except groq.RateLimitError as e:
                print_warning(
                    f"Rate Limit Exceeded: {e.message}. Retrying in {retry_delay} seconds..."
                )
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
