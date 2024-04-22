import json
import os

import colorama
import requests
from dotenv import load_dotenv

from .abc import LlmApi
from . import types_request, types_response


class OpenRouterAPI(LlmApi):
    def __init__(
        self,
        model: str,
    ):
        super().__init__(model=model)

        self.total_cost = 0

        load_dotenv()
        self._api_key = os.getenv("OPENROUTER_API_KEY")

    def response_from_messages(
        self,
        messages: list[types_request.Message],
        tools: list[types_request.Tool] | None = None,
        tag: str | None = None,
    ) -> str:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
        }
        request_data = {
            "model": self.model,
            "messages": messages,
            "repetition_penalty": 1.5,
        }
        if tools:
            request_data["tools"] = tools

        response = requests.post(
            url=url,
            headers=headers,
            data=json.dumps(request_data),
        )
        response.raise_for_status()
        response_json: types_response.Response = response.json()

        self.handle_usage(response_json, tag=tag)

        if "message" in response_json["choices"][0]:
            response_message = response_json["choices"][0]["message"]
        else:
            raise ValueError("Response does not contain a message")

        return response_message["content"]

    def handle_usage(self, response: types_response.Response, tag: str | None = None):
        usage = response.get("usage")
        if not usage:
            print_yellow(f"Response does not contain usage information")
            return

        input_tokens = usage["prompt_tokens"]
        output_tokens = usage["completion_tokens"]
        total_tokens = usage["total_tokens"]
        cost = usage["total_cost"]

        self.total_cost += cost

        print_yellow(
            f"{f'[{tag}] ' if tag else ''}"
            f"token usage: input {input_tokens} tokens, "
            f"output {output_tokens} tokens, "
            f"total {total_tokens} tokens, "
            f"cost {cost}, "
            f"total cost {self.total_cost}",
        )


def print_yellow(text: str):
    print(f"{colorama.Fore.YELLOW}{text}{colorama.Style.RESET_ALL}")
