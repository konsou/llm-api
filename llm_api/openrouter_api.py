import json
from typing import Literal

import requests

from llm_api.abc import LlmApi, ResponseAndUsage, Usage
from llm_api import types_request, types_response


class OpenRouterAPI(LlmApi):
    def __init__(
        self,
        model: str,
        api_key_name: str = "OPENROUTER_API_KEY",
        timeout: int = 5,
    ):
        super().__init__(model=model, api_key_name=api_key_name, timeout=timeout)

    @property
    def requires_alternating_roles(self) -> bool:
        raise NotImplementedError

    def _response_from_messages_implementation(
        self,
        messages: list[types_request.Message],
        tools: list[types_request.Tool] | None = None,
        tool_choice: Literal["auto", "required", "none"] = "auto",
        tag: str | None = None,
        response_format: Literal["json"] | None = None,
    ) -> ResponseAndUsage:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        request_data = {
            "model": self.model,
            "messages": messages,
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

        usage = self.parse_usage(response_json, tag=tag)

        if "message" in response_json["choices"][0]:
            response_message = response_json["choices"][0]["message"]
        else:
            raise ValueError("Response does not contain a message")

        return ResponseAndUsage(response_message["content"], usage)

    def parse_usage(
        self, response: types_response.Response, tag: str | None = None
    ) -> Usage:
        return Usage(
            input_tokens=response.get("usage", {}).get("prompt_tokens", 0),
            output_tokens=response.get("usage", {}).get("completion_tokens", 0),
            cost=response.get("usage", {}).get("total_cost", 0),
            tag=tag,
        )
