from abc import ABC, abstractmethod
from . import types_request, types_response


class LlmApi(ABC):
    def __init__(
        self,
        model: str,
        timeout: int = 5,
    ):
        self.model = model
        self.timeout = timeout

    @abstractmethod
    def response_from_messages(
        self,
        messages: list[types_request.Message],
        tools: list[types_request.Tool] | None = None,
        tag: str | None = None,
    ) -> str:
        pass

    def response_from_prompt(
        self,
        prompt: str,
        tag: str | None = None,
    ) -> str:
        return self.response_from_messages(
            [{"role": "user", "content": prompt}], tag=tag
        )

    @abstractmethod
    def handle_usage(
        self, response: types_response.Response, tag: str | None = None
    ) -> None:
        pass
