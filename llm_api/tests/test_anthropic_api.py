from unittest import TestCase
from unittest.mock import patch, Mock, MagicMock

import anthropic
import httpx
from anthropic.types import TextBlock, Usage, Message, ToolParam

import llm_api.abc
import llm_api.anthropic_api
from llm_api.anthropic_api import AnthropicApi

from .helpers import mock_tool_factory


def mock_response_message_factory() -> anthropic.types.Message:
    content = [
        TextBlock(
            text="test",
            type="text",
        ),
    ]
    u = Usage(input_tokens=1, output_tokens=2)
    return Message(
        id="test-id",
        content=content,
        model="test-model",
        role="assistant",
        stop_reason="end_turn",
        type="message",
        usage=u,
    )


class TestAnthropicApi(TestCase):
    def setUp(self):
        with patch("os.getenv", Mock(return_value="mock_api_key")):
            self.api = AnthropicApi(
                model="claude-3-5-sonnet-20240620",
                api_key_name="TEST_ANTHROPIC_API_KEY",
            )

    def test_api_key_not_set(self):
        with patch("os.getenv", Mock(return_value=None)):
            with self.assertRaises(llm_api.abc.ConfigurationError):
                AnthropicApi(model="test")

    def test_correct_api_key_used(self):
        def mock_getenv(key, default=None):
            if key == "ANTHROPIC_API_KEY":
                return "Test Anthropic API Key"
            return default

        with patch("os.getenv", mock_getenv):
            api = AnthropicApi(model="test")

        self.assertEqual("Test Anthropic API Key", api.api_key)

    def test_parse_usage(self):
        mock_response_message = Mock(spec=Message)
        mock_response_message.usage = Usage(
            input_tokens=1_000_000,
            output_tokens=2_000_000,
        )
        u = self.api.parse_usage(mock_response_message)
        self.assertIsInstance(u, llm_api.abc.Usage)
        self.assertEqual(1_000_000, u.input_tokens)
        self.assertEqual(2_000_000, u.output_tokens)
        self.assertEqual(
            33, u.cost
        )  # Pricing for claude-3-5-sonnet-20240620 on 2024-07
        self.assertIsNone(u.tag)

    def test_parse_usage_warns_when_unknown_model(self):
        self.api.model = "nonexistent"
        mock_response_message = Mock(spec=Message)
        mock_response_message.usage = Usage(
            input_tokens=1_000_000,
            output_tokens=2_000_000,
        )

        with patch("llm_api.anthropic_api.logger.warning") as mock_warning:
            u = self.api.parse_usage(mock_response_message)

        mock_warning.assert_called()
        self.assertIsInstance(u, llm_api.abc.Usage)
        self.assertEqual(1_000_000, u.input_tokens)
        self.assertEqual(2_000_000, u.output_tokens)
        self.assertEqual(0, u.cost)
        self.assertIsNone(u.tag)

    def test_parse_usage_with_tag(self):
        mock_response_message = Mock(spec=Message)
        mock_response_message.usage = Usage(
            input_tokens=1_000_000,
            output_tokens=2_000_000,
        )
        u = self.api.parse_usage(mock_response_message, tag="test tag")
        self.assertIsInstance(u, llm_api.abc.Usage)
        self.assertEqual(1_000_000, u.input_tokens)
        self.assertEqual(2_000_000, u.output_tokens)
        self.assertEqual(
            33, u.cost
        )  # Pricing for claude-3-5-sonnet-20240620 on 2024-07
        self.assertEqual("test tag", u.tag)

    def test_rate_limit(self):
        mock_create_message = MagicMock(
            side_effect=anthropic.RateLimitError(
                "Test Rate Limit Error",
                response=MagicMock(
                    spec=httpx.Response,
                    status_code=429,
                    headers={"request-id": "mock-id"},
                ),
                body=None,
            )
        )
        self.api._client.messages.create = mock_create_message
        with patch("time.sleep") as mock_sleep, self.assertRaises(
            anthropic.RateLimitError, msg="Should re-raise RateLimitError after 3 tries"
        ):
            self.api.response_from_messages(
                [{"role": "user", "content": "rate limit test"}]
            )
        self.assertEqual(
            3, len(mock_create_message.call_args_list), "Should try 3 times"
        )
        self.assertEqual(
            2, len(mock_sleep.call_args_list), "Should sleep 2 times between tries"
        )

    def test_tools_passed_to_anthropic(self):
        mock_response_message = mock_response_message_factory()
        mock_tools = mock_tool_factory()
        self.api._client.messages.create = Mock(return_value=mock_response_message)
        self.api._response_from_messages_implementation(
            [{"role": "user", "content": "test"}],
            tools=mock_tools,
        )
        tools_in_anthropic_format = [
            ToolParam(
                input_schema=mock_tools[0]["function"]["parameters"],  # type: ignore
                name="test",
                description="test function",
            )
        ]
        self.assertEqual(
            tools_in_anthropic_format,
            self.api._client.messages.create.call_args.kwargs["tools"],
            "Tools should be passed to anthropic api in correct format",
        )

    def test_tools_not_passed_to_anthropic(self):
        mock_response_message = mock_response_message_factory()
        self.api._client.messages.create = Mock(return_value=mock_response_message)
        self.api._response_from_messages_implementation(
            [{"role": "user", "content": "test"}],
        )
        self.assertNotIn(
            "tools",
            self.api._client.messages.create.call_args.kwargs,
            "Tools should not be passed to anthropic api in correct format",
        )

    def test_tool_choice_passed_to_anthropic(self):
        mock_response_message = mock_response_message_factory()
        mock_tools = mock_tool_factory()
        self.api._client.messages.create = Mock(return_value=mock_response_message)
        self.api._response_from_messages_implementation(
            [{"role": "user", "content": "test"}],
            tools=mock_tools,
            tool_choice="required",
        )
        self.assertEqual(
            {"type": "any"},
            self.api._client.messages.create.call_args.kwargs["tool_choice"],
            "Tool choice should be passed to anthropic api in correct format",
        )

    # TODO: actual tests for actual completions
