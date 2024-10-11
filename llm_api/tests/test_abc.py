from unittest import TestCase
from unittest.mock import Mock, patch

from llm_api import LlmApi
from llm_api.abc import ResponseAndUsage, ConfigurationError, Usage


class MockLlmApiSubclass(LlmApi):
    def __init__(
        self,
        model: str,
        api_key_name: str,
        timeout: int = 5,
    ):
        super().__init__(model=model, api_key_name=api_key_name, timeout=timeout)
        self._response_from_messages_implementation = Mock(
            name="_response_from_messages_implementation",
            return_value=ResponseAndUsage("Test response", Usage(1, 2, 3)),
        )


# Allow instantiating without implementing all abstract methods
MockLlmApiSubclass.__abstractmethods__ = frozenset()


class TestLlmApi(TestCase):
    def setUp(self):
        with patch("os.getenv", Mock(return_value="mock_api_key")):
            self.api = MockLlmApiSubclass(model="gpt-3", api_key_name="apikey")

    def test_response_from_messages(self):
        response = self.api.response_from_messages(
            [{"role": "user", "content": "Hello"}]
        )
        self.assertEqual("Test response", response)

    def test__response_from_messages_implementation_called(self):
        messages = [{"role": "user", "content": "Hello"}]
        self.api.response_from_messages(messages)
        self.api._response_from_messages_implementation.assert_called_once_with(
            messages=messages,
            tools=None,
            tool_choice="auto",
            tag=None,
            response_format=None,
        )

    def test__response_from_messages_implementation_called_with_response_format(self):
        messages = [{"role": "user", "content": "Hello"}]
        self.api.response_from_messages(messages, response_format="json")
        self.api._response_from_messages_implementation.assert_called_once_with(
            messages=messages,
            tools=None,
            tool_choice="auto",
            tag=None,
            response_format="json",
        )

    def test__response_from_messages_implementation_called_with_tool_choice(self):
        messages = [{"role": "user", "content": "Hello"}]
        self.api.response_from_messages(messages, tool_choice="none")
        self.api._response_from_messages_implementation.assert_called_once_with(
            messages=messages,
            tools=None,
            tool_choice="none",
            tag=None,
            response_format=None,
        )

    def test_handle_usage_once(self):
        self.api.handle_usage(input_tokens=1, output_tokens=2, cost=3)
        self.assertEqual(3, self.api.total_cost)

    def test_handle_usage_twice(self):
        self.api.handle_usage(input_tokens=1, output_tokens=2, cost=3)
        self.api.handle_usage(input_tokens=1, output_tokens=2, cost=3)
        self.assertEqual(6, self.api.total_cost)

    def test_handle_usage_no_cost(self):
        self.api.handle_usage(input_tokens=1, output_tokens=2)
        self.assertEqual(0, self.api.total_cost)

    def test_response_from_prompt(self):
        self.api.response_from_messages = Mock(
            name="response_from_messages", return_value="Test response"
        )
        result = self.api.response_from_prompt("Test prompt")
        self.api.response_from_messages.assert_called_once_with(
            [{"role": "user", "content": "Test prompt"}],
            tag=None,
            response_format=None,
        )
        self.assertEqual("Test response", result)

    def test_response_from_prompt_passes_response_format(self):
        self.api.response_from_messages = Mock(name="response_from_messages")
        self.api.response_from_prompt("Test prompt", response_format="json")
        self.api.response_from_messages.assert_called_once_with(
            [{"role": "user", "content": "Test prompt"}],
            tag=None,
            response_format="json",
        )

    def test_configuration_error_raised_when_no_api_key_set(self):
        with self.assertRaises(ConfigurationError):
            MockLlmApiSubclass(
                model="gpt-3", api_key_name="API_KEY_THAT_DOES_NOT_EXIST"
            )

    @patch("builtins.print")
    def test_handle_usage_no_printing(self, mock_print):
        self.api.handle_usage(input_tokens=1, output_tokens=2, cost=3)
        mock_print.assert_not_called()
