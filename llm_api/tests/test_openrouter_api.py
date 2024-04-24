from unittest import TestCase
from unittest.mock import patch, Mock

from llm_api import OpenRouterAPI, types_request
from llm_api.abc import Usage, ResponseAndUsage


class TestOpenRouterAPI(TestCase):
    def setUp(self):
        with patch("os.getenv", Mock(return_value="mock_api_key")):
            self.api = OpenRouterAPI(
                model="gpt-3", api_key_name="TEST_OPENROUTER_API_KEY"
            )

    def test_initialization(self):
        with patch("os.getenv", Mock(return_value="mock_api_key")):
            self.api = OpenRouterAPI(
                model="gpt-3", api_key_name="TEST_OPENROUTER_API_KEY", timeout=3
            )
        self.assertEqual("gpt-3", self.api.model)
        self.assertEqual("TEST_OPENROUTER_API_KEY", self.api.api_key_name)
        self.assertEqual("mock_api_key", self.api.api_key)
        self.assertEqual(3, self.api.timeout)

    @patch("requests.post")
    def test__response_from_messages_implementation(self, mock_post):
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "response message"}}]
        }
        mock_post.return_value = mock_response

        messages = [types_request.Message(role="user", content="Hello")]
        result = self.api._response_from_messages_implementation(messages)

        self.assertEqual(result, ResponseAndUsage("response message", Usage(0, 0, 0)))
        mock_post.assert_called_once_with(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api.api_key}"},
            data=f'{{"model": "{self.api.model}", "messages": [{{"role": "user", "content": "Hello"}}]}}',
        )

    def test_parse_usage(self):
        response = {
            "usage": {
                "completion_tokens": 1,
                "prompt_tokens": 2,
                "total_tokens": 3,
                "total_cost": 4.5,
            }
        }
        u = self.api.parse_usage(response)
        self.assertIsInstance(u, Usage)
        self.assertEqual(1, u.output_tokens)
        self.assertEqual(2, u.input_tokens)
        self.assertEqual(4.5, u.cost)
        self.assertIsNone(u.tag)

    def test_parse_usage_with_tag(self):
        response = {
            "usage": {
                "completion_tokens": 2,
                "prompt_tokens": 3,
                "total_tokens": 4,
                "total_cost": 5,
            }
        }
        u = self.api.parse_usage(response, tag="test_tag")
        self.assertIsInstance(u, Usage)
        self.assertEqual(2, u.output_tokens)
        self.assertEqual(3, u.input_tokens)
        self.assertEqual(5, u.cost)
        self.assertEqual("test_tag", u.tag)
