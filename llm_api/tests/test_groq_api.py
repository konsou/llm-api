from unittest import TestCase
from unittest.mock import patch, Mock

import groq.types.chat.chat_completion

import llm_api.abc
from llm_api.groq_api import GroqApi


class TestGroqApi(TestCase):
    def setUp(self):
        with patch("os.getenv", Mock(return_value="mock_api_key")):
            self.api = GroqApi(
                model="mixtral-8x7b-32768", api_key_name="TEST_GROQ_API_KEY"
            )

    def test_api_key_not_set(self):
        with patch("os.getenv", Mock(return_value=None)):
            with self.assertRaises(llm_api.abc.ConfigurationError):
                GroqApi(model="test")

    def test_correct_api_key_used(self):
        def mock_getenv(key, default=None):
            if key == "GROQ_API_KEY":
                return "Test Groq API Key"
            return default

        with patch("os.getenv", mock_getenv):
            api = GroqApi(model="test")

        self.assertEqual("Test Groq API Key", api.api_key)

    def test_parse_usage(self):
        mock_completion = Mock(spec=groq.types.chat.ChatCompletion)
        mock_completion.usage = groq.types.chat.chat_completion.Usage(
            prompt_tokens=1,
            completion_tokens=2,
            total_tokens=3,
        )
        u = self.api.parse_usage(mock_completion)
        self.assertIsInstance(u, llm_api.abc.Usage)
        self.assertEqual(1, u.input_tokens)
        self.assertEqual(2, u.output_tokens)
        self.assertEqual(0, u.cost)  # Groq is free at least on 2024-04-24
        self.assertIsNone(u.tag)

    def test_parse_usage_with_tag(self):
        mock_completion = Mock(spec=groq.types.chat.ChatCompletion)
        mock_completion.usage = groq.types.chat.chat_completion.Usage(
            prompt_tokens=1,
            completion_tokens=2,
            total_tokens=3,
        )
        u = self.api.parse_usage(mock_completion, tag="test tag")
        self.assertIsInstance(u, llm_api.abc.Usage)
        self.assertEqual(1, u.input_tokens)
        self.assertEqual(2, u.output_tokens)
        self.assertEqual(0, u.cost)  # Groq is free at least on 2024-04-24
        self.assertEqual("test tag", u.tag)

    # TODO: actual tests for actual completions
