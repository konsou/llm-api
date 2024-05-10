from unittest import TestCase
from unittest.mock import patch, Mock

import llm_api
from llm_api.ollama_api import OllamaApi


class TestOllamaApi(TestCase):
    def setUp(self):
        with patch("os.getenv", Mock(return_value="mock_api_key")):
            self.api = OllamaApi(
                model="mixtral-8x7b-32768",
                api_key_name="TEST_OLLAMA_API_KEY",
                base_url="http://localhost",
            )

    def test_api_key_not_set(self):
        with patch("os.getenv", Mock(return_value=None)):
            with self.assertRaises(llm_api.abc.ConfigurationError):
                OllamaApi(model="test", base_url="http://test.com")

    def test_correct_api_key_used(self):
        def mock_getenv(key, default=None):
            if key == "OLLAMA_API_KEY":
                return "Test Ollama API Key"
            return default

        with patch("os.getenv", mock_getenv):
            api = OllamaApi(model="test", base_url="http://test.com")

        self.assertEqual("Test Ollama API Key", api.api_key)

    def test_base_url_required(self):
        with (
            patch("os.getenv", Mock(return_value="mock_api_key")),
            self.assertRaises(
                llm_api.abc.ConfigurationError, msg="Should require base_url to be set"
            ),
        ):
            OllamaApi(model="test")

    # def test__response_from_messages_implementation(self):
    #     TODO: this
