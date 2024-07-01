import logging

from .abc import LlmApi
from .anthropic_api import AnthropicApi
from .groq_api import GroqApi
from .ollama_api import OllamaApi
from .openrouter_api import OpenRouterAPI

logger = logging.getLogger("llm_api")
logger.setLevel(logging.DEBUG)

logger.debug("Initialized")
