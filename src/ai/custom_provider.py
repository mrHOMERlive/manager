"""Custom OpenAI-compatible provider for remote servers."""
from typing import Optional

from src.ai.openai_provider import OpenAIProvider


class CustomProvider(OpenAIProvider):
    """Wraps OpenAIProvider with a custom base_url for OpenAI-compatible APIs."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "default",
        temperature: float = 0.1,
    ):
        super().__init__(
            api_key=api_key,
            model=model,
            temperature=temperature,
            base_url=base_url,
        )
