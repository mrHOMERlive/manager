"""Claude (Anthropic) AI provider implementation."""
import base64
from typing import Optional

from anthropic import AsyncAnthropic

from src.ai.provider import AIProvider


class ClaudeProvider(AIProvider):
    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.1,
    ):
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature

    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature or self.temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        response = await self.client.messages.create(**kwargs)
        return response.content[0].text

    async def generate_with_vision(
        self, prompt: str, image_data: bytes
    ) -> str:
        b64_image = base64.b64encode(image_data).decode("utf-8")
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64_image,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        return response.content[0].text

    async def transcribe_audio(
        self, audio_data: bytes, filename: str = "audio.ogg"
    ) -> str:
        raise NotImplementedError("Claude does not support audio transcription")

    async def text_to_speech(self, text: str) -> bytes:
        raise NotImplementedError("Claude does not support text-to-speech")

    async def create_embedding(self, text: str) -> list[float]:
        raise NotImplementedError("Claude does not support embeddings")
