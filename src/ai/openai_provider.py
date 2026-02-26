"""OpenAI AI provider implementation."""
import io
from typing import Optional
import base64

from openai import AsyncOpenAI

from src.ai.provider import AIProvider


class OpenAIProvider(AIProvider):
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        base_url: Optional[str] = None,
    ):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature

    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.temperature,
        )
        return response.choices[0].message.content

    async def generate_with_vision(
        self, prompt: str, image_data: bytes
    ) -> str:
        b64_image = base64.b64encode(image_data).decode("utf-8")
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_image}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
        )
        return response.choices[0].message.content

    async def transcribe_audio(
        self, audio_data: bytes, filename: str = "audio.ogg"
    ) -> str:
        audio_file = io.BytesIO(audio_data)
        audio_file.name = filename
        response = await self.client.audio.transcriptions.create(
            model="whisper-1", file=audio_file
        )
        return response.text

    async def text_to_speech(self, text: str) -> bytes:
        response = await self.client.audio.speech.create(
            model="tts-1", voice="alloy", input=text
        )
        return response.content

    async def create_embedding(self, text: str) -> list[float]:
        response = await self.client.embeddings.create(
            model="text-embedding-3-small", input=text
        )
        return response.data[0].embedding
