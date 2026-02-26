"""Google Gemini AI provider implementation."""
from typing import Optional

import google.generativeai as genai

from src.ai.provider import AIProvider


class GeminiProvider(AIProvider):
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.1,
    ):
        genai.configure(api_key=api_key)
        self.model_name = model
        self.temperature = temperature

    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_prompt,
        )
        response = await model.generate_content_async(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature or self.temperature,
            ),
        )
        return response.text

    async def generate_with_vision(
        self, prompt: str, image_data: bytes
    ) -> str:
        model = genai.GenerativeModel(model_name=self.model_name)
        image_part = {"mime_type": "image/jpeg", "data": image_data}
        response = await model.generate_content_async([prompt, image_part])
        return response.text

    async def transcribe_audio(
        self, audio_data: bytes, filename: str = "audio.ogg"
    ) -> str:
        raise NotImplementedError("Gemini does not support audio transcription")

    async def text_to_speech(self, text: str) -> bytes:
        raise NotImplementedError("Gemini does not support text-to-speech")

    async def create_embedding(self, text: str) -> list[float]:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
        )
        return result["embedding"]
