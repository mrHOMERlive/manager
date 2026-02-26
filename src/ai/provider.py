"""Base AI provider interface."""
from abc import ABC, abstractmethod
from typing import Optional


class AIProvider(ABC):
    @abstractmethod
    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str: ...

    @abstractmethod
    async def generate_with_vision(
        self, prompt: str, image_data: bytes
    ) -> str: ...

    @abstractmethod
    async def transcribe_audio(
        self, audio_data: bytes, filename: str = "audio.ogg"
    ) -> str: ...

    @abstractmethod
    async def text_to_speech(self, text: str) -> bytes: ...

    @abstractmethod
    async def create_embedding(self, text: str) -> list[float]: ...
