"""Text-to-speech processor."""

from src.ai.provider import AIProvider


class TTSProcessor:
    """Converts text to speech audio using AI capabilities."""

    def __init__(self, ai_provider: AIProvider) -> None:
        self.ai = ai_provider

    async def generate(self, text: str) -> bytes:
        """Generate speech audio from text.

        Args:
            text: The text to convert to speech.

        Returns:
            Raw audio bytes.
        """
        return await self.ai.text_to_speech(text)
