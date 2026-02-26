"""Audio processor for transcription and summarization."""

from src.ai.provider import AIProvider

# Threshold in bytes above which audio is considered "long"
LONG_AUDIO_THRESHOLD = 500 * 1024  # 500 KB


class AudioProcessor:
    """Processes audio files using AI capabilities."""

    def __init__(self, ai_provider: AIProvider) -> None:
        self.ai = ai_provider

    async def transcribe(
        self, audio_data: bytes, filename: str = "audio.ogg"
    ) -> str:
        """Transcribe audio data to text.

        Args:
            audio_data: Raw audio bytes.
            filename: Original filename for format detection.

        Returns:
            Transcribed text.
        """
        return await self.ai.transcribe_audio(audio_data, filename=filename)

    async def summarize(self, text: str) -> str:
        """Summarize a transcription text.

        Args:
            text: The transcribed text to summarize.

        Returns:
            A summary of the text.
        """
        prompt = (
            "Summarize the following transcription concisely:\n\n" + text
        )
        return await self.ai.generate_text(prompt)

    def is_long_audio(self, size_bytes: int) -> bool:
        """Check whether an audio file exceeds the long-audio threshold.

        Args:
            size_bytes: Size of the audio file in bytes.

        Returns:
            True if the audio is considered long.
        """
        return size_bytes > LONG_AUDIO_THRESHOLD
