"""Text formatter for cleaning up dictated text."""

from src.ai.provider import AIProvider


class TextFormatter:
    """Formats raw dictated text for social media and messaging."""

    def __init__(self, ai_provider: AIProvider) -> None:
        self.ai = ai_provider

    async def format_for_social(self, raw_text: str) -> str:
        """Clean up dictated text for social media posting.

        Uses AI to remove filler words, fix grammar, and produce
        clean text suitable for social media.

        Args:
            raw_text: Raw dictated text with potential filler words.

        Returns:
            Clean, formatted text.
        """
        prompt = (
            "Clean up the following dictated text for social media. "
            "Remove filler words (um, uh, like, you know), fix grammar "
            "and punctuation, and make it concise and clear. "
            "Return only the cleaned text, nothing else.\n\n"
            f"{raw_text}"
        )
        return await self.ai.generate_text(prompt)
