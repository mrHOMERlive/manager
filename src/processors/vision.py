"""Vision processor for image analysis."""

from typing import Optional

from src.ai.provider import AIProvider


class VisionProcessor:
    """Processes images using AI vision capabilities."""

    def __init__(self, ai_provider: AIProvider) -> None:
        self.ai = ai_provider

    async def describe(
        self, image_data: bytes, prompt: Optional[str] = None
    ) -> str:
        """Describe the contents of an image.

        Args:
            image_data: Raw image bytes.
            prompt: Optional prompt to guide the description.

        Returns:
            A text description of the image.
        """
        if prompt is None:
            prompt = "Describe this image in detail."
        return await self.ai.generate_with_vision(prompt, image_data)
