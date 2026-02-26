"""Document processor for text extraction and format detection."""

import os
from typing import Optional

from src.ai.provider import AIProvider

SUPPORTED_EXTENSIONS = {"pdf", "txt", "md", "csv", "docx"}


class DocumentProcessor:
    """Processes documents for text extraction."""

    def __init__(self, ai_provider: Optional[AIProvider] = None) -> None:
        self.ai = ai_provider

    @staticmethod
    def detect_format(filename: str) -> str:
        """Detect the document format from its filename.

        Args:
            filename: The name of the file.

        Returns:
            The file extension if supported, otherwise "unsupported".
        """
        _, ext = os.path.splitext(filename)
        ext = ext.lstrip(".").lower()
        if ext in SUPPORTED_EXTENSIONS:
            return ext
        return "unsupported"

    async def extract_text(
        self, file_data: bytes, filename: str
    ) -> str:
        """Extract text content from a document.

        Args:
            file_data: Raw file bytes.
            filename: Original filename for format detection.

        Returns:
            Extracted text content.

        Raises:
            ValueError: If the file format is unsupported.
        """
        fmt = self.detect_format(filename)
        if fmt == "unsupported":
            raise ValueError(
                f"Unsupported document format: {filename}"
            )

        if fmt == "txt" or fmt == "md" or fmt == "csv":
            return file_data.decode("utf-8")

        # For pdf and docx, delegate to AI for extraction
        if self.ai is not None:
            prompt = f"Extract all text from this {fmt} document."
            return await self.ai.generate_with_vision(prompt, file_data)

        raise ValueError(
            f"AI provider required to extract text from {fmt} files"
        )
