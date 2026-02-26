"""Tests for text formatter."""

import pytest
from unittest.mock import AsyncMock

from src.utils.text_formatter import TextFormatter


@pytest.mark.asyncio
async def test_format_for_social():
    ai = AsyncMock()
    ai.generate_text = AsyncMock(return_value="Clean formatted text")
    formatter = TextFormatter(ai_provider=ai)
    result = await formatter.format_for_social("messy uhh text")
    assert result == "Clean formatted text"
