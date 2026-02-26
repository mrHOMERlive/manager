"""Tests for all processors."""

import pytest
from unittest.mock import AsyncMock

from src.processors.vision import VisionProcessor
from src.processors.audio import AudioProcessor
from src.processors.documents import DocumentProcessor
from src.processors.tts import TTSProcessor


@pytest.mark.asyncio
async def test_vision_describe():
    ai = AsyncMock()
    ai.generate_with_vision = AsyncMock(return_value="A photo of a cat")
    proc = VisionProcessor(ai_provider=ai)
    result = await proc.describe(b"fake_image")
    assert result == "A photo of a cat"


@pytest.mark.asyncio
async def test_audio_transcribe():
    ai = AsyncMock()
    ai.transcribe_audio = AsyncMock(return_value="Hello world")
    proc = AudioProcessor(ai_provider=ai)
    result = await proc.transcribe(b"fake_audio", filename="voice.ogg")
    assert result == "Hello world"


@pytest.mark.asyncio
async def test_audio_is_long():
    proc = AudioProcessor(ai_provider=AsyncMock())
    assert proc.is_long_audio(600 * 1024) is True
    assert proc.is_long_audio(400 * 1024) is False


def test_document_detect_format():
    assert DocumentProcessor.detect_format("report.pdf") == "pdf"
    assert DocumentProcessor.detect_format("notes.md") == "md"
    assert DocumentProcessor.detect_format("data.csv") == "csv"
    assert DocumentProcessor.detect_format("photo.jpg") == "unsupported"


@pytest.mark.asyncio
async def test_tts_generate():
    ai = AsyncMock()
    ai.text_to_speech = AsyncMock(return_value=b"audio_bytes")
    proc = TTSProcessor(ai_provider=ai)
    result = await proc.generate("Hello")
    assert result == b"audio_bytes"
