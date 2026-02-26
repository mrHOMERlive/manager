"""Integration tests for the full message processing pipeline."""

import base64
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from httpx import AsyncClient, ASGITransport

from src.main import app
from src.core.config import settings


AUTH_HEADER = f"Bearer {settings.api_secret_key}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _b64(text: str) -> str:
    """Encode a string as base64."""
    return base64.b64encode(text.encode()).decode()


# ---------------------------------------------------------------------------
# Text message pipeline
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_text_message_echo():
    """Text messages without orchestrator return an echo."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/process",
            json={"message": {"text": "hello world"}},
            headers={"Authorization": AUTH_HEADER},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["text"] == "Echo: hello world"
    assert data["voice_data"] is None
    assert data["chunks"] is None


@pytest.mark.asyncio
async def test_text_command_detected():
    """Text starting with a trigger routes to the command handler."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/process",
            json={"message": {"text": "запомни купить молоко"}},
            headers={"Authorization": AUTH_HEADER},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "Command received: remember" in data["text"]
    assert "купить молоко" in data["text"]


@pytest.mark.asyncio
async def test_text_with_orchestrator():
    """When the orchestrator is wired, text goes through it."""
    mock_orchestrator = MagicMock()
    mock_orchestrator.process = AsyncMock(return_value={"text": "AI response", "voice": None})

    with patch("src.api.routes.app_context") as ctx:
        ctx.router.detect_type.return_value = __import__(
            "src.core.router", fromlist=["MessageType"]
        ).MessageType.TEXT
        ctx.command_handler.detect.return_value = __import__(
            "src.core.command_handler", fromlist=["Command"]
        ).Command.NONE
        ctx.orchestrator = mock_orchestrator
        ctx.tts_processor = None

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/process",
                json={"message": {"text": "what is the weather?"}},
                headers={"Authorization": AUTH_HEADER},
            )
    assert resp.status_code == 200
    assert resp.json()["text"] == "AI response"


# ---------------------------------------------------------------------------
# Auth checks
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_missing_auth_returns_401():
    """Requests without auth header are rejected."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/process",
            json={"message": {"text": "test"}},
        )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_wrong_auth_returns_401():
    """Requests with incorrect auth header are rejected."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/process",
            json={"message": {"text": "test"}},
            headers={"Authorization": "Bearer wrong-key"},
        )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Photo pipeline
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_photo_without_vision_processor():
    """Photo messages without a vision processor return a config message."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/process",
            json={"message": {"photo": _b64("fake-image-data")}},
            headers={"Authorization": AUTH_HEADER},
        )
    assert resp.status_code == 200
    assert "not configured" in resp.json()["text"].lower()


@pytest.mark.asyncio
async def test_photo_with_vision_processor():
    """Photo messages with a vision processor get described."""
    mock_vision = MagicMock()
    mock_vision.describe = AsyncMock(return_value="A cat sitting on a table")

    with patch("src.api.routes.app_context") as ctx:
        ctx.router.detect_type.return_value = __import__(
            "src.core.router", fromlist=["MessageType"]
        ).MessageType.PHOTO
        ctx.vision_processor = mock_vision
        ctx.orchestrator = None

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/process",
                json={"message": {"photo": _b64("fake-image")}},
                headers={"Authorization": AUTH_HEADER},
            )
    assert resp.status_code == 200
    assert resp.json()["text"] == "A cat sitting on a table"


# ---------------------------------------------------------------------------
# Voice / audio pipeline
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_voice_without_audio_processor():
    """Voice messages without an audio processor return a config message."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/process",
            json={"message": {"voice": _b64("fake-voice")}},
            headers={"Authorization": AUTH_HEADER},
        )
    assert resp.status_code == 200
    assert "not configured" in resp.json()["text"].lower()


@pytest.mark.asyncio
async def test_voice_transcription_echo():
    """Voice message is transcribed and echoed when no orchestrator."""
    mock_audio = MagicMock()
    mock_audio.transcribe = AsyncMock(return_value="hello from voice")

    with patch("src.api.routes.app_context") as ctx:
        ctx.router.detect_type.return_value = __import__(
            "src.core.router", fromlist=["MessageType"]
        ).MessageType.VOICE
        ctx.audio_processor = mock_audio
        ctx.command_handler.detect.return_value = __import__(
            "src.core.command_handler", fromlist=["Command"]
        ).Command.NONE
        ctx.orchestrator = None

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/process",
                json={"message": {"voice": _b64("audio-bytes")}},
                headers={"Authorization": AUTH_HEADER},
            )
    assert resp.status_code == 200
    assert resp.json()["text"] == "Transcription: hello from voice"


@pytest.mark.asyncio
async def test_voice_transcription_command():
    """Voice message transcribed to a command is handled as a command."""
    mock_audio = MagicMock()
    mock_audio.transcribe = AsyncMock(return_value="запомни позвонить врачу")

    with patch("src.api.routes.app_context") as ctx:
        ctx.router.detect_type.return_value = __import__(
            "src.core.router", fromlist=["MessageType"]
        ).MessageType.VOICE
        ctx.audio_processor = mock_audio
        ctx.command_handler.detect.return_value = __import__(
            "src.core.command_handler", fromlist=["Command"]
        ).Command.REMEMBER
        ctx.command_handler.extract_payload.return_value = "позвонить врачу"
        ctx.orchestrator = None

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/process",
                json={"message": {"voice": _b64("audio-bytes")}},
                headers={"Authorization": AUTH_HEADER},
            )
    assert resp.status_code == 200
    data = resp.json()
    assert "Command received: remember" in data["text"]


# ---------------------------------------------------------------------------
# Document pipeline
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_document_text_file():
    """Text documents are processed and returned as chunks."""
    content = "Hello, this is a test document with some content."
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/process",
            json={
                "message": {
                    "document": _b64(content),
                    "filename": "notes.txt",
                },
            },
            headers={"Authorization": AUTH_HEADER},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "Document processed" in data["text"]
    assert "1 chunk" in data["text"]
    assert data["chunks"] == [content]


@pytest.mark.asyncio
async def test_document_unsupported_format():
    """Unsupported document formats return an error message."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/process",
            json={
                "message": {
                    "document": _b64("binary"),
                    "filename": "file.exe",
                },
            },
            headers={"Authorization": AUTH_HEADER},
        )
    assert resp.status_code == 200
    assert "unsupported" in resp.json()["text"].lower()


# ---------------------------------------------------------------------------
# Unknown message type
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unknown_message_type():
    """Unknown message types return an appropriate response."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/process",
            json={"message": {"sticker": "some_sticker_id"}},
            headers={"Authorization": AUTH_HEADER},
        )
    assert resp.status_code == 200
    assert "unsupported" in resp.json()["text"].lower()
