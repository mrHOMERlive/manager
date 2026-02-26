"""API routes for SPRUT 3.0."""

import base64
import logging

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from typing import Optional

from src.core.config import settings
from src.core.app_context import app_context
from src.core.router import MessageType
from src.core.command_handler import Command
from src.utils.chunker import chunk_text

logger = logging.getLogger(__name__)

router = APIRouter()


class ProcessRequest(BaseModel):
    message: dict
    user_id: Optional[int] = None


class ProcessResponse(BaseModel):
    text: str
    voice_data: Optional[str] = None
    chunks: Optional[list[str]] = None


@router.get("/health")
async def health():
    return {"status": "healthy", "service": "sprut-api", "version": "3.0.0"}


@router.post("/api/process", response_model=ProcessResponse)
async def process_message(
    request: ProcessRequest,
    authorization: str = Header(default=""),
):
    # --- Auth check -----------------------------------------------------------
    expected = f"Bearer {settings.api_secret_key}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if request.user_id and request.user_id not in settings.allowed_user_id_list:
        raise HTTPException(status_code=403, detail="User not authorized")

    # --- Detect message type --------------------------------------------------
    msg = request.message
    msg_type = app_context.router.detect_type(msg)

    # --- Route to appropriate pipeline ----------------------------------------
    if msg_type == MessageType.TEXT:
        return await _handle_text(msg["text"])

    if msg_type == MessageType.PHOTO:
        return await _handle_photo(msg)

    if msg_type in (MessageType.VOICE, MessageType.AUDIO):
        return await _handle_audio(msg, msg_type)

    if msg_type == MessageType.DOCUMENT:
        return await _handle_document(msg)

    # Unknown type
    return ProcessResponse(text="Unsupported message type.")


# ---------------------------------------------------------------------------
# Pipeline handlers
# ---------------------------------------------------------------------------

async def _handle_text(text: str) -> ProcessResponse:
    """Handle a plain-text message: detect command or delegate to orchestrator."""
    command = app_context.command_handler.detect(text)

    if command != Command.NONE:
        payload = app_context.command_handler.extract_payload(text, command)
        return ProcessResponse(
            text=f"Command received: {command.value}. Payload: {payload}"
        )

    # No command -- delegate to orchestrator if available
    if app_context.orchestrator is not None:
        result = await app_context.orchestrator.process(text)
        voice_data = None
        if result.get("voice"):
            voice_data = await _generate_voice(result["text"])
        return ProcessResponse(text=result["text"], voice_data=voice_data)

    return ProcessResponse(text=f"Echo: {text}")


async def _handle_photo(msg: dict) -> ProcessResponse:
    """Handle a photo message: describe via vision, then pass to orchestrator."""
    if app_context.vision_processor is None:
        return ProcessResponse(text="Vision processing is not configured.")

    photo_data = _decode_file_data(msg["photo"])
    caption = msg.get("caption")
    description = await app_context.vision_processor.describe(photo_data, prompt=caption)

    # Pass the image description through the orchestrator
    if app_context.orchestrator is not None:
        prompt = f"The user sent a photo. Description: {description}"
        if caption:
            prompt += f"\nUser caption: {caption}"
        result = await app_context.orchestrator.process(prompt)
        voice_data = None
        if result.get("voice"):
            voice_data = await _generate_voice(result["text"])
        return ProcessResponse(text=result["text"], voice_data=voice_data)

    return ProcessResponse(text=description)


async def _handle_audio(msg: dict, msg_type: MessageType) -> ProcessResponse:
    """Handle voice/audio: transcribe, then route through command detection."""
    if app_context.audio_processor is None:
        return ProcessResponse(text="Audio processing is not configured.")

    key = "voice" if msg_type == MessageType.VOICE else "audio"
    audio_data = _decode_file_data(msg[key])
    filename = msg.get("filename", "audio.ogg")

    transcription = await app_context.audio_processor.transcribe(audio_data, filename=filename)

    # Route the transcribed text through the command handler
    command = app_context.command_handler.detect(transcription)
    if command != Command.NONE:
        payload = app_context.command_handler.extract_payload(transcription, command)
        return ProcessResponse(
            text=f"Command received: {command.value}. Payload: {payload}"
        )

    # No command -- pass to orchestrator
    if app_context.orchestrator is not None:
        result = await app_context.orchestrator.process(transcription)
        voice_data = None
        if result.get("voice"):
            voice_data = await _generate_voice(result["text"])
        return ProcessResponse(text=result["text"], voice_data=voice_data)

    return ProcessResponse(text=f"Transcription: {transcription}")


async def _handle_document(msg: dict) -> ProcessResponse:
    """Handle document: extract text and return chunks."""
    if app_context.document_processor is None:
        # Fallback: create a processor without AI for plain text files
        from src.processors.documents import DocumentProcessor
        processor = DocumentProcessor()
    else:
        processor = app_context.document_processor

    file_data = _decode_file_data(msg["document"])
    filename = msg.get("filename", "document.txt")

    try:
        extracted = await processor.extract_text(file_data, filename)
    except ValueError as exc:
        return ProcessResponse(text=str(exc))

    chunks = chunk_text(extracted)
    return ProcessResponse(
        text=f"Document processed: {filename} ({len(chunks)} chunk(s))",
        chunks=chunks,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_file_data(data) -> bytes:
    """Decode file data from a message field.

    Accepts raw bytes, a base64 string, or a dict with a ``data`` key.
    """
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return base64.b64decode(data)
    if isinstance(data, dict) and "data" in data:
        return base64.b64decode(data["data"])
    raise HTTPException(status_code=400, detail="Invalid file data format")


async def _generate_voice(text: str) -> Optional[str]:
    """Generate TTS audio and return as a base64 string, or None."""
    if app_context.tts_processor is None:
        return None
    try:
        audio_bytes = await app_context.tts_processor.generate(text)
        return base64.b64encode(audio_bytes).decode("utf-8")
    except Exception:
        logger.warning("TTS generation failed", exc_info=True)
        return None
