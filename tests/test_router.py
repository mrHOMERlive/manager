import pytest
from src.core.router import MessageRouter, MessageType


class TestMessageRouter:
    def test_detect_text_message(self):
        message = {"text": "Hello world"}
        assert MessageRouter.detect_type(message) == MessageType.TEXT

    def test_detect_photo_message(self):
        message = {"photo": [{"file_id": "abc123"}]}
        assert MessageRouter.detect_type(message) == MessageType.PHOTO

    def test_detect_voice_message(self):
        message = {"voice": {"file_id": "voice123", "duration": 5}}
        assert MessageRouter.detect_type(message) == MessageType.VOICE

    def test_detect_audio_message(self):
        message = {"audio": {"file_id": "audio123", "duration": 120}}
        assert MessageRouter.detect_type(message) == MessageType.AUDIO

    def test_detect_document_message(self):
        message = {"document": {"file_id": "doc123", "file_name": "report.pdf"}}
        assert MessageRouter.detect_type(message) == MessageType.DOCUMENT

    def test_detect_unknown_message(self):
        message = {"sticker": {"file_id": "sticker123"}}
        assert MessageRouter.detect_type(message) == MessageType.UNKNOWN

    def test_photo_takes_priority_over_text(self):
        """Photo key is checked before text, so a message with both returns PHOTO."""
        message = {"photo": [{"file_id": "abc"}], "text": "caption"}
        assert MessageRouter.detect_type(message) == MessageType.PHOTO

    def test_empty_message_is_unknown(self):
        assert MessageRouter.detect_type({}) == MessageType.UNKNOWN

    def test_message_type_values(self):
        """Verify enum string values."""
        assert MessageType.TEXT.value == "text"
        assert MessageType.PHOTO.value == "photo"
        assert MessageType.VOICE.value == "voice"
        assert MessageType.AUDIO.value == "audio"
        assert MessageType.DOCUMENT.value == "document"
        assert MessageType.UNKNOWN.value == "unknown"
