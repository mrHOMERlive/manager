from enum import Enum


class MessageType(str, Enum):
    TEXT = "text"
    PHOTO = "photo"
    VOICE = "voice"
    AUDIO = "audio"
    DOCUMENT = "document"
    UNKNOWN = "unknown"


class MessageRouter:
    @staticmethod
    def detect_type(message: dict) -> MessageType:
        if "photo" in message:
            return MessageType.PHOTO
        if "voice" in message:
            return MessageType.VOICE
        if "audio" in message:
            return MessageType.AUDIO
        if "document" in message:
            return MessageType.DOCUMENT
        if "text" in message:
            return MessageType.TEXT
        return MessageType.UNKNOWN
