"""Application context -- initializes all components."""

from typing import Optional

from src.core.config import settings, yaml_config
from src.ai.registry import AIRegistry
from src.core.router import MessageRouter
from src.core.command_handler import CommandHandler
from src.processors.vision import VisionProcessor
from src.processors.audio import AudioProcessor
from src.processors.documents import DocumentProcessor
from src.processors.tts import TTSProcessor
from src.agents.orchestrator import Orchestrator


class AppContext:
    """Holds all initialized components for the processing pipeline."""

    def __init__(self) -> None:
        self.router = MessageRouter()
        self.command_handler = CommandHandler()
        self.ai_registry: Optional[AIRegistry] = None
        self.vision_processor: Optional[VisionProcessor] = None
        self.audio_processor: Optional[AudioProcessor] = None
        self.document_processor: Optional[DocumentProcessor] = None
        self.tts_processor: Optional[TTSProcessor] = None
        self.orchestrator: Optional[Orchestrator] = None

    def initialize_ai(self) -> None:
        """Initialize AI providers from config.

        Builds the AIRegistry and creates processor instances
        for each role that has a configured provider.
        """
        try:
            self.ai_registry = AIRegistry.from_config(yaml_config, settings)
        except Exception:
            return  # AI providers optional during dev

        # Vision processor
        try:
            vision_provider = self.ai_registry.get_provider("vision")
            self.vision_processor = VisionProcessor(vision_provider)
        except KeyError:
            pass

        # Audio / transcription processor
        try:
            transcription_provider = self.ai_registry.get_provider("transcription")
            self.audio_processor = AudioProcessor(transcription_provider)
        except KeyError:
            pass

        # Document processor (uses vision provider for pdf/docx)
        try:
            doc_provider = self.ai_registry.get_provider("vision")
            self.document_processor = DocumentProcessor(doc_provider)
        except KeyError:
            self.document_processor = DocumentProcessor()

        # TTS processor
        try:
            tts_provider = self.ai_registry.get_provider("tts")
            self.tts_processor = TTSProcessor(tts_provider)
        except KeyError:
            pass

        # Orchestrator requires a text provider and an instructions store.
        # The instructions store depends on the database, so the orchestrator
        # will be fully initialized when the DB is available.

    def initialize_orchestrator(self, instructions_store) -> None:
        """Initialize the orchestrator once a database / vector store is ready.

        Args:
            instructions_store: A store with an async ``search`` method.
        """
        if self.ai_registry is None:
            return
        try:
            text_provider = self.ai_registry.get_provider("text")
            self.orchestrator = Orchestrator(text_provider, instructions_store)
        except KeyError:
            pass


app_context = AppContext()
