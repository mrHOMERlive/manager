"""SQLAlchemy models for SPRUT 3.0 memory tables."""

from datetime import datetime
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Integer, String, Text, DateTime, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class VectorMixin:
    """Mixin for tables with vector embeddings."""
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    meta: Mapped[Optional[dict]] = mapped_column("metadata", JSONB, default=dict)
    embedding = Column(Vector(1536))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class Instruction(VectorMixin, Base):
    __tablename__ = "instructions"


class AboutMe(VectorMixin, Base):
    __tablename__ = "about_me"


class Dialogue(VectorMixin, Base):
    __tablename__ = "dialogues"


class Thought(VectorMixin, Base):
    __tablename__ = "thoughts"


class Download(VectorMixin, Base):
    __tablename__ = "downloads"


class Transcription(Base):
    __tablename__ = "transcriptions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[Optional[str]] = mapped_column(Text)
    source_type: Mapped[Optional[str]] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


TABLE_MODELS = {
    "instructions": Instruction,
    "about_me": AboutMe,
    "dialogues": Dialogue,
    "thoughts": Thought,
    "downloads": Download,
    "transcriptions": Transcription,
}
