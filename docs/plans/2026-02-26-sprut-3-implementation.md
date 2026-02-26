# SPRUT 3.0 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a personal AI assistant (SPRUT 3.0) with Python FastAPI core + n8n orchestrator, multi-provider AI, PostgreSQL + pgvector memory, Telegram bot interface, and modular sub-agents.

**Architecture:** Python FastAPI service handles all business logic (message routing, AI calls, DB operations, file processing) exposed via REST API. n8n handles Telegram webhook, Google Drive/Obsidian sync. PostgreSQL + pgvector stores vector embeddings for semantic search across 5 knowledge databases. Docker Compose deploys everything.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy + asyncpg, pgvector, OpenAI/Anthropic/Google AI SDKs, Docker Compose, n8n, PostgreSQL 16, pytest + pytest-asyncio

**Design doc:** `docs/plans/2026-02-26-sprut-3-design.md`

---

## Phase 1: Project Scaffolding & Infrastructure

### Task 1: Project skeleton and dependencies

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `config.yaml`
- Create: `src/__init__.py`
- Create: `src/core/__init__.py`
- Create: `src/core/config.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

**Step 1: Create .gitignore**

```gitignore
__pycache__/
*.pyc
.env
config.yaml
*.egg-info/
.venv/
.pytest_cache/
dist/
build/
*.db
node_modules/
```

**Step 2: Create requirements.txt**

```txt
fastapi==0.115.0
uvicorn[standard]==0.30.0
sqlalchemy[asyncio]==2.0.35
asyncpg==0.30.0
pgvector==0.3.5
pydantic==2.9.0
pydantic-settings==2.5.0
pyyaml==6.0.2
httpx==0.27.0
openai==1.50.0
anthropic==0.34.0
google-generativeai==0.8.0
python-multipart==0.0.9
aiofiles==24.1.0

# Testing
pytest==8.3.0
pytest-asyncio==0.24.0
pytest-cov==5.0.0
```

**Step 3: Create .env.example**

```env
# Telegram
TELEGRAM_BOT_TOKEN=your_bot_token
ALLOWED_USER_IDS=123456789,987654321

# Database
DATABASE_URL=postgresql+asyncpg://sprut:sprut_password@postgres:5432/sprut

# AI Providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
CUSTOM_LLM_URL=http://your-server:8080
CUSTOM_LLM_KEY=...

# Internal API
API_SECRET_KEY=your_internal_secret

# n8n
N8N_WEBHOOK_URL=http://n8n:5678
```

**Step 4: Create config.yaml (template — will be .gitignored)**

```yaml
ai:
  default_text: "openai"
  default_vision: "openai"
  default_transcription: "openai"
  default_tts: "openai"
  default_embeddings: "openai"

  providers:
    openai:
      model: "gpt-4o-mini"
      temperature: 0.1
    claude:
      model: "claude-sonnet-4-20250514"
      temperature: 0.1
    gemini:
      model: "gemini-2.0-flash"
    custom:
      model: "default"

telegram:
  chunk_size: 3500
  audio_size_threshold_kb: 500

memory:
  embedding_dimensions: 1536
  search_top_k: 5
```

**Step 5: Create src/core/config.py**

```python
"""Configuration loader for SPRUT 3.0"""

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings from environment variables."""

    # Database
    database_url: str = Field(default="postgresql+asyncpg://sprut:sprut_password@localhost:5432/sprut")

    # Telegram
    telegram_bot_token: str = Field(default="")
    allowed_user_ids: str = Field(default="")

    # AI Keys
    openai_api_key: str = Field(default="")
    anthropic_api_key: str = Field(default="")
    gemini_api_key: str = Field(default="")
    custom_llm_url: str = Field(default="")
    custom_llm_key: str = Field(default="")

    # Internal
    api_secret_key: str = Field(default="dev-secret")

    model_config = {"env_file": ".env", "extra": "ignore"}

    @property
    def allowed_user_id_list(self) -> list[int]:
        if not self.allowed_user_ids:
            return []
        return [int(uid.strip()) for uid in self.allowed_user_ids.split(",") if uid.strip()]


def load_yaml_config(path: Optional[str] = None) -> dict:
    """Load YAML configuration file."""
    config_path = Path(path) if path else Path("config.yaml")
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


settings = Settings()
yaml_config = load_yaml_config()
```

**Step 6: Create test conftest and package inits**

```python
# tests/conftest.py
import pytest


@pytest.fixture
def yaml_config():
    """Sample YAML config for testing."""
    return {
        "ai": {
            "default_text": "openai",
            "default_vision": "openai",
            "default_transcription": "openai",
            "default_tts": "openai",
            "default_embeddings": "openai",
            "providers": {
                "openai": {"model": "gpt-4o-mini", "temperature": 0.1},
            },
        },
        "telegram": {"chunk_size": 3500, "audio_size_threshold_kb": 500},
        "memory": {"embedding_dimensions": 1536, "search_top_k": 5},
    }
```

Empty `__init__.py` files for: `src/`, `src/core/`, `tests/`.

**Step 7: Write test for config loader**

```python
# tests/test_config.py
from src.core.config import Settings, load_yaml_config


def test_settings_defaults():
    s = Settings(database_url="postgresql+asyncpg://test:test@localhost/test")
    assert "localhost" in s.database_url


def test_allowed_user_ids_parsing():
    s = Settings(allowed_user_ids="123,456,789")
    assert s.allowed_user_id_list == [123, 456, 789]


def test_allowed_user_ids_empty():
    s = Settings(allowed_user_ids="")
    assert s.allowed_user_id_list == []
```

**Step 8: Run tests**

Run: `pytest tests/test_config.py -v`
Expected: PASS

**Step 9: Commit**

```bash
git add .gitignore requirements.txt .env.example config.yaml src/ tests/
git commit -m "feat: project skeleton with config loader"
```

---

### Task 2: Docker Compose and database init

**Files:**
- Create: `docker-compose.yml`
- Create: `Dockerfile`
- Create: `db/init.sql`

**Step 1: Create db/init.sql**

```sql
CREATE EXTENSION IF NOT EXISTS vector;

-- Instructions: agent behavioral rules
CREATE TABLE IF NOT EXISTS instructions (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding vector(1536),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- About Me: personal facts
CREATE TABLE IF NOT EXISTS about_me (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding vector(1536),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Dialogues: conversation history
CREATE TABLE IF NOT EXISTS dialogues (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding vector(1536),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Thoughts: transcribed voice memos, notes
CREATE TABLE IF NOT EXISTS thoughts (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding vector(1536),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Downloads: extracted text from uploaded documents
CREATE TABLE IF NOT EXISTS downloads (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding vector(1536),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Transcriptions: raw audio transcripts (non-vector)
CREATE TABLE IF NOT EXISTS transcriptions (
    id SERIAL PRIMARY KEY,
    raw_text TEXT NOT NULL,
    summary TEXT,
    source_type VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for vector similarity search
CREATE INDEX IF NOT EXISTS idx_instructions_embedding ON instructions USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_about_me_embedding ON about_me USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_dialogues_embedding ON dialogues USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_thoughts_embedding ON thoughts USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_downloads_embedding ON downloads USING ivfflat (embedding vector_cosine_ops);
```

**Step 2: Create Dockerfile**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY config.yaml .

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Step 3: Create docker-compose.yml**

```yaml
version: "3.8"

services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: sprut
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-sprut_password}
      POSTGRES_DB: sprut
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./db/init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U sprut"]
      interval: 5s
      timeout: 5s
      retries: 5

  sprut-api:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - ./config.yaml:/app/config.yaml:ro
    depends_on:
      postgres:
        condition: service_healthy

  n8n:
    image: n8nio/n8n
    ports:
      - "5678:5678"
    environment:
      - N8N_BASIC_AUTH_ACTIVE=true
      - N8N_BASIC_AUTH_USER=${N8N_USER:-admin}
      - N8N_BASIC_AUTH_PASSWORD=${N8N_PASSWORD:-admin}
      - DB_TYPE=postgresdb
      - DB_POSTGRESDB_HOST=postgres
      - DB_POSTGRESDB_PORT=5432
      - DB_POSTGRESDB_DATABASE=sprut
      - DB_POSTGRESDB_USER=sprut
      - DB_POSTGRESDB_PASSWORD=${POSTGRES_PASSWORD:-sprut_password}
    volumes:
      - n8n_data:/home/node/.n8n
    depends_on:
      postgres:
        condition: service_healthy

  pgadmin:
    image: dpage/pgadmin4
    ports:
      - "5050:80"
    environment:
      PGADMIN_DEFAULT_EMAIL: ${PGADMIN_EMAIL:-admin@sprut.local}
      PGADMIN_DEFAULT_PASSWORD: ${PGADMIN_PASSWORD:-admin}
    volumes:
      - pgadmin_data:/var/lib/pgadmin
    depends_on:
      - postgres
    profiles:
      - tools

volumes:
  postgres_data:
  n8n_data:
  pgadmin_data:
```

**Step 4: Commit**

```bash
git add docker-compose.yml Dockerfile db/
git commit -m "feat: Docker Compose with PostgreSQL, n8n, and pgvector"
```

---

## Phase 2: Database Layer (Memory)

### Task 3: SQLAlchemy models

**Files:**
- Create: `src/memory/__init__.py`
- Create: `src/memory/models.py`
- Create: `tests/test_models.py`

**Step 1: Write test for models**

```python
# tests/test_models.py
from src.memory.models import Instruction, AboutMe, Dialogue, Thought, Download, Transcription


def test_instruction_model_fields():
    inst = Instruction(content="test rule", metadata={"key": "val"})
    assert inst.content == "test rule"
    assert inst.metadata == {"key": "val"}


def test_transcription_model_fields():
    t = Transcription(raw_text="hello world", source_type="voice")
    assert t.raw_text == "hello world"
    assert t.source_type == "voice"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_models.py -v`
Expected: FAIL with ImportError

**Step 3: Write SQLAlchemy models**

```python
# src/memory/models.py
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
    metadata: Mapped[Optional[dict]] = mapped_column(JSONB, default=dict)
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


# Map table name strings to model classes for dynamic access
TABLE_MODELS = {
    "instructions": Instruction,
    "about_me": AboutMe,
    "dialogues": Dialogue,
    "thoughts": Thought,
    "downloads": Download,
    "transcriptions": Transcription,
}
```

**Step 4: Run tests**

Run: `pytest tests/test_models.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/memory/ tests/test_models.py
git commit -m "feat: SQLAlchemy models with pgvector for all memory tables"
```

---

### Task 4: Vector store — CRUD and semantic search

**Files:**
- Create: `src/memory/vector_store.py`
- Create: `src/memory/database.py`
- Create: `tests/test_vector_store.py`

**Step 1: Create database session manager**

```python
# src/memory/database.py
"""Async database session management."""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.core.config import settings

engine = create_async_engine(settings.database_url, echo=False, pool_size=5, max_overflow=10)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_session() -> AsyncSession:
    async with async_session() as session:
        yield session
```

**Step 2: Write tests for vector store**

```python
# tests/test_vector_store.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.memory.vector_store import VectorStore


@pytest.fixture
def mock_session():
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    return session


@pytest.fixture
def store(mock_session):
    return VectorStore(session=mock_session, table_name="instructions")


@pytest.mark.asyncio
async def test_add_entry(store, mock_session):
    with patch.object(store, "_get_embedding", return_value=[0.1] * 1536):
        await store.add("Test rule", metadata={"source": "test"})
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_delete_entry(store, mock_session):
    mock_result = MagicMock()
    mock_result.rowcount = 1
    mock_session.execute.return_value = mock_result
    deleted = await store.delete(1)
    assert deleted is True


@pytest.mark.asyncio
async def test_delete_nonexistent(store, mock_session):
    mock_result = MagicMock()
    mock_result.rowcount = 0
    mock_session.execute.return_value = mock_result
    deleted = await store.delete(999)
    assert deleted is False
```

**Step 3: Run tests to verify they fail**

Run: `pytest tests/test_vector_store.py -v`
Expected: FAIL with ImportError

**Step 4: Implement vector store**

```python
# src/memory/vector_store.py
"""Vector store with pgvector for semantic search."""

from typing import Optional

from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.memory.models import TABLE_MODELS


class VectorStore:
    """CRUD + semantic search on vector-enabled tables."""

    def __init__(self, session: AsyncSession, table_name: str, embedding_fn=None):
        if table_name not in TABLE_MODELS:
            raise ValueError(f"Unknown table: {table_name}")
        self.session = session
        self.model = TABLE_MODELS[table_name]
        self.table_name = table_name
        self._embedding_fn = embedding_fn

    async def _get_embedding(self, text_content: str) -> list[float]:
        """Get embedding vector for text. Uses injected function or returns zeros."""
        if self._embedding_fn:
            return await self._embedding_fn(text_content)
        return [0.0] * 1536

    async def add(self, content: str, metadata: Optional[dict] = None) -> int:
        """Add an entry with embedding."""
        embedding = await self._get_embedding(content)
        entry = self.model(content=content, metadata=metadata or {}, embedding=embedding)
        self.session.add(entry)
        await self.session.commit()
        await self.session.refresh(entry)
        return entry.id

    async def delete(self, entry_id: int) -> bool:
        """Delete an entry by ID."""
        stmt = delete(self.model).where(self.model.id == entry_id)
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount > 0

    async def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Semantic search using cosine similarity."""
        query_embedding = await self._get_embedding(query)
        stmt = (
            select(
                self.model.id,
                self.model.content,
                self.model.metadata,
                self.model.embedding.cosine_distance(query_embedding).label("distance"),
            )
            .order_by("distance")
            .limit(top_k)
        )
        result = await self.session.execute(stmt)
        rows = result.all()
        return [
            {"id": r.id, "content": r.content, "metadata": r.metadata, "distance": r.distance}
            for r in rows
        ]

    async def get_all(self) -> list[dict]:
        """Get all entries (without embeddings)."""
        stmt = select(self.model.id, self.model.content, self.model.metadata)
        result = await self.session.execute(stmt)
        return [{"id": r.id, "content": r.content, "metadata": r.metadata} for r in result.all()]
```

**Step 5: Run tests**

Run: `pytest tests/test_vector_store.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/memory/database.py src/memory/vector_store.py tests/test_vector_store.py
git commit -m "feat: vector store with pgvector semantic search and CRUD"
```

---

## Phase 3: AI Provider Abstraction

### Task 5: AI provider interface and OpenAI implementation

**Files:**
- Create: `src/ai/__init__.py`
- Create: `src/ai/provider.py`
- Create: `src/ai/openai_provider.py`
- Create: `tests/test_ai_provider.py`

**Step 1: Write tests**

```python
# tests/test_ai_provider.py
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.ai.provider import AIProvider
from src.ai.openai_provider import OpenAIProvider


def test_base_provider_is_abstract():
    with pytest.raises(TypeError):
        AIProvider()


@pytest.mark.asyncio
async def test_openai_generate_text():
    provider = OpenAIProvider(api_key="test-key", model="gpt-4o-mini", temperature=0.1)
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Hello!"))]

    with patch.object(provider.client.chat.completions, "create", new_callable=AsyncMock, return_value=mock_response):
        result = await provider.generate_text("Hi", system_prompt="You are helpful")
        assert result == "Hello!"


@pytest.mark.asyncio
async def test_openai_create_embedding():
    provider = OpenAIProvider(api_key="test-key", model="gpt-4o-mini")
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1] * 1536)]

    with patch.object(provider.client.embeddings, "create", new_callable=AsyncMock, return_value=mock_response):
        result = await provider.create_embedding("test text")
        assert len(result) == 1536
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ai_provider.py -v`
Expected: FAIL

**Step 3: Implement base provider interface**

```python
# src/ai/provider.py
"""Base AI provider interface."""

from abc import ABC, abstractmethod
from typing import Optional


class AIProvider(ABC):
    """Abstract base class for AI providers."""

    @abstractmethod
    async def generate_text(
        self, prompt: str, system_prompt: Optional[str] = None, temperature: Optional[float] = None
    ) -> str:
        ...

    @abstractmethod
    async def generate_with_vision(self, prompt: str, image_data: bytes) -> str:
        ...

    @abstractmethod
    async def transcribe_audio(self, audio_data: bytes, filename: str = "audio.ogg") -> str:
        ...

    @abstractmethod
    async def text_to_speech(self, text: str) -> bytes:
        ...

    @abstractmethod
    async def create_embedding(self, text: str) -> list[float]:
        ...
```

**Step 4: Implement OpenAI provider**

```python
# src/ai/openai_provider.py
"""OpenAI AI provider implementation."""

import io
from typing import Optional, BinaryIO
import base64

from openai import AsyncOpenAI

from src.ai.provider import AIProvider


class OpenAIProvider(AIProvider):
    """OpenAI provider: GPT-4o mini, Whisper, TTS-1, embeddings."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.1, base_url: Optional[str] = None):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature

    async def generate_text(
        self, prompt: str, system_prompt: Optional[str] = None, temperature: Optional[float] = None
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.temperature,
        )
        return response.choices[0].message.content

    async def generate_with_vision(self, prompt: str, image_data: bytes) -> str:
        b64_image = base64.b64encode(image_data).decode("utf-8")
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}", "detail": "high"}},
                ],
            }],
        )
        return response.choices[0].message.content

    async def transcribe_audio(self, audio_data: bytes, filename: str = "audio.ogg") -> str:
        audio_file = io.BytesIO(audio_data)
        audio_file.name = filename
        response = await self.client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
        )
        return response.text

    async def text_to_speech(self, text: str) -> bytes:
        response = await self.client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text,
        )
        return response.content

    async def create_embedding(self, text: str) -> list[float]:
        response = await self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding
```

**Step 5: Run tests**

Run: `pytest tests/test_ai_provider.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/ai/ tests/test_ai_provider.py
git commit -m "feat: AI provider abstraction with OpenAI implementation"
```

---

### Task 6: Claude, Gemini, and Custom providers

**Files:**
- Create: `src/ai/claude_provider.py`
- Create: `src/ai/gemini_provider.py`
- Create: `src/ai/custom_provider.py`
- Create: `src/ai/registry.py`
- Create: `tests/test_ai_registry.py`

**Step 1: Write tests for registry**

```python
# tests/test_ai_registry.py
import pytest
from src.ai.registry import AIRegistry


def test_registry_registers_provider():
    registry = AIRegistry()
    registry.register("openai", provider_class="OpenAIProvider", api_key="test")
    assert "openai" in registry.providers


def test_registry_get_default():
    registry = AIRegistry()
    registry.register("openai", provider_class="OpenAIProvider", api_key="test")
    registry.set_default("text", "openai")
    provider = registry.get_provider("text")
    assert provider is not None


def test_registry_unknown_role():
    registry = AIRegistry()
    with pytest.raises(KeyError):
        registry.get_provider("nonexistent_role")
```

**Step 2: Implement Claude provider**

```python
# src/ai/claude_provider.py
"""Anthropic Claude AI provider."""

from typing import Optional
import base64

from anthropic import AsyncAnthropic

from src.ai.provider import AIProvider


class ClaudeProvider(AIProvider):
    """Claude provider for text generation."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514", temperature: float = 0.1):
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature

    async def generate_text(self, prompt: str, system_prompt: Optional[str] = None, temperature: Optional[float] = None) -> str:
        kwargs = {"model": self.model, "max_tokens": 4096, "temperature": temperature or self.temperature}
        if system_prompt:
            kwargs["system"] = system_prompt
        kwargs["messages"] = [{"role": "user", "content": prompt}]
        response = await self.client.messages.create(**kwargs)
        return response.content[0].text

    async def generate_with_vision(self, prompt: str, image_data: bytes) -> str:
        b64_image = base64.b64encode(image_data).decode("utf-8")
        response = await self.client.messages.create(
            model=self.model, max_tokens=4096,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64_image}},
                {"type": "text", "text": prompt},
            ]}],
        )
        return response.content[0].text

    async def transcribe_audio(self, audio_data: bytes, filename: str = "audio.ogg") -> str:
        raise NotImplementedError("Claude does not support audio transcription. Use OpenAI Whisper.")

    async def text_to_speech(self, text: str) -> bytes:
        raise NotImplementedError("Claude does not support TTS. Use OpenAI TTS.")

    async def create_embedding(self, text: str) -> list[float]:
        raise NotImplementedError("Claude does not support embeddings. Use OpenAI embeddings.")
```

**Step 3: Implement Gemini provider**

```python
# src/ai/gemini_provider.py
"""Google Gemini AI provider."""

from typing import Optional

import google.generativeai as genai

from src.ai.provider import AIProvider


class GeminiProvider(AIProvider):
    """Gemini provider for text generation."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        genai.configure(api_key=api_key)
        self.model_name = model

    async def generate_text(self, prompt: str, system_prompt: Optional[str] = None, temperature: Optional[float] = None) -> str:
        model = genai.GenerativeModel(self.model_name, system_instruction=system_prompt)
        response = await model.generate_content_async(prompt)
        return response.text

    async def generate_with_vision(self, prompt: str, image_data: bytes) -> str:
        model = genai.GenerativeModel(self.model_name)
        image_part = {"mime_type": "image/jpeg", "data": image_data}
        response = await model.generate_content_async([prompt, image_part])
        return response.text

    async def transcribe_audio(self, audio_data: bytes, filename: str = "audio.ogg") -> str:
        raise NotImplementedError("Use OpenAI Whisper for transcription.")

    async def text_to_speech(self, text: str) -> bytes:
        raise NotImplementedError("Use OpenAI TTS.")

    async def create_embedding(self, text: str) -> list[float]:
        result = genai.embed_content(model="models/text-embedding-004", content=text)
        return result["embedding"]
```

**Step 4: Implement Custom provider (OpenAI-compatible API)**

```python
# src/ai/custom_provider.py
"""Custom LLM provider (OpenAI-compatible API on remote server)."""

from src.ai.openai_provider import OpenAIProvider


class CustomProvider(OpenAIProvider):
    """Custom provider wrapping OpenAI-compatible remote server."""

    def __init__(self, base_url: str, api_key: str, model: str = "default", temperature: float = 0.1):
        super().__init__(api_key=api_key, model=model, temperature=temperature, base_url=base_url)
```

**Step 5: Implement AI registry**

```python
# src/ai/registry.py
"""AI provider registry for multi-provider support."""

from typing import Optional

from src.ai.provider import AIProvider
from src.ai.openai_provider import OpenAIProvider
from src.ai.claude_provider import ClaudeProvider
from src.ai.gemini_provider import GeminiProvider
from src.ai.custom_provider import CustomProvider

PROVIDER_CLASSES = {
    "openai": OpenAIProvider,
    "claude": ClaudeProvider,
    "gemini": GeminiProvider,
    "custom": CustomProvider,
}


class AIRegistry:
    """Registry for managing multiple AI providers."""

    def __init__(self):
        self.providers: dict[str, AIProvider] = {}
        self._defaults: dict[str, str] = {}

    def register(self, name: str, provider_class: str = None, **kwargs) -> None:
        """Register a provider by name."""
        if isinstance(provider_class, str):
            cls = PROVIDER_CLASSES.get(name, OpenAIProvider)
        else:
            cls = provider_class or PROVIDER_CLASSES.get(name, OpenAIProvider)
        self.providers[name] = cls(**kwargs)

    def set_default(self, role: str, provider_name: str) -> None:
        """Set default provider for a role (text, vision, transcription, tts, embeddings)."""
        self._defaults[role] = provider_name

    def get_provider(self, role: str) -> AIProvider:
        """Get the provider assigned to a role."""
        name = self._defaults.get(role)
        if not name or name not in self.providers:
            raise KeyError(f"No provider configured for role: {role}")
        return self.providers[name]

    @classmethod
    def from_config(cls, yaml_config: dict, settings) -> "AIRegistry":
        """Build registry from config.yaml and environment settings."""
        registry = cls()
        ai_config = yaml_config.get("ai", {})
        providers_config = ai_config.get("providers", {})

        # Register configured providers
        if settings.openai_api_key and "openai" in providers_config:
            pc = providers_config["openai"]
            registry.providers["openai"] = OpenAIProvider(
                api_key=settings.openai_api_key, model=pc.get("model", "gpt-4o-mini"), temperature=pc.get("temperature", 0.1)
            )
        if settings.anthropic_api_key and "claude" in providers_config:
            pc = providers_config["claude"]
            registry.providers["claude"] = ClaudeProvider(
                api_key=settings.anthropic_api_key, model=pc.get("model", "claude-sonnet-4-20250514")
            )
        if settings.gemini_api_key and "gemini" in providers_config:
            pc = providers_config["gemini"]
            registry.providers["gemini"] = GeminiProvider(api_key=settings.gemini_api_key, model=pc.get("model", "gemini-2.0-flash"))
        if settings.custom_llm_url and "custom" in providers_config:
            pc = providers_config["custom"]
            registry.providers["custom"] = CustomProvider(
                base_url=settings.custom_llm_url, api_key=settings.custom_llm_key, model=pc.get("model", "default")
            )

        # Set defaults
        for role in ("text", "vision", "transcription", "tts", "embeddings"):
            default_name = ai_config.get(f"default_{role}")
            if default_name:
                registry.set_default(role, default_name)

        return registry
```

**Step 6: Run tests**

Run: `pytest tests/test_ai_registry.py tests/test_ai_provider.py -v`
Expected: PASS

**Step 7: Commit**

```bash
git add src/ai/ tests/test_ai_registry.py
git commit -m "feat: Claude, Gemini, Custom providers and AI registry"
```

---

## Phase 4: Message Processing

### Task 7: Message router

**Files:**
- Create: `src/core/router.py`
- Create: `tests/test_router.py`

**Step 1: Write tests**

```python
# tests/test_router.py
from src.core.router import MessageRouter, MessageType


def test_detect_text():
    msg = {"text": "hello", "chat": {"id": 123}}
    assert MessageRouter.detect_type(msg) == MessageType.TEXT


def test_detect_photo():
    msg = {"photo": [{"file_id": "abc"}], "chat": {"id": 123}}
    assert MessageRouter.detect_type(msg) == MessageType.PHOTO


def test_detect_voice():
    msg = {"voice": {"file_id": "abc"}, "chat": {"id": 123}}
    assert MessageRouter.detect_type(msg) == MessageType.VOICE


def test_detect_audio():
    msg = {"audio": {"file_id": "abc", "mime_type": "audio/mpeg"}, "chat": {"id": 123}}
    assert MessageRouter.detect_type(msg) == MessageType.AUDIO


def test_detect_document():
    msg = {"document": {"file_id": "abc", "file_name": "report.pdf"}, "chat": {"id": 123}}
    assert MessageRouter.detect_type(msg) == MessageType.DOCUMENT


def test_detect_unknown():
    msg = {"sticker": {"file_id": "abc"}, "chat": {"id": 123}}
    assert MessageRouter.detect_type(msg) == MessageType.UNKNOWN
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_router.py -v`
Expected: FAIL

**Step 3: Implement router**

```python
# src/core/router.py
"""Message type router — determines processing pipeline."""

from enum import Enum


class MessageType(str, Enum):
    TEXT = "text"
    PHOTO = "photo"
    VOICE = "voice"
    AUDIO = "audio"
    DOCUMENT = "document"
    UNKNOWN = "unknown"


class MessageRouter:
    """Routes incoming Telegram messages to appropriate pipelines."""

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
```

**Step 4: Run tests**

Run: `pytest tests/test_router.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/core/router.py tests/test_router.py
git commit -m "feat: message router with type detection"
```

---

### Task 8: Command handler

**Files:**
- Create: `src/core/command_handler.py`
- Create: `tests/test_command_handler.py`

**Step 1: Write tests**

```python
# tests/test_command_handler.py
from src.core.command_handler import CommandHandler, Command


def test_detect_remember():
    assert CommandHandler.detect("Запомни: всегда отвечай кратко") == Command.REMEMBER


def test_detect_write_about_me():
    assert CommandHandler.detect("Запиши обо мне: я люблю кофе") == Command.WRITE_ABOUT_ME


def test_detect_delete_about_me():
    assert CommandHandler.detect("Удали обо мне факт про кофе") == Command.DELETE_ABOUT_ME


def test_detect_delete_instruction():
    assert CommandHandler.detect("Удали инструкцию про фитнес") == Command.DELETE_INSTRUCTION


def test_detect_record_thought():
    assert CommandHandler.detect("Запиши мысль: нужно больше спать") == Command.RECORD_THOUGHT


def test_detect_ask_terminal():
    assert CommandHandler.detect("Спроси терминал: какой uptime") == Command.ASK_TERMINAL


def test_detect_no_command():
    assert CommandHandler.detect("Привет, как дела?") == Command.NONE


def test_extract_payload_remember():
    payload = CommandHandler.extract_payload("Запомни: всегда отвечай кратко", Command.REMEMBER)
    assert payload == "всегда отвечай кратко"


def test_case_insensitive():
    assert CommandHandler.detect("запомни: тест") == Command.REMEMBER
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_command_handler.py -v`
Expected: FAIL

**Step 3: Implement command handler**

```python
# src/core/command_handler.py
"""Command trigger detection and routing."""

from enum import Enum
from typing import Optional


class Command(str, Enum):
    REMEMBER = "remember"
    WRITE_ABOUT_ME = "write_about_me"
    DELETE_ABOUT_ME = "delete_about_me"
    DELETE_INSTRUCTION = "delete_instruction"
    RECORD_THOUGHT = "record_thought"
    ASK_TERMINAL = "ask_terminal"
    NONE = "none"


# Trigger phrases mapped to commands (checked in order)
TRIGGERS = [
    ("запомни", Command.REMEMBER),
    ("запиши обо мне", Command.WRITE_ABOUT_ME),
    ("удали обо мне", Command.DELETE_ABOUT_ME),
    ("удали инструкцию", Command.DELETE_INSTRUCTION),
    ("запиши мысль", Command.RECORD_THOUGHT),
    ("спроси терминал", Command.ASK_TERMINAL),
]


class CommandHandler:
    """Detects command triggers in text messages."""

    @staticmethod
    def detect(text: str) -> Command:
        """Check if text starts with a known trigger phrase."""
        lower = text.strip().lower()
        for trigger, command in TRIGGERS:
            if lower.startswith(trigger):
                return command
        return Command.NONE

    @staticmethod
    def extract_payload(text: str, command: Command) -> str:
        """Extract the payload after the trigger phrase."""
        lower = text.strip().lower()
        for trigger, cmd in TRIGGERS:
            if cmd == command and lower.startswith(trigger):
                payload = text.strip()[len(trigger):].lstrip(": ")
                return payload
        return text
```

**Step 4: Run tests**

Run: `pytest tests/test_command_handler.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/core/command_handler.py tests/test_command_handler.py
git commit -m "feat: command handler with trigger detection"
```

---

### Task 9: Text chunker utility

**Files:**
- Create: `src/utils/__init__.py`
- Create: `src/utils/chunker.py`
- Create: `tests/test_chunker.py`

**Step 1: Write tests**

```python
# tests/test_chunker.py
from src.utils.chunker import chunk_text


def test_short_text_no_split():
    chunks = chunk_text("Hello world", max_size=3500)
    assert chunks == ["Hello world"]


def test_long_text_splits():
    text = "A" * 7000
    chunks = chunk_text(text, max_size=3500)
    assert len(chunks) == 2
    assert all(len(c) <= 3500 for c in chunks)


def test_splits_at_newline():
    text = "Line one\n" * 500  # ~4500 chars
    chunks = chunk_text(text, max_size=3500)
    assert len(chunks) >= 2
    assert chunks[0].endswith("\n") or len(chunks[0]) <= 3500


def test_empty_text():
    chunks = chunk_text("")
    assert chunks == [""]
```

**Step 2: Implement chunker**

```python
# src/utils/chunker.py
"""Text chunking for Telegram message limits."""


def chunk_text(text: str, max_size: int = 3500) -> list[str]:
    """Split text into chunks, preferring line breaks as split points."""
    if len(text) <= max_size:
        return [text]

    chunks = []
    remaining = text
    while remaining:
        if len(remaining) <= max_size:
            chunks.append(remaining)
            break

        # Find last newline within limit
        split_at = remaining.rfind("\n", 0, max_size)
        if split_at == -1 or split_at < max_size // 2:
            # No good newline, split at max_size
            split_at = max_size

        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip("\n")

    return chunks
```

**Step 3: Run tests**

Run: `pytest tests/test_chunker.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/utils/ tests/test_chunker.py
git commit -m "feat: text chunker utility for Telegram message limits"
```

---

## Phase 5: Agents

### Task 10: Main orchestrator agent (Vector Prompt)

**Files:**
- Create: `src/agents/__init__.py`
- Create: `src/agents/orchestrator.py`
- Create: `tests/test_orchestrator.py`

**Step 1: Write tests**

```python
# tests/test_orchestrator.py
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.agents.orchestrator import Orchestrator


@pytest.fixture
def mock_ai():
    ai = AsyncMock()
    ai.generate_text = AsyncMock(return_value="Agent response")
    return ai


@pytest.fixture
def mock_memory():
    memory = AsyncMock()
    memory.search = AsyncMock(return_value=[
        {"content": "If asked about fitness, use fitness_trainer tool", "distance": 0.1}
    ])
    return memory


@pytest.mark.asyncio
async def test_orchestrator_injects_hidden_prompt(mock_ai, mock_memory):
    orch = Orchestrator(ai_provider=mock_ai, instructions_store=mock_memory)
    await orch.process("How to lose weight?")
    # Verify the hidden prompt injection was added
    call_args = mock_ai.generate_text.call_args
    prompt = call_args.kwargs.get("prompt") or call_args[0][0]
    assert "инструкцию" in prompt.lower() or "instruction" in prompt.lower()


@pytest.mark.asyncio
async def test_orchestrator_queries_instructions(mock_ai, mock_memory):
    orch = Orchestrator(ai_provider=mock_ai, instructions_store=mock_memory)
    await orch.process("test question")
    mock_memory.search.assert_called_once()


@pytest.mark.asyncio
async def test_orchestrator_returns_response(mock_ai, mock_memory):
    orch = Orchestrator(ai_provider=mock_ai, instructions_store=mock_memory)
    result = await orch.process("test")
    assert result["text"] == "Agent response"
    assert result["voice"] is None
```

**Step 2: Implement orchestrator**

```python
# src/agents/orchestrator.py
"""Main orchestrator agent using Vector Prompt methodology."""

from typing import Optional

from src.ai.provider import AIProvider
from src.memory.vector_store import VectorStore

SYSTEM_PROMPT = """Ты — СПРУТ, персональный AI-ассистент. Твои обязанности:
1. Отвечать на вопросы пользователя, используя найденные инструкции
2. Использовать инструменты и суб-агентов когда указано в инструкциях
3. Быть кратким и точным

Доступные инструменты: fitness_trainer, doctor, auto_mechanic, kaizen, terminal

Если в инструкциях указано использовать конкретный инструмент — используй его.
Если нет подходящей инструкции — отвечай самостоятельно."""

HIDDEN_INJECTION = "Обязательно зайди в инструкцию и найди правило, на какой инструмент или агент отправить запрос, или ответь самостоятельно. "


class Orchestrator:
    """Main agent with Vector Prompt — minimal system prompt, DB-driven intelligence."""

    def __init__(self, ai_provider: AIProvider, instructions_store: VectorStore):
        self.ai = ai_provider
        self.instructions = instructions_store

    async def process(self, user_text: str, context: Optional[str] = None) -> dict:
        """Process user input through the Vector Prompt pipeline.

        Returns dict with 'text' (response) and 'voice' (bool if TTS requested).
        """
        # 1. Query instructions DB
        relevant_rules = await self.instructions.search(user_text, top_k=5)
        rules_text = "\n".join(f"- {r['content']}" for r in relevant_rules) if relevant_rules else "Нет подходящих инструкций."

        # 2. Build prompt with hidden injection + context
        prompt_parts = [HIDDEN_INJECTION]
        if context:
            prompt_parts.append(f"Контекст: {context}")
        prompt_parts.append(f"Найденные инструкции:\n{rules_text}")
        prompt_parts.append(f"Запрос пользователя: {user_text}")
        full_prompt = "\n\n".join(prompt_parts)

        # 3. Generate response
        response_text = await self.ai.generate_text(prompt=full_prompt, system_prompt=SYSTEM_PROMPT, temperature=0.1)

        # 4. Check for voice tag
        has_voice = "[VOICE]" in response_text
        clean_text = response_text.replace("[VOICE]", "").strip()

        return {"text": clean_text, "voice": has_voice if has_voice else None}
```

**Step 3: Run tests**

Run: `pytest tests/test_orchestrator.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/agents/ tests/test_orchestrator.py
git commit -m "feat: main orchestrator agent with Vector Prompt methodology"
```

---

### Task 11: Superinstruction agent

**Files:**
- Create: `src/agents/superinstruction.py`
- Create: `tests/test_superinstruction.py`

**Step 1: Write tests**

```python
# tests/test_superinstruction.py
import pytest
from unittest.mock import AsyncMock

from src.agents.superinstruction import SuperinstructionAgent


@pytest.fixture
def mock_ai():
    ai = AsyncMock()
    ai.generate_text = AsyncMock(return_value='{"action": "save", "formatted_rule": "Always respond in Russian"}')
    return ai


@pytest.fixture
def mock_store():
    store = AsyncMock()
    store.search = AsyncMock(return_value=[])  # No conflicts
    store.add = AsyncMock(return_value=1)
    return store


@pytest.mark.asyncio
async def test_saves_when_no_conflicts(mock_ai, mock_store):
    agent = SuperinstructionAgent(ai_provider=mock_ai, instructions_store=mock_store)
    result = await agent.process("всегда отвечай на русском")
    assert result["saved"] is True
    mock_store.add.assert_called_once()


@pytest.mark.asyncio
async def test_rejects_on_conflict(mock_ai, mock_store):
    mock_store.search.return_value = [{"content": "Always respond in English", "distance": 0.05}]
    mock_ai.generate_text.return_value = '{"action": "reject", "reason": "Contradicts existing rule about language"}'
    agent = SuperinstructionAgent(ai_provider=mock_ai, instructions_store=mock_store)
    result = await agent.process("всегда отвечай на русском")
    assert result["saved"] is False
    assert "reason" in result
```

**Step 2: Implement superinstruction agent**

```python
# src/agents/superinstruction.py
"""Superinstruction agent — validates and saves new behavioral rules."""

import json
from typing import Optional

from src.ai.provider import AIProvider
from src.memory.vector_store import VectorStore

SYSTEM_PROMPT = """Ты — агент проверки инструкций. Твоя задача:
1. Проверить новую инструкцию на конфликты с существующими
2. Если конфликт найден — отклонить и объяснить почему
3. Если конфликтов нет — сформулировать чёткое правило и сохранить

Ответь СТРОГО в JSON формате:
- Если сохранять: {"action": "save", "formatted_rule": "чёткая формулировка правила"}
- Если отклонить: {"action": "reject", "reason": "объяснение конфликта"}"""


class SuperinstructionAgent:
    """Handles 'Запомни' command — validates rules before saving."""

    def __init__(self, ai_provider: AIProvider, instructions_store: VectorStore):
        self.ai = ai_provider
        self.store = instructions_store

    async def process(self, rule_text: str) -> dict:
        """Validate and save a new instruction rule."""
        # 1. Search for similar/conflicting rules
        existing = await self.store.search(rule_text, top_k=5)
        existing_text = "\n".join(f"- {r['content']}" for r in existing) if existing else "Нет похожих инструкций."

        # 2. Ask AI to check for conflicts
        prompt = f"Существующие инструкции:\n{existing_text}\n\nНовая инструкция: {rule_text}"
        response = await self.ai.generate_text(prompt=prompt, system_prompt=SYSTEM_PROMPT)

        # 3. Parse AI decision
        try:
            decision = json.loads(response)
        except json.JSONDecodeError:
            return {"saved": False, "reason": "AI returned invalid response", "raw": response}

        if decision.get("action") == "save":
            formatted = decision.get("formatted_rule", rule_text)
            entry_id = await self.store.add(formatted, metadata={"source": "user_command"})
            return {"saved": True, "id": entry_id, "rule": formatted}
        else:
            return {"saved": False, "reason": decision.get("reason", "Unknown conflict")}
```

**Step 3: Run tests**

Run: `pytest tests/test_superinstruction.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/agents/superinstruction.py tests/test_superinstruction.py
git commit -m "feat: superinstruction agent with conflict detection"
```

---

### Task 12: Terminal sub-agent and sub-agent base

**Files:**
- Create: `src/agents/terminal.py`
- Create: `src/agents/sub_agents/__init__.py`
- Create: `src/agents/sub_agents/base.py`
- Create: `tests/test_sub_agents.py`

**Step 1: Write tests**

```python
# tests/test_sub_agents.py
import pytest
from unittest.mock import AsyncMock

from src.agents.sub_agents.base import SubAgent
from src.agents.terminal import TerminalAgent


@pytest.mark.asyncio
async def test_sub_agent_process():
    ai = AsyncMock()
    ai.generate_text = AsyncMock(return_value="Sub-agent response")
    store = AsyncMock()
    store.search = AsyncMock(return_value=[])

    agent = SubAgent(name="test", system_prompt="You are test agent", ai_provider=ai, memory_store=store)
    result = await agent.process("test question")
    assert result == "Sub-agent response"


@pytest.mark.asyncio
async def test_terminal_agent():
    ai = AsyncMock()
    ai.generate_text = AsyncMock(return_value="Server uptime: 30 days")

    agent = TerminalAgent(ai_provider=ai)
    result = await agent.process("какой uptime сервера")
    assert "uptime" in result.lower() or "30" in result
```

**Step 2: Implement sub-agent base**

```python
# src/agents/sub_agents/base.py
"""Base class for specialized sub-agents."""

from typing import Optional

from src.ai.provider import AIProvider
from src.memory.vector_store import VectorStore


class SubAgent:
    """Base sub-agent with own system prompt and memory."""

    def __init__(self, name: str, system_prompt: str, ai_provider: AIProvider, memory_store: Optional[VectorStore] = None):
        self.name = name
        self.system_prompt = system_prompt
        self.ai = ai_provider
        self.memory = memory_store

    async def process(self, query: str) -> str:
        """Process a query using this sub-agent's expertise."""
        context = ""
        if self.memory:
            results = await self.memory.search(query, top_k=3)
            if results:
                context = "Relevant context:\n" + "\n".join(f"- {r['content']}" for r in results)

        prompt = f"{context}\n\nUser query: {query}" if context else query
        return await self.ai.generate_text(prompt=prompt, system_prompt=self.system_prompt)
```

**Step 3: Implement terminal agent**

```python
# src/agents/terminal.py
"""Terminal sub-agent — queries server via shell commands (simulated)."""

from src.ai.provider import AIProvider
from src.agents.sub_agents.base import SubAgent

TERMINAL_SYSTEM_PROMPT = """Ты — агент для работы с терминалом сервера.
Пользователь задаёт вопрос о состоянии сервера.
Сформулируй ответ на основе предоставленных данных.
Отвечай кратко и по делу."""


class TerminalAgent(SubAgent):
    """Handles 'Спроси терминал' commands."""

    def __init__(self, ai_provider: AIProvider):
        super().__init__(
            name="terminal",
            system_prompt=TERMINAL_SYSTEM_PROMPT,
            ai_provider=ai_provider,
            memory_store=None,
        )
```

**Step 4: Run tests**

Run: `pytest tests/test_sub_agents.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/agents/terminal.py src/agents/sub_agents/ tests/test_sub_agents.py
git commit -m "feat: sub-agent base class and terminal agent"
```

---

## Phase 6: Processors

### Task 13: Vision, audio, document processors and TTS

**Files:**
- Create: `src/processors/__init__.py`
- Create: `src/processors/vision.py`
- Create: `src/processors/audio.py`
- Create: `src/processors/documents.py`
- Create: `src/processors/tts.py`
- Create: `tests/test_processors.py`

**Step 1: Write tests**

```python
# tests/test_processors.py
import pytest
from unittest.mock import AsyncMock

from src.processors.vision import VisionProcessor
from src.processors.audio import AudioProcessor
from src.processors.documents import DocumentProcessor
from src.processors.tts import TTSProcessor


@pytest.mark.asyncio
async def test_vision_describe_image():
    ai = AsyncMock()
    ai.generate_with_vision = AsyncMock(return_value="A photo of a cat")
    proc = VisionProcessor(ai_provider=ai)
    result = await proc.describe(b"fake_image_data")
    assert result == "A photo of a cat"


@pytest.mark.asyncio
async def test_audio_transcribe():
    ai = AsyncMock()
    ai.transcribe_audio = AsyncMock(return_value="Hello world transcription")
    proc = AudioProcessor(ai_provider=ai)
    result = await proc.transcribe(b"fake_audio", filename="voice.ogg")
    assert result == "Hello world transcription"


@pytest.mark.asyncio
async def test_audio_is_long():
    proc = AudioProcessor(ai_provider=AsyncMock())
    assert proc.is_long_audio(600 * 1024) is True  # 600KB
    assert proc.is_long_audio(400 * 1024) is False  # 400KB


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
    result = await proc.generate(text="Hello")
    assert result == b"audio_bytes"
```

**Step 2: Implement processors**

```python
# src/processors/vision.py
"""Image processing — description via vision model."""

from src.ai.provider import AIProvider

VISION_PROMPT = "Опиши это изображение подробно. Что на нём изображено?"


class VisionProcessor:
    def __init__(self, ai_provider: AIProvider):
        self.ai = ai_provider

    async def describe(self, image_data: bytes, prompt: str = VISION_PROMPT) -> str:
        return await self.ai.generate_with_vision(prompt, image_data)
```

```python
# src/processors/audio.py
"""Audio processing — transcription and summarization."""

from typing import Optional

from src.ai.provider import AIProvider

SUMMARY_PROMPT = "Кратко суммаризируй этот текст, выдели ключевые мысли и action items:\n\n{text}"


class AudioProcessor:
    def __init__(self, ai_provider: AIProvider, threshold_kb: int = 500):
        self.ai = ai_provider
        self.threshold_bytes = threshold_kb * 1024

    def is_long_audio(self, size_bytes: int) -> bool:
        return size_bytes > self.threshold_bytes

    async def transcribe(self, audio_data: bytes, filename: str = "audio.ogg") -> str:
        return await self.ai.transcribe_audio(audio_data, filename)

    async def summarize(self, text: str) -> str:
        return await self.ai.generate_text(prompt=SUMMARY_PROMPT.format(text=text))
```

```python
# src/processors/documents.py
"""Document processing — text extraction from various formats."""

from pathlib import Path
from typing import Optional

from src.ai.provider import AIProvider

SUPPORTED_FORMATS = {"pdf", "txt", "md", "csv", "docx"}


class DocumentProcessor:
    def __init__(self, ai_provider: Optional[AIProvider] = None):
        self.ai = ai_provider

    @staticmethod
    def detect_format(filename: str) -> str:
        ext = Path(filename).suffix.lstrip(".").lower()
        return ext if ext in SUPPORTED_FORMATS else "unsupported"

    async def extract_text(self, file_data: bytes, filename: str) -> str:
        fmt = self.detect_format(filename)
        if fmt == "txt" or fmt == "md" or fmt == "csv":
            return file_data.decode("utf-8", errors="replace")
        if fmt == "pdf" and self.ai:
            # Use AI to extract text from PDF
            return await self.ai.generate_text(
                prompt=f"Extract all text content from this document: {filename}"
            )
        return f"Unsupported format: {fmt}"
```

```python
# src/processors/tts.py
"""Text-to-Speech generation."""

from src.ai.provider import AIProvider


class TTSProcessor:
    def __init__(self, ai_provider: AIProvider):
        self.ai = ai_provider

    async def generate(self, text: str) -> bytes:
        return await self.ai.text_to_speech(text)
```

**Step 3: Run tests**

Run: `pytest tests/test_processors.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/processors/ tests/test_processors.py
git commit -m "feat: vision, audio, document, and TTS processors"
```

---

### Task 14: Text formatter utility

**Files:**
- Create: `src/utils/text_formatter.py`
- Create: `tests/test_text_formatter.py`

**Step 1: Write tests**

```python
# tests/test_text_formatter.py
import pytest
from unittest.mock import AsyncMock

from src.utils.text_formatter import TextFormatter


@pytest.mark.asyncio
async def test_format_for_social():
    ai = AsyncMock()
    ai.generate_text = AsyncMock(return_value="Clean formatted text for social media")
    formatter = TextFormatter(ai_provider=ai)
    result = await formatter.format_for_social("messy dictated text uhh yeah so basically...")
    assert result == "Clean formatted text for social media"
```

**Step 2: Implement**

```python
# src/utils/text_formatter.py
"""Text formatting utilities — clean up dictation for social media."""

from src.ai.provider import AIProvider

FORMAT_PROMPT = """Перед тобой текст, который был надиктован голосом. Он может быть грязным, с повторами и словами-паразитами.

Задача: отформатируй его в чистый, структурированный текст, подходящий для публикации в социальных сетях. Сохрани смысл, убери мусор, добавь структуру.

Текст: {text}"""


class TextFormatter:
    def __init__(self, ai_provider: AIProvider):
        self.ai = ai_provider

    async def format_for_social(self, raw_text: str) -> str:
        return await self.ai.generate_text(prompt=FORMAT_PROMPT.format(text=raw_text))
```

**Step 3: Run tests**

Run: `pytest tests/test_text_formatter.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/utils/text_formatter.py tests/test_text_formatter.py
git commit -m "feat: text formatter for dictation cleanup"
```

---

## Phase 7: FastAPI Service

### Task 15: FastAPI app and API routes

**Files:**
- Create: `src/main.py`
- Create: `src/api/__init__.py`
- Create: `src/api/routes.py`
- Create: `tests/test_api.py`

**Step 1: Write tests**

```python
# tests/test_api.py
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch

from src.main import app


@pytest.mark.asyncio
async def test_health_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


@pytest.mark.asyncio
async def test_process_requires_auth():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/process", json={"message": {"text": "hello"}})
        assert response.status_code in (401, 403)
```

**Step 2: Implement FastAPI app**

```python
# src/main.py
"""SPRUT 3.0 — FastAPI entry point."""

from fastapi import FastAPI

from src.api.routes import router

app = FastAPI(title="SPRUT 3.0", version="3.0.0")
app.include_router(router)
```

```python
# src/api/routes.py
"""API routes for SPRUT 3.0."""

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from typing import Optional

from src.core.config import settings

router = APIRouter()


class ProcessRequest(BaseModel):
    message: dict
    user_id: Optional[int] = None


class ProcessResponse(BaseModel):
    text: str
    voice_data: Optional[str] = None  # base64 encoded audio
    chunks: Optional[list[str]] = None


@router.get("/health")
async def health():
    return {"status": "healthy", "service": "sprut-api", "version": "3.0.0"}


@router.post("/api/process", response_model=ProcessResponse)
async def process_message(
    request: ProcessRequest,
    authorization: str = Header(default=""),
):
    """Process an incoming Telegram message."""
    # Auth check
    expected = f"Bearer {settings.api_secret_key}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # User ID check
    if request.user_id and request.user_id not in settings.allowed_user_id_list:
        raise HTTPException(status_code=403, detail="User not authorized")

    # TODO: Wire up router → command handler → agents → processors
    # Placeholder response
    return ProcessResponse(text="SPRUT 3.0 is running. Processing not yet implemented.")
```

**Step 3: Run tests**

Run: `pytest tests/test_api.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/main.py src/api/ tests/test_api.py
git commit -m "feat: FastAPI app with health and process endpoints"
```

---

## Phase 8: Integration — Wire Everything Together

### Task 16: Full processing pipeline in API route

**Files:**
- Modify: `src/api/routes.py` — wire router, command handler, agents, processors

**Step 1: Write integration test**

```python
# tests/test_integration.py
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch, MagicMock

from src.main import app
from src.core.config import settings


@pytest.mark.asyncio
async def test_text_message_pipeline():
    """Test full text processing: route → command check → agent → response."""
    settings.api_secret_key = "test-secret"
    settings.allowed_user_ids = "123"

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/process",
            json={"message": {"text": "Привет", "chat": {"id": 123}}, "user_id": 123},
            headers={"Authorization": "Bearer test-secret"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "text" in data
```

This test verifies the pipeline works end-to-end. The actual AI calls will be mocked at the provider level during testing.

**Step 2: Wire up the full pipeline in routes.py**

Update `src/api/routes.py` to import and use:
- `MessageRouter.detect_type()` for routing
- `CommandHandler.detect()` for commands
- `Orchestrator.process()` for default text
- `SuperinstructionAgent.process()` for "запомни"
- Processor classes for photo/voice/document

This is the largest single integration step. Wire each message type to its pipeline as documented in the design doc Section 5 (Input Processing Pipelines).

**Step 3: Run all tests**

Run: `pytest tests/ -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add src/ tests/
git commit -m "feat: wire full processing pipeline in API routes"
```

---

## Phase 9: n8n Workflows

### Task 17: n8n workflow templates

**Files:**
- Create: `n8n/workflows/telegram_main.json`
- Create: `n8n/workflows/gdrive_sync.json`
- Create: `n8n/workflows/obsidian_sync.json`
- Create: `n8n/README.md`

**Step 1: Create Telegram main workflow JSON**

Create n8n workflow with nodes:
1. **Telegram Trigger** — webhook receiving all message types
2. **Code Node** — security filter checking user_id against allowed list
3. **HTTP Request** — POST to `http://sprut-api:8000/api/process` with Bearer auth
4. **Switch** — check if response has voice_data
5. **Telegram Send Message** — send text response
6. **Telegram Send Voice** — send voice if present

**Step 2: Create Google Drive sync workflow**

Basic workflow for syncing files to Google Drive "Memory" folder.

**Step 3: Create Obsidian sync workflow**

Workflow for saving Markdown files to Google Drive folder synced with Obsidian.

**Step 4: Write n8n README with import instructions**

**Step 5: Commit**

```bash
git add n8n/
git commit -m "feat: n8n workflow templates for Telegram, Google Drive, Obsidian"
```

---

## Phase 10: Documentation and Final Setup

### Task 18: Architecture docs and README

**Files:**
- Create: `docs/architecture.md`
- Create: `README.md`

**Step 1: Write architecture.md**

Document the full architecture as described in the design doc, including:
- System overview diagram
- Component descriptions
- Data flow
- Database schema
- AI provider configuration
- Command trigger reference

**Step 2: Write README.md**

Include:
- Quick start guide
- Prerequisites (Docker, API keys)
- Setup steps (copy .env.example, configure, docker-compose up)
- Usage guide
- Project structure

**Step 3: Commit**

```bash
git add docs/architecture.md README.md
git commit -m "docs: architecture documentation and README"
```

---

### Task 19: Final verification

**Step 1: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: ALL PASS

**Step 2: Verify Docker build**

Run: `docker build -t sprut-api .`
Expected: Build succeeds

**Step 3: Verify docker-compose config**

Run: `docker-compose config`
Expected: Valid configuration output

**Step 4: Final commit if needed**

```bash
git add -A
git commit -m "chore: final cleanup and verification"
```

---

## Summary

| Phase | Tasks | What it delivers |
|-------|-------|-----------------|
| 1. Scaffolding | 1-2 | Project structure, Docker, DB init |
| 2. Database | 3-4 | SQLAlchemy models, vector store |
| 3. AI Layer | 5-6 | Multi-provider abstraction |
| 4. Processing | 7-9 | Router, commands, chunker |
| 5. Agents | 10-12 | Orchestrator, superinstruction, sub-agents |
| 6. Processors | 13-14 | Vision, audio, documents, TTS, formatter |
| 7. API | 15 | FastAPI endpoints |
| 8. Integration | 16 | Full pipeline wiring |
| 9. n8n | 17 | Workflow templates |
| 10. Docs | 18-19 | Architecture docs, README, verification |
