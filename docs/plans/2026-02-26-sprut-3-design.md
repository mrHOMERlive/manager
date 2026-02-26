# SPRUT 3.0 Design Document

**Date:** 2026-02-26
**Status:** Approved
**Approach:** Python Core (FastAPI) + n8n Orchestrator
**Based on:** SPRUT 2.0 architecture from NotebookLM documentation

---

## 1. Overview

SPRUT 3.0 is a personal AI assistant that replicates and extends the SPRUT 2.0 system. The original was built entirely within n8n; this version uses a hybrid architecture: a Python FastAPI service handles all business logic, AI calls, and database operations, while n8n handles Telegram webhook integration, Google Drive sync, and Obsidian vault synchronization.

**Key improvements over SPRUT 2.0:**
- Multi-provider AI support (OpenAI, Claude, Gemini, custom LLMs)
- Testable Python codebase with clear separation of concerns
- All databases consolidated in local PostgreSQL + pgvector (including sub-agents)
- Docker Compose for one-command deployment

---

## 2. Architecture

```
Docker Compose (server)
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  Telegram ──webhook──▶ n8n (5678) ──REST API──▶          │
│  Telegram ◀──response── n8n ◀──────────────────          │
│                          │                               │
│                 Google Drive / Obsidian sync              │
│                          │                               │
│  ┌───────────────────────┴──────────────────────────┐   │
│  │         SPRUT Python Service — FastAPI (8000)     │   │
│  │                                                    │   │
│  │  Router → Command Handler → Main Agent            │   │
│  │  AI Layer (multi-provider) → Sub-Agent Engine     │   │
│  │  File/Audio/Vision Processors → Memory Manager    │   │
│  └───────────────────────┬──────────────────────────┘   │
│                          │                               │
│  ┌───────────────────────┴──────────────────────────┐   │
│  │       PostgreSQL 16 + pgvector (5432)             │   │
│  │  instructions | about_me | dialogues              │   │
│  │  thoughts | downloads | transcriptions            │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  pgAdmin (optional, 5050)                                │
└──────────────────────────────────────────────────────────┘
```

### Data Flow

1. Telegram message → n8n webhook trigger
2. n8n Code node checks authorized `user_id` → rejects unauthorized
3. n8n calls `POST /api/process` on Python service with message payload
4. Python Router determines message type (text/photo/voice/document)
5. Processing through appropriate pipeline
6. Response returned to n8n → sent to Telegram

### Docker Compose Services

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| `sprut-api` | Custom Dockerfile | 8000 | FastAPI Python service |
| `n8n` | n8nio/n8n | 5678 | Workflow orchestration |
| `postgres` | pgvector/pgvector:pg16 | 5432 | Database with vector extension |
| `pgadmin` | dpage/pgadmin4 | 5050 | DB admin UI (optional) |

---

## 3. AI Layer — Multi-Provider

### Provider Abstraction

```
AIProvider (base interface)
├── OpenAIProvider    — GPT-4o mini, Whisper, TTS-1, embeddings
├── ClaudeProvider    — Sonnet/Haiku for text
├── GeminiProvider    — For terminal queries
└── CustomProvider    — User's remote LLM server
```

Each provider implements a common interface:
- `generate_text(prompt, system_prompt, temperature)` → str
- `generate_with_vision(prompt, image_data)` → str
- `transcribe_audio(audio_data)` → str
- `text_to_speech(text)` → bytes
- `create_embedding(text)` → list[float]

### Configuration (config.yaml)

```yaml
ai:
  default_text: "openai"
  default_vision: "openai"
  default_transcription: "openai"
  default_tts: "openai"
  default_embeddings: "openai"

  providers:
    openai:
      api_key: "${OPENAI_API_KEY}"
      model: "gpt-4o-mini"
      temperature: 0.1
    claude:
      api_key: "${ANTHROPIC_API_KEY}"
      model: "claude-sonnet-4-20250514"
    gemini:
      api_key: "${GEMINI_API_KEY}"
      model: "gemini-2.0-flash"
    custom:
      base_url: "${CUSTOM_LLM_URL}"
      api_key: "${CUSTOM_LLM_KEY}"
```

---

## 4. Agent Architecture

### Main Orchestrator (Vector Prompt Methodology)

The main agent uses a minimal system prompt containing only:
- Basic duties ("You are a personal assistant")
- List of available tools

Before every user query, a hidden prompt injection is prepended:
> "Обязательно зайди в инструкцию и найди правило, на какой инструмент или агент отправить запрос, или ответь самостоятельно"

This forces the agent to query the Instructions vector DB before responding. The agent's "intelligence" comes from the database, not from a massive prompt.

**Key settings:**
- Temperature: 0.1 (strict instruction following)
- Conversational memory: DISABLED (prevents hallucination loops)
- Each request is independent — context comes from DB queries

### Sub-Agents

| Agent | Purpose | Trigger | Memory |
|-------|---------|---------|--------|
| Fitness Trainer | Workouts, nutrition | Instructions DB rule | Own vector DB |
| Doctor | Medical questions | Instructions DB rule | Own vector DB |
| Auto Mechanic | Car questions | Instructions DB rule | Own vector DB |
| Kaizen | Self-improvement, habits | Instructions DB rule | Own vector DB |
| Superinstruction | Add/validate rules | "Запомни" command | Instructions DB |
| Terminal | Server queries | "Спроси терминал" command | None |

Sub-agents are isolated pipelines with their own system prompts, tools, and memory. The main agent delegates to them based on rules found in the Instructions DB.

### Superinstruction Agent ("Запомни" Command)

1. User sends "Запомни: [rule]"
2. Command handler bypasses main agent → triggers Superinstruction agent
3. Agent searches Instructions DB for duplicates or contradictions
4. If conflict found → rejects with explanation to user
5. If safe → formats rule and saves to Instructions DB
6. Confirms success to user

---

## 5. Input Processing Pipelines

### Message Router

```
Telegram Message
    ├── text      → Command Switch
    ├── photo     → Vision Pipeline
    ├── voice     → Audio Pipeline
    └── document  → Document Pipeline
```

### Command Switch (text messages)

| Trigger Phrase | Action |
|----------------|--------|
| "Запомни" | Superinstruction Agent → validate + save to Instructions DB |
| "Запиши обо мне" | Direct save to About Me DB |
| "Удали обо мне" | Agent finds ID → deletes from About Me DB |
| "Удали инструкцию" | Agent finds ID → deletes from Instructions DB |
| "Запиши мысль" | Format → save as MD to Obsidian + Thoughts DB |
| "Спроси терминал" | Terminal Sub-Agent (Gemini CLI) → response |
| [no trigger] | Hidden prompt inject → Main Orchestrator Agent |

### Vision Pipeline (photos)

1. Send "Обрабатываю изображение..." to Telegram
2. Download image from Telegram
3. GPT-4o mini Vision (high detail) → image description
4. Append context ("Description of image sent by user")
5. Pass text to Main Agent via Command Switch

### Audio Pipeline (voice/audio files)

1. Check file size: under or over 500KB
2. Transcribe via OpenAI Whisper (25MB limit)
3. Short voice notes → pass transcript to Command Switch as text
4. Long recordings (>500KB):
   - AI summarization → action items
   - Save raw transcript to transcriptions table
   - Format as Markdown → sync to Obsidian via Google Drive
   - Save to Thoughts vector DB

### Document Pipeline (PDF/TXT/MD/CSV/DOCX)

1. Detect format by file extension
2. Extract text (AI Assistant for complex formats, direct parsing for simple)
3. Chunk text at 3500 characters → send chunks to Telegram
4. Save original document to Google Drive ("Memory" folder)
5. Parse and inject full text into Downloads vector DB

### Voice Output (TTS)

If the Main Agent includes a voice tag in its response:
1. Extract text content for TTS
2. Generate audio via OpenAI TTS-1
3. Send both text message and voice message to Telegram

---

## 6. Database Schema

All tables use PostgreSQL with pgvector extension for semantic search.

### Standard vector table schema

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE instructions (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding vector(1536),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Same schema for: about_me, dialogues, thoughts, downloads
```

### Transcriptions table (non-vector)

```sql
CREATE TABLE transcriptions (
    id SERIAL PRIMARY KEY,
    raw_text TEXT NOT NULL,
    summary TEXT,
    source_type VARCHAR(50),  -- 'voice', 'dictaphone', 'mp3'
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Tables

| Table | Content | Vector Search |
|-------|---------|---------------|
| `instructions` | Agent behavioral rules, tool usage guidelines | Yes |
| `about_me` | Personal facts about the user | Yes |
| `dialogues` | Conversation history | Yes |
| `thoughts` | Transcribed voice memos, notes | Yes |
| `downloads` | Extracted text from uploaded documents | Yes |
| `transcriptions` | Raw audio transcripts with summaries | No |

---

## 7. Project Structure

```
sprut/
├── docker-compose.yml
├── .env.example
├── config.yaml
│
├── src/
│   ├── main.py                 # FastAPI app entry point
│   ├── api/
│   │   └── routes.py           # POST /api/process, GET /health
│   ├── core/
│   │   ├── router.py           # Message type detection
│   │   ├── command_handler.py  # Command trigger switch
│   │   └── config.py           # Config loader
│   ├── agents/
│   │   ├── orchestrator.py     # Main Agent (Vector Prompt)
│   │   ├── superinstruction.py # Superinstruction Agent
│   │   ├── terminal.py         # Terminal Sub-Agent
│   │   └── sub_agents/         # Fitness, Doctor, Mechanic, Kaizen
│   ├── ai/
│   │   ├── provider.py         # AIProvider base interface
│   │   ├── openai_provider.py
│   │   ├── claude_provider.py
│   │   ├── gemini_provider.py
│   │   └── custom_provider.py
│   ├── memory/
│   │   ├── vector_store.py     # pgvector CRUD + semantic search
│   │   └── models.py           # SQLAlchemy models
│   ├── processors/
│   │   ├── vision.py           # Image → text description
│   │   ├── audio.py            # Whisper transcription + summarize
│   │   ├── documents.py        # PDF/TXT/MD/CSV text extraction
│   │   └── tts.py              # Text-to-Speech generation
│   └── utils/
│       ├── text_formatter.py   # Format dictation for social media
│       └── chunker.py          # Split text at 3500 char boundaries
│
├── n8n/
│   └── workflows/
│       ├── telegram_main.json  # Telegram webhook → SPRUT API
│       ├── gdrive_sync.json    # Google Drive sync workflow
│       └── obsidian_sync.json  # Obsidian vault sync workflow
│
├── db/
│   └── init.sql                # Table creation + pgvector setup
│
├── tests/
│   ├── test_router.py
│   ├── test_agents.py
│   ├── test_memory.py
│   └── test_processors.py
│
├── docs/
│   └── architecture.md
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 8. Security

- **Telegram auth:** First node checks `user_id` against allowed list in `.env`
- **API keys:** Environment variables only, never in code
- **Internal API:** Bearer token for n8n → FastAPI communication (Docker internal network)
- **Sensitive files:** `.env`, `config.yaml` in `.gitignore`
- **No credentials in git:** Per project CLAUDE.md rules

---

## 9. Error Handling

- **AI calls:** Retry with exponential backoff (3 attempts), optional fallback to another provider
- **Telegram limits:** Auto-chunking at 3500 characters (4096 limit with margin)
- **Audio limits:** Whisper 25MB max, user-friendly error on exceed
- **Unsupported formats:** Clear error message to user
- **Database:** SQLAlchemy connection pool, graceful reconnect
- **n8n → API:** Timeout handling, retry on 5xx

---

## 10. Testing Strategy

- **Unit tests:** Router, command handler, AI provider abstraction, vector store CRUD
- **Integration tests:** Full pipeline (text → agent → response) with mocked AI
- **Framework:** pytest + pytest-asyncio (FastAPI is async)
- **CI-ready:** Tests run without Docker dependencies (mocked DB)

---

## 11. Deployment

Single command deployment:
```bash
docker-compose up -d
```

**Persistent volumes:**
- PostgreSQL data
- n8n configuration and workflows
- Google Drive auth state

**Health monitoring:**
- `GET /health` endpoint on FastAPI
- n8n built-in health check
- PostgreSQL connection check

---

## Decisions Log

| Decision | Rationale |
|----------|-----------|
| FastAPI over Flask | Async support, automatic OpenAPI docs, better performance |
| pgvector over Pinecone/Weaviate | Self-hosted, no external dependencies, integrated with PostgreSQL |
| SQLAlchemy over raw SQL | ORM for maintainability, migration support |
| Multi-provider abstraction | User has keys for multiple LLMs, easy switching |
| Disabled conversation memory | Matches SPRUT 2.0 design — prevents hallucination loops |
| Hidden prompt injection | Matches Vector Prompt methodology from original SPRUT |
| Docker Compose | Single-command deployment, service isolation, volume persistence |
| Sub-agents as isolated pipelines | Prevents main agent confusion, matches original architecture |
