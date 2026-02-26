# SPRUT 3.0 Architecture

## System Overview

SPRUT 3.0 is a personal AI assistant with a hybrid architecture: a Python FastAPI
service handles all business logic, AI calls, and database operations, while n8n
handles Telegram webhook integration and external sync workflows.

```
Docker Compose
+--------------------------------------------------------------+
|                                                              |
|  Telegram --webhook--> n8n (5678) --REST API-->              |
|  Telegram <--response-- n8n <--------------                  |
|                          |                                   |
|                 Google Drive / Obsidian sync                  |
|                          |                                   |
|  +--------------------------------------------------+       |
|  |       SPRUT Python Service -- FastAPI (8000)      |       |
|  |                                                    |       |
|  |  Router -> Command Handler -> Main Agent           |       |
|  |  AI Layer (multi-provider) -> Sub-Agent Engine     |       |
|  |  File/Audio/Vision Processors -> Memory Manager    |       |
|  +---------------------------+-----------------------+       |
|                              |                               |
|  +---------------------------+-----------------------+       |
|  |       PostgreSQL 16 + pgvector (5432)              |       |
|  |  instructions | about_me | dialogues               |       |
|  |  thoughts | downloads | transcriptions             |       |
|  +---------------------------------------------------+       |
|                                                              |
|  pgAdmin (optional, 5050)                                    |
+--------------------------------------------------------------+
```

## Docker Compose Services

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| `sprut-api` | Custom Dockerfile | 8000 | FastAPI Python service, all business logic |
| `n8n` | n8nio/n8n | 5678 | Workflow orchestration, Telegram webhooks |
| `postgres` | pgvector/pgvector:pg16 | 5432 | Database with vector extension |
| `pgadmin` | dpage/pgadmin4 | 5050 | DB admin UI (optional, `tools` profile) |

### Service Dependencies

- `sprut-api` depends on `postgres` (healthy)
- `n8n` depends on `postgres` (healthy)
- `pgadmin` is optional (enabled with `docker compose --profile tools up`)

### Internal Networking

All services communicate over the Docker Compose default network. The n8n service
calls the FastAPI service at `http://sprut-api:8000` using Docker DNS resolution.
No internal ports are exposed to the host unless explicitly mapped.

## Data Flow

```
1. User sends message to Telegram bot
              |
              v
2. n8n Telegram Trigger (webhook)
              |
              v
3. n8n Code node: Security check
   - Validates user_id against ALLOWED_USER_IDS
   - Rejects unauthorized users
   - Extracts message type and payload
              |
              v
4. n8n HTTP Request: POST http://sprut-api:8000/api/process
   - Bearer token authentication
   - Payload: { message: {...}, user_id: int }
              |
              v
5. FastAPI Router: Determines message type
   - text    -> Command Handler
   - photo   -> Vision Pipeline
   - voice   -> Audio Pipeline
   - document -> Document Pipeline
              |
              v
6. Processing pipeline executes
   - Command triggers route to specific handlers
   - Default text goes through Main Orchestrator Agent
   - Agent queries Instructions DB for routing rules
              |
              v
7. Response returned to n8n
   - { text: "...", voice_data: "..." (optional) }
              |
              v
8. n8n Switch node: Check for voice_data
   - Has voice -> Send text + voice message to Telegram
   - Text only -> Send text message to Telegram
```

## Database Schema

All tables use PostgreSQL 16 with the pgvector extension for semantic search.
The database is initialized by `db/init.sql` on first startup.

### Vector-Enabled Tables

These five tables share a common schema with a `vector(1536)` embedding column
for semantic similarity search via cosine distance:

```sql
CREATE TABLE <table_name> (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding vector(1536),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

| Table | Content | Purpose |
|-------|---------|---------|
| `instructions` | Agent behavioral rules, tool usage guidelines | Vector Prompt: agent consults this before every response |
| `about_me` | Personal facts about the user | Context for personalized responses |
| `dialogues` | Conversation history | Reference for past interactions |
| `thoughts` | Transcribed voice memos, notes | User thought archive |
| `downloads` | Extracted text from uploaded documents | Document knowledge base |

### Non-Vector Table

```sql
CREATE TABLE transcriptions (
    id BIGSERIAL PRIMARY KEY,
    raw_text TEXT NOT NULL,
    summary TEXT,
    source_type VARCHAR(50),  -- 'voice', 'dictaphone', 'mp3'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

| Table | Content | Purpose |
|-------|---------|---------|
| `transcriptions` | Raw audio transcripts with AI summaries | Archive of full transcriptions |

### Indexes

IVFFlat indexes are created on the `embedding` column of all vector-enabled tables
for efficient cosine similarity search (`vector_cosine_ops`, lists=100).

## AI Provider Configuration

SPRUT 3.0 supports multiple AI providers through a unified abstraction layer.
Each provider implements a common interface:

- `generate_text(prompt, system_prompt, temperature)` -- text completion
- `generate_with_vision(prompt, image_data)` -- image understanding
- `transcribe_audio(audio_data)` -- speech-to-text
- `text_to_speech(text)` -- text-to-speech
- `create_embedding(text)` -- vector embedding generation

### Provider Hierarchy

```
AIProvider (base interface)
  +-- OpenAIProvider    -- GPT-4o mini, Whisper, TTS-1, embeddings
  +-- ClaudeProvider    -- Claude Sonnet/Haiku for text
  +-- GeminiProvider    -- Gemini Flash for terminal queries
  +-- CustomProvider    -- User's remote LLM server
```

### Configuration File (config.yaml)

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
```

API keys are stored in environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`GEMINI_API_KEY`, `CUSTOM_LLM_KEY`), never in config files.

## Command Trigger Reference

Text messages are checked against these trigger phrases (case-insensitive,
prefix match). If no trigger matches, the message is routed to the Main
Orchestrator Agent.

| Trigger Phrase | Command | Action |
|----------------|---------|--------|
| `Запомни` | REMEMBER | Superinstruction Agent validates and saves rule to Instructions DB |
| `Запиши обо мне` | WRITE_ABOUT_ME | Direct save to About Me DB |
| `Удали обо мне` | DELETE_ABOUT_ME | Agent finds matching entry and deletes from About Me DB |
| `Удали инструкцию` | DELETE_INSTRUCTION | Agent finds matching entry and deletes from Instructions DB |
| `Запиши мысль` | RECORD_THOUGHT | Format as Markdown, save to Obsidian + Thoughts DB |
| `Спроси терминал` | ASK_TERMINAL | Terminal Sub-Agent (Gemini CLI) executes and returns response |

## Sub-Agent Reference

Sub-agents are isolated pipelines with their own system prompts, tools, and memory.
The Main Orchestrator delegates to them based on rules found in the Instructions DB.

| Agent | Purpose | Trigger | Memory |
|-------|---------|---------|--------|
| Main Orchestrator | General assistant, routes to sub-agents | Default (no trigger) | Instructions DB (read-only) |
| Superinstruction | Add/validate behavioral rules | "Запомни" command | Instructions DB (read/write) |
| Terminal | Server queries via Gemini CLI | "Спроси терминал" command | None |
| Fitness Trainer | Workouts, nutrition plans | Instructions DB routing rule | Own vector DB partition |
| Doctor | Medical questions, health advice | Instructions DB routing rule | Own vector DB partition |
| Auto Mechanic | Car maintenance, diagnostics | Instructions DB routing rule | Own vector DB partition |
| Kaizen | Self-improvement, habit tracking | Instructions DB routing rule | Own vector DB partition |

### Superinstruction Agent Flow

1. User sends "Запомни: [rule]"
2. Command handler bypasses Main Agent, triggers Superinstruction Agent
3. Agent searches Instructions DB for duplicates or contradictions
4. If conflict found, rejects with explanation to user
5. If safe, formats rule and saves to Instructions DB with embedding
6. Confirms success to user

## Vector Prompt Methodology

The Main Orchestrator Agent uses a methodology called "Vector Prompt" where the
agent's intelligence comes from the database rather than from a massive system prompt.

### How It Works

1. **Minimal system prompt**: Contains only basic duties ("You are a personal
   assistant") and a list of available tools.

2. **Hidden prompt injection**: Before every user query, a hidden instruction is
   prepended:
   > "Обязательно зайди в инструкцию и найди правило, на какой инструмент или
   > агент отправить запрос, или ответь самостоятельно"
   (Translation: "You must check the instructions and find the rule for which
   tool or agent to route the request to, or answer on your own")

3. **Database-driven behavior**: This forces the agent to query the Instructions
   vector DB using semantic search before generating any response. The matching
   instructions tell the agent how to behave, which sub-agent to delegate to,
   and what tools to use.

4. **Key settings**:
   - Temperature: 0.1 (strict instruction following)
   - Conversational memory: DISABLED (prevents hallucination loops)
   - Each request is independent; context comes from DB queries

### Benefits

- Behavior can be modified at runtime by adding/removing instructions (via
  the "Запомни" command) without redeploying
- No massive system prompt that consumes tokens
- Instructions are semantically searched, so only relevant rules are loaded
- Sub-agent routing is data-driven, not hardcoded

## Input Processing Pipelines

### Message Router

```
Telegram Message
    +-- text      -> Command Switch (trigger phrase detection)
    +-- photo     -> Vision Pipeline (GPT-4o mini Vision)
    +-- voice     -> Audio Pipeline (Whisper transcription)
    +-- document  -> Document Pipeline (text extraction)
```

### Vision Pipeline

1. Send status message to Telegram
2. Download image from Telegram
3. GPT-4o mini Vision (high detail) generates image description
4. Append context and pass to Main Agent via Command Switch

### Audio Pipeline

1. Check file size (threshold: 500KB)
2. Transcribe via OpenAI Whisper (25MB limit)
3. Short voice notes: pass transcript to Command Switch as text
4. Long recordings (>500KB): AI summarization, save raw transcript to
   `transcriptions` table, format as Markdown for Obsidian, save to Thoughts DB

### Document Pipeline

1. Detect format by file extension (PDF/TXT/MD/CSV/DOCX)
2. Extract text content
3. Chunk text at 3500 characters, send chunks to Telegram
4. Save original document to Google Drive "Memory" folder
5. Parse and save full text to Downloads vector DB

## Error Handling

- **AI calls**: Retry with exponential backoff (3 attempts), optional fallback
  to another provider
- **Telegram limits**: Auto-chunking at 3500 characters (4096 limit with margin)
- **Audio limits**: Whisper 25MB max, user-friendly error on exceed
- **Unsupported formats**: Clear error message to user
- **Database**: SQLAlchemy connection pool with graceful reconnect
- **n8n to API**: Timeout handling (120s), retry on 5xx responses

## Security

- **Telegram auth**: n8n Code node validates `user_id` against allowed list
- **API auth**: Bearer token for n8n to FastAPI communication
- **API keys**: Environment variables only, never in code or config files
- **Sensitive files**: `.env` and `config.yaml` listed in `.gitignore`
- **Docker isolation**: Services communicate over internal Docker network
