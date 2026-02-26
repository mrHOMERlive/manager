# SPRUT 3.0

Personal AI assistant with multi-provider AI support, Telegram integration via n8n,
and PostgreSQL vector memory. Built with a hybrid architecture: Python FastAPI handles
business logic and AI processing, while n8n orchestrates Telegram webhooks and
external sync workflows.

## Prerequisites

- **Docker** and **Docker Compose** (v2+)
- **Telegram Bot Token** (from [@BotFather](https://t.me/BotFather))
- **API keys** for at least one AI provider:
  - OpenAI API key (primary: GPT-4o mini, Whisper, TTS, embeddings)
  - Anthropic API key (optional: Claude for text)
  - Google Gemini API key (optional: terminal queries)
  - Custom LLM server URL and key (optional)

## Quick Start

### 1. Clone and configure

```bash
git clone <repository-url>
cd sprut
cp .env.example .env
```

### 2. Edit `.env` with your values

```bash
# Required
TELEGRAM_BOT_TOKEN=your_bot_token_from_botfather
ALLOWED_USER_IDS=your_telegram_user_id
OPENAI_API_KEY=sk-...
API_SECRET_KEY=generate_a_random_secret

# Optional providers
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
```

### 3. Start all services

```bash
docker compose up -d
```

### 4. Import n8n workflows

1. Open n8n at `http://localhost:5678`
2. Import workflow files from `n8n/workflows/`
3. Configure credentials (see `n8n/README.md` for details)
4. Activate the Telegram Main workflow

### 5. Test

Send a message to your Telegram bot. You should receive a response from SPRUT.

## Project Structure

```
sprut/
├── docker-compose.yml          # All services definition
├── Dockerfile                  # Python service image
├── .env.example                # Environment variable template
├── config.yaml                 # AI provider configuration
│
├── src/                        # Python FastAPI service
│   ├── main.py                 # App entry point
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
│   │   ├── openai_provider.py  # OpenAI (GPT, Whisper, TTS)
│   │   ├── claude_provider.py  # Anthropic Claude
│   │   ├── gemini_provider.py  # Google Gemini
│   │   └── custom_provider.py  # Custom LLM server
│   ├── memory/
│   │   ├── vector_store.py     # pgvector CRUD + semantic search
│   │   └── models.py           # SQLAlchemy models
│   └── processors/
│       ├── vision.py           # Image description
│       ├── audio.py            # Whisper transcription
│       ├── documents.py        # PDF/TXT/MD/CSV extraction
│       └── tts.py              # Text-to-Speech
│
├── n8n/
│   └── workflows/
│       ├── telegram_main.json  # Telegram webhook -> SPRUT API
│       ├── gdrive_sync.json    # Google Drive sync
│       └── obsidian_sync.json  # Obsidian vault sync
│
├── db/
│   └── init.sql                # Table creation + pgvector setup
│
├── tests/                      # pytest test suite
│   ├── test_router.py
│   ├── test_agents.py
│   ├── test_memory.py
│   └── test_processors.py
│
└── docs/
    └── architecture.md         # Full architecture documentation
```

## Configuration

### Environment Variables (.env)

| Variable | Required | Description |
|----------|----------|-------------|
| `TELEGRAM_BOT_TOKEN` | Yes | Telegram bot token from BotFather |
| `ALLOWED_USER_IDS` | Yes | Comma-separated Telegram user IDs |
| `OPENAI_API_KEY` | Yes | OpenAI API key for GPT, Whisper, TTS, embeddings |
| `API_SECRET_KEY` | Yes | Bearer token for n8n to FastAPI auth |
| `DATABASE_URL` | No | PostgreSQL connection string (has default) |
| `ANTHROPIC_API_KEY` | No | Anthropic API key for Claude |
| `GEMINI_API_KEY` | No | Google Gemini API key |
| `CUSTOM_LLM_URL` | No | Custom LLM server base URL |
| `CUSTOM_LLM_KEY` | No | Custom LLM server API key |
| `N8N_BASIC_AUTH_USER` | No | n8n web UI username (default: admin) |
| `N8N_BASIC_AUTH_PASSWORD` | No | n8n web UI password (default: admin) |

### AI Provider Config (config.yaml)

Controls which provider is used for each task type (text, vision, transcription,
TTS, embeddings) and model-specific settings. API keys are read from environment
variables, not from this file.

### Switching Providers

To use Claude instead of OpenAI for text generation, change `config.yaml`:

```yaml
ai:
  default_text: "claude"
```

## Command Reference

Send these commands as Telegram messages to trigger specific actions:

| Command | Description |
|---------|-------------|
| `Запомни: [rule]` | Save a new behavioral rule to Instructions DB |
| `Запиши обо мне: [fact]` | Save a personal fact to About Me DB |
| `Удали обо мне: [description]` | Find and delete a fact from About Me DB |
| `Удали инструкцию: [description]` | Find and delete a rule from Instructions DB |
| `Запиши мысль: [text]` | Save a thought to Obsidian + Thoughts DB |
| `Спроси терминал: [query]` | Query the server via Terminal Sub-Agent |

Any message without a trigger phrase is processed by the Main Orchestrator Agent
using the Vector Prompt methodology.

## Services

| Service | URL | Description |
|---------|-----|-------------|
| SPRUT API | http://localhost:8000 | FastAPI service |
| SPRUT API Health | http://localhost:8000/health | Health check endpoint |
| n8n | http://localhost:5678 | Workflow editor UI |
| pgAdmin | http://localhost:5050 | Database admin (requires `--profile tools`) |

### Starting with pgAdmin

```bash
docker compose --profile tools up -d
```

## Development

### Running tests

```bash
pip install -r requirements.txt
pytest tests/
```

### API documentation

When the service is running, OpenAPI docs are available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Architecture

For detailed architecture documentation including data flow diagrams, database
schema, sub-agent reference, and the Vector Prompt methodology, see
[docs/architecture.md](docs/architecture.md).
