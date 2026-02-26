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
