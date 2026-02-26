import pytest
from src.ai.registry import AIRegistry


def test_registry_get_default():
    registry = AIRegistry()
    # Manually set up a provider for testing
    from src.ai.openai_provider import OpenAIProvider
    registry.providers["openai"] = OpenAIProvider(api_key="test")
    registry.set_default("text", "openai")
    provider = registry.get_provider("text")
    assert provider is not None


def test_registry_unknown_role():
    registry = AIRegistry()
    with pytest.raises(KeyError):
        registry.get_provider("nonexistent_role")
