import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.ai.provider import AIProvider
from src.ai.openai_provider import OpenAIProvider


def test_base_provider_is_abstract():
    with pytest.raises(TypeError):
        AIProvider()


@pytest.mark.asyncio
async def test_openai_generate_text():
    provider = OpenAIProvider(
        api_key="test-key", model="gpt-4o-mini", temperature=0.1
    )
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Hello!"))]
    with patch.object(
        provider.client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        result = await provider.generate_text(
            "Hi", system_prompt="You are helpful"
        )
        assert result == "Hello!"


@pytest.mark.asyncio
async def test_openai_create_embedding():
    provider = OpenAIProvider(api_key="test-key", model="gpt-4o-mini")
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1] * 1536)]
    with patch.object(
        provider.client.embeddings,
        "create",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        result = await provider.create_embedding("test text")
        assert len(result) == 1536
