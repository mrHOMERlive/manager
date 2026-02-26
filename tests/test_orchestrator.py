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
