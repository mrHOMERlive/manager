"""Tests for SubAgent base class and TerminalAgent."""

import pytest
from unittest.mock import AsyncMock

from src.agents.sub_agents.base import SubAgent
from src.agents.terminal import TerminalAgent, TERMINAL_SYSTEM_PROMPT


@pytest.fixture
def mock_ai():
    """Create a mock AI provider."""
    ai = AsyncMock()
    ai.generate_text.return_value = "AI response"
    return ai


@pytest.fixture
def mock_memory():
    """Create a mock memory store."""
    return AsyncMock()


class TestSubAgent:
    @pytest.mark.asyncio
    async def test_process_without_memory(self, mock_ai):
        """SubAgent without memory should pass query directly to AI."""
        agent = SubAgent(
            name="test-agent",
            system_prompt="You are a test agent.",
            ai_provider=mock_ai,
            memory_store=None,
        )

        result = await agent.process("hello")

        assert result == "AI response"
        mock_ai.generate_text.assert_awaited_once_with(
            prompt="hello",
            system_prompt="You are a test agent.",
        )

    @pytest.mark.asyncio
    async def test_process_with_memory_and_results(self, mock_ai, mock_memory):
        """SubAgent with memory should enrich the prompt with search results."""
        mock_memory.search.return_value = [
            {"content": "Fact A"},
            {"content": "Fact B"},
        ]

        agent = SubAgent(
            name="test-agent",
            system_prompt="You are a test agent.",
            ai_provider=mock_ai,
            memory_store=mock_memory,
        )

        result = await agent.process("tell me something")

        assert result == "AI response"
        mock_memory.search.assert_awaited_once_with("tell me something", top_k=3)

        # Check that the prompt includes context from memory
        call_kwargs = mock_ai.generate_text.call_args.kwargs
        assert "Relevant context:" in call_kwargs["prompt"]
        assert "- Fact A" in call_kwargs["prompt"]
        assert "- Fact B" in call_kwargs["prompt"]
        assert "tell me something" in call_kwargs["prompt"]

    @pytest.mark.asyncio
    async def test_process_with_memory_empty_results(self, mock_ai, mock_memory):
        """SubAgent with memory but no search results should use plain query."""
        mock_memory.search.return_value = []

        agent = SubAgent(
            name="test-agent",
            system_prompt="You are a test agent.",
            ai_provider=mock_ai,
            memory_store=mock_memory,
        )

        result = await agent.process("hello")

        assert result == "AI response"
        mock_ai.generate_text.assert_awaited_once_with(
            prompt="hello",
            system_prompt="You are a test agent.",
        )


class TestTerminalAgent:
    @pytest.mark.asyncio
    async def test_process(self, mock_ai):
        """TerminalAgent should use the terminal system prompt and no memory."""
        agent = TerminalAgent(ai_provider=mock_ai)

        assert agent.name == "terminal"
        assert agent.system_prompt == TERMINAL_SYSTEM_PROMPT
        assert agent.memory is None

        result = await agent.process("how to list files")

        assert result == "AI response"
        mock_ai.generate_text.assert_awaited_once_with(
            prompt="how to list files",
            system_prompt=TERMINAL_SYSTEM_PROMPT,
        )
