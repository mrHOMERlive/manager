"""Tests for the Superinstruction agent."""

import json

import pytest
from unittest.mock import AsyncMock

from src.agents.superinstruction import SuperinstructionAgent


@pytest.fixture
def mock_ai():
    """Create a mock AI provider."""
    return AsyncMock()


@pytest.fixture
def mock_store():
    """Create a mock instructions store."""
    store = AsyncMock()
    return store


@pytest.mark.asyncio
async def test_saves_when_no_conflicts(mock_ai, mock_store):
    """When no existing rules conflict, the agent should save the new rule."""
    # Store returns no existing rules
    mock_store.search.return_value = []

    # AI says save
    ai_response = json.dumps({
        "action": "save",
        "formatted_rule": "Always greet users politely",
    })
    mock_ai.generate_text.return_value = ai_response

    # Store returns an ID for the new entry
    mock_store.add.return_value = 42

    agent = SuperinstructionAgent(ai_provider=mock_ai, instructions_store=mock_store)
    result = await agent.process("greet users politely")

    assert result["saved"] is True
    assert result["id"] == 42
    assert result["rule"] == "Always greet users politely"

    mock_store.search.assert_awaited_once_with("greet users politely", top_k=5)
    mock_store.add.assert_awaited_once_with("Always greet users politely")


@pytest.mark.asyncio
async def test_rejects_on_conflict(mock_ai, mock_store):
    """When an existing rule conflicts, the agent should reject the new rule."""
    # Store returns a similar existing rule
    mock_store.search.return_value = [
        {
            "id": 1,
            "content": "Never greet users",
            "metadata": {},
            "distance": 0.15,
        }
    ]

    # AI says reject
    ai_response = json.dumps({
        "action": "reject",
        "reason": "Conflicts with existing rule: 'Never greet users'",
    })
    mock_ai.generate_text.return_value = ai_response

    agent = SuperinstructionAgent(ai_provider=mock_ai, instructions_store=mock_store)
    result = await agent.process("always greet users")

    assert result["saved"] is False
    assert "Conflicts with existing rule" in result["reason"]

    mock_store.search.assert_awaited_once_with("always greet users", top_k=5)
    mock_store.add.assert_not_awaited()
