"""Tests for the VectorStore CRUD operations."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.memory.vector_store import VectorStore


@pytest.fixture
def mock_session():
    """Create a mock async database session."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    return session


@pytest.fixture
def store(mock_session):
    """Return a VectorStore for the instructions table."""
    return VectorStore(session=mock_session, table_name="instructions")


def test_invalid_table_name(mock_session):
    """VectorStore should reject unknown table names."""
    with pytest.raises(ValueError, match="Unknown table"):
        VectorStore(session=mock_session, table_name="nonexistent")


@pytest.mark.asyncio
async def test_add_entry(store, mock_session):
    """add() should insert a model instance and commit."""
    mock_session.refresh = AsyncMock(side_effect=lambda obj: setattr(obj, "id", 1))

    with patch.object(store, "_get_embedding", return_value=[0.1] * 1536):
        result = await store.add("Test rule", metadata={"source": "test"})

    mock_session.add.assert_called_once()
    mock_session.commit.assert_called_once()

    # Verify the model instance was created with correct attributes
    added_obj = mock_session.add.call_args[0][0]
    assert added_obj.content == "Test rule"
    assert added_obj.meta == {"source": "test"}


@pytest.mark.asyncio
async def test_add_entry_default_metadata(store, mock_session):
    """add() with no metadata should default to an empty dict."""
    mock_session.refresh = AsyncMock(side_effect=lambda obj: setattr(obj, "id", 2))

    with patch.object(store, "_get_embedding", return_value=[0.0] * 1536):
        await store.add("Another rule")

    added_obj = mock_session.add.call_args[0][0]
    assert added_obj.meta == {}


@pytest.mark.asyncio
async def test_delete_entry(store, mock_session):
    """delete() should return True when a row is removed."""
    mock_result = MagicMock()
    mock_result.rowcount = 1
    mock_session.execute.return_value = mock_result

    deleted = await store.delete(1)
    assert deleted is True
    mock_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_delete_nonexistent(store, mock_session):
    """delete() should return False when the row does not exist."""
    mock_result = MagicMock()
    mock_result.rowcount = 0
    mock_session.execute.return_value = mock_result

    deleted = await store.delete(999)
    assert deleted is False


@pytest.mark.asyncio
async def test_get_embedding_with_custom_fn():
    """_get_embedding should delegate to the injected function."""
    custom_fn = AsyncMock(return_value=[1.0] * 1536)
    session = AsyncMock()
    vs = VectorStore(session=session, table_name="instructions", embedding_fn=custom_fn)

    result = await vs._get_embedding("hello")
    custom_fn.assert_awaited_once_with("hello")
    assert result == [1.0] * 1536


@pytest.mark.asyncio
async def test_get_embedding_without_custom_fn():
    """_get_embedding should return a zero vector when no function is provided."""
    session = AsyncMock()
    vs = VectorStore(session=session, table_name="instructions")

    result = await vs._get_embedding("hello")
    assert result == [0.0] * 1536
    assert len(result) == 1536
