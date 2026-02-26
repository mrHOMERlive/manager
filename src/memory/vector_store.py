"""Vector store with pgvector for semantic search."""

from typing import Optional

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.memory.models import TABLE_MODELS


class VectorStore:
    """CRUD + semantic search on vector-enabled tables."""

    def __init__(
        self,
        session: AsyncSession,
        table_name: str,
        embedding_fn=None,
    ):
        if table_name not in TABLE_MODELS:
            raise ValueError(f"Unknown table: {table_name}")
        self.session = session
        self.model = TABLE_MODELS[table_name]
        self.table_name = table_name
        self._embedding_fn = embedding_fn

    async def _get_embedding(self, text_content: str) -> list[float]:
        """Return an embedding vector for the given text.

        Uses the injected embedding function when available,
        otherwise falls back to a zero vector (useful for tests).
        """
        if self._embedding_fn:
            return await self._embedding_fn(text_content)
        return [0.0] * 1536

    async def add(self, content: str, metadata: Optional[dict] = None) -> int:
        """Insert a new entry and return its id."""
        embedding = await self._get_embedding(content)
        # The model attribute is 'meta' (mapped to the 'metadata' DB column).
        entry = self.model(
            content=content,
            meta=metadata or {},
            embedding=embedding,
        )
        self.session.add(entry)
        await self.session.commit()
        await self.session.refresh(entry)
        return entry.id

    async def delete(self, entry_id: int) -> bool:
        """Delete an entry by id. Returns True if a row was actually removed."""
        stmt = delete(self.model).where(self.model.id == entry_id)
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount > 0

    async def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Semantic search using cosine distance on pgvector embeddings."""
        query_embedding = await self._get_embedding(query)
        stmt = (
            select(
                self.model.id,
                self.model.content,
                self.model.meta.label("metadata"),
                self.model.embedding.cosine_distance(query_embedding).label("distance"),
            )
            .order_by("distance")
            .limit(top_k)
        )
        result = await self.session.execute(stmt)
        rows = result.all()
        return [
            {
                "id": r.id,
                "content": r.content,
                "metadata": r.metadata,
                "distance": r.distance,
            }
            for r in rows
        ]

    async def get_all(self) -> list[dict]:
        """Return every entry in the table (no distance calculation)."""
        stmt = select(
            self.model.id,
            self.model.content,
            self.model.meta.label("metadata"),
        )
        result = await self.session.execute(stmt)
        return [
            {
                "id": r.id,
                "content": r.content,
                "metadata": r.metadata,
            }
            for r in result.all()
        ]
