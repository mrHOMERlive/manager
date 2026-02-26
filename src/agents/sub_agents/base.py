"""Base class for sub-agents."""

from typing import Optional

from src.ai.provider import AIProvider
from src.memory.vector_store import VectorStore


class SubAgent:
    """A sub-agent that uses AI with optional memory-backed context."""

    def __init__(
        self,
        name: str,
        system_prompt: str,
        ai_provider: AIProvider,
        memory_store: Optional[VectorStore] = None,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.ai = ai_provider
        self.memory = memory_store

    async def process(self, query: str) -> str:
        """Process a query, optionally enriching it with memory context."""
        context = ""
        if self.memory:
            results = await self.memory.search(query, top_k=3)
            if results:
                context = "Relevant context:\n" + "\n".join(
                    f"- {r['content']}" for r in results
                )

        prompt = f"{context}\n\nUser query: {query}" if context else query
        return await self.ai.generate_text(
            prompt=prompt, system_prompt=self.system_prompt
        )
