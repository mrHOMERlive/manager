"""Main orchestrator agent (Vector Prompt)."""

from typing import Optional

SYSTEM_PROMPT = (
    "You are SPRUT, a personal AI assistant. "
    "Your duties: answer questions, manage tasks, and help the user. "
    "Available tools: fitness_trainer, scheduler, notes, web_search."
)

HIDDEN_INJECTION = (
    "Обязательно зайди в инструкцию и найди правило, "
    "которое относится к запросу пользователя. "
    "Следуй этому правилу при ответе."
)


class Orchestrator:
    """Builds a vector-augmented prompt and delegates to an AI provider."""

    def __init__(self, ai_provider, instructions_store):
        self.ai_provider = ai_provider
        self.instructions_store = instructions_store

    async def process(
        self, user_text: str, context: Optional[str] = None
    ) -> dict:
        """Process a user query through the orchestrator pipeline.

        1. Query instructions DB for relevant rules.
        2. Build prompt: hidden injection + context + rules + user text.
        3. Call AI provider.
        4. Check for [VOICE] tag in response.
        5. Return {"text": clean_text, "voice": True/None}.
        """
        # 1. Retrieve relevant instructions
        instructions = await self.instructions_store.search(
            user_text, top_k=5
        )

        # 2. Build the augmented prompt
        parts: list[str] = [HIDDEN_INJECTION]

        if context:
            parts.append(f"Context: {context}")

        if instructions:
            rules = "\n".join(
                f"- {item['content']}" for item in instructions
            )
            parts.append(f"Relevant rules:\n{rules}")

        parts.append(f"User: {user_text}")

        prompt = "\n\n".join(parts)

        # 3. Call the AI provider
        response = await self.ai_provider.generate_text(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.1,
        )

        # 4. Check for [VOICE] tag
        voice = True if "[VOICE]" in response else None
        clean_text = response.replace("[VOICE]", "").strip()

        # 5. Return structured result
        return {"text": clean_text, "voice": voice}
