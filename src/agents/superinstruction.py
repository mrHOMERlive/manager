"""Superinstruction agent - handles the 'Запомни' command.

Validates new rules against existing ones before saving to the instructions store.
"""

import json
from typing import Any

from src.ai.provider import AIProvider
from src.memory.vector_store import VectorStore

SYSTEM_PROMPT = (
    "You are a rule management assistant. Your job is to check whether a new rule "
    "conflicts with existing rules.\n\n"
    "You will receive a list of existing rules and a new rule to evaluate.\n"
    "If the new rule does NOT conflict with any existing rule, format it clearly "
    "and respond with a JSON object:\n"
    '{"action": "save", "formatted_rule": "<clean version of the rule>"}\n\n'
    "If the new rule CONFLICTS with an existing rule, respond with a JSON object:\n"
    '{"action": "reject", "reason": "<explanation of the conflict>"}\n\n'
    "Respond ONLY with valid JSON, no extra text."
)


class SuperinstructionAgent:
    """Agent that validates and saves new rules (superinstructions)."""

    def __init__(self, ai_provider: AIProvider, instructions_store: VectorStore):
        self.ai = ai_provider
        self.store = instructions_store

    async def process(self, rule_text: str) -> dict[str, Any]:
        """Validate a new rule against existing ones and save if no conflicts.

        Returns:
            {"saved": True, "id": <int>, "rule": <str>} on success
            {"saved": False, "reason": <str>} on conflict/rejection
        """
        # 1. Search existing rules for potential conflicts
        existing_rules = await self.store.search(rule_text, top_k=5)

        # 2. Build prompt with existing rules context
        if existing_rules:
            rules_text = "\n".join(
                f"- {r['content']}" for r in existing_rules
            )
            prompt = (
                f"Existing rules:\n{rules_text}\n\n"
                f"New rule to evaluate:\n{rule_text}"
            )
        else:
            prompt = (
                "There are no existing rules.\n\n"
                f"New rule to evaluate:\n{rule_text}"
            )

        # 3. Ask AI to check for conflicts
        response = await self.ai.generate_text(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
        )

        # 4. Parse AI response
        result = json.loads(response)

        # 5. Act on AI decision
        if result["action"] == "save":
            formatted_rule = result["formatted_rule"]
            entry_id = await self.store.add(formatted_rule)
            return {"saved": True, "id": entry_id, "rule": formatted_rule}
        else:
            return {"saved": False, "reason": result["reason"]}
