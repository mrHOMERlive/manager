"""Terminal sub-agent for handling terminal/command-line related queries."""

from src.agents.sub_agents.base import SubAgent
from src.ai.provider import AIProvider

TERMINAL_SYSTEM_PROMPT = (
    "You are a terminal command assistant. You help users with shell commands, "
    "terminal operations, and command-line tools. Provide clear, concise, and "
    "accurate responses about terminal usage. When suggesting commands, explain "
    "what they do and any important flags or options."
)


class TerminalAgent(SubAgent):
    """Sub-agent specialized for terminal/command-line queries."""

    def __init__(self, ai_provider: AIProvider):
        super().__init__(
            name="terminal",
            system_prompt=TERMINAL_SYSTEM_PROMPT,
            ai_provider=ai_provider,
            memory_store=None,
        )
