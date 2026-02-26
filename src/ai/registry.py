"""AI provider registry — stores providers and maps roles to them."""
from __future__ import annotations

from typing import TYPE_CHECKING

from src.ai.provider import AIProvider

if TYPE_CHECKING:
    from src.core.config import Settings


class AIRegistry:
    """Central registry that maps roles to AI providers."""

    def __init__(self) -> None:
        self.providers: dict[str, AIProvider] = {}
        self._defaults: dict[str, str] = {}

    def register(self, name: str, provider: AIProvider) -> None:
        """Register a provider by name."""
        self.providers[name] = provider

    def set_default(self, role: str, provider_name: str) -> None:
        """Map a role (text/vision/transcription/tts/embeddings) to a provider."""
        self._defaults[role] = provider_name

    def get_provider(self, role: str) -> AIProvider:
        """Return the provider for a role. Raises KeyError if missing."""
        if role not in self._defaults:
            raise KeyError(f"No provider configured for role '{role}'")
        name = self._defaults[role]
        if name not in self.providers:
            raise KeyError(f"Provider '{name}' not registered")
        return self.providers[name]

    @classmethod
    def from_config(cls, yaml_config: dict, settings: Settings) -> AIRegistry:
        """Build a registry from config.yaml values and env-based Settings."""
        registry = cls()
        ai_cfg = yaml_config.get("ai", {})
        providers_cfg = ai_cfg.get("providers", {})

        # --- OpenAI -----------------------------------------------------------
        if "openai" in providers_cfg and settings.openai_api_key:
            from src.ai.openai_provider import OpenAIProvider

            pcfg = providers_cfg["openai"]
            registry.register(
                "openai",
                OpenAIProvider(
                    api_key=settings.openai_api_key,
                    model=pcfg.get("model", "gpt-4o-mini"),
                    temperature=pcfg.get("temperature", 0.1),
                ),
            )

        # --- Claude -----------------------------------------------------------
        if "claude" in providers_cfg and settings.anthropic_api_key:
            from src.ai.claude_provider import ClaudeProvider

            pcfg = providers_cfg["claude"]
            registry.register(
                "claude",
                ClaudeProvider(
                    api_key=settings.anthropic_api_key,
                    model=pcfg.get("model", "claude-sonnet-4-20250514"),
                    temperature=pcfg.get("temperature", 0.1),
                ),
            )

        # --- Gemini -----------------------------------------------------------
        if "gemini" in providers_cfg and settings.gemini_api_key:
            from src.ai.gemini_provider import GeminiProvider

            pcfg = providers_cfg["gemini"]
            registry.register(
                "gemini",
                GeminiProvider(
                    api_key=settings.gemini_api_key,
                    model=pcfg.get("model", "gemini-2.0-flash"),
                    temperature=pcfg.get("temperature", 0.1),
                ),
            )

        # --- Custom (OpenAI-compatible) ---------------------------------------
        if "custom" in providers_cfg and settings.custom_llm_url:
            from src.ai.custom_provider import CustomProvider

            pcfg = providers_cfg["custom"]
            registry.register(
                "custom",
                CustomProvider(
                    api_key=settings.custom_llm_key or "no-key",
                    base_url=settings.custom_llm_url,
                    model=pcfg.get("model", "default"),
                    temperature=pcfg.get("temperature", 0.1),
                ),
            )

        # --- Role defaults ----------------------------------------------------
        role_map = {
            "text": "default_text",
            "vision": "default_vision",
            "transcription": "default_transcription",
            "tts": "default_tts",
            "embeddings": "default_embeddings",
        }
        for role, config_key in role_map.items():
            provider_name = ai_cfg.get(config_key)
            if provider_name and provider_name in registry.providers:
                registry.set_default(role, provider_name)

        return registry
