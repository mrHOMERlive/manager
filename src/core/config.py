"""Configuration loader for SPRUT 3.0"""

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings from environment variables."""

    # Database
    database_url: str = Field(default="postgresql+asyncpg://sprut:sprut_password@localhost:5432/sprut")

    # Telegram
    telegram_bot_token: str = Field(default="")
    allowed_user_ids: str = Field(default="")

    # AI Keys
    openai_api_key: str = Field(default="")
    anthropic_api_key: str = Field(default="")
    gemini_api_key: str = Field(default="")
    custom_llm_url: str = Field(default="")
    custom_llm_key: str = Field(default="")

    # Internal
    api_secret_key: str = Field(default="dev-secret")

    model_config = {"env_file": ".env", "extra": "ignore"}

    @property
    def allowed_user_id_list(self) -> list[int]:
        if not self.allowed_user_ids:
            return []
        return [int(uid.strip()) for uid in self.allowed_user_ids.split(",") if uid.strip()]


def load_yaml_config(path: Optional[str] = None) -> dict:
    """Load YAML configuration file."""
    config_path = Path(path) if path else Path("config.yaml")
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


settings = Settings()
yaml_config = load_yaml_config()
