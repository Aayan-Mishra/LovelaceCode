"""Global configuration stored in user's Application Support directory."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from platformdirs import user_config_dir, user_data_dir
from pydantic import BaseModel, Field


APP_NAME = "lovelace-code"
APP_AUTHOR = "Spestly"


def get_global_config_dir() -> Path:
    """Get the global config directory (e.g., ~/Library/Application Support/lovelace-code/)."""
    return Path(user_config_dir(APP_NAME, APP_AUTHOR))


def get_global_data_dir() -> Path:
    """Get the global data directory for caches, models, etc."""
    return Path(user_data_dir(APP_NAME, APP_AUTHOR))


def get_global_config_path() -> Path:
    """Get path to the global config file."""
    return get_global_config_dir() / "config.json"


def get_models_cache_dir() -> Path:
    """Get the directory for cached/downloaded models."""
    return get_global_data_dir() / "models"


class GlobalConfig(BaseModel):
    """Global user configuration (not project-specific)."""

    # User info
    username: str = Field(default="")
    email: str = Field(default="")

    # Onboarding
    onboarding_completed: bool = Field(default=False)
    onboarding_version: int = Field(default=0)  # Bump to re-trigger onboarding

    # Default preferences
    default_backend: Literal[
        "api", "local", "llama", 
        "openai", "anthropic", "gemini", "xai", "openrouter", "openai-compatible", "groq"
    ] = Field(default="api")
    default_model: str = Field(default="Spestly/Lovelace-1-3B")
    theme: Literal["dark", "light", "auto"] = Field(default="dark")

    # API keys (stored locally)
    hf_token: str = Field(default="", description="HuggingFace API token")
    openai_api_key: str = Field(default="", description="OpenAI API key")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    google_api_key: str = Field(default="", description="Google/Gemini API key")
    xai_api_key: str = Field(default="", description="xAI (Grok) API key")
    openrouter_api_key: str = Field(default="", description="OpenRouter API key")
    groq_api_key: str = Field(default="", description="Groq API key")
    openai_compatible_base_url: str = Field(default="", description="OpenAI-compatible base URL")
    openai_compatible_api_key: str = Field(default="", description="OpenAI-compatible API key")

    # Telemetry (opt-in)
    telemetry_enabled: bool = Field(default=False)

    # Statistics
    total_sessions: int = Field(default=0)
    total_messages: int = Field(default=0)
    first_run_at: str = Field(default="")
    last_run_at: str = Field(default="")

    # Feature flags
    auto_update_check: bool = Field(default=True)
    show_tips: bool = Field(default=True)


# Current onboarding version - bump this to re-trigger onboarding for existing users
CURRENT_ONBOARDING_VERSION = 1


def load_global_config() -> GlobalConfig:
    """Load global config from disk, or return defaults."""
    path = get_global_config_path()
    if not path.exists():
        return GlobalConfig()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return GlobalConfig.model_validate(data)
    except Exception:
        return GlobalConfig()


def save_global_config(config: GlobalConfig) -> None:
    """Save global config to disk."""
    path = get_global_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config.model_dump(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def is_first_run() -> bool:
    """Check if this is the first time running the app."""
    config = load_global_config()
    return not config.onboarding_completed


def needs_onboarding() -> bool:
    """Check if onboarding is needed (first run or version bump)."""
    config = load_global_config()
    return (
        not config.onboarding_completed
        or config.onboarding_version < CURRENT_ONBOARDING_VERSION
    )


def setup_api_keys_from_config(config: GlobalConfig) -> None:
    """Set API keys from config as environment variables if not already set."""
    import os
    
    # Map config fields to environment variables
    key_mappings = [
        ("hf_token", ["HUGGINGFACEHUB_API_TOKEN", "HF_TOKEN"]),
        ("openai_api_key", ["OPENAI_API_KEY"]),
        ("anthropic_api_key", ["ANTHROPIC_API_KEY"]),
        ("google_api_key", ["GOOGLE_API_KEY", "GEMINI_API_KEY"]),
        ("xai_api_key", ["XAI_API_KEY"]),
        ("openrouter_api_key", ["OPENROUTER_API_KEY"]),
        ("groq_api_key", ["GROQ_API_KEY"]),
        ("openai_compatible_base_url", ["OPENAI_COMPATIBLE_BASE_URL"]),
        ("openai_compatible_api_key", ["OPENAI_COMPATIBLE_API_KEY"]),
    ]
    
    for config_field, env_vars in key_mappings:
        value = getattr(config, config_field, "")
        if value:
            for env_var in env_vars:
                # Only set if not already in environment
                if not os.getenv(env_var):
                    os.environ[env_var] = value


def mark_session_start() -> GlobalConfig:
    """Update session statistics and return config."""
    config = load_global_config()
    now = datetime.now(timezone.utc).isoformat()

    if not config.first_run_at:
        config.first_run_at = now

    config.last_run_at = now
    config.total_sessions += 1
    save_global_config(config)
    return config


def increment_message_count() -> None:
    """Increment total message count."""
    config = load_global_config()
    config.total_messages += 1
    save_global_config(config)
