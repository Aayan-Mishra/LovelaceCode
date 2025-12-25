from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


DEFAULT_FAST_MODEL = "Spestly/Lovelace-1-3B"
DEFAULT_DEEP_MODEL = "Spestly/Lovelace-1-7B"

BackendType = Literal[
    "api",  # HuggingFace Inference API
    "local",  # Local transformers
    "llama",  # llama.cpp / GGUF
    "openai",  # OpenAI API
    "anthropic",  # Anthropic API
    "gemini",  # Google Gemini API
    "xai",  # xAI (Grok) API
    "openrouter",  # OpenRouter API
    "groq",  # Groq API
    "openai-compatible",  # Any OpenAI-compatible endpoint
]


class LovelaceConfig(BaseModel):
    # Backend settings
    backend: BackendType = Field(default="api", description="Model backend: api, local, or llama")
    model: str = Field(default=DEFAULT_FAST_MODEL)
    deep_model: str = Field(default=DEFAULT_DEEP_MODEL)
    
    # Generation settings
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_new_tokens: int = Field(default=1024, ge=16, le=8192)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1, le=500)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)
    
    # Behavior
    auto_approve_tools: bool = Field(default=False, description="Auto-approve tool calls")
    stream_output: bool = Field(default=True, description="Stream model output")
    
    # Metadata
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ProjectMemory(BaseModel):
    """Persistent memory for a project."""
    
    summary: str = Field(default="", description="AI-generated project summary")
    key_files: list[str] = Field(default_factory=list, description="Important files")
    notes: list[str] = Field(default_factory=list, description="User/AI notes")
    tech_stack: list[str] = Field(default_factory=list, description="Detected technologies")


def load_config(path: Path) -> LovelaceConfig:
    if not path.exists():
        return LovelaceConfig()
    data = json.loads(path.read_text(encoding="utf-8"))
    return LovelaceConfig.model_validate(data)


def save_config(path: Path, config: LovelaceConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config.model_dump(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_memory(path: Path) -> ProjectMemory:
    if not path.exists():
        return ProjectMemory()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return ProjectMemory.model_validate(data)
    except Exception:
        return ProjectMemory()


def save_memory(path: Path, memory: ProjectMemory) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(memory.model_dump(), indent=2) + "\n",
        encoding="utf-8",
    )
