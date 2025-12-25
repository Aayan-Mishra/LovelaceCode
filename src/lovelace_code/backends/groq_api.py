"""Groq API backend - Ultra-fast inference with LPUs."""

from __future__ import annotations

import os
from typing import Iterator

from .base import Backend, BackendError, GenerationConfig
from .registry import register_backend


@register_backend("groq")
class GroqBackend(Backend):
    """Groq API backend using their OpenAI-compatible API.
    
    Groq provides ultra-fast inference using their custom LPU hardware.
    Supports Llama, Mixtral, Gemma, and other open models.
    
    Environment variable: GROQ_API_KEY
    """

    name = "groq"
    
    # Groq-specific models (as of late 2025)
    GROQ_MODELS = {
        # Llama 3.3
        "llama-3.3-70b-versatile": {"context": 131072, "description": "Llama 3.3 70B - versatile, fast"},
        "llama-3.3-70b-specdec": {"context": 8192, "description": "Llama 3.3 70B - speculative decoding"},
        # Llama 3.1
        "llama-3.1-8b-instant": {"context": 131072, "description": "Llama 3.1 8B - instant responses"},
        "llama-3.1-70b-versatile": {"context": 131072, "description": "Llama 3.1 70B - versatile"},
        # Llama 3.2 Vision
        "llama-3.2-11b-vision-preview": {"context": 8192, "description": "Llama 3.2 11B Vision"},
        "llama-3.2-90b-vision-preview": {"context": 8192, "description": "Llama 3.2 90B Vision"},
        # Llama Guard
        "llama-guard-3-8b": {"context": 8192, "description": "Llama Guard 3 - content moderation"},
        # Mixtral
        "mixtral-8x7b-32768": {"context": 32768, "description": "Mixtral 8x7B MoE"},
        # Gemma
        "gemma2-9b-it": {"context": 8192, "description": "Gemma 2 9B Instruct"},
        # Qwen
        "qwen-qwq-32b": {"context": 131072, "description": "Qwen QwQ 32B - reasoning"},
        # Whisper (audio)
        "whisper-large-v3": {"context": 448, "description": "Whisper Large V3 - transcription"},
        "whisper-large-v3-turbo": {"context": 448, "description": "Whisper Large V3 Turbo"},
        "distil-whisper-large-v3-en": {"context": 448, "description": "Distil Whisper - English"},
    }

    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    def __init__(self):
        self._client = None

    def _get_client(self):
        """Lazy-load the OpenAI client configured for Groq."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise BackendError(
                    "OpenAI SDK required for Groq backend. Install with: pip install openai"
                ) from exc

            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise BackendError(
                    "GROQ_API_KEY environment variable not set. "
                    "Get your API key at https://console.groq.com/keys"
                )

            self._client = OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
            )
        return self._client

    def is_available(self) -> bool:
        """Check if Groq API is available."""
        try:
            from openai import OpenAI  # noqa: F401
            return bool(os.environ.get("GROQ_API_KEY"))
        except ImportError:
            return False

    def load_model(self, model_id: str, **kwargs) -> None:
        """Prepare model for inference (no-op for API backends)."""
        # API backends don't need to load models locally
        # Just validate the client can be created
        _ = self._get_client()

    def generate(self, prompt: str, config: GenerationConfig) -> str:
        """Generate a response using Groq API."""
        client = self._get_client()
        model = config.model or self.DEFAULT_MODEL

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.temperature,
                max_tokens=config.max_new_tokens,
                top_p=config.top_p,
                stream=False,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            raise BackendError(f"Groq API error: {exc}") from exc

    def stream(self, prompt: str, config: GenerationConfig) -> Iterator[str]:
        """Stream a response from Groq API."""
        client = self._get_client()
        model = config.model or self.DEFAULT_MODEL

        try:
            stream = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.temperature,
                max_tokens=config.max_new_tokens,
                top_p=config.top_p,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as exc:
            raise BackendError(f"Groq API streaming error: {exc}") from exc

    @classmethod
    def list_models(cls) -> list[str]:
        """List available Groq models."""
        return list(cls.GROQ_MODELS.keys())

    @classmethod
    def get_model_info(cls, model: str) -> dict:
        """Get info about a specific model."""
        if model in cls.GROQ_MODELS:
            return cls.GROQ_MODELS[model]
        return {"context": 8192, "description": "Unknown Groq model"}
