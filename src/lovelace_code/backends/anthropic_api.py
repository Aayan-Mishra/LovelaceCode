"""Anthropic API backend."""

from __future__ import annotations

import os
from typing import Iterable

from .base import Backend, BackendError, GenerationConfig
from .registry import register_backend


@register_backend("anthropic")
class AnthropicBackend(Backend):
    """Uses Anthropic API (Claude Opus 4.5, Sonnet 4.5, Haiku 4.5, etc.)."""

    def __init__(self):
        self._client = None
        self._model_id = None

    def is_available(self) -> bool:
        try:
            import anthropic  # noqa: F401
            return True
        except ImportError:
            return False

    def _get_client(self):
        if self._client is None:
            import anthropic

            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise BackendError(
                    "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable."
                )
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def load_model(self, model_id: str, **kwargs) -> None:
        self._model_id = model_id

    def generate(self, prompt: str, config: GenerationConfig) -> str:
        client = self._get_client()
        model = config.model or self._model_id

        try:
            response = client.messages.create(
                model=model,
                max_tokens=config.max_new_tokens,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.temperature,
                top_p=config.top_p,
            )
            # Anthropic returns content blocks
            if response.content:
                return "".join(
                    block.text for block in response.content if hasattr(block, "text")
                )
            return ""
        except Exception as exc:
            raise BackendError(f"Anthropic API error: {exc}") from exc

    def stream(self, prompt: str, config: GenerationConfig) -> Iterable[str]:
        client = self._get_client()
        model = config.model or self._model_id

        try:
            with client.messages.stream(
                model=model,
                max_tokens=config.max_new_tokens,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.temperature,
                top_p=config.top_p,
            ) as stream:
                for text in stream.text_stream:
                    yield text
        except Exception as exc:
            raise BackendError(f"Anthropic API stream error: {exc}") from exc
