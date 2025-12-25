"""OpenAI API backend."""

from __future__ import annotations

import os
from typing import Iterable

from .base import Backend, BackendError, GenerationConfig
from .registry import register_backend


@register_backend("openai")
class OpenAIBackend(Backend):
    """Uses OpenAI API (GPT-5.x, o-series, etc.)."""

    def __init__(self):
        self._client = None
        self._model_id = None

    def is_available(self) -> bool:
        try:
            import openai  # noqa: F401
            return True
        except ImportError:
            return False

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise BackendError(
                    "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
                )
            self._client = OpenAI(api_key=api_key)
        return self._client

    def load_model(self, model_id: str, **kwargs) -> None:
        self._model_id = model_id

    def generate(self, prompt: str, config: GenerationConfig) -> str:
        client = self._get_client()
        model = config.model or self._model_id

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                stop=config.stop_sequences if config.stop_sequences else None,
            )
            content = response.choices[0].message.content
            return content if content else ""
        except Exception as exc:
            raise BackendError(f"OpenAI API error: {exc}") from exc

    def stream(self, prompt: str, config: GenerationConfig) -> Iterable[str]:
        client = self._get_client()
        model = config.model or self._model_id

        try:
            stream = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                stop=config.stop_sequences if config.stop_sequences else None,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as exc:
            raise BackendError(f"OpenAI API stream error: {exc}") from exc
