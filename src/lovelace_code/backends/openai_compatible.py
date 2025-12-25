"""OpenAI-compatible API backend for custom endpoints."""

from __future__ import annotations

import os
from typing import Iterable

from .base import Backend, BackendError, GenerationConfig
from .registry import register_backend


@register_backend("openai-compatible")
class OpenAICompatibleBackend(Backend):
    """Uses any OpenAI-compatible API endpoint (LocalAI, LM Studio, ollama, vLLM, etc.)."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        self._client = None
        self._model_id = None
        self._base_url = base_url
        self._api_key = api_key

    def is_available(self) -> bool:
        try:
            import openai  # noqa: F401
            return True
        except ImportError:
            return False

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            base_url = self._base_url or os.getenv("OPENAI_COMPATIBLE_BASE_URL")
            api_key = self._api_key or os.getenv("OPENAI_COMPATIBLE_API_KEY", "not-needed")

            if not base_url:
                raise BackendError(
                    "OpenAI-compatible base URL not set. "
                    "Set OPENAI_COMPATIBLE_BASE_URL environment variable or pass base_url."
                )
            
            self._client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )
        return self._client

    def load_model(self, model_id: str, **kwargs) -> None:
        self._model_id = model_id
        # Allow setting base_url/api_key via kwargs
        if "base_url" in kwargs:
            self._base_url = kwargs["base_url"]
            self._client = None  # Reset client to use new URL
        if "api_key" in kwargs:
            self._api_key = kwargs["api_key"]
            self._client = None

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
            raise BackendError(f"OpenAI-compatible API error: {exc}") from exc

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
            raise BackendError(f"OpenAI-compatible API stream error: {exc}") from exc
