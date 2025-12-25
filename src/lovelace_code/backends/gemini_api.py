"""Google Gemini API backend."""

from __future__ import annotations

import os
from typing import Iterable

from .base import Backend, BackendError, GenerationConfig
from .registry import register_backend


@register_backend("gemini")
class GeminiBackend(Backend):
    """Uses Google Gemini API (Gemini 3 Pro, Gemini 3 Flash, etc.)."""

    def __init__(self):
        self._client = None
        self._model_id = None

    def is_available(self) -> bool:
        try:
            from google import genai  # noqa: F401
            return True
        except ImportError:
            return False

    def _get_client(self):
        if self._client is None:
            from google import genai

            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise BackendError(
                    "Google API key not found. "
                    "Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable."
                )
            self._client = genai.Client(api_key=api_key)
        return self._client

    def load_model(self, model_id: str, **kwargs) -> None:
        self._model_id = model_id

    def generate(self, prompt: str, config: GenerationConfig) -> str:
        client = self._get_client()
        model = config.model or self._model_id

        try:
            from google.genai import types
            
            generation_config = types.GenerateContentConfig(
                max_output_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                stop_sequences=config.stop_sequences if config.stop_sequences else None,
            )

            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=generation_config,
            )
            
            if response.text:
                return response.text
            return ""
        except Exception as exc:
            raise BackendError(f"Gemini API error: {exc}") from exc

    def stream(self, prompt: str, config: GenerationConfig) -> Iterable[str]:
        client = self._get_client()
        model = config.model or self._model_id

        try:
            from google.genai import types
            
            generation_config = types.GenerateContentConfig(
                max_output_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                stop_sequences=config.stop_sequences if config.stop_sequences else None,
            )

            for chunk in client.models.generate_content_stream(
                model=model,
                contents=prompt,
                config=generation_config,
            ):
                if chunk.text:
                    yield chunk.text
        except Exception as exc:
            raise BackendError(f"Gemini API stream error: {exc}") from exc
