"""Hugging Face Inference API backend."""

from __future__ import annotations

import os
from typing import Iterable

from .base import Backend, BackendError, GenerationConfig
from .registry import register_backend


@register_backend("api")
class HuggingFaceAPIBackend(Backend):
    """Uses HuggingFace Inference API (remote, no GPU required)."""

    def __init__(self):
        self._client = None

    def is_available(self) -> bool:
        try:
            from huggingface_hub import InferenceClient  # noqa: F401

            return True
        except ImportError:
            return False

    def _get_client(self):
        if self._client is None:
            from huggingface_hub import InferenceClient

            token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
            self._client = InferenceClient(token=token) if token else InferenceClient()
        return self._client

    def load_model(self, model_id: str, **kwargs) -> None:
        # API backend doesn't preload
        self._model_id = model_id

    def generate(self, prompt: str, config: GenerationConfig) -> str:
        client = self._get_client()
        try:
            result = client.text_generation(
                prompt,
                model=config.model,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty,
                return_full_text=False,
            )
            return result if isinstance(result, str) else str(result)
        except Exception as exc:
            raise BackendError(f"API error: {exc}") from exc

    def stream(self, prompt: str, config: GenerationConfig) -> Iterable[str]:
        client = self._get_client()
        try:
            for chunk in client.text_generation(
                prompt,
                model=config.model,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty,
                stream=True,
                return_full_text=False,
            ):
                if isinstance(chunk, str):
                    yield chunk
                else:
                    token = getattr(getattr(chunk, "token", None), "text", None)
                    if token:
                        yield token
        except Exception as exc:
            raise BackendError(f"API stream error: {exc}") from exc
