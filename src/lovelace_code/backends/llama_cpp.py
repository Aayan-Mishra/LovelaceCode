"""llama.cpp backend for GGUF models."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

from .base import Backend, BackendError, GenerationConfig
from .registry import register_backend


@register_backend("llama")
class LlamaCppBackend(Backend):
    """Runs GGUF models via llama-cpp-python (very efficient on CPU/Metal)."""

    def __init__(self):
        self._llm = None
        self._model_path: str | None = None

    def is_available(self) -> bool:
        try:
            from llama_cpp import Llama  # noqa: F401

            return True
        except ImportError:
            return False

    def load_model(self, model_id: str, **kwargs) -> None:
        """Load a GGUF model.
        
        model_id can be:
        - A local path to a .gguf file
        - A HuggingFace repo ID (will download the model)
        """
        if self._model_path == model_id and self._llm is not None:
            return

        try:
            from llama_cpp import Llama

            n_ctx = kwargs.get("n_ctx", 4096)
            n_gpu_layers = kwargs.get("n_gpu_layers", -1)  # -1 = all layers on GPU

            # Check if it's a local file
            if Path(model_id).exists() and model_id.endswith(".gguf"):
                model_path = model_id
                print(f"[Lovelace] Loading local GGUF: {model_path}", file=sys.stderr)
            else:
                # Try to download from HuggingFace
                print(f"[Lovelace] Downloading GGUF from HuggingFace: {model_id}", file=sys.stderr)
                from huggingface_hub import hf_hub_download

                # Assume the repo has a .gguf file; try common patterns
                try:
                    model_path = hf_hub_download(
                        repo_id=model_id,
                        filename="model.gguf",
                    )
                except Exception:
                    # Try to find any gguf file
                    from huggingface_hub import list_repo_files

                    files = list_repo_files(model_id)
                    gguf_files = [f for f in files if f.endswith(".gguf")]
                    if not gguf_files:
                        raise BackendError(f"No .gguf files found in {model_id}")
                    # Pick the smallest one or Q4
                    preferred = [f for f in gguf_files if "Q4" in f or "q4" in f]
                    chosen = preferred[0] if preferred else gguf_files[0]
                    model_path = hf_hub_download(repo_id=model_id, filename=chosen)
                    print(f"[Lovelace] Downloaded: {chosen}", file=sys.stderr)

            self._llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )
            self._model_path = model_id
            print("[Lovelace] GGUF model loaded!", file=sys.stderr)

        except ImportError as exc:
            raise BackendError(
                "llama-cpp-python not installed. Install with: pip install lovelace-code[llama]"
            ) from exc
        except Exception as exc:
            raise BackendError(f"Failed to load GGUF model: {exc}") from exc

    def generate(self, prompt: str, config: GenerationConfig) -> str:
        if self._llm is None:
            self.load_model(config.model)

        try:
            output = self._llm(
                prompt,
                max_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repeat_penalty=config.repetition_penalty,
                stop=config.stop_sequences or None,
            )
            return output["choices"][0]["text"]
        except Exception as exc:
            raise BackendError(f"Generation error: {exc}") from exc

    def stream(self, prompt: str, config: GenerationConfig) -> Iterable[str]:
        if self._llm is None:
            self.load_model(config.model)

        try:
            for output in self._llm(
                prompt,
                max_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repeat_penalty=config.repetition_penalty,
                stop=config.stop_sequences or None,
                stream=True,
            ):
                token = output["choices"][0]["text"]
                if token:
                    yield token
        except Exception as exc:
            raise BackendError(f"Streaming error: {exc}") from exc

    def unload_model(self) -> None:
        if self._llm is not None:
            del self._llm
            self._llm = None
        self._model_path = None
