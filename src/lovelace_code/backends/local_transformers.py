"""Local transformers backend (runs models on GPU/CPU)."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Iterable

from .base import Backend, BackendError, GenerationConfig
from .registry import register_backend

if TYPE_CHECKING:
    pass


@register_backend("local")
class LocalTransformersBackend(Backend):
    """Runs models locally via Hugging Face Transformers."""

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._model_id: str | None = None
        self._device = None

    def is_available(self) -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401

            return True
        except ImportError:
            return False

    def load_model(self, model_id: str, **kwargs) -> None:
        if self._model_id == model_id and self._model is not None:
            return  # Already loaded

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Determine device
            if torch.cuda.is_available():
                self._device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"

            print(f"[Lovelace] Loading {model_id} on {self._device}...", file=sys.stderr)

            # Base load kwargs
            load_kwargs = {
                "trust_remote_code": True,
            }

            # Device-specific settings
            if self._device == "cuda":
                load_kwargs["device_map"] = "auto"
                # Try to use 4-bit quantization if available
                try:
                    from transformers import BitsAndBytesConfig
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                except ImportError:
                    load_kwargs["torch_dtype"] = torch.float16
            elif self._device == "mps":
                # MPS doesn't support device_map="auto", load to CPU first then move
                load_kwargs["torch_dtype"] = torch.float32  # MPS works better with float32
                load_kwargs["low_cpu_mem_usage"] = True
            else:
                # CPU
                load_kwargs["low_cpu_mem_usage"] = True

            self._tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            
            # Set pad token if not set
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            self._model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

            # Move to device for MPS and CPU
            if self._device in ("mps", "cpu"):
                self._model = self._model.to(self._device)
            
            # Set to eval mode
            self._model.eval()

            self._model_id = model_id
            print(f"[Lovelace] Model loaded successfully!", file=sys.stderr)

        except Exception as exc:
            raise BackendError(f"Failed to load model '{model_id}': {exc}") from exc

    def generate(self, prompt: str, config: GenerationConfig) -> str:
        if self._model is None:
            self.load_model(config.model)

        try:
            import torch

            inputs = self._tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                # Build generation kwargs carefully
                gen_kwargs = {
                    **inputs,
                    "max_new_tokens": config.max_new_tokens,
                    "pad_token_id": self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
                    "eos_token_id": self._tokenizer.eos_token_id,
                }
                
                # Only add sampling params if doing sampling
                if config.temperature > 0:
                    gen_kwargs.update({
                        "do_sample": True,
                        "temperature": config.temperature,
                        "top_p": config.top_p,
                        "top_k": config.top_k if config.top_k > 0 else 50,
                        "repetition_penalty": config.repetition_penalty,
                    })
                else:
                    gen_kwargs["do_sample"] = False
                
                outputs = self._model.generate(**gen_kwargs)

            # Decode only the new tokens
            input_len = inputs["input_ids"].shape[1]
            generated = outputs[0][input_len:]
            return self._tokenizer.decode(generated, skip_special_tokens=True)

        except Exception as exc:
            raise BackendError(f"Generation error: {exc}") from exc

    def stream(self, prompt: str, config: GenerationConfig) -> Iterable[str]:
        if self._model is None:
            self.load_model(config.model)

        try:
            from transformers import TextIteratorStreamer
            from threading import Thread
            import torch

            inputs = self._tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            streamer = TextIteratorStreamer(
                self._tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            gen_kwargs = {
                **inputs,
                "streamer": streamer,
                "max_new_tokens": config.max_new_tokens,
                "pad_token_id": self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
                "eos_token_id": self._tokenizer.eos_token_id,
            }
            
            # Only add sampling params if doing sampling
            if config.temperature > 0:
                gen_kwargs.update({
                    "do_sample": True,
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "top_k": config.top_k if config.top_k > 0 else 50,
                    "repetition_penalty": config.repetition_penalty,
                })
            else:
                gen_kwargs["do_sample"] = False

            thread = Thread(target=self._model.generate, kwargs=gen_kwargs)
            thread.start()

            for text in streamer:
                yield text

            thread.join()

        except ImportError:
            # Fallback to non-streaming
            yield self.generate(prompt, config)
        except Exception as exc:
            raise BackendError(f"Streaming error: {exc}") from exc

    def unload_model(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._model_id = None

        # Try to free GPU memory
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
