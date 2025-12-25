"""Base backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from lovelace_code.config import LovelaceConfig


class BackendError(RuntimeError):
    """Raised when a backend fails."""


@dataclass
class GenerationConfig:
    model: str
    temperature: float = 0.2
    max_new_tokens: int = 1024
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.1
    stop_sequences: list[str] = field(default_factory=list)


class Backend(ABC):
    """Abstract backend interface."""

    name: str = "base"

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this backend can be used."""
        ...

    @abstractmethod
    def load_model(self, model_id: str, **kwargs) -> None:
        """Load or prepare a model for inference."""
        ...

    @abstractmethod
    def generate(self, prompt: str, config: GenerationConfig) -> str:
        """Generate a response (blocking)."""
        ...

    @abstractmethod
    def stream(self, prompt: str, config: GenerationConfig) -> Iterable[str]:
        """Stream tokens one by one."""
        ...

    def unload_model(self) -> None:
        """Optionally unload the model to free memory."""
        pass

    @classmethod
    def from_config(cls, config: "LovelaceConfig") -> "Backend":
        """Factory method — can be overridden."""
        return cls()
