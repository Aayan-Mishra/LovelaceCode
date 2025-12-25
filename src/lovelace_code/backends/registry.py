"""Backend registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Backend

_BACKENDS: dict[str, type["Backend"]] = {}


def register_backend(name: str):
    """Decorator to register a backend class."""

    def decorator(cls: type["Backend"]):
        _BACKENDS[name] = cls
        cls.name = name
        return cls

    return decorator


def get_backend(name: str) -> "Backend":
    """Instantiate a backend by name."""
    if name not in _BACKENDS:
        # Lazy import backends to trigger registration
        _ensure_backends_loaded()
    if name not in _BACKENDS:
        available = ", ".join(_BACKENDS.keys()) or "(none)"
        raise ValueError(f"Unknown backend '{name}'. Available: {available}")
    return _BACKENDS[name]()


def list_backends() -> list[str]:
    """List all registered backend names."""
    _ensure_backends_loaded()
    return list(_BACKENDS.keys())


def _ensure_backends_loaded():
    """Import backend modules to trigger registration."""
    # pylint: disable=import-outside-toplevel,unused-import
    from . import hf_api, local_transformers, llama_cpp  # noqa: F401
    from . import openai_api, anthropic_api, openai_compatible  # noqa: F401
    from . import xai_api, openrouter_api, gemini_api  # noqa: F401
    from . import groq_api  # noqa: F401
