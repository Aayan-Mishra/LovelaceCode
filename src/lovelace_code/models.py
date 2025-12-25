"""Model management for Lovelace Code."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ModelInfo:
    """Information about a supported model."""
    id: str
    name: str
    size: str
    context_length: int
    backend: Literal[
        "api",  # HuggingFace API
        "local",  # Local transformers
        "llama",  # llama.cpp
        "any",  # Works with multiple backends
        "openai",  # OpenAI API
        "anthropic",  # Anthropic API
        "gemini",  # Google Gemini API
        "xai",  # xAI (Grok) API
        "openrouter",  # OpenRouter API
        "openai-compatible",  # OpenAI-compatible API
    ]
    description: str
    recommended: bool = False
    provider: str = "huggingface"  # huggingface, openai, anthropic, google, xai, openrouter, meta


# ─────────────────────────────────────────────────────────────────────────────
# Lovelace Models (Featured - Open Source)
# ─────────────────────────────────────────────────────────────────────────────
LOVELACE_MODELS: list[ModelInfo] = [
    ModelInfo(
        id="Spestly/Lovelace-1-3B",
        name="Lovelace 1 (3B)",
        size="3B",
        context_length=4096,
        backend="any",
        description="Fast, efficient coding assistant",
        recommended=True,
        provider="huggingface",
    ),
    ModelInfo(
        id="Spestly/Lovelace-1-7B",
        name="Lovelace 1 (7B)",
        size="7B",
        context_length=4096,
        backend="any",
        description="More capable coding assistant",
        recommended=True,
        provider="huggingface",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI Models (API)
# ─────────────────────────────────────────────────────────────────────────────
OPENAI_MODELS: list[ModelInfo] = [
    ModelInfo(
        id="gpt-5.2-pro",
        name="GPT-5.2 Pro",
        size="Flagship",
        context_length=256000,
        backend="openai",
        description="OpenAI's flagship high-capability model",
        recommended=True,
        provider="openai",
    ),
    ModelInfo(
        id="gpt-5.2-thinking",
        name="GPT-5.2 Thinking",
        size="Flagship",
        context_length=256000,
        backend="openai",
        description="Reasoning-focused variant with chain-of-thought",
        provider="openai",
    ),
    ModelInfo(
        id="gpt-5.2-instant",
        name="GPT-5.2 Instant",
        size="Fast",
        context_length=128000,
        backend="openai",
        description="Fast, cost-efficient variant",
        provider="openai",
    ),
    ModelInfo(
        id="gpt-5.1-high",
        name="GPT-5.1 High",
        size="Large",
        context_length=128000,
        backend="openai",
        description="Earlier 5.x performance leader",
        provider="openai",
    ),
    ModelInfo(
        id="o4-mini",
        name="o4-mini",
        size="Efficient",
        context_length=128000,
        backend="openai",
        description="Efficient reasoning model (o-series)",
        recommended=True,
        provider="openai",
    ),
    ModelInfo(
        id="o3-mini",
        name="o3-mini",
        size="Efficient",
        context_length=128000,
        backend="openai",
        description="Compact reasoning model",
        provider="openai",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Anthropic Models (API)
# ─────────────────────────────────────────────────────────────────────────────
ANTHROPIC_MODELS: list[ModelInfo] = [
    ModelInfo(
        id="claude-opus-4.5",
        name="Claude Opus 4.5",
        size="Flagship",
        context_length=200000,
        backend="anthropic",
        description="Top-tier reasoning + tool use",
        recommended=True,
        provider="anthropic",
    ),
    ModelInfo(
        id="claude-sonnet-4.5",
        name="Claude Sonnet 4.5",
        size="Balanced",
        context_length=200000,
        backend="anthropic",
        description="Balanced reasoning + multimodal",
        recommended=True,
        provider="anthropic",
    ),
    ModelInfo(
        id="claude-haiku-4.5",
        name="Claude Haiku 4.5",
        size="Light",
        context_length=200000,
        backend="anthropic",
        description="Lighter, faster variant",
        provider="anthropic",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Google DeepMind Models (API)
# ─────────────────────────────────────────────────────────────────────────────
GOOGLE_MODELS: list[ModelInfo] = [
    ModelInfo(
        id="gemini-3-pro",
        name="Gemini 3 Pro",
        size="Flagship",
        context_length=1000000,
        backend="gemini",
        description="Multimodal deep reasoning model",
        recommended=True,
        provider="google",
    ),
    ModelInfo(
        id="gemini-3-flash",
        name="Gemini 3 Flash",
        size="Fast",
        context_length=1000000,
        backend="gemini",
        description="Faster, cheaper variant",
        provider="google",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# xAI Models (API)
# ─────────────────────────────────────────────────────────────────────────────
XAI_MODELS: list[ModelInfo] = [
    ModelInfo(
        id="grok-4",
        name="Grok-4",
        size="Large",
        context_length=128000,
        backend="xai",
        description="General reasoning + real-time focus",
        recommended=True,
        provider="xai",
    ),
    ModelInfo(
        id="grok-4.1",
        name="Grok-4.1",
        size="Large",
        context_length=128000,
        backend="xai",
        description="Refined speed + reduced hallucination",
        provider="xai",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Zhipu AI / GLM Models (API)
# ─────────────────────────────────────────────────────────────────────────────
ZHIPU_MODELS: list[ModelInfo] = [
    ModelInfo(
        id="glm-4.7",
        name="GLM-4.7",
        size="Large",
        context_length=128000,
        backend="zhipu",
        description="Strong coding/reasoning (Dec 2025)",
        recommended=True,
        provider="zhipu",
    ),
    ModelInfo(
        id="glm-4.6v-flash",
        name="GLM-4.6V-Flash",
        size="Fast",
        context_length=128000,
        backend="zhipu",
        description="Efficient multimodal variant",
        provider="zhipu",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Meta / Llama 4 Models (API + Local)
# ─────────────────────────────────────────────────────────────────────────────
LLAMA4_MODELS: list[ModelInfo] = [
    ModelInfo(
        id="meta-llama/Llama-4-Scout",
        name="Llama 4 Scout",
        size="17B",
        context_length=128000,
        backend="any",
        description="Lightweight multimodal model",
        recommended=True,
        provider="meta",
    ),
    ModelInfo(
        id="meta-llama/Llama-4-Maverick",
        name="Llama 4 Maverick",
        size="70B",
        context_length=128000,
        backend="api",
        description="Larger multimodal variant",
        provider="meta",
    ),
    ModelInfo(
        id="meta-llama/Llama-4-Behemoth",
        name="Llama 4 Behemoth",
        size="405B",
        context_length=128000,
        backend="api",
        description="Powerful research model (preview)",
        provider="meta",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Other Emerging Open Models
# ─────────────────────────────────────────────────────────────────────────────
EMERGING_MODELS: list[ModelInfo] = [
    ModelInfo(
        id="deepseek-ai/DeepSeek-V3-0324",
        name="DeepSeek V3",
        size="685B MoE",
        context_length=128000,
        backend="api",
        description="High-cap open model with strong benchmarks",
        recommended=True,
        provider="deepseek",
    ),
    ModelInfo(
        id="apertus/Apertus-8B",
        name="Apertus 8B",
        size="8B",
        context_length=32768,
        backend="any",
        description="Multilingual, transparent open model",
        provider="apertus",
    ),
    ModelInfo(
        id="apertus/Apertus-70B",
        name="Apertus 70B",
        size="70B",
        context_length=32768,
        backend="api",
        description="Large Apertus variant",
        provider="apertus",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Open-Source Code Models (HuggingFace)
# ─────────────────────────────────────────────────────────────────────────────
OPENSOURCE_CODE_MODELS: list[ModelInfo] = [
    ModelInfo(
        id="codellama/CodeLlama-7b-hf",
        name="Code Llama 7B",
        size="7B",
        context_length=16384,
        backend="any",
        description="Meta's code-specialized Llama",
        provider="huggingface",
    ),
    ModelInfo(
        id="codellama/CodeLlama-13b-hf",
        name="Code Llama 13B",
        size="13B",
        context_length=16384,
        backend="any",
        description="Larger Code Llama variant",
        provider="huggingface",
    ),
    ModelInfo(
        id="codellama/CodeLlama-34b-hf",
        name="Code Llama 34B",
        size="34B",
        context_length=16384,
        backend="api",
        description="Most capable Code Llama (API recommended)",
        provider="huggingface",
    ),
    ModelInfo(
        id="bigcode/starcoder2-3b",
        name="StarCoder2 3B",
        size="3B",
        context_length=16384,
        backend="any",
        description="Efficient code model from BigCode",
        provider="huggingface",
    ),
    ModelInfo(
        id="bigcode/starcoder2-7b",
        name="StarCoder2 7B",
        size="7B",
        context_length=16384,
        backend="any",
        description="Balanced StarCoder2 variant",
        provider="huggingface",
    ),
    ModelInfo(
        id="bigcode/starcoder2-15b",
        name="StarCoder2 15B",
        size="15B",
        context_length=16384,
        backend="any",
        description="Larger StarCoder2 variant",
        provider="huggingface",
    ),
    ModelInfo(
        id="Qwen/Qwen2.5-Coder-7B",
        name="Qwen2.5 Coder 7B",
        size="7B",
        context_length=32768,
        backend="any",
        description="Alibaba's latest code model",
        provider="huggingface",
    ),
    ModelInfo(
        id="Qwen/Qwen2.5-Coder-14B",
        name="Qwen2.5 Coder 14B",
        size="14B",
        context_length=32768,
        backend="any",
        description="Larger Qwen coder variant",
        provider="huggingface",
    ),
    ModelInfo(
        id="microsoft/phi-2",
        name="Phi-2",
        size="2.7B",
        context_length=2048,
        backend="any",
        description="Microsoft's efficient small model",
        provider="huggingface",
    ),
    ModelInfo(
        id="microsoft/Phi-3-mini-4k-instruct",
        name="Phi-3 Mini",
        size="3.8B",
        context_length=4096,
        backend="any",
        description="Microsoft's latest small model",
        provider="huggingface",
    ),
    ModelInfo(
        id="mistralai/Mistral-7B-Instruct-v0.2",
        name="Mistral 7B Instruct",
        size="7B",
        context_length=32768,
        backend="any",
        description="Mistral's efficient instruct model",
        provider="huggingface",
    ),
    ModelInfo(
        id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        name="Mixtral 8x7B",
        size="46.7B (MoE)",
        context_length=32768,
        backend="api",
        description="Mistral's MoE model (API recommended)",
        provider="huggingface",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# GGUF Models for llama.cpp
# ─────────────────────────────────────────────────────────────────────────────
GGUF_MODELS: list[ModelInfo] = [
    ModelInfo(
        id="TheBloke/CodeLlama-7B-GGUF",
        name="Code Llama 7B GGUF",
        size="7B",
        context_length=16384,
        backend="llama",
        description="Quantized Code Llama for llama.cpp",
        provider="huggingface",
    ),
    ModelInfo(
        id="TheBloke/CodeLlama-13B-GGUF",
        name="Code Llama 13B GGUF",
        size="13B",
        context_length=16384,
        backend="llama",
        description="Quantized Code Llama 13B",
        provider="huggingface",
    ),
    ModelInfo(
        id="TheBloke/deepseek-coder-6.7B-base-GGUF",
        name="DeepSeek Coder 6.7B GGUF",
        size="6.7B",
        context_length=16384,
        backend="llama",
        description="Quantized DeepSeek Coder",
        provider="huggingface",
    ),
    ModelInfo(
        id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        name="Mistral 7B GGUF",
        size="7B",
        context_length=32768,
        backend="llama",
        description="Quantized Mistral Instruct",
        provider="huggingface",
    ),
    ModelInfo(
        id="TheBloke/phi-2-GGUF",
        name="Phi-2 GGUF",
        size="2.7B",
        context_length=2048,
        backend="llama",
        description="Quantized Phi-2",
        provider="huggingface",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Combined Lists
# ─────────────────────────────────────────────────────────────────────────────

# All API models (commercial providers)
API_MODELS: list[ModelInfo] = (
    OPENAI_MODELS + 
    ANTHROPIC_MODELS + 
    GOOGLE_MODELS + 
    XAI_MODELS + 
    ZHIPU_MODELS
)

# All recommended models (for main list display)
RECOMMENDED_MODELS: list[ModelInfo] = (
    LOVELACE_MODELS +
    API_MODELS +
    LLAMA4_MODELS +
    EMERGING_MODELS +
    OPENSOURCE_CODE_MODELS
)

# Models optimized for local inference (smaller, quantized friendly)
LOCAL_OPTIMIZED_MODELS = [
    m for m in RECOMMENDED_MODELS 
    if m.backend in ("any", "local", "llama") and m.provider == "huggingface"
]

# All models including GGUF
ALL_MODELS = RECOMMENDED_MODELS + GGUF_MODELS


# ─────────────────────────────────────────────────────────────────────────────
# Provider Information
# ─────────────────────────────────────────────────────────────────────────────
PROVIDERS = {
    "openai": {
        "name": "OpenAI",
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "icon": "🧠",
    },
    "anthropic": {
        "name": "Anthropic",
        "api_key_env": "ANTHROPIC_API_KEY",
        "base_url": "https://api.anthropic.com",
        "icon": "🧠",
    },
    "google": {
        "name": "Google DeepMind",
        "api_key_env": "GOOGLE_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com",
        "icon": "🧠",
    },
    "xai": {
        "name": "xAI",
        "api_key_env": "XAI_API_KEY",
        "base_url": "https://api.x.ai",
        "icon": "🧠",
    },
    "zhipu": {
        "name": "Zhipu AI",
        "api_key_env": "ZHIPU_API_KEY",
        "base_url": "https://open.bigmodel.cn/api",
        "icon": "🧠",
    },
    "huggingface": {
        "name": "Hugging Face",
        "api_key_env": "HF_TOKEN",
        "base_url": "https://api-inference.huggingface.co",
        "icon": "🤗",
    },
    "meta": {
        "name": "Meta",
        "api_key_env": "HF_TOKEN",
        "base_url": "https://api-inference.huggingface.co",
        "icon": "🦙",
    },
    "deepseek": {
        "name": "DeepSeek",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "icon": "🔍",
    },
    "apertus": {
        "name": "Apertus",
        "api_key_env": "HF_TOKEN",
        "base_url": "https://api-inference.huggingface.co",
        "icon": "🌐",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────
def get_model_info(model_id: str) -> ModelInfo | None:
    """Get info for a model by ID."""
    for model in ALL_MODELS:
        if model.id == model_id:
            return model
    return None


def search_models(query: str) -> list[ModelInfo]:
    """Search models by name or ID."""
    query = query.lower()
    return [
        m for m in ALL_MODELS
        if query in m.id.lower() or query in m.name.lower() or query in m.description.lower()
    ]


def get_models_for_backend(backend: str) -> list[ModelInfo]:
    """Get models suitable for a specific backend."""
    if backend == "llama":
        return [m for m in ALL_MODELS if m.backend in ("llama", "any")]
    elif backend == "local":
        return [m for m in ALL_MODELS if m.backend in ("local", "any") and "GGUF" not in m.name]
    elif backend == "api":
        return [m for m in ALL_MODELS if m.backend in ("api", "any")]
    return ALL_MODELS


def get_models_by_provider(provider: str) -> list[ModelInfo]:
    """Get models from a specific provider."""
    return [m for m in ALL_MODELS if m.provider == provider]


def get_api_models() -> list[ModelInfo]:
    """Get all commercial API models."""
    return API_MODELS


def estimate_vram_gb(size: str) -> float:
    """Estimate VRAM needed for a model size (rough estimate)."""
    size = size.upper().replace("B", "").replace("(MOE)", "").replace("MoE", "").strip()
    
    # Handle special cases
    if size in ("FLAGSHIP", "BALANCED", "LIGHT", "FAST", "LARGE", "EFFICIENT"):
        return 0.0  # API models don't need local VRAM
    
    try:
        params = float(size)
    except ValueError:
        return 0.0
    
    # Rough estimate: ~1GB per 1B params for 4-bit quantized
    return params * 1.0
