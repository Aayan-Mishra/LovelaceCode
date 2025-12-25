"""First-run onboarding flow for Lovelace Code."""

from __future__ import annotations

import getpass
import os
from pathlib import Path

from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.style import Style
from rich.text import Text

from lovelace_code.global_config import (
    CURRENT_ONBOARDING_VERSION,
    GlobalConfig,
    get_global_config_dir,
    get_models_cache_dir,
    load_global_config,
    save_global_config,
)
from lovelace_code.ui import Colors, STYLES, render_logo, create_styled_panel, create_success_message


# Provider information for onboarding
PROVIDERS = {
    "openai": {
        "name": "OpenAI",
        "env_var": "OPENAI_API_KEY",
        "config_field": "openai_api_key",
        "url": "https://platform.openai.com/api-keys",
        "description": "GPT-5.x, o-series reasoning models",
    },
    "anthropic": {
        "name": "Anthropic",
        "env_var": "ANTHROPIC_API_KEY",
        "config_field": "anthropic_api_key",
        "url": "https://console.anthropic.com/settings/keys",
        "description": "Claude Opus/Sonnet/Haiku models",
    },
    "gemini": {
        "name": "Google Gemini",
        "env_var": "GOOGLE_API_KEY",
        "config_field": "google_api_key",
        "url": "https://aistudio.google.com/app/apikey",
        "description": "Gemini 3 Pro/Flash models",
    },
    "xai": {
        "name": "xAI",
        "env_var": "XAI_API_KEY",
        "config_field": "xai_api_key",
        "url": "https://console.x.ai/",
        "description": "Grok-4, Grok-4.1 models",
    },
    "openrouter": {
        "name": "OpenRouter",
        "env_var": "OPENROUTER_API_KEY",
        "config_field": "openrouter_api_key",
        "url": "https://openrouter.ai/keys",
        "description": "Access many models via single API",
    },
    "groq": {
        "name": "Groq",
        "env_var": "GROQ_API_KEY",
        "config_field": "groq_api_key",
        "url": "https://console.groq.com/keys",
        "description": "Ultra-fast inference with LPUs",
    },
    "api": {
        "name": "HuggingFace",
        "env_var": "HF_TOKEN",
        "config_field": "hf_token",
        "url": "https://huggingface.co/settings/tokens",
        "description": "Open-source models via Inference API",
    },
}


def run_onboarding(console: Console) -> GlobalConfig:
    """Run the full onboarding flow and return the configured GlobalConfig."""
    
    config = load_global_config()
    
    console.clear()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Welcome screen with logo
    # ─────────────────────────────────────────────────────────────────────────
    render_logo(console)
    console.print()
    
    welcome_text = Text()
    welcome_text.append("Welcome to Lovelace Code!\n\n", style=Style(color=Colors.LIGHT_PURPLE, bold=True))
    welcome_text.append("An open-source, agentic coding assistant powered by open models.\n", style=Style(color=Colors.LAVENDER))
    welcome_text.append("Let's get you set up in just a few steps.", style=Style(color=Colors.SAKURA))
    
    welcome_panel = Panel(
        welcome_text,
        border_style=Colors.SOFT_PURPLE,
        padding=(1, 2),
        box=ROUNDED,
    )
    console.print(welcome_panel)
    console.print()
    
    console.print(Text("Press Enter to continue...", style=Style(color=Colors.DIM)), end="")
    input()
    console.clear()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: User info
    # ─────────────────────────────────────────────────────────────────────────
    step_title = Text()
    step_title.append("Step 1/4: ", style=Style(color=Colors.SAKURA))
    step_title.append("About You", style=Style(color=Colors.LIGHT_PURPLE, bold=True))
    console.print(Panel(step_title, border_style=Colors.SOFT_PURPLE, box=ROUNDED))
    console.print()
    
    default_username = getpass.getuser()
    config.username = Prompt.ask(
        Text("What should we call you?", style=Style(color=Colors.LAVENDER)),
        default=default_username.capitalize(),
    )
    
    config.email = Prompt.ask(
        Text("Email (optional, for updates)", style=Style(color=Colors.LAVENDER)),
        default="",
    )
    
    console.print()
    success = Text()
    success.append("✓ ", style=Style(color=Colors.SUCCESS))
    success.append(f"Nice to meet you, ", style=Style(color=Colors.SUCCESS))
    success.append(config.username, style=Style(color=Colors.SAKURA, bold=True))
    success.append("!", style=Style(color=Colors.SUCCESS))
    console.print(success)
    console.print()
    console.print(Text("Press Enter to continue...", style=Style(color=Colors.DIM)), end="")
    input()
    console.clear()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: Backend selection
    # ─────────────────────────────────────────────────────────────────────────
    step_title = Text()
    step_title.append("Step 2/4: ", style=Style(color=Colors.SAKURA))
    step_title.append("Choose Your AI Provider", style=Style(color=Colors.LIGHT_PURPLE, bold=True))
    console.print(Panel(step_title, border_style=Colors.SOFT_PURPLE, box=ROUNDED))
    console.print()
    
    console.print(Text("☁️  Cloud API Providers:\n", style=Style(color=Colors.WHITE, bold=True)))
    
    # Anthropic
    opt_anthropic = Text()
    opt_anthropic.append("  1. ", style=Style(color=Colors.DIM))
    opt_anthropic.append("anthropic", style=Style(color=Colors.ORCHID, bold=True))
    opt_anthropic.append("     — Claude Opus/Sonnet/Haiku 4.5", style=Style(color=Colors.LAVENDER))
    console.print(opt_anthropic)
    console.print(Text("                     Top-tier reasoning & tool use", style=Style(color=Colors.DIM)))
    console.print()
    
    # OpenAI
    opt_openai = Text()
    opt_openai.append("  2. ", style=Style(color=Colors.DIM))
    opt_openai.append("openai", style=Style(color=Colors.ORCHID, bold=True))
    opt_openai.append("        — GPT-5.x, o-series reasoning models", style=Style(color=Colors.LAVENDER))
    console.print(opt_openai)
    console.print(Text("                     OpenAI's flagship models", style=Style(color=Colors.DIM)))
    console.print()
    
    # Gemini
    opt_gemini = Text()
    opt_gemini.append("  3. ", style=Style(color=Colors.DIM))
    opt_gemini.append("gemini", style=Style(color=Colors.ORCHID, bold=True))
    opt_gemini.append("        — Gemini 3 Pro/Flash", style=Style(color=Colors.LAVENDER))
    console.print(opt_gemini)
    console.print(Text("                     Google's multimodal models", style=Style(color=Colors.DIM)))
    console.print()
    
    # xAI
    opt_xai = Text()
    opt_xai.append("  4. ", style=Style(color=Colors.DIM))
    opt_xai.append("xai", style=Style(color=Colors.ORCHID, bold=True))
    opt_xai.append("           — Grok-4, Grok-4.1", style=Style(color=Colors.LAVENDER))
    console.print(opt_xai)
    console.print(Text("                     xAI's real-time focused models", style=Style(color=Colors.DIM)))
    console.print()
    
    # OpenRouter
    opt_openrouter = Text()
    opt_openrouter.append("  5. ", style=Style(color=Colors.DIM))
    opt_openrouter.append("openrouter", style=Style(color=Colors.ORCHID, bold=True))
    opt_openrouter.append("    — Access 100+ models via single API", style=Style(color=Colors.LAVENDER))
    console.print(opt_openrouter)
    console.print(Text("                     Unified API for many providers", style=Style(color=Colors.DIM)))
    console.print()
    
    # Groq
    opt_groq = Text()
    opt_groq.append("  6. ", style=Style(color=Colors.DIM))
    opt_groq.append("groq", style=Style(color=Colors.ORCHID, bold=True))
    opt_groq.append("          — Ultra-fast LPU inference", style=Style(color=Colors.LAVENDER))
    console.print(opt_groq)
    console.print(Text("                     Llama, Mixtral, Gemma at blazing speed", style=Style(color=Colors.DIM)))
    console.print()
    
    # HuggingFace API
    opt_hf = Text()
    opt_hf.append("  7. ", style=Style(color=Colors.DIM))
    opt_hf.append("api", style=Style(color=Colors.ORCHID, bold=True))
    opt_hf.append("           — HuggingFace Inference API", style=Style(color=Colors.LAVENDER))
    console.print(opt_hf)
    console.print(Text("                     Open-source models (Lovelace, etc.)", style=Style(color=Colors.DIM)))
    console.print()
    
    console.print(Text("\n💻 Local Inference:\n", style=Style(color=Colors.WHITE, bold=True)))
    
    # Local
    opt_local = Text()
    opt_local.append("  8. ", style=Style(color=Colors.DIM))
    opt_local.append("local", style=Style(color=Colors.ORCHID, bold=True))
    opt_local.append("         — Run models locally with Transformers", style=Style(color=Colors.LAVENDER))
    console.print(opt_local)
    console.print(Text("                     Requires GPU (CUDA/MPS)", style=Style(color=Colors.DIM)))
    console.print()
    
    # Llama
    opt_llama = Text()
    opt_llama.append("  9. ", style=Style(color=Colors.DIM))
    opt_llama.append("llama", style=Style(color=Colors.ORCHID, bold=True))
    opt_llama.append("         — Run GGUF models with llama.cpp", style=Style(color=Colors.LAVENDER))
    console.print(opt_llama)
    console.print(Text("                     Best for CPU/Apple Silicon", style=Style(color=Colors.DIM)))
    console.print()
    
    # OpenAI-compatible
    opt_compat = Text()
    opt_compat.append(" 10. ", style=Style(color=Colors.DIM))
    opt_compat.append("openai-compatible", style=Style(color=Colors.ORCHID, bold=True))
    opt_compat.append(" — Custom OpenAI-compatible endpoint", style=Style(color=Colors.LAVENDER))
    console.print(opt_compat)
    console.print(Text("                          LocalAI, LM Studio, ollama, vLLM", style=Style(color=Colors.DIM)))
    console.print()
    
    backend_choice = Prompt.ask(
        Text("\nChoose your default backend", style=Style(color=Colors.LAVENDER)),
        choices=["anthropic", "openai", "gemini", "xai", "openrouter", "groq", "api", "local", "llama", "openai-compatible",
                 "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        default="anthropic",
    )
    
    # Map number choices
    backend_map = {
        "1": "anthropic", "2": "openai", "3": "gemini", "4": "xai",
        "5": "openrouter", "6": "groq", "7": "api", "8": "local", "9": "llama", "10": "openai-compatible"
    }
    config.default_backend = backend_map.get(backend_choice, backend_choice)  # type: ignore
    
    console.print()
    console.print(create_success_message(f"Default backend set to {config.default_backend}"))
    console.print()
    console.print(Text("Press Enter to continue...", style=Style(color=Colors.DIM)), end="")
    input()
    console.clear()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 3: API Key (based on selected backend)
    # ─────────────────────────────────────────────────────────────────────────
    step_title = Text()
    step_title.append("Step 3/4: ", style=Style(color=Colors.SAKURA))
    step_title.append("API Access", style=Style(color=Colors.LIGHT_PURPLE, bold=True))
    console.print(Panel(step_title, border_style=Colors.SOFT_PURPLE, box=ROUNDED))
    console.print()
    
    selected_backend = config.default_backend
    
    if selected_backend in ("local", "llama"):
        # Local backends don't need API keys
        console.print(Text("✓ No API key required for local inference!", style=Style(color=Colors.SUCCESS)))
        console.print()
        if selected_backend == "local":
            console.print(Text("Make sure you have PyTorch installed with CUDA or MPS support.", style=Style(color=Colors.LAVENDER)))
        else:
            console.print(Text("Make sure llama-cpp-python is installed.", style=Style(color=Colors.LAVENDER)))
    elif selected_backend == "openai-compatible":
        # OpenAI-compatible needs base URL
        console.print(Text("Configure your OpenAI-compatible endpoint:\n", style=Style(color=Colors.LAVENDER)))
        
        base_url = Prompt.ask(
            Text("Base URL (e.g., http://localhost:1234/v1)", style=Style(color=Colors.ORCHID)),
            default="",
        )
        if base_url.strip():
            config.openai_compatible_base_url = base_url.strip()
        
        api_key = Prompt.ask(
            Text("API Key (optional, press Enter to skip)", style=Style(color=Colors.ORCHID)),
            password=True,
            default="",
        )
        if api_key.strip():
            config.openai_compatible_api_key = api_key.strip()
            
        if config.openai_compatible_base_url:
            console.print(create_success_message("Endpoint configured!"))
        else:
            skip_text = Text()
            skip_text.append("○ ", style=Style(color=Colors.WARNING))
            skip_text.append("Skipped (set OPENAI_COMPATIBLE_BASE_URL env var later)", style=Style(color=Colors.WARNING))
            console.print(skip_text)
    else:
        # Cloud API providers
        provider_info = PROVIDERS.get(selected_backend, PROVIDERS["api"])
        
        # Check for existing token in env
        existing_token = os.getenv(provider_info["env_var"])
        
        if existing_token:
            console.print(create_success_message(f"Found {provider_info['name']} API key in environment variables!"))
            console.print()
        else:
            console.print(Text(f"A {provider_info['name']} API key is required for:\n", style=Style(color=Colors.LAVENDER)))
            console.print(Text(f"  • {provider_info['description']}", style=Style(color=Colors.WHITE)))
            console.print()
            
            link = Text()
            link.append("Get your API key at: ", style=Style(color=Colors.DIM))
            link.append(provider_info["url"], style=Style(color=Colors.LAVENDER, underline=True))
            console.print(link)
            console.print()
            
            if Confirm.ask(Text("Would you like to enter an API key now?", style=Style(color=Colors.LAVENDER)), default=True):
                token = Prompt.ask(Text(f"{provider_info['name']} API Key", style=Style(color=Colors.ORCHID)), password=True)
                if token.strip():
                    setattr(config, provider_info["config_field"], token.strip())
                    console.print(create_success_message("API key saved securely!"))
                else:
                    _print_skip_message(console, provider_info["env_var"])
            else:
                _print_skip_message(console, provider_info["env_var"])
    
    console.print()
    
    # Optionally configure additional providers
    if selected_backend not in ("local", "llama", "openai-compatible"):
        if Confirm.ask(Text("Would you like to configure additional API providers?", style=Style(color=Colors.LAVENDER)), default=False):
            console.print()
            _configure_additional_providers(console, config, selected_backend)
    
    console.print(Text("Press Enter to continue...", style=Style(color=Colors.DIM)), end="")
    input()
    console.clear()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 4: Preferences
    # ─────────────────────────────────────────────────────────────────────────
    step_title = Text()
    step_title.append("Step 4/4: ", style=Style(color=Colors.SAKURA))
    step_title.append("Preferences", style=Style(color=Colors.LIGHT_PURPLE, bold=True))
    console.print(Panel(step_title, border_style=Colors.SOFT_PURPLE, box=ROUNDED))
    console.print()
    
    config.telemetry_enabled = Confirm.ask(
        Text("Send anonymous usage statistics to help improve Lovelace Code?", style=Style(color=Colors.LAVENDER)),
        default=False,
    )
    
    config.auto_update_check = Confirm.ask(
        Text("Check for updates on startup?", style=Style(color=Colors.LAVENDER)),
        default=True,
    )
    
    config.show_tips = Confirm.ask(
        Text("Show helpful tips during sessions?", style=Style(color=Colors.LAVENDER)),
        default=True,
    )
    
    console.print()
    console.print(create_success_message("Preferences saved!"))
    console.print()
    console.print(Text("Press Enter to continue...", style=Style(color=Colors.DIM)), end="")
    input()
    console.clear()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Finish
    # ─────────────────────────────────────────────────────────────────────────
    config.onboarding_completed = True
    config.onboarding_version = CURRENT_ONBOARDING_VERSION
    save_global_config(config)
    
    # Create necessary directories
    get_global_config_dir().mkdir(parents=True, exist_ok=True)
    get_models_cache_dir().mkdir(parents=True, exist_ok=True)
    
    # Show completion screen
    render_logo(console)
    console.print()
    
    complete_text = Text()
    complete_text.append("🎉 Setup Complete!\n\n", style=Style(color=Colors.SUCCESS, bold=True))
    complete_text.append("Your configuration:\n", style=Style(color=Colors.WHITE, bold=True))
    complete_text.append(f"  • Username: ", style=Style(color=Colors.LAVENDER))
    complete_text.append(f"{config.username}\n", style=Style(color=Colors.SAKURA))
    complete_text.append(f"  • Backend: ", style=Style(color=Colors.LAVENDER))
    complete_text.append(f"{config.default_backend}\n", style=Style(color=Colors.SAKURA))
    complete_text.append(f"  • Model: ", style=Style(color=Colors.LAVENDER))
    complete_text.append(f"{config.default_model}\n", style=Style(color=Colors.SAKURA))
    complete_text.append(f"  • Config: ", style=Style(color=Colors.LAVENDER))
    complete_text.append(f"{get_global_config_dir()}\n\n", style=Style(color=Colors.DIM))
    complete_text.append("Next steps:\n", style=Style(color=Colors.WHITE, bold=True))
    complete_text.append("  1. ", style=Style(color=Colors.DIM))
    complete_text.append("cd your-project\n", style=Style(color=Colors.ORCHID))
    complete_text.append("  2. ", style=Style(color=Colors.DIM))
    complete_text.append("lovelace init\n", style=Style(color=Colors.ORCHID))
    complete_text.append("  3. ", style=Style(color=Colors.DIM))
    complete_text.append("lovelace\n\n", style=Style(color=Colors.ORCHID))
    complete_text.append("Run ", style=Style(color=Colors.DIM))
    complete_text.append("lovelace --help", style=Style(color=Colors.ORCHID, bold=True))
    complete_text.append(" for more commands", style=Style(color=Colors.DIM))
    
    complete_panel = Panel(
        complete_text,
        border_style=Colors.SUCCESS,
        title=Text("✦ Welcome to Lovelace Code", style=Style(color=Colors.LIGHT_PURPLE, bold=True)),
        padding=(1, 2),
        box=ROUNDED,
    )
    console.print(complete_panel)
    console.print()
    
    return config


def _print_skip_message(console: Console, env_var: str) -> None:
    """Print a styled skip message."""
    skip_text = Text()
    skip_text.append("○ ", style=Style(color=Colors.WARNING))
    skip_text.append(f"Skipped (set {env_var} env var or run ", style=Style(color=Colors.WARNING))
    skip_text.append("lovelace login", style=Style(color=Colors.ORCHID, bold=True))
    skip_text.append(")", style=Style(color=Colors.WARNING))
    console.print(skip_text)


def _configure_additional_providers(console: Console, config: GlobalConfig, skip_backend: str) -> None:
    """Configure additional API providers."""
    for backend, info in PROVIDERS.items():
        if backend == skip_backend:
            continue
        if backend in ("api",):  # Skip HuggingFace in additional config for now
            continue
            
        existing = os.getenv(info["env_var"])
        current_value = getattr(config, info["config_field"], "")
        
        if existing or current_value:
            status_text = Text()
            status_text.append(f"  ✓ {info['name']}: ", style=Style(color=Colors.SUCCESS))
            status_text.append("configured", style=Style(color=Colors.DIM))
            console.print(status_text)
            continue
        
        if Confirm.ask(Text(f"Configure {info['name']}?", style=Style(color=Colors.LAVENDER)), default=False):
            token = Prompt.ask(Text(f"  {info['name']} API Key", style=Style(color=Colors.ORCHID)), password=True)
            if token.strip():
                setattr(config, info["config_field"], token.strip())
                console.print(create_success_message(f"  {info['name']} API key saved!"))
            else:
                console.print(Text(f"  ○ Skipped", style=Style(color=Colors.DIM)))
        console.print()


def run_quick_onboarding(console: Console) -> GlobalConfig:
    """Run a minimal onboarding for users who want to skip the full flow."""
    config = load_global_config()
    
    config.username = getpass.getuser().capitalize()
    config.default_backend = "anthropic"  # Default to Anthropic now
    config.onboarding_completed = True
    config.onboarding_version = CURRENT_ONBOARDING_VERSION
    
    save_global_config(config)
    
    console.print(create_success_message("Quick setup complete!"))
    path_text = Text()
    path_text.append("  Config saved to: ", style=Style(color=Colors.DIM))
    path_text.append(str(get_global_config_dir()), style=Style(color=Colors.LAVENDER))
    console.print(path_text)
    console.print()
    
    return config
