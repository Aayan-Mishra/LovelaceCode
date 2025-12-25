# Lovelace Code 🚀

**Lovelace Code** is a **terminal-first, model-agnostic agentic coding environment** designed for developers who want powerful AI assistance **without being locked to a single model or vendor**.

It provides a lightweight local runtime where **any capable language model**—open or proprietary—can act as a software agent through a shared tool interface, with persistent per-repository state.

> Think *Claude Code*, but **vendor-independent**, **local-first**, and **extensible by design**.

---

## Why Lovelace? 🧠

Most AI coding tools hard-wire:
- a single provider
- a fixed UX
- opaque agent logic

Lovelace flips the stack:

```

Tool Runtime (stable)
└─ Agent Protocol
└─ Interchangeable Models

```

Models are engines.  
Lovelace is the runtime.

---

## Core Features ✨

- **Terminal-first agentic workflow** — no IDE lock-in
- **Model-agnostic architecture** (local + API backends)
- **Per-repo persistent state** stored under `.lovelace/`
- **Pluggable backends** for Hugging Face and hosted APIs
- **Fast iteration loop** for coding, refactoring, debugging
- **Minimal surface area** — easy to extend, easy to reason about

---

## Default Local Models 🧩

Out of the box, Lovelace ships with open, local-friendly models:

- `Spestly/Lovelace-1-3B` — fast, efficient coding assistant
- `Spestly/Lovelace-1-7B` — more capable reasoning and code generation

These are ideal for:
- local development
- offline workflows
- experimentation without API costs

---

## API / Hosted Models (Optional) 🌐

Lovelace can also interface with **state-of-the-art hosted models** via optional API backends.

These models are **not required** — they are simply additional engines Lovelace can drive.

> Proprietary APIs are treated as interchangeable backends, not dependencies.

```

╭─────────────────────────┬───────────┬───────────┬─────────┬──────────────────────────────────────────╮
│ Model ID                │ Provider  │      Size │ Context │ Description                              │
├─────────────────────────┼───────────┼───────────┼─────────┼──────────────────────────────────────────┤
│ ● Spestly/Lovelace-1-3B │    HF     │        3B │      4K │ Fast, efficient coding assistant         │
│ ★ Spestly/Lovelace-1-7B │    HF     │        7B │      4K │ More capable coding assistant            │
│ ★ gpt-5.2-pro           │  OpenAI   │  Flagship │    256K │ High-capability general reasoning        │
│ gpt-5.2-thinking        │  OpenAI   │  Flagship │    256K │ Reasoning-focused variant                │
│ gpt-5.2-instant         │  OpenAI   │      Fast │    128K │ Low-latency, cost-efficient              │
│ ★ o4-mini               │  OpenAI   │ Efficient │    128K │ Efficient reasoning (o-series)           │
│ ★ claude-opus-4.5       │ Anthropic │  Flagship │    200K │ Top-tier reasoning + tool use            │
│ ★ claude-sonnet-4.5     │ Anthropic │  Balanced │    200K │ Balanced reasoning and speed             │
│ claude-haiku-4.5        │ Anthropic │     Light │    200K │ Lightweight, fast variant                │
│ ★ gemini-3-pro          │  Google   │  Flagship │      1M │ Large-context multimodal reasoning       │
│ gemini-3-flash          │  Google   │      Fast │      1M │ Faster, cheaper Gemini variant           │
│ ★ grok-4                │    xAI    │     Large │    128K │ General reasoning + real-time focus      │
│ ★ glm-4.7               │   Zhipu   │     Large │    128K │ Strong coding and reasoning              │
│ glm-4.6v-flash          │   Zhipu   │      Fast │    128K │ Efficient multimodal variant             │
╰─────────────────────────┴───────────┴───────────┴─────────┴────────────────────────────────────────╯

````

---

## Quickstart (Development) 💻

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
````

Enable Lovelace in any project:

```bash
lovelace init
lovelace
```

This creates a `.lovelace/` directory and starts an interactive agent session.

---

## Commands 🧭

* `lovelace init` — initialize repo-scoped Lovelace state
* `lovelace` — start an interactive agent session

Inside a session:

* `/model` — switch the active model backend
* `/config` — inspect runtime configuration

---

## Project State & Files 📁

Each repository using Lovelace contains a `.lovelace/` folder:

* `.lovelace/config.json` — per-repo configuration
* `.lovelace/activity.log` — chronological action log
* `.lovelace/memory.md` — lightweight long-term agent memory

This keeps agent context **local, inspectable, and version-controllable**.

---

## Backends & Extensibility 🔧

Backend implementations live under:

```
src/lovelace_code/backends/
```

To add a new backend:

1. Implement the backend interface
2. Register it in `backends/registry.py`
3. Add basic tests for expected behavior

Lovelace is intentionally minimal to make backend development trivial.

---

## Troubleshooting ⚠️

* Ensure required API tokens are set (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`)
* Check `.lovelace/activity.log` for recent errors
* Inspect backend implementations for model-specific issues

---

## Philosophy 🧩

Lovelace Code is built around one principle:

> **Agentic tooling should outlive models.**

Models will change.
Vendors will change.
The runtime should not.

---

## License & Contributions 📜

Add a `LICENSE` file if publishing publicly.

Contributions are welcome:

* new backends
* tooling improvements
* protocol refinements

Open an issue or PR to discuss changes.

---

*Lovelace Code — vendor-independent agentic AI for developers who care about control.* 💡
