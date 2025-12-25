"""Lovelace Code CLI - Main entry point."""

from __future__ import annotations

import getpass
import json
import re
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.style import Style
from rich.text import Text

from lovelace_code import __version__
from lovelace_code.activity import append_activity, read_recent_activity
from lovelace_code.backends import Backend, BackendError, GenerationConfig, get_backend, list_backends
from lovelace_code.config import LovelaceConfig, load_config, save_config
from lovelace_code.global_config import (
    get_global_config_dir,
    load_global_config,
    mark_session_start,
    needs_onboarding,
    save_global_config,
    setup_api_keys_from_config,
)
from lovelace_code.paths import (
    activity_log_path,
    config_path,
    find_project_root,
    lovelace_dir,
    memory_path,
)
from lovelace_code.prompts import build_conversation_prompt, build_system_prompt
from lovelace_code.tools import ToolResult, execute_tool, list_tools, get_tool
from lovelace_code.ui import (
    Colors,
    STYLES,
    WelcomeData,
    create_styled_panel,
    create_success_message,
    create_warning_message,
    create_error_message,
    create_info_message,
    create_key_value,
    create_header,
    render_login_success,
    render_thinking,
    render_tool_call,
    render_tool_result,
    render_welcome,
    render_init_success,
    render_doctor_report,
    render_config_panel,
    render_stats_panel,
    render_help_panel,
    render_tools_panel,
    render_models_table,
    render_model_info,
    render_model_current,
    render_user_message,
    render_assistant_prefix,
    render_streaming_assistant,
)

app = typer.Typer(
    name="lovelace",
    help="Lovelace Code: an agentic terminal tool powered by open models",
    add_completion=False,
    no_args_is_help=False,
)

# Models subcommand group
models_app = typer.Typer(
    name="models",
    help="Manage and browse available models",
)
app.add_typer(models_app, name="models")

console = Console()


def _check_first_run() -> None:
    """Check if this is first run and trigger onboarding if needed."""
    if needs_onboarding():
        from lovelace_code.onboarding import run_onboarding
        
        # Ask if they want full or quick setup
        console.print()
        console.print("[bold]Welcome to Lovelace Code![/bold]")
        console.print()
        
        if Confirm.ask("Would you like to run the setup wizard?", default=True):
            run_onboarding(console)
        else:
            from lovelace_code.onboarding import run_quick_onboarding
            run_quick_onboarding(console)


# ─────────────────────────────────────────────────────────────────────────────
# lovelace init
# ─────────────────────────────────────────────────────────────────────────────
@app.command()
def init(
    path: Path = typer.Argument(
        default=Path("."),
        help="Directory to initialize (defaults to current folder)",
    ),
    backend: str = typer.Option(
        None,
        "--backend", "-b",
        help="Backend to use (defaults to global config setting)",
    ),
    model: str = typer.Option(
        None,
        "--model", "-m",
        help="Model to use (defaults to global config setting)",
    ),
) -> None:
    """Initialize Lovelace Code in the current (or specified) project."""
    # Load global config for defaults
    global_config = load_global_config()
    
    # Use global config defaults if not specified
    if backend is None:
        backend = global_config.default_backend or "api"
    if model is None:
        model = global_config.default_model or "Spestly/Lovelace-1-3B"
    
    root = find_project_root(path)
    ldir = lovelace_dir(root)

    if ldir.exists():
        console.print(f"[yellow]Lovelace already initialized at {ldir}[/yellow]")
        console.print("Run [bold]lovelace[/bold] to start a session.")
        raise typer.Exit(0)

    # Validate backend
    valid_backends = ("api", "local", "llama", "openai", "anthropic", "gemini", "xai", "openrouter", "groq", "openai-compatible")
    if backend not in valid_backends:
        console.print(f"[red]Invalid backend: {backend}[/red]")
        console.print(f"Choose from: {', '.join(valid_backends)}")
        raise typer.Exit(1)

    ldir.mkdir(parents=True, exist_ok=True)
    
    cfg = LovelaceConfig(backend=backend, model=model)  # type: ignore
    save_config(config_path(root), cfg)
    
    # Initialize memory
    memory_path(root).write_text(
        json.dumps({"summary": "", "key_files": [], "notes": [], "tech_stack": []}, indent=2),
        encoding="utf-8",
    )
    
    append_activity(activity_log_path(root), "Initialized Lovelace Code")

    # Render success UI
    render_init_success(console, str(ldir), backend, model)
    
    if backend == "local":
        console.print()
        console.print(create_info_message("Local backend requires: pip install lovelace-code[local]"))
    elif backend == "llama":
        console.print()
        console.print(create_info_message("Llama backend requires: pip install lovelace-code[llama]"))


# ─────────────────────────────────────────────────────────────────────────────
# lovelace login (simulated)
# ─────────────────────────────────────────────────────────────────────────────
@app.command()
def login() -> None:
    """Authenticate with Hugging Face (for API backend)."""
    import os
    from rich.table import Table
    
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
    
    if token:
        console.print()
        console.print(create_success_message("Already authenticated via environment variable."))
        return
    
    # Login UI
    content = Table.grid(padding=(0, 1))
    content.add_column()
    content.add_row(create_info_message("Enter your Hugging Face API token"))
    content.add_row(Text(""))
    
    link_text = Text()
    link_text.append("  Get one at: ", style=STYLES["muted"])
    link_text.append("https://huggingface.co/settings/tokens", style=Style(color=Colors.LAVENDER, underline=True))
    content.add_row(link_text)
    
    panel = create_styled_panel(content, title="✦ Authentication")
    console.print()
    console.print(panel)
    console.print()
    
    token = Prompt.ask(Text("Token", style=Style(color=Colors.ORCHID)), password=True)
    
    if token:
        # Save to ~/.huggingface/token
        hf_dir = Path.home() / ".huggingface"
        hf_dir.mkdir(exist_ok=True)
        (hf_dir / "token").write_text(token.strip())
        
        render_login_success(console)
    else:
        console.print()
        console.print(create_warning_message("No token provided."))


# ─────────────────────────────────────────────────────────────────────────────
# lovelace (main interactive command)
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-v", help="Show version and exit"),
    plan: bool = typer.Option(False, "--plan", "-p", help="Start in plan mode"),
) -> None:
    """Launch an interactive Lovelace Code session."""
    if version:
        console.print(f"Lovelace Code v{__version__}")
        raise typer.Exit(0)

    # Skip if a subcommand was invoked
    if ctx.invoked_subcommand is not None:
        return

    # Check for first run / onboarding
    _check_first_run()
    
    # Setup API keys from config (if stored)
    global_cfg = load_global_config()
    setup_api_keys_from_config(global_cfg)
    
    # Mark session start for stats
    global_cfg = mark_session_start()

    root = find_project_root()
    cfg_file = config_path(root)

    if not cfg_file.exists():
        console.print(
            "[yellow]No .lovelace/ folder found.[/yellow]\n"
            "Run [bold]lovelace init[/bold] to initialize this project."
        )
        raise typer.Exit(1)

    cfg = load_config(cfg_file)
    activity = read_recent_activity(activity_log_path(root), limit=5)

    # Welcome banner - use username from global config if available
    user_label = global_cfg.username or getpass.getuser().capitalize()
    welcome_data = WelcomeData(
        user_label=user_label,
        project_path=str(root),
        model_label=cfg.model,
        backend_label=cfg.backend,
        recent_activity=activity,
        version=__version__,
    )
    
    console.clear()
    render_welcome(console, welcome_data)
    console.print()

    # Start REPL
    session = Session(root, cfg, plan_mode=plan)
    session.run()


# ─────────────────────────────────────────────────────────────────────────────
# Session class - handles the interactive REPL
# ─────────────────────────────────────────────────────────────────────────────
class Session:
    """Interactive chat session."""

    def __init__(self, root: Path, config: LovelaceConfig, plan_mode: bool = False):
        self.root = root
        self.config = config
        self.plan_mode = plan_mode
        self.history: list[tuple[str, str]] = []
        self.backend: Backend | None = None

    def run(self) -> None:
        """Main REPL loop."""
        while True:
            try:
                # Purple prompt like Claude Code
                user_input = Prompt.ask(Text("> ", style=Style(color=Colors.LIGHT_PURPLE, bold=True)))
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Goodbye![/dim]")
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            # Handle slash commands
            if user_input.startswith("/"):
                should_exit = self._handle_command(user_input)
                if should_exit:
                    break
                continue

            # Process message
            self._process_message(user_input)

    def _get_backend(self) -> Backend:
        """Lazy-load the backend."""
        if self.backend is None:
            try:
                self.backend = get_backend(self.config.backend)
                if not self.backend.is_available():
                    console.print(f"[red]Backend '{self.config.backend}' is not available.[/red]")
                    console.print("Try installing dependencies or switching backends with /backend")
                    raise typer.Exit(1)
            except Exception as exc:
                console.print(f"[red]Failed to initialize backend: {exc}[/red]")
                raise typer.Exit(1)
        return self.backend

    def _process_message(self, user_input: str) -> None:
        """Process a user message and generate a response."""
        self.history.append(("user", user_input))
        
        # Build prompt
        system = build_system_prompt(str(self.root), self.root.name)
        prompt = build_conversation_prompt(system, self.history, self.plan_mode)

        gen_config = GenerationConfig(
            model=self.config.model,
            temperature=self.config.temperature,
            max_new_tokens=self.config.max_new_tokens,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            repetition_penalty=self.config.repetition_penalty,
        )

        console.print()
        response_parts: list[str] = []
        
        try:
            backend = self._get_backend()
            
            if self.config.stream_output:
                # Show thinking indicator first, then stream with assistant prefix
                with Live(render_thinking(console), console=console, refresh_per_second=12) as live:
                    for chunk in backend.stream(prompt, gen_config):
                        response_parts.append(chunk)
                        # Show streaming with assistant bullet prefix
                        live.update(render_streaming_assistant("".join(response_parts)))
            else:
                with console.status("[dim]● thinking...[/dim]"):
                    response = backend.generate(prompt, gen_config)
                    response_parts.append(response)
                # Show with assistant prefix
                console.print(render_streaming_assistant(response))

        except BackendError as exc:
            console.print(f"[red]● Error:[/red] {exc}")
            self.history.pop()
            return

        assistant_text = "".join(response_parts).strip()
        self.history.append(("assistant", assistant_text))
        
        # Check for tool calls and capture results
        tool_results = self._handle_tool_calls(assistant_text)

        # If tools were executed, ask the model to continue the response using tool outputs
        if tool_results:
            max_followups = 3
            followups = 0
            while followups < max_followups:
                followups += 1
                console.print(create_info_message("Continuing assistant response after tool outputs..."))

                # Rebuild prompt with updated history (tool outputs were added by _handle_tool_calls)
                prompt = build_conversation_prompt(system, self.history, self.plan_mode)

                continuation_parts: list[str] = []
                try:
                    if self.config.stream_output:
                        with Live(render_thinking(console), console=console, refresh_per_second=12) as live:
                            for chunk in backend.stream(prompt, gen_config):
                                continuation_parts.append(chunk)
                                live.update(render_streaming_assistant("".join(continuation_parts)))
                        # Print final continuation
                        console.print(render_streaming_assistant("".join(continuation_parts)))
                    else:
                        with console.status("[dim]● thinking...[/dim]"):
                            cont = backend.generate(prompt, gen_config)
                            continuation_parts.append(cont)
                        console.print(render_streaming_assistant(cont))
                except BackendError as exc:
                    console.print(f"[red]● Error while generating continuation:[/red] {exc}")
                    break

                continuation_text = "".join(continuation_parts).strip()
                if not continuation_text:
                    break

                # Append continuation to history
                self.history.append(("assistant", continuation_text))

                # Check for additional tool calls in the continuation
                new_tool_results = self._handle_tool_calls(continuation_text)
                if not new_tool_results:
                    break

            console.print()

        # Log activity
        append_activity(activity_log_path(self.root), f"Chat: {user_input[:40]}…")
        console.print()

    def _handle_tool_calls(self, response: str) -> list[tuple[str, "ToolResult"]]:
        """Parse and execute tool calls from the response.

        Supports multiple formats:
         - ```tool\n{...}\n``` blocks (preferred)
         - Inline JSON objects containing a top-level "tool" key
         - Heuristic lines like "tool: repo_indexer" with optional JSON args

        Returns a list of (tool_name, ToolResult) for each executed tool.
        """
        executed_results: list[tuple[str, "ToolResult"]] = []

        # First try explicit ```tool blocks
        tool_pattern = r"```tool\s*\n({.*?})\s*\n```"
        matches = list(re.findall(tool_pattern, response, re.DOTALL))

        # If none found, look for any inline JSON object that includes a "tool" key
        if not matches:
            # Extract balanced JSON objects to avoid matching across unrelated braces
            objs: list[str] = []
            stack = []
            start = None
            for i, ch in enumerate(response):
                if ch == '{':
                    if not stack:
                        start = i
                    stack.append(ch)
                elif ch == '}':
                    if stack:
                        stack.pop()
                        if not stack and start is not None:
                            snippet = response[start:i+1]
                            if '"tool"' in snippet:
                                objs.append(snippet)
                            start = None
            matches = objs

        # Also support simple "tool: name" lines as a fallback
        fallback_calls: list[dict] = []
        simple_pattern = r"\btool\b\s*[:=]\s*([A-Za-z0-9_\-]+)(?:.*?args\s*[:=]\s*({.*?}))?"
        for m in re.finditer(simple_pattern, response, re.IGNORECASE | re.DOTALL):
            tool_name = m.group(1)
            args_text = m.group(2)
            args = {}
            if args_text:
                try:
                    args = json.loads(args_text)
                except Exception:
                    # If parsing fails, leave args empty and log
                    console.print(f"[yellow]Couldn't parse args for tool '{tool_name}', ignoring args.[/yellow]")
            fallback_calls.append({"tool": tool_name, "args": args})

        # Combine explicit matches and fallback_calls
        processed = set()
        for match in matches:
            try:
                data = json.loads(match)
                tool_name = data.get("tool")
                args = data.get("args", {})
                if not tool_name:
                    continue
                key = f"{tool_name}:{json.dumps(args, sort_keys=True)}"
                if key in processed:
                    continue
                processed.add(key)

                # Check required params if available on the tool definition
                try:
                    tool_obj = get_tool(tool_name)
                    required_params = tool_obj.parameters.get("required", []) or []
                except Exception:
                    required_params = []

                missing = [p for p in required_params if p not in args]
                if missing:
                    # Prompt user to supply missing arguments interactively
                    cancelled = False
                    tool_props = getattr(tool_obj, "parameters", {}).get("properties", {})
                    for p in missing:
                        prop = tool_props.get(p, {})
                        desc = prop.get("description", "")
                        typ = prop.get("type")
                        prompt_msg = f"Provide value for '{p}'"
                        if desc:
                            prompt_msg += f" ({desc})"
                        try:
                            val = Prompt.ask(prompt_msg, default="")
                        except Exception:
                            val = ""

                        if val == "":
                            console.print("  [dim]Skipped[/dim]")
                            cancelled = True
                            break

                        # Try JSON parsing first for structured values
                        try:
                            parsed = json.loads(val)
                        except Exception:
                            # Fall back to basic type coercion based on schema if available
                            if typ == "integer":
                                try:
                                    parsed = int(val)
                                except Exception:
                                    parsed = val
                            elif typ in ("number", "float"):
                                try:
                                    parsed = float(val)
                                except Exception:
                                    parsed = val
                            elif typ == "boolean":
                                parsed = val.lower() in ("true", "1", "yes", "y")
                            else:
                                parsed = val

                        args[p] = parsed

                    if cancelled:
                        continue

                render_tool_call(console, tool_name, args)
                if not self.config.auto_approve_tools:
                    if not Confirm.ask("  Execute?", default=True):
                        console.print("  [dim]Skipped[/dim]")
                        continue

                result = execute_tool(tool_name, **args)
                render_tool_result(console, tool_name, result.success, result.output)
                self.history.append(("tool", f"Tool {tool_name} returned: {result.output[:500]}"))
                # Collect executed result for higher-level handling
                executed_results.append((tool_name, result))

            except json.JSONDecodeError:
                console.print(f"[yellow]Invalid tool call JSON[/yellow]")
            except Exception as exc:
                console.print(f"[red]Tool error:[/red] {exc}")

        for call in fallback_calls:
            tool_name = call.get("tool")
            args = call.get("args", {})
            key = f"{tool_name}:{json.dumps(args, sort_keys=True)}"
            if key in processed:
                continue
            processed.add(key)

            try:
                tool_obj = get_tool(tool_name)
                required_params = tool_obj.parameters.get("required", []) or []
            except Exception:
                required_params = []

            missing = [p for p in required_params if p not in args]
            if missing:
                cancelled = False
                tool_props = getattr(tool_obj, "parameters", {}).get("properties", {})
                for p in missing:
                    prop = tool_props.get(p, {})
                    desc = prop.get("description", "")
                    typ = prop.get("type")
                    prompt_msg = f"Provide value for '{p}'"
                    if desc:
                        prompt_msg += f" ({desc})"
                    try:
                        val = Prompt.ask(prompt_msg, default="")
                    except Exception:
                        val = ""

                    if val == "":
                        console.print("  [dim]Skipped[/dim]")
                        cancelled = True
                        break

                    try:
                        parsed = json.loads(val)
                    except Exception:
                        if typ == "integer":
                            try:
                                parsed = int(val)
                            except Exception:
                                parsed = val
                        elif typ in ("number", "float"):
                            try:
                                parsed = float(val)
                            except Exception:
                                parsed = val
                        elif typ == "boolean":
                            parsed = val.lower() in ("true", "1", "yes", "y")
                        else:
                            parsed = val

                    args[p] = parsed

                if cancelled:
                    continue

            render_tool_call(console, tool_name, args)
            if not self.config.auto_approve_tools:
                if not Confirm.ask("  Execute?", default=True):
                    console.print("  [dim]Skipped[/dim]")
                    continue

            try:
                result = execute_tool(tool_name, **args)
                render_tool_result(console, tool_name, result.success, result.output)
                self.history.append(("tool", f"Tool {tool_name} returned: {result.output[:500]}"))
                # Collect executed result for higher-level handling
                executed_results.append((tool_name, result))
            except Exception as exc:
                console.print(f"[red]Tool error:[/red] {exc}")

        return executed_results
        for call in fallback_calls:
            tool_name = call.get("tool")
            args = call.get("args", {})
            key = f"{tool_name}:{json.dumps(args, sort_keys=True)}"
            if key in processed:
                continue
            processed.add(key)

            try:
                tool_obj = get_tool(tool_name)
                required_params = tool_obj.parameters.get("required", []) or []
            except Exception:
                required_params = []

            missing = [p for p in required_params if p not in args]
            if missing:
                cancelled = False
                tool_props = getattr(tool_obj, "parameters", {}).get("properties", {})
                for p in missing:
                    prop = tool_props.get(p, {})
                    desc = prop.get("description", "")
                    typ = prop.get("type")
                    prompt_msg = f"Provide value for '{p}'"
                    if desc:
                        prompt_msg += f" ({desc})"
                    try:
                        val = Prompt.ask(prompt_msg, default="")
                    except Exception:
                        val = ""

                    if val == "":
                        console.print("  [dim]Skipped[/dim]")
                        cancelled = True
                        break

                    try:
                        parsed = json.loads(val)
                    except Exception:
                        if typ == "integer":
                            try:
                                parsed = int(val)
                            except Exception:
                                parsed = val
                        elif typ in ("number", "float"):
                            try:
                                parsed = float(val)
                            except Exception:
                                parsed = val
                        elif typ == "boolean":
                            parsed = val.lower() in ("true", "1", "yes", "y")
                        else:
                            parsed = val

                    args[p] = parsed

                if cancelled:
                    continue

            render_tool_call(console, tool_name, args)
            if not self.config.auto_approve_tools:
                if not Confirm.ask("  Execute?", default=True):
                    console.print("  [dim]Skipped[/dim]")
                    continue

            try:
                result = execute_tool(tool_name, **args)
                render_tool_result(console, tool_name, result.success, result.output)
                self.history.append(("tool", f"Tool {tool_name} returned: {result.output[:500]}"))
                # Collect executed result for higher-level handling
                executed_results.append((tool_name, result))
            except Exception as exc:
                console.print(f"[red]Tool error:[/red] {exc}")

    def _handle_command(self, cmd: str) -> bool:
        """Handle slash commands. Returns True if should exit."""
        parts = cmd.lower().split()
        keyword = parts[0]

        if keyword in {"/exit", "/quit", "/q"}:
            console.print("[dim]Goodbye![/dim]")
            return True

        if keyword == "/help":
            self._show_help()
            return False

        if keyword == "/config":
            self._show_config()
            return False

        if keyword == "/model":
            self._handle_model_command(parts)
            return False

        if keyword == "/backend":
            self._handle_backend_command(parts)
            return False

        if keyword == "/tools":
            self._show_tools()
            return False

        if keyword == "/plan":
            self.plan_mode = not self.plan_mode
            status = "ON" if self.plan_mode else "OFF"
            console.print(f"[cyan]Plan mode: {status}[/cyan]")
            return False

        if keyword == "/clear":
            self.history.clear()
            console.print("[dim]Conversation cleared.[/dim]")
            return False

        if keyword == "/history":
            self._show_history()
            return False

        if keyword == "/auto":
            self.config.auto_approve_tools = not self.config.auto_approve_tools
            status = "ON" if self.config.auto_approve_tools else "OFF"
            save_config(config_path(self.root), self.config)
            console.print(f"[cyan]Auto-approve tools: {status}[/cyan]")
            return False

        console.print(f"[yellow]Unknown command: {cmd}[/yellow]. Try /help")
        return False

    def _show_help(self) -> None:
        render_help_panel(console)

    def _show_config(self) -> None:
        render_config_panel(
            console,
            self.config.model_dump(),
            str(config_path(self.root)),
            title="Project Configuration",
        )

    def _handle_model_command(self, parts: list[str]) -> None:
        if len(parts) == 1:
            render_model_current(
                console,
                project_model=self.config.model,
                project_deep_model=self.config.deep_model,
                project_config_path=str(config_path(self.root)),
                global_default=load_global_config().default_model,
            )
        else:
            new_model = parts[1]
            self.config.model = new_model
            save_config(config_path(self.root), self.config)
            
            # Unload current model if local
            if self.backend:
                self.backend.unload_model()
                self.backend = None
            
            console.print()
            console.print(create_success_message(f"Model set to {new_model}"))

    def _handle_backend_command(self, parts: list[str]) -> None:
        if len(parts) == 1:
            from rich.table import Table
            
            content = Table.grid(padding=(0, 1))
            content.add_column()
            content.add_row(create_key_value("Current backend", self.config.backend))
            content.add_row(Text(""))
            content.add_row(create_header("Available backends", "◈"))
            content.add_row(Text(""))
            
            for b in list_backends():
                backend_text = Text()
                if b == self.config.backend:
                    backend_text.append(f"  ● {b}", style=Style(color=Colors.SUCCESS))
                else:
                    backend_text.append(f"  ○ {b}", style=Style(color=Colors.LAVENDER))
                content.add_row(backend_text)
            
            content.add_row(Text(""))
            hint_text = Text()
            hint_text.append("  Usage: ", style=STYLES["muted"])
            hint_text.append("/backend <name>", style=Style(color=Colors.ORCHID))
            content.add_row(hint_text)
            
            panel = create_styled_panel(content, title="✦ Backend")
            console.print()
            console.print(panel)
        else:
            new_backend = parts[1]
            if new_backend not in list_backends():
                console.print()
                console.print(create_error_message(f"Unknown backend: {new_backend}"))
                console.print(create_info_message(f"Available: {', '.join(list_backends())}"))
                return
            
            # Unload current backend
            if self.backend:
                self.backend.unload_model()
                self.backend = None
            
            self.config.backend = new_backend  # type: ignore
            save_config(config_path(self.root), self.config)
            console.print()
            console.print(create_success_message(f"Backend set to {new_backend}"))

    def _show_tools(self) -> None:
        tools = list_tools()
        render_tools_panel(console, tools)

    def _show_history(self) -> None:
        from rich.table import Table
        
        activity = read_recent_activity(activity_log_path(self.root), limit=20)
        
        content = Table.grid(padding=(0, 1))
        content.add_column()
        
        for entry in activity:
            entry_text = Text()
            entry_text.append("  ◈ ", style=Style(color=Colors.ORCHID))
            entry_text.append(entry, style=Style(color=Colors.LAVENDER))
            content.add_row(entry_text)
        
        if not activity:
            content.add_row(create_info_message("No recent activity"))
        
        panel = create_styled_panel(content, title="✦ Recent Activity")
        console.print()
        console.print(panel)


# ─────────────────────────────────────────────────────────────────────────────
# Additional commands
# ─────────────────────────────────────────────────────────────────────────────
@app.command()
def config() -> None:
    """Show current configuration."""
    root = find_project_root()
    cfg_file = config_path(root)
    
    if not cfg_file.exists():
        console.print()
        console.print(create_warning_message("Not initialized. Run 'lovelace init' first."))
        raise typer.Exit(1)
    
    cfg = load_config(cfg_file)
    render_config_panel(console, cfg.model_dump(), str(cfg_file), title="Project Configuration")


@app.command()
def doctor() -> None:
    """Check system requirements and backend availability."""
    import os
    
    # Collect data
    py_version = sys.version_info[:3]
    
    # Check all backends
    all_backends = [
        "api",  # HuggingFace
        "openai",
        "anthropic", 
        "gemini",
        "xai",
        "openrouter",
        "openai-compatible",
        "local",
        "llama",
    ]
    
    backends_status = []
    for backend_name in all_backends:
        try:
            backend = get_backend(backend_name)
            available = backend.is_available()
            backends_status.append((backend_name, available, None))
        except Exception as exc:
            backends_status.append((backend_name, False, str(exc)))
    
    # Check API keys
    api_keys_status = {
        "HuggingFace": bool(os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")),
        "OpenAI": bool(os.getenv("OPENAI_API_KEY")),
        "Anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "Google/Gemini": bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
        "xAI": bool(os.getenv("XAI_API_KEY")),
        "OpenRouter": bool(os.getenv("OPENROUTER_API_KEY")),
    }
    
    # Check GPU
    gpu_info = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info = torch.cuda.get_device_name(0)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            gpu_info = "Apple Silicon (MPS)"
    except ImportError:
        pass
    
    render_doctor_report(
        console,
        python_version=py_version,
        backends=backends_status,
        api_keys=api_keys_status,
        gpu_info=gpu_info,
        config_path=str(get_global_config_dir()),
    )


@app.command()
def setup() -> None:
    """Re-run the setup wizard."""
    from lovelace_code.onboarding import run_onboarding
    
    console.print()
    if Confirm.ask(
        Text("Re-run the setup wizard?", style=Style(color=Colors.LAVENDER)),
        default=True
    ):
        run_onboarding(console)
    else:
        console.print(create_info_message("Cancelled."))


@app.command("config-global")
def config_global(
    show: bool = typer.Option(True, "--show", "-s", help="Show global configuration"),
    reset: bool = typer.Option(False, "--reset", help="Reset global configuration"),
    set_key: str = typer.Option(None, "--set", help="Set a config value (format: key=value)"),
    backend: str = typer.Option(None, "--backend", "-b", help="Set default backend"),
    model: str = typer.Option(None, "--model", "-m", help="Set default model"),
) -> None:
    """View or manage global configuration."""
    global_cfg = load_global_config()
    
    # Handle --set key=value
    if set_key:
        if "=" not in set_key:
            console.print(create_error_message("Use format: --set key=value"))
            raise typer.Exit(1)
        key, value = set_key.split("=", 1)
        key = key.strip()
        
        # Validate key exists
        if not hasattr(global_cfg, key):
            valid_keys = [k for k in global_cfg.model_dump().keys() if not k.startswith("_")]
            console.print(create_error_message(f"Unknown key: {key}"))
            console.print(f"Valid keys: {', '.join(valid_keys)}")
            raise typer.Exit(1)
        
        # Handle boolean values
        if value.lower() in ("true", "yes", "1"):
            value = True
        elif value.lower() in ("false", "no", "0"):
            value = False
        
        setattr(global_cfg, key, value)
        save_global_config(global_cfg)
        console.print(create_success_message(f"Set {key} = {value}"))
        return
    
    # Handle --backend shortcut
    if backend:
        valid_backends = ["api", "local", "llama", "openai", "anthropic", "gemini", "xai", "openrouter", "groq", "openai-compatible"]
        if backend not in valid_backends:
            console.print(create_error_message(f"Invalid backend: {backend}"))
            console.print(f"Valid backends: {', '.join(valid_backends)}")
            raise typer.Exit(1)
        global_cfg.default_backend = backend
        save_global_config(global_cfg)
        console.print(create_success_message(f"Default backend set to: {backend}"))
        return
    
    # Handle --model shortcut
    if model:
        global_cfg.default_model = model
        save_global_config(global_cfg)
        console.print(create_success_message(f"Default model set to: {model}"))
        return
    
    if reset:
        console.print()
        if Confirm.ask(
            Text("Reset all global settings?", style=Style(color=Colors.WARNING)),
            default=False
        ):
            from lovelace_code.global_config import GlobalConfig
            save_global_config(GlobalConfig())
            console.print(create_success_message("Global configuration reset."))
        return
    
    render_config_panel(
        console,
        global_cfg.model_dump(),
        str(get_global_config_dir() / "config.json"),
        title="Global Configuration",
    )


@app.command()
def stats() -> None:
    """Show usage statistics."""
    global_cfg = load_global_config()
    
    render_stats_panel(
        console,
        total_sessions=global_cfg.total_sessions,
        total_messages=global_cfg.total_messages,
        first_run=global_cfg.first_run_at,
        last_run=global_cfg.last_run_at,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Models subcommands
# ─────────────────────────────────────────────────────────────────────────────
@models_app.command("list")
def models_list(
    local: bool = typer.Option(False, "--local", "-l", help="Show models optimized for local inference"),
    gguf: bool = typer.Option(False, "--gguf", "-g", help="Show GGUF models for llama.cpp"),
    api: bool = typer.Option(False, "--api", "-a", help="Show commercial API models (OpenAI, Anthropic, etc.)"),
    provider: str = typer.Option(None, "--provider", "-p", help="Filter by provider (openai, anthropic, google, xai, zhipu, huggingface)"),
    search: str = typer.Option(None, "--search", "-s", help="Search models by name"),
    set_model: str = typer.Option(None, "--set", help="Set this model as default"),
) -> None:
    """List available models or set the default model."""
    from lovelace_code.models import (
        RECOMMENDED_MODELS,
        GGUF_MODELS,
        LOCAL_OPTIMIZED_MODELS,
        API_MODELS,
        LOVELACE_MODELS,
        get_model_info,
        search_models,
        estimate_vram_gb,
        get_models_by_provider,
    )
    
    # If --set is provided, set the model
    if set_model:
        _set_model(set_model)
        return
    
    # Determine which models to show
    if search:
        models = search_models(search)
        title = f"Search results for '{search}'"
    elif provider:
        models = get_models_by_provider(provider.lower())
        title = f"Models from {provider.title()}"
    elif api:
        models = API_MODELS
        title = "Commercial API Models"
    elif gguf:
        models = GGUF_MODELS
        title = "GGUF Models (for llama.cpp backend)"
    elif local:
        models = LOCAL_OPTIMIZED_MODELS
        title = "Models optimized for local inference"
    else:
        # Default: show Lovelace + API models (most useful)
        models = LOVELACE_MODELS + API_MODELS
        title = "Featured Models"
    
    if not models:
        console.print()
        console.print(create_warning_message("No models found."))
        return
    
    # Get current model
    try:
        root = find_project_root()
        cfg = load_config(config_path(root))
        current_model = cfg.model
    except Exception:
        global_cfg = load_global_config()
        current_model = global_cfg.default_model
    
    render_models_table(console, models, current_model, title)
    
    # Show hints for other categories
    hints = Text()
    hints.append("\n  Other categories: ", style=STYLES["muted"])
    hints.append("--api", style=Style(color=Colors.ORCHID))
    hints.append(" (commercial)  ", style=STYLES["muted"])
    hints.append("--local", style=Style(color=Colors.ORCHID))
    hints.append(" (open-source)  ", style=STYLES["muted"])
    hints.append("--gguf", style=Style(color=Colors.ORCHID))
    hints.append(" (quantized)", style=STYLES["muted"])
    console.print(hints)


@models_app.command("set")
def models_set(
    model_id: str = typer.Argument(..., help="Model ID to set (e.g., 'codellama/CodeLlama-7b-hf')"),
    globally: bool = typer.Option(False, "--global", "-g", help="Set as global default"),
) -> None:
    """Set the active model (use any HuggingFace model ID)."""
    _set_model(model_id, globally=globally)


@models_app.command("current")
def models_current() -> None:
    """Show the currently active model."""
    try:
        root = find_project_root()
        cfg = load_config(config_path(root))
        project_model = cfg.model
        project_deep = cfg.deep_model
        project_path = str(config_path(root))
    except Exception:
        project_model = None
        project_deep = None
        project_path = None
    
    global_cfg = load_global_config()
    
    render_model_current(
        console,
        project_model=project_model,
        project_deep_model=project_deep,
        project_config_path=project_path,
        global_default=global_cfg.default_model,
    )


@models_app.command("search")
def models_search(
    query: str = typer.Argument(..., help="Search query"),
) -> None:
    """Search for models by name or description."""
    from lovelace_code.models import search_models as do_search
    
    models = do_search(query)
    
    if not models:
        console.print()
        console.print(create_warning_message(f"No models found matching '{query}'"))
        console.print(create_info_message("You can use any HuggingFace model ID with 'lovelace models set <id>'"))
        return
    
    # Get current model
    try:
        root = find_project_root()
        cfg = load_config(config_path(root))
        current_model = cfg.model
    except Exception:
        global_cfg = load_global_config()
        current_model = global_cfg.default_model
    
    render_models_table(console, models, current_model, f"Models matching '{query}'")


@models_app.command("info")
def models_info(
    model_id: str = typer.Argument(..., help="Model ID to get info for"),
) -> None:
    """Get detailed information about a model."""
    from lovelace_code.models import get_model_info, estimate_vram_gb
    
    info = get_model_info(model_id)
    
    if info:
        render_model_info(
            console,
            model_id=info.id,
            name=info.name,
            size=info.size,
            context_length=info.context_length,
            backend=info.backend,
            description=info.description,
            vram_estimate=estimate_vram_gb(info.size),
            recommended=info.recommended,
            provider=info.provider,
        )
    else:
        console.print()
        console.print(create_info_message(f"Model '{model_id}' not in curated list."))
        console.print(create_info_message(f"You can still use it with 'lovelace models set {model_id}'"))
        console.print()
        
        link_text = Text()
        link_text.append("  HuggingFace: ", style=STYLES["muted"])
        link_text.append(f"https://huggingface.co/{model_id}", style=Style(color=Colors.LAVENDER, underline=True))
        console.print(link_text)


def _set_model(model_id: str, globally: bool = False) -> None:
    """Internal helper to set a model."""
    from lovelace_code.models import get_model_info
    
    # Check if it's a known model
    info = get_model_info(model_id)
    
    console.print()
    if info:
        console.print(create_info_message(f"Setting model: {info.name} ({info.size})"))
    else:
        console.print(create_info_message(f"Setting custom model: {model_id}"))
        console.print(create_warning_message("Not in curated list - make sure the model ID is correct"))
    
    if globally:
        # Set global default
        global_cfg = load_global_config()
        global_cfg.default_model = model_id
        save_global_config(global_cfg)
        console.print(create_success_message(f"Global default model set to {model_id}"))
    else:
        # Set project model
        try:
            root = find_project_root()
            cfg_file = config_path(root)
            
            if not cfg_file.exists():
                console.print(create_warning_message("No project initialized. Use --global or run 'lovelace init' first."))
                return
            
            cfg = load_config(cfg_file)
            cfg.model = model_id
            save_config(cfg_file, cfg)
            console.print(create_success_message(f"Project model set to {model_id}"))
        except Exception as e:
            console.print(create_error_message(str(e)))
            console.print(create_info_message("Use --global to set as global default instead."))


if __name__ == "__main__":
    app()
