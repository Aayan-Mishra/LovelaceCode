from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rich.align import Align
from rich.box import ROUNDED, HEAVY, DOUBLE, SIMPLE
from rich.columns import Columns
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.style import Style
from rich.table import Table
from rich.text import Text
from rich import box


# ─────────────────────────────────────────────────────────────────────────────
# Color Palette - Purple & Cherry Blossom Pink
# ─────────────────────────────────────────────────────────────────────────────
class Colors:
    """Lovelace Code color palette - Purple & Cherry Blossom Pink."""
    
    # Primary purples
    DEEP_PURPLE = "#6B21A8"      # Deep purple
    PURPLE = "#9333EA"           # Main purple
    LIGHT_PURPLE = "#A855F7"     # Light purple
    SOFT_PURPLE = "#C084FC"      # Soft purple
    PALE_PURPLE = "#E9D5FF"      # Pale purple
    
    # Cherry blossom pinks
    CHERRY_DARK = "#BE185D"      # Dark cherry
    CHERRY = "#DB2777"           # Main cherry
    CHERRY_LIGHT = "#EC4899"     # Light cherry
    SAKURA = "#F472B6"           # Sakura pink
    BLOSSOM = "#FBCFE8"          # Soft blossom
    
    # Accents
    LAVENDER = "#DDD6FE"         # Lavender
    ORCHID = "#D946EF"           # Orchid
    MAGENTA = "#E879F9"          # Magenta
    
    # Neutrals
    WHITE = "#FAFAFA"
    LIGHT_GRAY = "#A1A1AA"
    DIM = "#71717A"
    DARK = "#27272A"
    
    # Semantic
    SUCCESS = "#22C55E"
    WARNING = "#F59E0B"
    ERROR = "#EF4444"
    INFO = "#8B5CF6"


# Pre-defined styles for consistent usage
STYLES = {
    "primary": Style(color=Colors.PURPLE),
    "secondary": Style(color=Colors.CHERRY),
    "accent": Style(color=Colors.SAKURA),
    "highlight": Style(color=Colors.ORCHID, bold=True),
    "muted": Style(color=Colors.DIM),
    "success": Style(color=Colors.SUCCESS),
    "warning": Style(color=Colors.WARNING),
    "error": Style(color=Colors.ERROR),
    "info": Style(color=Colors.INFO),
    "title": Style(color=Colors.LIGHT_PURPLE, bold=True),
    "subtitle": Style(color=Colors.SAKURA),
    "border": Style(color=Colors.SOFT_PURPLE),
    "header": Style(color=Colors.CHERRY_LIGHT, bold=True),
    "link": Style(color=Colors.LAVENDER, underline=True),
    "key": Style(color=Colors.ORCHID, bold=True),
    "value": Style(color=Colors.WHITE),
}


# ─────────────────────────────────────────────────────────────────────────────
# ASCII Art Logo with gradient effect
# ─────────────────────────────────────────────────────────────────────────────
LOGO_ART = r"""
██╗      ██████╗ ██╗   ██╗███████╗██╗      █████╗  ██████╗███████╗
██║     ██╔═══██╗██║   ██║██╔════╝██║     ██╔══██╗██╔════╝██╔════╝
██║     ██║   ██║██║   ██║█████╗  ██║     ███████║██║     █████╗  
██║     ██║   ██║╚██╗ ██╔╝██╔══╝  ██║     ██╔══██║██║     ██╔══╝  
███████╗╚██████╔╝ ╚████╔╝ ███████╗███████╗██║  ██║╚██████╗███████╗
╚══════╝ ╚═════╝   ╚═══╝  ╚══════╝╚══════╝╚═╝  ╚═╝ ╚═════╝╚══════╝
 ██████╗ ██████╗ ██████╗ ███████╗
██╔════╝██╔═══██╗██╔══██╗██╔════╝
██║     ██║   ██║██║  ██║█████╗  
██║     ██║   ██║██║  ██║██╔══╝  
╚██████╗╚██████╔╝██████╔╝███████╗
 ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝
"""

SMALL_LOGO = r"""
    __    _____  _   _ ______ _        _    _____ ______ 
   / /   / ___ \| | | |  ____| |      / \  / ____|  ____|
  / /   | |   | | | | | |__  | |     / _ \| |    | |__   
 / /    | |   | | | | |  __| | |    / ___ | |    |  __|  
/ /___  | |___| | |_| | |____| |___/ /   \ \____| |____ 
\_____\  \_____/ \___/|______|_____/_/     \_\____|______|
"""

# Retro computer monitor mascot (compact)
MASCOT = r"""
  .------.
  |.----.|
  ||    ||
  |'----'|
  |  <>  |
  '------'
"""

# Smaller mascot for side-by-side display
MASCOT_MINI = r"""
 ┌──────┐
 │ ▀▀▀▀ │
 │ ▄▄▄▄ │
 └──┬┬──┘
   ─┴┴─
"""

# LOVELACE text for side-by-side with mascot
LOVELACE_TEXT = r"""
██╗      ██████╗ ██╗   ██╗███████╗██╗      █████╗  ██████╗███████╗
██║     ██╔═══██╗██║   ██║██╔════╝██║     ██╔══██╗██╔════╝██╔════╝
██║     ██║   ██║██║   ██║█████╗  ██║     ███████║██║     █████╗  
██║     ██║   ██║╚██╗ ██╔╝██╔══╝  ██║     ██╔══██║██║     ██╔══╝  
███████╗╚██████╔╝ ╚████╔╝ ███████╗███████╗██║  ██║╚██████╗███████╗
╚══════╝ ╚═════╝   ╚═══╝  ╚══════╝╚══════╝╚═╝  ╚═╝ ╚═════╝╚══════╝
"""

# Gradient colors for logo (top to bottom)
LOGO_GRADIENT = [
    Colors.DEEP_PURPLE,
    Colors.PURPLE,
    Colors.LIGHT_PURPLE,
    Colors.ORCHID,
    Colors.CHERRY,
    Colors.CHERRY_LIGHT,
    Colors.SAKURA,
    Colors.BLOSSOM,
]


@dataclass(frozen=True)
class WelcomeData:
    user_label: str
    project_path: str
    model_label: str
    backend_label: str
    recent_activity: list[str]
    version: str = "0.1.0"


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────
def gradient_text(text: str, colors: list[str] | None = None) -> Text:
    """Create text with a gradient effect."""
    if colors is None:
        colors = LOGO_GRADIENT
    
    lines = text.strip().split("\n")
    result = Text()
    
    for i, line in enumerate(lines):
        color_idx = min(i * len(colors) // max(len(lines), 1), len(colors) - 1)
        result.append(line + "\n", style=Style(color=colors[color_idx]))
    
    return result


def create_styled_panel(
    content: RenderableType,
    title: str = "",
    subtitle: str = "",
    border_style: str = Colors.SOFT_PURPLE,
    padding: tuple[int, int] = (1, 2),
) -> Panel:
    """Create a consistently styled panel."""
    title_text = Text(title, style=STYLES["title"]) if title else None
    subtitle_text = Text(subtitle, style=STYLES["muted"]) if subtitle else None
    
    return Panel(
        content,
        title=title_text,
        subtitle=subtitle_text,
        border_style=border_style,
        padding=padding,
        box=ROUNDED,
    )


def create_header(text: str, icon: str = "✦") -> Text:
    """Create a styled header."""
    header = Text()
    header.append(f"{icon} ", style=Style(color=Colors.SAKURA))
    header.append(text, style=STYLES["title"])
    return header


def create_key_value(key: str, value: str, separator: str = ":") -> Text:
    """Create a styled key-value pair."""
    line = Text()
    line.append(f"  {key}", style=STYLES["key"])
    line.append(f"{separator} ", style=STYLES["muted"])
    line.append(str(value), style=STYLES["value"])
    return line


def create_success_message(text: str) -> Text:
    """Create a success message with icon."""
    msg = Text()
    msg.append("✓ ", style=STYLES["success"])
    msg.append(text, style=Style(color=Colors.SUCCESS))
    return msg


def create_error_message(text: str) -> Text:
    """Create an error message with icon."""
    msg = Text()
    msg.append("✗ ", style=STYLES["error"])
    msg.append(text, style=Style(color=Colors.ERROR))
    return msg


def create_warning_message(text: str) -> Text:
    """Create a warning message with icon."""
    msg = Text()
    msg.append("⚠ ", style=STYLES["warning"])
    msg.append(text, style=Style(color=Colors.WARNING))
    return msg


def create_info_message(text: str) -> Text:
    """Create an info message with icon."""
    msg = Text()
    msg.append("ℹ ", style=STYLES["info"])
    msg.append(text, style=Style(color=Colors.LAVENDER))
    return msg


def create_styled_table(
    title: str = "",
    show_header: bool = True,
    box_style: box.Box = ROUNDED,
) -> Table:
    """Create a consistently styled table."""
    return Table(
        title=title,
        title_style=STYLES["title"],
        show_header=show_header,
        header_style=Style(color=Colors.CHERRY_LIGHT, bold=True),
        border_style=Colors.SOFT_PURPLE,
        box=box_style,
        padding=(0, 1),
        row_styles=[Style(color=Colors.WHITE), Style(color=Colors.LAVENDER)],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Render functions
# ─────────────────────────────────────────────────────────────────────────────
def render_logo(console: Console, style: str | None = None) -> None:
    """Render the large ASCII logo with gradient."""
    console.print(gradient_text(LOGO_ART))


def render_welcome(console: Console, data: WelcomeData) -> None:
    """Render the welcome screen with dashboard layout."""
    
    # Title with version
    title = Text()
    title.append("✦ ", style=Style(color=Colors.SAKURA))
    title.append("Lovelace Code", style=Style(color=Colors.LIGHT_PURPLE, bold=True))
    title.append(f" v{data.version}", style=STYLES["muted"])

    # ─────────────────────────────────────────────────────────────────────────
    # Top section: Mascot on left + LOVELACE text on right
    # ─────────────────────────────────────────────────────────────────────────
    
    # Create mascot column with gradient
    mascot_lines = MASCOT_MINI.strip().splitlines()
    mascot_text = Text()
    mascot_colors = [Colors.DEEP_PURPLE, Colors.PURPLE, Colors.LIGHT_PURPLE, 
                     Colors.ORCHID, Colors.SAKURA]
    for i, line in enumerate(mascot_lines):
        color_idx = min(i * len(mascot_colors) // len(mascot_lines), len(mascot_colors) - 1)
        mascot_text.append(line + "\n", style=Style(color=mascot_colors[color_idx]))
    
    # Create LOVELACE text with gradient
    lovelace_lines = LOVELACE_TEXT.strip().splitlines()
    lovelace_text = Text()
    for i, line in enumerate(lovelace_lines):
        color_idx = min(i * len(LOGO_GRADIENT) // len(lovelace_lines), len(LOGO_GRADIENT) - 1)
        lovelace_text.append(line + "\n", style=Style(color=LOGO_GRADIENT[color_idx]))
    
    # Side by side layout: mascot | lovelace text
    header_grid = Table.grid(padding=(0, 2))
    header_grid.add_column(justify="center", vertical="middle")
    header_grid.add_column(justify="left")
    header_grid.add_row(mascot_text, lovelace_text)
    
    # Welcome message
    welcome_text = Text()
    welcome_text.append("\nWelcome back ", style=Style(color=Colors.WHITE))
    welcome_text.append(data.user_label, style=Style(color=Colors.SAKURA, bold=True))
    welcome_text.append("!", style=Style(color=Colors.WHITE))
    
    # Model & backend info
    model_info = Text()
    model_info.append("◈ ", style=Style(color=Colors.ORCHID))
    model_info.append(data.model_label, style=Style(color=Colors.LIGHT_PURPLE, bold=True))
    
    context_info = Text()
    context_info.append(f"Backend: ", style=STYLES["muted"])
    context_info.append(data.backend_label, style=Style(color=Colors.SAKURA))
    context_info.append("  •  ", style=STYLES["muted"])
    context_info.append(data.project_path, style=Style(color=Colors.LAVENDER))
    
    # Combine top section
    top = Table.grid(padding=(0, 1))
    top.add_column(justify="center")
    top.add_row(header_grid)
    top.add_row(welcome_text)
    top.add_row(Text(""))
    top.add_row(model_info)
    top.add_row(context_info)

    # ─────────────────────────────────────────────────────────────────────────
    # Bottom section: Activity + What's new (side by side)
    # ─────────────────────────────────────────────────────────────────────────
    
    # Recent activity table
    activity_tbl = Table(
        show_header=True,
        header_style=Style(color=Colors.CHERRY_LIGHT, bold=True),
        border_style=Colors.SOFT_PURPLE,
        box=SIMPLE,
        padding=(0, 2),
    )
    activity_tbl.add_column("Recent activity", style=Style(color=Colors.WHITE))
    
    if data.recent_activity:
        for row in data.recent_activity[-5:]:
            if "\t" in row:
                ts, msg = row.split("\t", 1)
                activity_tbl.add_row(f"[{Colors.DIM}]{ts}[/{Colors.DIM}]  {msg}")
            else:
                activity_tbl.add_row(row)
    else:
        activity_tbl.add_row(f"[{Colors.DIM}](no recent activity)[/{Colors.DIM}]")
    
    activity_tbl.add_row(f"[{Colors.DIM}]... /history for more[/{Colors.DIM}]")

    # What's new table
    whats_new = Table(
        show_header=True,
        header_style=Style(color=Colors.SAKURA, bold=True),
        border_style=Colors.SOFT_PURPLE,
        box=SIMPLE,
        padding=(0, 2),
    )
    whats_new.add_column("What's new", style=Style(color=Colors.WHITE))
    whats_new.add_row(f"[{Colors.ORCHID}]/plan[/{Colors.ORCHID}] for complex multi-step tasks")
    whats_new.add_row(f"[{Colors.ORCHID}]/tools[/{Colors.ORCHID}] to list available tools")
    whats_new.add_row(f"[{Colors.ORCHID}]/backend[/{Colors.ORCHID}] to switch local/api")
    whats_new.add_row(f"[{Colors.DIM}]... /help for more[/{Colors.DIM}]")

    bottom = Columns([activity_tbl, whats_new], equal=True, expand=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Combine into panel
    # ─────────────────────────────────────────────────────────────────────────
    body = Group(Align.center(top), Text(""), bottom)
    panel = Panel(
        body,
        title=title,
        border_style=Colors.SOFT_PURPLE,
        padding=(1, 2),
        box=ROUNDED,
    )

    console.print(panel)
    console.print()
    
    # Input hint
    hint = Text()
    hint.append("> ", style=Style(color=Colors.ORCHID, bold=True))
    hint.append("Type a message, or ", style=STYLES["muted"])
    hint.append("/help", style=Style(color=Colors.LIGHT_PURPLE, bold=True))
    hint.append(" for commands", style=STYLES["muted"])
    console.print(hint)


def render_login_success(console: Console) -> None:
    """Render the login success screen with large logo."""
    console.print()
    
    # Welcome box
    welcome_text = Text()
    welcome_text.append("✦ ", style=Style(color=Colors.SAKURA))
    welcome_text.append("Welcome to the Lovelace Code open-source preview!", style=Style(color=Colors.WHITE, bold=True))
    
    welcome_box = Panel(
        welcome_text,
        border_style=Colors.SOFT_PURPLE,
        padding=(0, 2),
        box=ROUNDED,
    )
    console.print(welcome_box)
    console.print()
    
    # Big logo with gradient
    render_logo(console)
    
    console.print()
    console.print(create_success_message("Login successful. Press Enter to continue"))


# ─────────────────────────────────────────────────────────────────────────────
# Chat Message Rendering (Claude Code-like UI)
# ─────────────────────────────────────────────────────────────────────────────
def render_user_message(console: Console, message: str) -> None:
    """Render a user message with visual indicator."""
    text = Text()
    text.append("> ", style=Style(color=Colors.LIGHT_PURPLE, bold=True))
    text.append(message, style=Style(color=Colors.WHITE))
    console.print(text)


def render_assistant_prefix(console: Console) -> Text:
    """Render the assistant message prefix (bullet point)."""
    text = Text()
    text.append("● ", style=Style(color=Colors.SAKURA, bold=True))
    return text


def render_assistant_message(console: Console, content: str) -> None:
    """Render a complete assistant message."""
    from rich.markdown import Markdown
    
    # Add prefix bullet
    text = Text()
    text.append("● ", style=Style(color=Colors.SAKURA, bold=True))
    console.print(text, end="")
    
    # Render content as markdown, indented
    console.print(Markdown(content))


def render_streaming_assistant(content: str) -> Text:
    """Return text for streaming assistant response with prefix."""
    text = Text()
    text.append("● ", style=Style(color=Colors.SAKURA, bold=True))
    text.append(content, style=Style(color=Colors.WHITE))
    return text


def render_thinking(console: Console) -> Text:
    """Return a 'thinking' indicator."""
    text = Text()
    text.append("● ", style=Style(color=Colors.ORCHID))
    text.append("thinking…", style=Style(color=Colors.SAKURA, italic=True))
    return text


def render_tool_call(console: Console, tool_name: str, args: dict) -> None:
    """Render a tool call indicator."""
    text = Text()
    text.append("◈ ", style=Style(color=Colors.ORCHID))
    text.append(tool_name, style=Style(color=Colors.LIGHT_PURPLE, bold=True))
    text.append("(", style=STYLES["muted"])
    text.append(", ".join(f"{k}={v!r}" for k, v in list(args.items())[:3]), style=Style(color=Colors.SAKURA))
    text.append(")", style=STYLES["muted"])
    console.print(text)


def render_tool_result(console: Console, tool_name: str, success: bool, output: str) -> None:
    """Render tool execution result."""
    status_color = Colors.SUCCESS if success else Colors.ERROR
    status_icon = "✓" if success else "✗"
    
    text = Text()
    text.append(f"  └─ {status_icon} ", style=Style(color=status_color))
    
    # Truncate output for display
    preview = output[:200].replace("\n", " ")
    if len(output) > 200:
        preview += "…"
    text.append(preview, style=STYLES["muted"])
    
    console.print(text)


def render_streaming_update(console: Console, content: str) -> Panel:
    """Return a panel for streaming content."""
    return Panel(
        Text(content),
        border_style=Colors.SOFT_PURPLE,
        padding=(0, 1),
        box=ROUNDED,
    )


# ─────────────────────────────────────────────────────────────────────────────
# UI Components for CLI commands
# ─────────────────────────────────────────────────────────────────────────────
def render_init_success(
    console: Console,
    ldir: str,
    backend: str,
    model: str,
) -> None:
    """Render initialization success message."""
    content = Table.grid(padding=(0, 1))
    content.add_column()
    
    content.add_row(create_success_message(f"Lovelace Code initialized in {ldir}"))
    content.add_row(Text(""))
    content.add_row(create_key_value("Backend", backend))
    content.add_row(create_key_value("Model", model))
    content.add_row(Text(""))
    
    run_text = Text()
    run_text.append("  Run ", style=STYLES["muted"])
    run_text.append("lovelace", style=Style(color=Colors.LIGHT_PURPLE, bold=True))
    run_text.append(" to start chatting!", style=STYLES["muted"])
    content.add_row(run_text)
    
    panel = create_styled_panel(
        content,
        title="✦ Project Initialized",
    )
    console.print()
    console.print(panel)


def render_doctor_report(
    console: Console,
    python_version: tuple[int, int, int],
    backends: list[tuple[str, bool, str | None]],
    api_keys: dict[str, bool],
    gpu_info: str | None,
    config_path: str,
) -> None:
    """Render the doctor report."""
    content = Table.grid(padding=(0, 1))
    content.add_column()
    
    # Python version
    py_ok = python_version >= (3, 10, 0)
    py_text = Text()
    py_text.append("  Python  ", style=Style(color=Colors.LAVENDER))
    py_text.append(f"{python_version[0]}.{python_version[1]}.{python_version[2]} ", style=Style(color=Colors.WHITE))
    if py_ok:
        py_text.append("✓", style=STYLES["success"])
    else:
        py_text.append("✗ (need 3.10+)", style=STYLES["error"])
    content.add_row(py_text)
    content.add_row(Text(""))
    
    # Backends section - grouped by type
    content.add_row(create_header("Cloud API Backends", "◈"))
    content.add_row(Text(""))
    
    cloud_backends = ["api", "openai", "anthropic", "gemini", "xai", "openrouter", "openai-compatible"]
    local_backends = ["local", "llama"]
    
    for name, available, error in backends:
        if name not in cloud_backends:
            continue
        line = Text()
        display_name = {
            "api": "huggingface",
            "openai-compatible": "openai-compat",
        }.get(name, name)
        line.append(f"  {display_name:16}", style=Style(color=Colors.LAVENDER))
        if error:
            line.append(f"✗ error", style=STYLES["error"])
        elif available:
            line.append("✓ available", style=STYLES["success"])
        else:
            line.append("○ missing SDK", style=STYLES["warning"])
        content.add_row(line)
    
    content.add_row(Text(""))
    content.add_row(create_header("Local Backends", "◈"))
    content.add_row(Text(""))
    
    for name, available, error in backends:
        if name not in local_backends:
            continue
        line = Text()
        line.append(f"  {name:16}", style=Style(color=Colors.LAVENDER))
        if error:
            line.append(f"✗ error", style=STYLES["error"])
        elif available:
            line.append("✓ available", style=STYLES["success"])
        else:
            line.append("○ not installed", style=STYLES["warning"])
        content.add_row(line)
    
    content.add_row(Text(""))
    
    # API Keys section
    content.add_row(create_header("API Keys", "◈"))
    content.add_row(Text(""))
    
    for provider, is_set in api_keys.items():
        key_text = Text()
        key_text.append(f"  {provider:16}", style=Style(color=Colors.LAVENDER))
        if is_set:
            key_text.append("✓ configured", style=STYLES["success"])
        else:
            key_text.append("○ not set", style=STYLES["warning"])
        content.add_row(key_text)
    
    content.add_row(Text(""))
    
    # Hardware
    content.add_row(create_header("Hardware", "◈"))
    content.add_row(Text(""))
    hw_text = Text()
    hw_text.append("  GPU  ", style=Style(color=Colors.LAVENDER))
    if gpu_info:
        hw_text.append(f"✓ {gpu_info}", style=STYLES["success"])
    else:
        hw_text.append("○ CPU only", style=STYLES["warning"])
    content.add_row(hw_text)
    
    content.add_row(Text(""))
    
    # Paths
    content.add_row(create_header("Paths", "◈"))
    content.add_row(Text(""))
    content.add_row(create_key_value("Global config", config_path))
    
    panel = create_styled_panel(content, title="✦ Lovelace Code Doctor")
    console.print()
    console.print(panel)


def render_config_panel(
    console: Console,
    config: dict[str, Any],
    config_file: str,
    title: str = "Configuration",
) -> None:
    """Render a configuration panel."""
    content = Table.grid(padding=(0, 1))
    content.add_column()
    
    path_text = Text()
    path_text.append("  File: ", style=STYLES["muted"])
    path_text.append(config_file, style=Style(color=Colors.LAVENDER))
    content.add_row(path_text)
    content.add_row(Text(""))
    
    for key, value in config.items():
        # Hide sensitive data
        if key == "hf_token" and value:
            value = value[:8] + "..." + value[-4:] if len(str(value)) > 12 else "***"
        content.add_row(create_key_value(key, str(value)))
    
    panel = create_styled_panel(content, title=f"✦ {title}")
    console.print()
    console.print(panel)


def render_stats_panel(
    console: Console,
    total_sessions: int,
    total_messages: int,
    first_run: str | None,
    last_run: str | None,
) -> None:
    """Render statistics panel."""
    content = Table.grid(padding=(0, 1))
    content.add_column()
    
    # Stats with icons
    sessions_text = Text()
    sessions_text.append("  ◈ ", style=Style(color=Colors.ORCHID))
    sessions_text.append("Total sessions  ", style=Style(color=Colors.LAVENDER))
    sessions_text.append(str(total_sessions), style=Style(color=Colors.SAKURA, bold=True))
    content.add_row(sessions_text)
    
    messages_text = Text()
    messages_text.append("  ◈ ", style=Style(color=Colors.ORCHID))
    messages_text.append("Total messages  ", style=Style(color=Colors.LAVENDER))
    messages_text.append(str(total_messages), style=Style(color=Colors.SAKURA, bold=True))
    content.add_row(messages_text)
    
    content.add_row(Text(""))
    
    if first_run:
        content.add_row(create_key_value("First run", first_run[:10]))
    if last_run:
        content.add_row(create_key_value("Last run", last_run[:10]))
    
    panel = create_styled_panel(content, title="✦ Usage Statistics")
    console.print()
    console.print(panel)


def render_help_panel(console: Console) -> None:
    """Render the help panel."""
    # Commands table
    cmd_table = Table(
        show_header=False,
        box=None,
        padding=(0, 2),
    )
    cmd_table.add_column(style=Style(color=Colors.ORCHID, bold=True))
    cmd_table.add_column(style=Style(color=Colors.LAVENDER))
    
    commands = [
        ("/help", "Show this help"),
        ("/model [name]", "View or change model"),
        ("/backend [name]", "View or change backend (api, local, llama)"),
        ("/config", "Show current configuration"),
        ("/tools", "List available tools"),
        ("/plan", "Toggle plan mode"),
        ("/auto", "Toggle auto-approve for tools"),
        ("/clear", "Clear conversation history"),
        ("/history", "Show recent activity"),
        ("/exit", "Exit Lovelace Code"),
    ]
    
    for cmd, desc in commands:
        cmd_table.add_row(cmd, desc)
    
    # Tips section
    tips_text = Text()
    tips_text.append("\n")
    tips_text.append("  Tips\n", style=STYLES["header"])
    tips_text.append(f"  • Use ", style=Style(color=Colors.LAVENDER))
    tips_text.append("--plan", style=Style(color=Colors.ORCHID, bold=True))
    tips_text.append(" flag when starting for complex tasks\n", style=Style(color=Colors.LAVENDER))
    tips_text.append(f"  • Set ", style=Style(color=Colors.LAVENDER))
    tips_text.append("HF_TOKEN", style=Style(color=Colors.ORCHID, bold=True))
    tips_text.append(" env var for API access\n", style=Style(color=Colors.LAVENDER))
    tips_text.append(f"  • Install ", style=Style(color=Colors.LAVENDER))
    tips_text.append("lovelace-code[local]", style=Style(color=Colors.ORCHID, bold=True))
    tips_text.append(" for local inference", style=Style(color=Colors.LAVENDER))
    
    content = Group(cmd_table, tips_text)
    panel = create_styled_panel(content, title="✦ Help")
    console.print(panel)


def render_tools_panel(console: Console, tools: list[Any]) -> None:
    """Render the available tools panel."""
    content = Table.grid(padding=(0, 1))
    content.add_column()
    
    for tool in tools:
        tool_text = Text()
        tool_text.append(f"  ◈ ", style=Style(color=Colors.ORCHID))
        tool_text.append(tool.name, style=Style(color=Colors.LIGHT_PURPLE, bold=True))
        content.add_row(tool_text)
        
        desc_text = Text()
        desc_text.append(f"    {tool.description}", style=Style(color=Colors.LAVENDER))
        content.add_row(desc_text)
        content.add_row(Text(""))
    
    panel = create_styled_panel(content, title="✦ Available Tools")
    console.print()
    console.print(panel)


def render_models_table(
    console: Console,
    models: list[Any],
    current_model: str,
    title: str = "Available Models",
) -> None:
    """Render the models table."""
    table = Table(
        show_header=True,
        header_style=Style(color=Colors.CHERRY_LIGHT, bold=True),
        border_style=Colors.SOFT_PURPLE,
        box=ROUNDED,
        padding=(0, 1),
    )
    
    table.add_column("Model ID", style=Style(color=Colors.WHITE), no_wrap=True)
    table.add_column("Provider", justify="center", style=Style(color=Colors.LAVENDER))
    table.add_column("Size", justify="right", style=Style(color=Colors.SAKURA))
    table.add_column("Context", justify="right", style=Style(color=Colors.LAVENDER))
    table.add_column("Description", style=Style(color=Colors.LIGHT_GRAY))
    
    # Provider icons/colors
    provider_styles = {
        "openai": f"[{Colors.SUCCESS}]OpenAI[/{Colors.SUCCESS}]",
        "anthropic": f"[{Colors.ORCHID}]Anthropic[/{Colors.ORCHID}]",
        "google": f"[{Colors.INFO}]Google[/{Colors.INFO}]",
        "xai": f"[{Colors.CHERRY}]xAI[/{Colors.CHERRY}]",
        "zhipu": f"[{Colors.LIGHT_PURPLE}]Zhipu[/{Colors.LIGHT_PURPLE}]",
        "huggingface": f"[{Colors.SAKURA}]HF[/{Colors.SAKURA}]",
        "meta": f"[{Colors.INFO}]Meta[/{Colors.INFO}]",
        "deepseek": f"[{Colors.ORCHID}]DeepSeek[/{Colors.ORCHID}]",
        "apertus": f"[{Colors.LAVENDER}]Apertus[/{Colors.LAVENDER}]",
    }
    
    for model in models:
        model_id = model.id
        if model_id == current_model:
            model_id = f"[{Colors.SUCCESS}]● {model_id}[/{Colors.SUCCESS}]"
        elif hasattr(model, 'recommended') and model.recommended:
            model_id = f"[{Colors.SAKURA}]★[/{Colors.SAKURA}] {model_id}"
        
        # Get provider style
        provider = getattr(model, 'provider', 'huggingface')
        provider_display = provider_styles.get(provider, provider)
        
        # Format context
        ctx = model.context_length
        if ctx >= 1000000:
            ctx_str = f"{ctx // 1000000}M"
        elif ctx >= 1000:
            ctx_str = f"{ctx // 1000}K"
        else:
            ctx_str = str(ctx)
        
        table.add_row(
            model_id,
            provider_display,
            model.size,
            ctx_str,
            model.description,
        )
    
    console.print()
    console.print(f"[{Colors.LIGHT_PURPLE}]✦ {title}[/{Colors.LIGHT_PURPLE}]")
    console.print()
    console.print(table)
    console.print()
    
    legend = Text()
    legend.append("  ● = current  ", style=STYLES["muted"])
    legend.append("★ = recommended", style=Style(color=Colors.SAKURA))
    console.print(legend)
    
    hints = Text()
    hints.append("\n  Use ", style=STYLES["muted"])
    hints.append("lovelace models list --set <model_id>", style=Style(color=Colors.ORCHID))
    hints.append(" to change model", style=STYLES["muted"])
    console.print(hints)


def render_model_info(
    console: Console,
    model_id: str,
    name: str | None = None,
    size: str | None = None,
    context_length: int | None = None,
    backend: str | None = None,
    description: str | None = None,
    vram_estimate: float | None = None,
    recommended: bool = False,
    provider: str | None = None,
) -> None:
    """Render detailed model information."""
    content = Table.grid(padding=(0, 1))
    content.add_column()
    
    if name:
        name_text = Text()
        name_text.append(f"  {name}", style=Style(color=Colors.LIGHT_PURPLE, bold=True))
        content.add_row(name_text)
        content.add_row(Text(""))
    
    content.add_row(create_key_value("ID", model_id))
    if provider:
        content.add_row(create_key_value("Provider", provider.title()))
    if size:
        content.add_row(create_key_value("Size", size))
    if context_length:
        # Format context nicely
        if context_length >= 1000000:
            ctx_str = f"{context_length:,} tokens ({context_length // 1000000}M)"
        else:
            ctx_str = f"{context_length:,} tokens"
        content.add_row(create_key_value("Context", ctx_str))
    if backend:
        content.add_row(create_key_value("Backend", backend))
    if description:
        content.add_row(create_key_value("Description", description))
    if vram_estimate and vram_estimate > 0:
        content.add_row(create_key_value("Est. VRAM (4-bit)", f"~{vram_estimate:.1f} GB"))
    
    if recommended:
        content.add_row(Text(""))
        rec_text = Text()
        rec_text.append("  ★ ", style=Style(color=Colors.SAKURA))
        rec_text.append("Recommended", style=Style(color=Colors.SAKURA, bold=True))
        content.add_row(rec_text)
    
    content.add_row(Text(""))
    
    # Show appropriate link based on provider
    if provider in ("openai", "anthropic", "google", "xai", "zhipu"):
        link_text = Text()
        link_text.append("  Provider API required", style=STYLES["muted"])
        content.add_row(link_text)
    else:
        link_text = Text()
        link_text.append("  HuggingFace: ", style=STYLES["muted"])
        link_text.append(f"https://huggingface.co/{model_id}", style=Style(color=Colors.LAVENDER, underline=True))
        content.add_row(link_text)
    
    panel = create_styled_panel(content, title="✦ Model Info")
    console.print()
    console.print(panel)


def render_model_current(
    console: Console,
    project_model: str | None,
    project_deep_model: str | None,
    project_config_path: str | None,
    global_default: str,
) -> None:
    """Render current model information."""
    content = Table.grid(padding=(0, 1))
    content.add_column()
    
    if project_model:
        content.add_row(create_header("Project Configuration", "◈"))
        content.add_row(Text(""))
        content.add_row(create_key_value("Model", project_model))
        if project_deep_model:
            content.add_row(create_key_value("Deep model", project_deep_model))
        if project_config_path:
            path_text = Text()
            path_text.append("    ", style="")
            path_text.append(project_config_path, style=STYLES["muted"])
            content.add_row(path_text)
        content.add_row(Text(""))
    else:
        notice = Text()
        notice.append("  No project initialized in current directory", style=Style(color=Colors.WARNING, italic=True))
        content.add_row(notice)
        content.add_row(Text(""))
    
    content.add_row(create_header("Global Default", "◈"))
    content.add_row(Text(""))
    content.add_row(create_key_value("Model", global_default))
    
    panel = create_styled_panel(content, title="✦ Current Model")
    console.print()
    console.print(panel)

