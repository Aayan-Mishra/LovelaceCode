"""Tool registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Tool, ToolResult

_TOOLS: dict[str, "Tool"] = {}


def register_tool(tool: "Tool | type") -> "Tool":
    """Register a tool instance or class.

    Supports usage as a decorator on a Tool class (in which case the class
    will be instantiated) or a direct instance registration.
    """
    # If decorator used on a class, instantiate it
    if isinstance(tool, type):
        instance = tool()
        _TOOLS[instance.name] = instance
        return instance

    _TOOLS[tool.name] = tool
    return tool


def get_tool(name: str) -> "Tool":
    """Get a tool by name."""
    _ensure_tools_loaded()
    if name not in _TOOLS:
        available = ", ".join(_TOOLS.keys()) or "(none)"
        raise ValueError(f"Unknown tool '{name}'. Available: {available}")
    return _TOOLS[name]


def list_tools() -> list["Tool"]:
    """List all registered tools."""
    _ensure_tools_loaded()
    return list(_TOOLS.values())


def execute_tool(name: str, **kwargs) -> "ToolResult":
    """Execute a tool by name."""
    tool = get_tool(name)
    return tool.execute(**kwargs)


def _ensure_tools_loaded():
    """Import tool modules to trigger registration."""
    if _TOOLS:
        return
    # pylint: disable=import-outside-toplevel,unused-import
    from . import (  # noqa: F401
        file_tools,
        shell_tools,
        git_tools,
        repo_tools,
        task_tools,
        code_tools,
        model_tools,
    )
