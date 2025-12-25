"""Base tool interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """Result of a tool execution."""

    success: bool
    output: str
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class Tool(ABC):
    """Abstract tool interface."""

    name: str = "base"
    description: str = ""
    parameters: dict[str, Any] = {}

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with the given parameters."""
        ...

    def to_schema(self) -> dict[str, Any]:
        """Return a JSON schema for this tool (for prompting)."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
