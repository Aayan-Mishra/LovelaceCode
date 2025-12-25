"""Shell command tools."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from .base import Tool, ToolResult
from .registry import register_tool


class RunCommandTool(Tool):
    name = "run_command"
    description = "Run a shell command and return its output"
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Command to execute"},
            "cwd": {"type": "string", "description": "Working directory (optional)"},
            "timeout": {"type": "integer", "description": "Timeout in seconds (default: 60)"},
        },
        "required": ["command"],
    }

    # Commands that are never allowed
    BLOCKED_PATTERNS = [
        "rm -rf /",
        "rm -rf ~",
        "mkfs",
        "dd if=",
        ":(){:|:&};:",  # Fork bomb
        "> /dev/sd",
        "chmod -R 777 /",
    ]

    def execute(
        self, command: str, cwd: str | None = None, timeout: int = 60
    ) -> ToolResult:
        # Safety check
        cmd_lower = command.lower()
        for blocked in self.BLOCKED_PATTERNS:
            if blocked in cmd_lower:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Blocked dangerous command pattern: {blocked}",
                )

        try:
            working_dir = Path(cwd).resolve() if cwd else Path.cwd()
            if not working_dir.exists():
                return ToolResult(
                    success=False, output="", error=f"Directory not found: {cwd}"
                )

            result = subprocess.run(
                command,
                shell=True,
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"

            # Truncate very long outputs
            if len(output) > 50000:
                output = output[:50000] + "\n... (truncated)"

            return ToolResult(
                success=result.returncode == 0,
                output=output,
                data={"returncode": result.returncode, "cwd": str(working_dir)},
                error=None if result.returncode == 0 else f"Exit code: {result.returncode}",
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output="",
                error=f"Command timed out after {timeout} seconds",
            )
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))


# Register
register_tool(RunCommandTool())
