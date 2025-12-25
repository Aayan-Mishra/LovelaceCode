"""File manipulation tools."""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path

from .base import Tool, ToolResult
from .registry import register_tool


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read the contents of a file"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to read"},
            "start_line": {"type": "integer", "description": "Start line (1-indexed, optional)"},
            "end_line": {"type": "integer", "description": "End line (1-indexed, optional)"},
        },
        "required": ["path"],
    }

    def execute(self, path: str, start_line: int | None = None, end_line: int | None = None) -> ToolResult:
        try:
            file_path = Path(path).resolve()
            if not file_path.exists():
                return ToolResult(success=False, output="", error=f"File not found: {path}")
            if not file_path.is_file():
                return ToolResult(success=False, output="", error=f"Not a file: {path}")

            content = file_path.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines(keepends=True)

            if start_line is not None or end_line is not None:
                start = (start_line or 1) - 1
                end = end_line or len(lines)
                lines = lines[start:end]
                content = "".join(lines)

            return ToolResult(
                success=True,
                output=content,
                data={"path": str(file_path), "lines": len(lines)},
            )
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))


class WriteFileTool(Tool):
    name = "write_file"
    description = "Write content to a file (creates directories if needed)"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to write"},
            "content": {"type": "string", "description": "Content to write"},
        },
        "required": ["path", "content"],
    }

    def execute(self, path: str, content: str) -> ToolResult:
        try:
            file_path = Path(path).resolve()
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            return ToolResult(
                success=True,
                output=f"Wrote {len(content)} bytes to {file_path}",
                data={"path": str(file_path), "bytes": len(content)},
            )
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))


class EditFileTool(Tool):
    name = "edit_file"
    description = "Replace a specific string in a file with new content"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to edit"},
            "old_string": {"type": "string", "description": "Exact string to replace"},
            "new_string": {"type": "string", "description": "Replacement string"},
        },
        "required": ["path", "old_string", "new_string"],
    }

    def execute(self, path: str, old_string: str, new_string: str) -> ToolResult:
        try:
            file_path = Path(path).resolve()
            if not file_path.exists():
                return ToolResult(success=False, output="", error=f"File not found: {path}")

            content = file_path.read_text(encoding="utf-8")
            if old_string not in content:
                return ToolResult(
                    success=False,
                    output="",
                    error="String to replace not found in file",
                )

            count = content.count(old_string)
            new_content = content.replace(old_string, new_string, 1)
            file_path.write_text(new_content, encoding="utf-8")

            return ToolResult(
                success=True,
                output=f"Replaced 1 of {count} occurrences in {file_path}",
                data={"path": str(file_path), "occurrences": count},
            )
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))


class ListDirectoryTool(Tool):
    name = "list_directory"
    description = "List files and directories in a path"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory path to list"},
            "recursive": {"type": "boolean", "description": "List recursively (default: false)"},
            "max_depth": {"type": "integer", "description": "Max recursion depth (default: 3)"},
        },
        "required": ["path"],
    }

    def execute(
        self, path: str, recursive: bool = False, max_depth: int = 3
    ) -> ToolResult:
        try:
            dir_path = Path(path).resolve()
            if not dir_path.exists():
                return ToolResult(success=False, output="", error=f"Path not found: {path}")
            if not dir_path.is_dir():
                return ToolResult(success=False, output="", error=f"Not a directory: {path}")

            entries: list[str] = []

            def walk(p: Path, depth: int):
                if depth > max_depth:
                    return
                try:
                    for item in sorted(p.iterdir()):
                        # Skip hidden and common ignore patterns
                        if item.name.startswith(".") or item.name in {
                            "node_modules",
                            "__pycache__",
                            "venv",
                            ".venv",
                            "dist",
                            "build",
                        }:
                            continue
                        rel = item.relative_to(dir_path)
                        suffix = "/" if item.is_dir() else ""
                        entries.append(f"{rel}{suffix}")
                        if recursive and item.is_dir():
                            walk(item, depth + 1)
                except PermissionError:
                    pass

            walk(dir_path, 0)
            output = "\n".join(entries[:500])  # Cap output
            if len(entries) > 500:
                output += f"\n... and {len(entries) - 500} more"

            return ToolResult(
                success=True,
                output=output,
                data={"path": str(dir_path), "count": len(entries)},
            )
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))


class SearchFilesTool(Tool):
    name = "search_files"
    description = "Search for files by name pattern or content"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory to search in"},
            "pattern": {"type": "string", "description": "Glob pattern for filenames (e.g., '*.py')"},
            "content": {"type": "string", "description": "Search for this text in file contents"},
        },
        "required": ["path"],
    }

    def execute(
        self, path: str, pattern: str = "*", content: str | None = None
    ) -> ToolResult:
        try:
            dir_path = Path(path).resolve()
            if not dir_path.exists():
                return ToolResult(success=False, output="", error=f"Path not found: {path}")

            matches: list[str] = []
            for root, dirs, files in os.walk(dir_path):
                # Skip common directories
                dirs[:] = [
                    d for d in dirs if not d.startswith(".") and d not in {
                        "node_modules", "__pycache__", "venv", ".venv", "dist", "build"
                    }
                ]

                for filename in files:
                    if not fnmatch.fnmatch(filename, pattern):
                        continue
                    file_path = Path(root) / filename

                    if content:
                        try:
                            text = file_path.read_text(encoding="utf-8", errors="ignore")
                            if content.lower() in text.lower():
                                rel = file_path.relative_to(dir_path)
                                # Find line numbers
                                for i, line in enumerate(text.splitlines(), 1):
                                    if content.lower() in line.lower():
                                        matches.append(f"{rel}:{i}: {line.strip()[:80]}")
                                        if len(matches) >= 100:
                                            break
                        except Exception:
                            pass
                    else:
                        rel = file_path.relative_to(dir_path)
                        matches.append(str(rel))

                    if len(matches) >= 100:
                        break
                if len(matches) >= 100:
                    break

            output = "\n".join(matches)
            if len(matches) >= 100:
                output += "\n... (truncated at 100 results)"

            return ToolResult(
                success=True,
                output=output or "(no matches)",
                data={"path": str(dir_path), "count": len(matches)},
            )
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))


# Register all tools
register_tool(ReadFileTool())
register_tool(WriteFileTool())
register_tool(EditFileTool())
register_tool(ListDirectoryTool())
register_tool(SearchFilesTool())
