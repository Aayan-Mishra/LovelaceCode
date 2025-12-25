"""Git tools."""

from __future__ import annotations

from pathlib import Path

from .base import Tool, ToolResult
from .registry import register_tool


def _get_repo(path: str | None):
    """Get a git.Repo object for the given path."""
    try:
        import git

        repo_path = Path(path).resolve() if path else Path.cwd()
        return git.Repo(repo_path, search_parent_directories=True)
    except ImportError:
        raise RuntimeError("GitPython not installed")
    except Exception as exc:
        raise RuntimeError(f"Not a git repository: {exc}")


class GitStatusTool(Tool):
    name = "git_status"
    description = "Get the current git status (changed files, branch, etc.)"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Repository path (optional)"},
        },
        "required": [],
    }

    def execute(self, path: str | None = None) -> ToolResult:
        try:
            repo = _get_repo(path)

            lines = [f"Branch: {repo.active_branch.name}"]

            # Staged changes
            staged = [item.a_path for item in repo.index.diff("HEAD")]
            if staged:
                lines.append(f"\nStaged ({len(staged)}):")
                for f in staged[:20]:
                    lines.append(f"  + {f}")

            # Unstaged changes
            unstaged = [item.a_path for item in repo.index.diff(None)]
            if unstaged:
                lines.append(f"\nModified ({len(unstaged)}):")
                for f in unstaged[:20]:
                    lines.append(f"  M {f}")

            # Untracked
            untracked = repo.untracked_files[:20]
            if untracked:
                lines.append(f"\nUntracked ({len(repo.untracked_files)}):")
                for f in untracked:
                    lines.append(f"  ? {f}")

            return ToolResult(
                success=True,
                output="\n".join(lines),
                data={
                    "branch": repo.active_branch.name,
                    "staged": len(staged),
                    "modified": len(unstaged),
                    "untracked": len(repo.untracked_files),
                },
            )
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))


class GitDiffTool(Tool):
    name = "git_diff"
    description = "Show git diff for a file or the entire repo"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Repository path (optional)"},
            "file": {"type": "string", "description": "Specific file to diff (optional)"},
            "staged": {"type": "boolean", "description": "Show staged changes (default: false)"},
        },
        "required": [],
    }

    def execute(
        self, path: str | None = None, file: str | None = None, staged: bool = False
    ) -> ToolResult:
        try:
            repo = _get_repo(path)

            if staged:
                diff = repo.git.diff("--cached", file) if file else repo.git.diff("--cached")
            else:
                diff = repo.git.diff(file) if file else repo.git.diff()

            if not diff:
                return ToolResult(success=True, output="(no changes)", data={})

            # Truncate if too long
            if len(diff) > 30000:
                diff = diff[:30000] + "\n... (truncated)"

            return ToolResult(success=True, output=diff, data={})
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))


class GitCommitTool(Tool):
    name = "git_commit"
    description = "Stage files and create a commit"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Repository path (optional)"},
            "message": {"type": "string", "description": "Commit message"},
            "files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Files to stage (or 'all' for everything)",
            },
        },
        "required": ["message"],
    }

    def execute(
        self, message: str, path: str | None = None, files: list[str] | None = None
    ) -> ToolResult:
        try:
            repo = _get_repo(path)

            # Stage files
            if files and files != ["all"]:
                repo.index.add(files)
            else:
                repo.git.add("-A")

            # Commit
            commit = repo.index.commit(message)

            return ToolResult(
                success=True,
                output=f"Committed: {commit.hexsha[:8]} {message}",
                data={"sha": commit.hexsha, "message": message},
            )
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))


class GitLogTool(Tool):
    name = "git_log"
    description = "Show recent commit history"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Repository path (optional)"},
            "count": {"type": "integer", "description": "Number of commits (default: 10)"},
        },
        "required": [],
    }

    def execute(self, path: str | None = None, count: int = 10) -> ToolResult:
        try:
            repo = _get_repo(path)

            lines = []
            for commit in repo.iter_commits(max_count=count):
                date = commit.committed_datetime.strftime("%Y-%m-%d %H:%M")
                lines.append(f"{commit.hexsha[:8]} {date} {commit.message.splitlines()[0][:60]}")

            return ToolResult(
                success=True,
                output="\n".join(lines) or "(no commits)",
                data={"count": len(lines)},
            )
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))


# Register all
register_tool(GitStatusTool())
register_tool(GitDiffTool())
register_tool(GitCommitTool())
register_tool(GitLogTool())
