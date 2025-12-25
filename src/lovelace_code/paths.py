from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Best-effort project root resolution.

    We treat the current working directory as the project root. If you want a stricter
    heuristic (git root, etc.), we can extend this later.
    """

    return (start or Path.cwd()).resolve()


def lovelace_dir(project_root: Path) -> Path:
    return project_root / ".lovelace"


def config_path(project_root: Path) -> Path:
    return lovelace_dir(project_root) / "config.json"


def activity_log_path(project_root: Path) -> Path:
    return lovelace_dir(project_root) / "activity.log"


def memory_path(project_root: Path) -> Path:
    return lovelace_dir(project_root) / "memory.md"
