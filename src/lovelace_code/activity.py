from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def append_activity(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    log_path.open("a", encoding="utf-8").write(f"{ts}\t{message}\n")


def read_recent_activity(log_path: Path, limit: int = 6) -> list[str]:
    if not log_path.exists():
        return []
    lines = log_path.read_text(encoding="utf-8").splitlines()
    return lines[-limit:]
