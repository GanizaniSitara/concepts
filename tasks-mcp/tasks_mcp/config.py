from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8876
DEFAULT_TRANSPORT = "streamable-http"
DEFAULT_PREFIX = "TASK"

STATUS_DIRECTORY_NAMES = {
    "backlog": "backlog",
    "in-progress": "in-progress",
    "blocked": "blocked",
    "done": "done",
}

STATUS_ALIASES = {
    "todo": "backlog",
    "queued": "backlog",
    "backlog": "backlog",
    "in_progress": "in-progress",
    "in-progress": "in-progress",
    "progress": "in-progress",
    "working": "in-progress",
    "blocked": "blocked",
    "block": "blocked",
    "stalled": "blocked",
    "done": "done",
    "complete": "done",
    "completed": "done",
    "closed": "done",
}

READABLE_ASSET_SUFFIXES = {
    ".css",
    ".csv",
    ".html",
    ".js",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".sql",
    ".toml",
    ".ts",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}

REFERENCE_FIELDS = ("parent", "depends_on", "blocked_by")


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_tasks_root() -> Path:
    return Path.home() / "tasks"


def default_index_dir(base_dir: Path | None = None) -> Path:
    root = base_dir if base_dir is not None else repo_root()
    return root / ".data" / "whoosh"


def default_log_dir(base_dir: Path | None = None) -> Path:
    root = base_dir if base_dir is not None else repo_root()
    return root / ".data" / "logs"


@dataclass(frozen=True)
class Settings:
    tasks_root: Path
    index_dir: Path
    host: str
    port: int
    transport: str
    default_prefix: str

    @classmethod
    def from_env(cls) -> "Settings":
        tasks_root = Path(
            os.environ.get("TASKS_ROOT", str(default_tasks_root()))
        ).expanduser().resolve()
        index_dir = Path(
            os.environ.get("TASKS_INDEX_DIR", str(default_index_dir()))
        ).expanduser().resolve()
        return cls(
            tasks_root=tasks_root,
            index_dir=index_dir,
            host=os.environ.get("TASKS_MCP_HOST", DEFAULT_HOST),
            port=int(os.environ.get("TASKS_MCP_PORT", str(DEFAULT_PORT))),
            transport=os.environ.get("TASKS_MCP_TRANSPORT", DEFAULT_TRANSPORT),
            default_prefix=os.environ.get("TASKS_MCP_DEFAULT_PREFIX", DEFAULT_PREFIX).upper(),
        )


def normalize_status(value: str) -> str:
    key = value.strip().lower()
    if key not in STATUS_ALIASES:
        allowed = ", ".join(sorted(STATUS_DIRECTORY_NAMES))
        raise ValueError(f"Unknown status '{value}'. Expected one of: {allowed}")
    return STATUS_ALIASES[key]


def status_dir(tasks_root: Path, status: str) -> Path:
    normalized = normalize_status(status)
    return tasks_root / STATUS_DIRECTORY_NAMES[normalized]


def normalize_prefix(value: str) -> str:
    cleaned = "".join(ch for ch in str(value).upper() if ch.isalnum())
    if not cleaned:
        raise ValueError("Prefix must contain at least one letter or digit")
    return cleaned


def canonical_ticket_id(prefix: str, number: int) -> str:
    normalized_prefix = normalize_prefix(prefix)
    width = max(3, len(str(number)))
    return f"{normalized_prefix}-{number:0{width}d}"
