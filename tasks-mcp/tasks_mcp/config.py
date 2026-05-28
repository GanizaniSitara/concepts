from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re

import yaml

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


STRICT_PREFIX_RE = re.compile(r"^[A-Z][A-Z0-9]+$")
STRICT_PREFIX_MIN_LEN = 3


def _allowed_prefixes_path() -> Path:
    override = os.environ.get("TASKS_MCP_ALLOWED_PREFIXES")
    if override:
        return Path(override).expanduser().resolve()
    return repo_root() / "config" / "allowed_prefixes.yaml"


def _coerce_prefix_set(raw) -> set[str]:
    if isinstance(raw, dict):
        return {str(k).upper() for k in raw.keys()}
    if isinstance(raw, list):
        return {str(p).upper() for p in raw}
    return set()


def load_allowed_prefixes() -> dict[str, set[str]] | None:
    """Load the prefix allowlist.

    Returns a dict with two keys:
      - "prefixes": set of 3+ letter codes, the primary allowlist
      - "two_letter_legacy": frozen set of grandfathered 2-letter codes
    Returns None if no file exists (lenient mode — used in tests via env override).
    """
    path = _allowed_prefixes_path()
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {
        "prefixes": _coerce_prefix_set(data.get("prefixes")),
        "two_letter_legacy": _coerce_prefix_set(data.get("two_letter_legacy")),
    }


def validate_prefix_for_creation(prefix: str) -> str:
    """Strict prefix validation for new task creation.

    Rules (when allowlist file is present):
      - Three or more letters: must appear in `prefixes:`.
      - Exactly two letters: must appear in the frozen `two_letter_legacy:`
        list. Two-letter codes not on that list are rejected outright.
      - One character or empty: rejected by format check.

    Returns the normalised prefix. Raises ValueError on any violation.
    """
    cleaned = normalize_prefix(prefix)
    if not STRICT_PREFIX_RE.fullmatch(cleaned):
        raise ValueError(
            f"Invalid prefix '{prefix}': must be uppercase letters/digits, "
            f"start with a letter, minimum 2 characters."
        )

    allowlist = load_allowed_prefixes()
    if allowlist is None:
        # Lenient mode: no config file (used by tests). Format check above is enough.
        return cleaned

    legacy_two = allowlist["two_letter_legacy"]
    primary = allowlist["prefixes"]

    if len(cleaned) < STRICT_PREFIX_MIN_LEN:
        if cleaned in legacy_two:
            return cleaned
        raise ValueError(
            f"Prefix '{cleaned}' is a two-letter code and is not on the frozen "
            f"two-letter allowlist at {_allowed_prefixes_path()}. New project "
            f"codes MUST be three letters or longer. Pick a 3+ letter prefix "
            f"and ask the user before adding it to the allowlist."
        )

    if cleaned in primary:
        return cleaned

    sample = ", ".join(sorted(primary)[:8])
    raise ValueError(
        f"Prefix '{cleaned}' is not in the allowlist at "
        f"{_allowed_prefixes_path()}. New project codes require explicit user "
        f"approval — ask the user FIRST, then the user adds the entry. Agents "
        f"must not edit the allowlist unilaterally. Existing prefixes include: "
        f"{sample}, ..."
    )


def canonical_ticket_id(prefix: str, number: int) -> str:
    normalized_prefix = normalize_prefix(prefix)
    width = max(3, len(str(number)))
    return f"{normalized_prefix}-{number:0{width}d}"
