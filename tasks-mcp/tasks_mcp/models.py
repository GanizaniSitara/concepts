from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Ticket:
    ticket_id: str
    title: str
    status: str
    path: Path
    stem: str
    body: str
    frontmatter: dict[str, Any]
    prefix: str | None = None
    project: str | None = None
    number: int | None = None
    slug: str = ""
    priority: str | None = None
    created: str | None = None
    updated: str | None = None
    tags: list[str] = field(default_factory=list)
    companion_dir: Path | None = None
    asset_paths: list[str] = field(default_factory=list)
    asset_blob: str = ""
    source_mtime: float = 0.0
    content_hash: str = ""


@dataclass(slots=True)
class MigrationAction:
    current_ticket_id: str
    current_path: Path
    current_status: str
    target_ticket_id: str
    target_path: Path
    target_status: str
    target_prefix: str
    target_number: int
    target_slug: str
    current_companion_dir: Path | None = None
    target_companion_dir: Path | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "current_task_id": self.current_ticket_id,
            "current_path": str(self.current_path),
            "current_status": self.current_status,
            "target_task_id": self.target_ticket_id,
            "target_path": str(self.target_path),
            "target_status": self.target_status,
            "target_prefix": self.target_prefix,
            "target_number": self.target_number,
            "target_slug": self.target_slug,
            "current_companion_dir": str(self.current_companion_dir) if self.current_companion_dir else None,
            "target_companion_dir": str(self.target_companion_dir) if self.target_companion_dir else None,
            "reason": self.reason,
        }
