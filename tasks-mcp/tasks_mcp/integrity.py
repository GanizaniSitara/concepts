"""Corpus-level integrity primitives: lint, pivot, repair.

The MCP is the source of truth for task state, so it also owns corpus integrity.
Consumers should never need to roll their own dedupe, pivot, or normalization.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
import re

from .config import STATUS_DIRECTORY_NAMES, normalize_status, status_dir
from .markdown_store import TaskRepository, UNSET
from .models import Ticket


_PRIORITY_CANONICAL_RE = re.compile(r"^P[0-5]$")

_PRIORITY_WORD_MAP: dict[str, str] = {
    "critical": "P1",
    "crit": "P1",
    "urgent": "P1",
    "highest": "P1",
    "high": "P1",
    "h": "P1",
    "medium": "P3",
    "med": "P3",
    "normal": "P3",
    "m": "P3",
    "low": "P4",
    "l": "P4",
    "lowest": "P5",
    "trivial": "P5",
    "nice-to-have": "P5",
}

STATUS_SORT_ORDER = {
    "in-progress": 0,
    "blocked": 1,
    "backlog": 2,
    "done": 3,
}

PIVOT_AXES = ("project", "prefix", "status", "priority")


def normalize_priority(value: object) -> tuple[str | None, bool]:
    """Return (normalized, changed). changed=False when value was already P0..P5 or empty."""
    if value is None:
        return None, False
    text = str(value).strip()
    if not text:
        return None, False
    upper = text.upper()
    if _PRIORITY_CANONICAL_RE.fullmatch(upper):
        return upper, upper != text
    # Integer form: "1" -> P1, "2" -> P2
    if text.isdigit():
        n = int(text)
        if 0 <= n <= 5:
            return f"P{n}", True
        return text, False
    lower = text.lower()
    mapped = _PRIORITY_WORD_MAP.get(lower)
    if mapped:
        return mapped, True
    return text, False


def is_canonical_priority(value: object) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    if not text:
        return True
    return bool(_PRIORITY_CANONICAL_RE.fullmatch(text.upper())) and text == text.upper()


def _raw_frontmatter_status(ticket: Ticket) -> str | None:
    raw = ticket.frontmatter.get("status")
    if raw is None:
        return None
    try:
        return normalize_status(str(raw))
    except ValueError:
        return None


def _folder_status(ticket: Ticket) -> str:
    return ticket.path.parent.name


def _ticket_key(ticket_id: str) -> str:
    return str(ticket_id).strip().upper()


def _canonical_copy(tickets: list[Ticket]) -> Ticket:
    """Pick the canonical copy among duplicates: prefer open statuses, then most
    recently modified, then most recently updated frontmatter date, then path."""
    def key(t: Ticket) -> tuple:
        return (
            STATUS_SORT_ORDER.get(t.status, 99),
            -(t.source_mtime or 0.0),
            -(int(t.updated.replace("-", "")) if t.updated and t.updated.replace("-", "").isdigit() else 0),
            str(t.path),
        )
    return sorted(tickets, key=key)[0]


class IntegrityService:
    def __init__(self, repository: TaskRepository, index):
        self.repository = repository
        self.index = index

    # ------------------------------------------------------------------ lint

    def lint(self) -> dict[str, object]:
        tickets = self.repository.scan_tickets()
        by_id: dict[str, list[Ticket]] = {}
        for t in tickets:
            by_id.setdefault(_ticket_key(t.ticket_id), []).append(t)

        duplicate_task_ids: list[dict[str, object]] = []
        split_brain_content: list[dict[str, object]] = []
        for key, group in by_id.items():
            if len(group) <= 1:
                continue
            duplicate_task_ids.append(
                {
                    "task_id": group[0].ticket_id,
                    "count": len(group),
                    "files": [str(t.path) for t in group],
                    "statuses": [t.status for t in group],
                }
            )
            hashes = {t.content_hash for t in group}
            priorities = {(t.priority or "") for t in group}
            if len(hashes) > 1 or len(priorities) > 1:
                conflicts = [
                    f"{t.status}→{t.priority or 'none'}"
                    for t in group
                ]
                split_brain_content.append(
                    {
                        "task_id": group[0].ticket_id,
                        "conflicts": conflicts,
                        "files": [str(t.path) for t in group],
                    }
                )

        status_mismatches: list[dict[str, object]] = []
        for t in tickets:
            raw = _raw_frontmatter_status(t)
            folder = _folder_status(t)
            if raw is not None and folder in STATUS_DIRECTORY_NAMES and raw != folder:
                status_mismatches.append(
                    {
                        "path": str(t.path),
                        "task_id": t.ticket_id,
                        "folder_status": folder,
                        "frontmatter_status": raw,
                    }
                )

        orphan_companion_dirs: list[dict[str, object]] = []
        for canonical in STATUS_DIRECTORY_NAMES:
            folder = status_dir(self.repository.tasks_root, canonical)
            if not folder.exists():
                continue
            for child in sorted(folder.iterdir()):
                if not child.is_dir():
                    continue
                expected_stub = child.with_suffix(".md")
                if not expected_stub.exists():
                    orphan_companion_dirs.append(
                        {
                            "dir": str(child),
                            "expected_stub": str(expected_stub),
                        }
                    )

        priority_non_normalized: list[dict[str, object]] = []
        for t in tickets:
            if t.priority is None:
                continue
            if not is_canonical_priority(t.priority):
                priority_non_normalized.append(
                    {
                        "task_id": t.ticket_id,
                        "path": str(t.path),
                        "priority": t.priority,
                    }
                )

        missing_prefix: list[dict[str, object]] = []
        for t in tickets:
            if t.prefix is None:
                missing_prefix.append(
                    {
                        "task_id": t.ticket_id,
                        "path": str(t.path),
                    }
                )

        whoosh_drift = self._whoosh_drift(tickets)

        counts = {
            "files_on_disk": len(tickets),
            "unique_task_ids": len(by_id),
            "delta": len(tickets) - len(by_id),
        }

        integrity = {
            "duplicate_task_ids": duplicate_task_ids,
            "status_mismatches": status_mismatches,
            "orphan_companion_dirs": orphan_companion_dirs,
            "split_brain_content": split_brain_content,
            "priority_non_normalized": {
                "count": len(priority_non_normalized),
                "samples": priority_non_normalized[:25],
            },
            "missing_prefix": missing_prefix,
            "whoosh_drift": whoosh_drift,
        }

        return {
            "integrity": integrity,
            "counts": counts,
            "warnings": _summarize_warnings(integrity, counts),
        }

    def _whoosh_drift(self, tickets: list[Ticket]) -> dict[str, object]:
        live_paths = {str(t.path) for t in tickets}
        live_ids = {t.ticket_id for t in tickets}
        stale_paths: list[str] = []
        stale_ids: list[str] = []
        try:
            with self.index.ix.searcher() as searcher:
                for doc in searcher.all_stored_fields():
                    doc_path = doc.get("path")
                    doc_id = doc.get("ticket_id")
                    if doc_path and doc_path not in live_paths:
                        stale_paths.append(doc_path)
                    if doc_id and doc_id not in live_ids:
                        stale_ids.append(doc_id)
            doc_count = 0
            with self.index.ix.searcher() as searcher:
                doc_count = searcher.doc_count_all()
        except Exception as exc:  # pragma: no cover - defensive
            return {"error": str(exc), "stale_entries": 0}
        return {
            "stale_entries": len(stale_paths),
            "document_count": doc_count,
            "stale_paths": stale_paths[:25],
            "stale_ticket_ids": sorted(set(stale_ids))[:25],
        }

    # ------------------------------------------------------------------ pivot

    def pivot(
        self,
        *,
        rows: str = "project",
        cols: str = "status",
        status: list[str] | None = None,
        priorities: list[str] | None = None,
        projects: list[str] | None = None,
        dedupe: str = "task_id",
        normalize_priority_enabled: bool = True,
    ) -> dict[str, object]:
        if rows not in PIVOT_AXES:
            raise ValueError(f"rows must be one of {PIVOT_AXES}, got {rows!r}")
        if cols not in PIVOT_AXES:
            raise ValueError(f"cols must be one of {PIVOT_AXES}, got {cols!r}")
        if rows == cols:
            raise ValueError("rows and cols must differ")
        if dedupe not in ("task_id", "none"):
            raise ValueError("dedupe must be 'task_id' or 'none'")

        tickets = self.repository.scan_tickets()

        status_filter: set[str] | None = None
        if status:
            status_filter = {normalize_status(s) for s in status}
        priority_filter: set[str] | None = set(priorities) if priorities else None
        project_filter: set[str] | None = {p.upper() for p in projects} if projects else None

        if dedupe == "task_id":
            by_id: dict[str, list[Ticket]] = {}
            for t in tickets:
                by_id.setdefault(_ticket_key(t.ticket_id), []).append(t)
            tickets = [_canonical_copy(group) for group in by_id.values()]

        def axis_value(ticket: Ticket, axis: str) -> str:
            if axis == "project":
                return ticket.project or ticket.prefix or "(none)"
            if axis == "prefix":
                return ticket.prefix or "(none)"
            if axis == "status":
                return ticket.status
            if axis == "priority":
                raw = ticket.priority
                if normalize_priority_enabled:
                    normalized, _ = normalize_priority(raw)
                    return normalized or "(none)"
                return raw or "(none)"
            return "(none)"  # pragma: no cover

        filtered: list[Ticket] = []
        for t in tickets:
            if status_filter and t.status not in status_filter:
                continue
            if project_filter and (t.project or t.prefix or "").upper() not in project_filter:
                continue
            if priority_filter:
                raw = t.priority
                if normalize_priority_enabled:
                    normalized, _ = normalize_priority(raw)
                    effective = normalized or "(none)"
                else:
                    effective = raw or "(none)"
                if effective not in priority_filter:
                    continue
            filtered.append(t)

        counts: dict[tuple[str, str], int] = {}
        row_totals: dict[str, int] = {}
        col_totals: dict[str, int] = {}
        for t in filtered:
            r = axis_value(t, rows)
            c = axis_value(t, cols)
            counts[(r, c)] = counts.get((r, c), 0) + 1
            row_totals[r] = row_totals.get(r, 0) + 1
            col_totals[c] = col_totals.get(c, 0) + 1

        row_labels = sorted(row_totals.keys())
        col_labels = sorted(col_totals.keys())
        cells = [[counts.get((r, c), 0) for c in col_labels] for r in row_labels]

        lint_report = self.lint()
        integrity_warnings = lint_report["warnings"]

        return {
            "rows_axis": rows,
            "cols_axis": cols,
            "rows": row_labels,
            "cols": col_labels,
            "cells": cells,
            "row_totals": [row_totals[r] for r in row_labels],
            "col_totals": [col_totals[c] for c in col_labels],
            "totals": {
                "total": sum(row_totals.values()),
                "rows": row_totals,
                "cols": col_totals,
            },
            "filter": {
                "status": sorted(status_filter) if status_filter else None,
                "priorities": sorted(priority_filter) if priority_filter else None,
                "projects": sorted(project_filter) if project_filter else None,
                "dedupe": dedupe,
                "normalize_priority": normalize_priority_enabled,
            },
            "integrity_warnings": integrity_warnings,
        }

    # ----------------------------------------------------------------- repair

    def repair(
        self,
        *,
        fixes: list[str],
        dry_run: bool = True,
    ) -> dict[str, object]:
        allowed = {
            "normalize_priority",
            "resolve_status_mismatch",
            "flag_split_brain",
            "rebuild_whoosh",
        }
        unknown = [f for f in fixes if f not in allowed]
        if unknown:
            raise ValueError(
                f"Unknown fix(es) {unknown!r}; expected subset of {sorted(allowed)}"
            )

        tickets = self.repository.scan_tickets()
        by_id: dict[str, list[Ticket]] = {}
        for t in tickets:
            by_id.setdefault(_ticket_key(t.ticket_id), []).append(t)

        plan: list[dict[str, object]] = []
        unresolved: list[dict[str, object]] = []

        if "normalize_priority" in fixes:
            for t in tickets:
                if t.priority is None:
                    continue
                if is_canonical_priority(t.priority):
                    continue
                normalized, changed = normalize_priority(t.priority)
                if not changed:
                    unresolved.append(
                        {
                            "fix": "normalize_priority",
                            "task_id": t.ticket_id,
                            "path": str(t.path),
                            "priority": t.priority,
                            "reason": "no_mapping_rule",
                        }
                    )
                    continue
                plan.append(
                    {
                        "fix": "normalize_priority",
                        "task_id": t.ticket_id,
                        "path": str(t.path),
                        "before": {"priority": t.priority},
                        "after": {"priority": normalized},
                    }
                )

        if "resolve_status_mismatch" in fixes:
            for t in tickets:
                raw = _raw_frontmatter_status(t)
                folder = _folder_status(t)
                if raw is None or folder not in STATUS_DIRECTORY_NAMES:
                    continue
                if raw == folder:
                    continue
                # Only auto-resolve for tickets that are unique: duplicate IDs
                # are split-brain territory.
                if len(by_id.get(_ticket_key(t.ticket_id), [])) > 1:
                    unresolved.append(
                        {
                            "fix": "resolve_status_mismatch",
                            "task_id": t.ticket_id,
                            "path": str(t.path),
                            "folder_status": folder,
                            "frontmatter_status": raw,
                            "reason": "duplicate_task_id",
                        }
                    )
                    continue
                plan.append(
                    {
                        "fix": "resolve_status_mismatch",
                        "task_id": t.ticket_id,
                        "path": str(t.path),
                        "before": {"folder": folder},
                        "after": {"folder": raw},
                    }
                )

        if "flag_split_brain" in fixes:
            for key, group in by_id.items():
                if len(group) <= 1:
                    continue
                hashes = {t.content_hash for t in group}
                priorities = {(t.priority or "") for t in group}
                if len(hashes) <= 1 and len(priorities) <= 1:
                    continue
                for t in group:
                    if "SPLIT-BRAIN" in t.tags:
                        continue
                    plan.append(
                        {
                            "fix": "flag_split_brain",
                            "task_id": t.ticket_id,
                            "path": str(t.path),
                            "before": {"tags": list(t.tags)},
                            "after": {"tags": list(t.tags) + ["SPLIT-BRAIN"]},
                        }
                    )

        if "rebuild_whoosh" in fixes:
            plan.append(
                {
                    "fix": "rebuild_whoosh",
                    "task_id": None,
                    "path": None,
                    "before": None,
                    "after": None,
                }
            )

        if dry_run:
            return {
                "applied": False,
                "dry_run": True,
                "plan": plan,
                "unresolved": unresolved,
            }

        applied: list[dict[str, object]] = []
        for step in plan:
            fix = step["fix"]
            if fix == "normalize_priority":
                self.repository.update_ticket(
                    step["task_id"],
                    priority=step["after"]["priority"],
                )
                applied.append(step)
            elif fix == "resolve_status_mismatch":
                self.repository.update_ticket(
                    step["task_id"],
                    status=step["after"]["folder"],
                )
                applied.append(step)
            elif fix == "flag_split_brain":
                # Find the specific duplicate by path and tag it.
                target_path = Path(step["path"])
                ticket = self.repository.parse_ticket(target_path, target_path.parent.name)
                if ticket is None:
                    unresolved.append(
                        {
                            "fix": fix,
                            "task_id": step["task_id"],
                            "path": step["path"],
                            "reason": "unparseable_ticket",
                        }
                    )
                    continue
                new_tags = list(ticket.tags) + ["SPLIT-BRAIN"]
                ticket.tags = new_tags
                ticket.frontmatter = dict(ticket.frontmatter)
                self.repository._write_ticket(ticket)  # direct write: preserves path
                applied.append(step)
            elif fix == "rebuild_whoosh":
                result = self.index.rebuild()
                applied.append({**step, "result": result})

        return {
            "applied": True,
            "dry_run": False,
            "plan": plan,
            "applied_steps": applied,
            "unresolved": unresolved,
        }


def _summarize_warnings(integrity: dict[str, object], counts: dict[str, int]) -> list[str]:
    warnings: list[str] = []
    if integrity["duplicate_task_ids"]:
        warnings.append(
            f"duplicate_task_ids: {len(integrity['duplicate_task_ids'])} ids have >1 file on disk"
        )
    if integrity["status_mismatches"]:
        warnings.append(
            f"status_mismatches: {len(integrity['status_mismatches'])} files filed under a folder that disagrees with their frontmatter"
        )
    if integrity["orphan_companion_dirs"]:
        warnings.append(
            f"orphan_companion_dirs: {len(integrity['orphan_companion_dirs'])} companion folders without a stub"
        )
    if integrity["split_brain_content"]:
        warnings.append(
            f"split_brain_content: {len(integrity['split_brain_content'])} duplicate ids with diverging content"
        )
    p_count = integrity["priority_non_normalized"]["count"]
    if p_count:
        warnings.append(f"priority_non_normalized: {p_count} tickets with non-canonical priority")
    if integrity["missing_prefix"]:
        warnings.append(f"missing_prefix: {len(integrity['missing_prefix'])} tickets without a project prefix")
    drift = integrity["whoosh_drift"].get("stale_entries", 0)
    if drift:
        warnings.append(f"whoosh_drift: {drift} stale index entries")
    if counts["delta"]:
        warnings.append(
            f"file/id delta: {counts['files_on_disk']} files vs {counts['unique_task_ids']} unique ids"
        )
    return warnings
