from __future__ import annotations

from pathlib import Path
import shutil
import tempfile

import pytest

from tasks_mcp.config import Settings
from tasks_mcp.indexer import TicketIndex
from tasks_mcp.integrity import (
    IntegrityService,
    is_canonical_priority,
    normalize_priority,
)
from tasks_mcp.markdown_store import TaskRepository


def make_temp_root() -> Path:
    return Path(tempfile.mkdtemp(prefix="tasks-mcp-integrity-"))


def make_settings(tmp_root: Path) -> Settings:
    return Settings(
        tasks_root=tmp_root / "tasks",
        index_dir=tmp_root / "whoosh-index",
        host="127.0.0.1",
        port=8876,
        transport="streamable-http",
        default_prefix="TASK",
    )


def make_service(tmp_root: Path) -> IntegrityService:
    settings = make_settings(tmp_root)
    repo = TaskRepository(settings)
    repo.ensure_structure()
    for name in ("backlog", "in-progress", "blocked", "done"):
        (settings.tasks_root / name).mkdir(parents=True, exist_ok=True)
    index = TicketIndex(settings.index_dir, repo)
    return IntegrityService(repo, index)


def write_ticket_file(
    root: Path,
    folder: str,
    stem: str,
    *,
    task_id: str,
    title: str,
    status: str | None,
    priority: str | None = None,
    project: str | None = None,
    body: str = "",
    tags: list[str] | None = None,
) -> Path:
    target = root / folder / f"{stem}.md"
    lines = ["---", f"task: {task_id}", f"title: {title}"]
    if status is not None:
        lines.append(f"status: {status}")
    if priority is not None:
        lines.append(f"priority: {priority}")
    if project is not None:
        lines.append(f"project: {project}")
    if tags:
        lines.append(f"tags: [{', '.join(tags)}]")
    lines.append("---")
    lines.append("")
    lines.append(body or "body.")
    lines.append("")
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


# ------------------------------------------------------------ priority helpers


def test_normalize_priority_maps_words_and_digits() -> None:
    assert normalize_priority("high") == ("P1", True)
    assert normalize_priority("CRITICAL") == ("P1", True)
    assert normalize_priority("medium") == ("P3", True)
    assert normalize_priority("low") == ("P4", True)
    assert normalize_priority("1") == ("P1", True)
    assert normalize_priority("p2") == ("P2", True)
    assert normalize_priority("P1") == ("P1", False)
    assert normalize_priority(None) == (None, False)
    assert normalize_priority("") == (None, False)
    # Unmappable surfaces unchanged
    assert normalize_priority("urgent asap please") == ("urgent asap please", False)


def test_is_canonical_priority() -> None:
    assert is_canonical_priority("P1")
    assert is_canonical_priority("P5")
    assert is_canonical_priority(None)
    assert is_canonical_priority("")
    assert not is_canonical_priority("p1")
    assert not is_canonical_priority("high")
    assert not is_canonical_priority("1")
    assert not is_canonical_priority("P6")


# --------------------------------------------------------------------- lint


def test_lint_flags_duplicates_and_status_mismatches() -> None:
    tmp_root = make_temp_root()
    try:
        service = make_service(tmp_root)
        root = service.repository.tasks_root

        # Duplicate AAA-22 in backlog + done
        write_ticket_file(
            root, "backlog", "AAA-022-example",
            task_id="AAA-022", title="Example", status="backlog", project="AAA"
        )
        write_ticket_file(
            root, "done", "AAA-022-example",
            task_id="AAA-022", title="Example done", status="done", project="AAA"
        )
        # Status mismatch: file in in-progress/ but frontmatter says backlog
        write_ticket_file(
            root, "in-progress", "TECH-010-drift",
            task_id="TECH-010", title="Drift", status="backlog", project="TECH"
        )
        # Orphan companion dir
        (root / "backlog" / "004-orphan-companion").mkdir()
        # Non-canonical priority
        write_ticket_file(
            root, "backlog", "TASK-001-mixed-prio",
            task_id="TASK-001", title="Mixed prio", status="backlog",
            priority="high", project="TASK"
        )
        # Missing prefix: pure-numeric task id
        write_ticket_file(
            root, "backlog", "002-legacy",
            task_id="002", title="Legacy numeric", status="backlog"
        )

        report = service.lint()
        integrity = report["integrity"]

        assert any(d["task_id"] == "AAA-022" and d["count"] == 2 for d in integrity["duplicate_task_ids"])

        assert any(
            m["task_id"] == "TECH-010"
            and m["folder_status"] == "in-progress"
            and m["frontmatter_status"] == "backlog"
            for m in integrity["status_mismatches"]
        )

        assert any(
            "004-orphan-companion" in o["dir"]
            for o in integrity["orphan_companion_dirs"]
        )

        assert integrity["priority_non_normalized"]["count"] >= 1
        assert any(s["priority"] == "high" for s in integrity["priority_non_normalized"]["samples"])

        assert any(m["task_id"] == "002" for m in integrity["missing_prefix"])

        counts = report["counts"]
        assert counts["files_on_disk"] == 5
        assert counts["unique_task_ids"] == 4  # AAA-022 collapses
        assert counts["delta"] == 1
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_lint_detects_split_brain_when_priorities_diverge() -> None:
    tmp_root = make_temp_root()
    try:
        service = make_service(tmp_root)
        root = service.repository.tasks_root
        write_ticket_file(
            root, "backlog", "BBB-008-a",
            task_id="BBB-008", title="Test work", status="backlog",
            priority="P4", project="BBB", body="Backlog body."
        )
        write_ticket_file(
            root, "in-progress", "BBB-008-b",
            task_id="BBB-008", title="Test work", status="in-progress",
            priority="P1", project="BBB", body="In-progress body."
        )
        report = service.lint()
        assert any(s["task_id"] == "BBB-008" for s in report["integrity"]["split_brain_content"])
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


# --------------------------------------------------------------------- pivot


def test_pivot_dedupes_by_task_id_and_normalizes_priority() -> None:
    tmp_root = make_temp_root()
    try:
        service = make_service(tmp_root)
        root = service.repository.tasks_root

        # Two copies of TECH-001, should be counted once.
        write_ticket_file(
            root, "backlog", "TECH-001-a",
            task_id="TECH-001", title="T1", status="backlog",
            priority="high", project="TECH"
        )
        write_ticket_file(
            root, "in-progress", "TECH-001-b",
            task_id="TECH-001", title="T1", status="in-progress",
            priority="P1", project="TECH"
        )
        # Unique tickets
        write_ticket_file(
            root, "backlog", "TECH-002",
            task_id="TECH-002", title="T2", status="backlog",
            priority="critical", project="TECH"
        )
        write_ticket_file(
            root, "backlog", "AAA-001",
            task_id="AAA-001", title="F1", status="backlog",
            priority="medium", project="AAA"
        )

        result = service.pivot(rows="project", cols="priority", status=["backlog", "in-progress"])
        # TECH-001 dedupes to the in-progress copy (open-status preference)
        assert sum(sum(row) for row in result["cells"]) == 3
        assert "TECH" in result["rows"]
        assert "AAA" in result["rows"]
        assert "P1" in result["cols"]
        assert "P3" in result["cols"]

        tech_index = result["rows"].index("TECH")
        p1_index = result["cols"].index("P1")
        # TECH/P1 should have both TECH-001 (high→P1) deduped to 1 AND TECH-002 (critical→P1) = 2
        assert result["cells"][tech_index][p1_index] == 2

        assert "integrity_warnings" in result
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_pivot_is_stable_across_calls() -> None:
    tmp_root = make_temp_root()
    try:
        service = make_service(tmp_root)
        root = service.repository.tasks_root
        for i in range(5):
            write_ticket_file(
                root, "backlog", f"TECH-{i + 1:03d}",
                task_id=f"TECH-{i + 1:03d}", title=f"t{i}", status="backlog",
                priority="P2", project="TECH"
            )
        a = service.pivot(rows="project", cols="priority")
        b = service.pivot(rows="project", cols="priority")
        assert a["rows"] == b["rows"]
        assert a["cols"] == b["cols"]
        assert a["cells"] == b["cells"]
        assert a["totals"] == b["totals"]
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


# -------------------------------------------------------------------- repair


def test_repair_normalize_priority_dry_run_then_apply() -> None:
    tmp_root = make_temp_root()
    try:
        service = make_service(tmp_root)
        root = service.repository.tasks_root
        write_ticket_file(
            root, "backlog", "TASK-001-high",
            task_id="TASK-001", title="High", status="backlog",
            priority="high", project="TASK"
        )
        write_ticket_file(
            root, "backlog", "TASK-002-medium",
            task_id="TASK-002", title="Medium", status="backlog",
            priority="medium", project="TASK"
        )
        write_ticket_file(
            root, "backlog", "TASK-003-good",
            task_id="TASK-003", title="Good", status="backlog",
            priority="P2", project="TASK"
        )
        write_ticket_file(
            root, "backlog", "TASK-004-weird",
            task_id="TASK-004", title="Weird", status="backlog",
            priority="yolo", project="TASK"
        )

        dry = service.repair(fixes=["normalize_priority"], dry_run=True)
        assert dry["dry_run"] is True
        assert dry["applied"] is False
        plan_ids = {step["task_id"] for step in dry["plan"]}
        assert plan_ids == {"TASK-001", "TASK-002"}
        assert any(u["task_id"] == "TASK-004" for u in dry["unresolved"])

        # lint before: priority warnings > 0
        before = service.lint()
        assert before["integrity"]["priority_non_normalized"]["count"] >= 2

        applied = service.repair(fixes=["normalize_priority"], dry_run=False)
        assert applied["dry_run"] is False
        assert applied["applied"] is True
        assert len(applied["applied_steps"]) == 2

        # lint after: TASK-001/TASK-002 fixed, TASK-004 remains (unmappable)
        after = service.lint()
        remaining = {s["task_id"] for s in after["integrity"]["priority_non_normalized"]["samples"]}
        assert "TASK-001" not in remaining
        assert "TASK-002" not in remaining
        assert "TASK-004" in remaining
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_repair_resolve_status_mismatch() -> None:
    tmp_root = make_temp_root()
    try:
        service = make_service(tmp_root)
        root = service.repository.tasks_root
        # File under in-progress/ but frontmatter says backlog -> fix to backlog
        write_ticket_file(
            root, "in-progress", "TECH-010-drift",
            task_id="TECH-010", title="Drift", status="backlog", project="TECH"
        )
        applied = service.repair(fixes=["resolve_status_mismatch"], dry_run=False)
        assert len(applied["applied_steps"]) == 1
        assert (root / "backlog" / "TECH-010-drift.md").exists()
        assert not (root / "in-progress" / "TECH-010-drift.md").exists()

        # lint after: no more mismatches for TECH-010
        after = service.lint()
        assert not any(m["task_id"] == "TECH-010" for m in after["integrity"]["status_mismatches"])
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_repair_flag_split_brain_tags_duplicates() -> None:
    tmp_root = make_temp_root()
    try:
        service = make_service(tmp_root)
        root = service.repository.tasks_root
        write_ticket_file(
            root, "backlog", "BBB-008-a",
            task_id="BBB-008", title="Test work", status="backlog",
            priority="P4", project="BBB", body="Backlog body."
        )
        write_ticket_file(
            root, "in-progress", "BBB-008-b",
            task_id="BBB-008", title="Test work", status="in-progress",
            priority="P1", project="BBB", body="In-progress body."
        )
        applied = service.repair(fixes=["flag_split_brain"], dry_run=False)
        assert len(applied["applied_steps"]) == 2
        tickets = service.repository.scan_tickets()
        mail = [t for t in tickets if t.ticket_id == "BBB-008"]
        assert len(mail) == 2
        for t in mail:
            assert "SPLIT-BRAIN" in t.tags
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_repair_rebuild_whoosh_runs() -> None:
    tmp_root = make_temp_root()
    try:
        service = make_service(tmp_root)
        root = service.repository.tasks_root
        write_ticket_file(
            root, "backlog", "TASK-001",
            task_id="TASK-001", title="T", status="backlog", project="TASK"
        )
        applied = service.repair(fixes=["rebuild_whoosh"], dry_run=False)
        assert applied["applied_steps"][0]["fix"] == "rebuild_whoosh"
        assert "result" in applied["applied_steps"][0]
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_repair_rejects_unknown_fix() -> None:
    tmp_root = make_temp_root()
    try:
        service = make_service(tmp_root)
        with pytest.raises(ValueError, match="Unknown fix"):
            service.repair(fixes=["nuke_everything"], dry_run=True)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_lint_then_repair_then_lint_priority_warnings_drop_to_zero() -> None:
    """Acceptance criterion: lint → repair --normalize_priority → lint
    shows priority warnings drop to zero (for mappable values)."""
    tmp_root = make_temp_root()
    try:
        service = make_service(tmp_root)
        root = service.repository.tasks_root
        for i, raw in enumerate(["high", "critical", "medium", "low", "1"], start=1):
            write_ticket_file(
                root, "backlog", f"TASK-{i:03d}",
                task_id=f"TASK-{i:03d}", title=f"t{i}", status="backlog",
                priority=raw, project="TASK"
            )

        before = service.lint()
        assert before["integrity"]["priority_non_normalized"]["count"] == 5

        service.repair(fixes=["normalize_priority"], dry_run=False)

        after = service.lint()
        assert after["integrity"]["priority_non_normalized"]["count"] == 0
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
