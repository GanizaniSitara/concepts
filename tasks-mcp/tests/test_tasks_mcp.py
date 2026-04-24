from __future__ import annotations

from pathlib import Path
import shutil
import tempfile

import pytest

from tasks_mcp.config import Settings
from tasks_mcp.indexer import TicketIndex
from tasks_mcp.markdown_store import DuplicateTicketError, TaskRepository


def make_temp_root() -> Path:
    return Path(tempfile.mkdtemp(prefix="tasks-mcp-case-"))


def make_settings(tmp_root: Path) -> Settings:
    tasks_root = tmp_root / "tasks"
    index_dir = tmp_root / "whoosh-index"
    return Settings(
        tasks_root=tasks_root,
        index_dir=index_dir,
        host="127.0.0.1",
        port=8876,
        transport="streamable-http",
        default_prefix="TASK",
    )


def test_create_and_move_ticket() -> None:
    tmp_root = make_temp_root()
    try:
        settings = make_settings(tmp_root)
        repo = TaskRepository(settings)
        index = TicketIndex(settings.index_dir, repo)

        created = repo.create_ticket(
            title="Create MCP server",
            description="Build a local task MCP service.",
            prefix="PRJ",
            project="PRJ",
            status="backlog",
        )
        assert created.ticket_id == "PRJ-001"
        assert created.path.name == "PRJ-001-create-mcp-server.md"
        assert created.companion_dir is not None
        assert created.companion_dir.is_dir()

        results = index.search(query="create mcp service", limit=5, include_content=True)
        assert any(item["task_id"] == "PRJ-001" and item["status"] == "backlog" for item in results)

        moved = repo.update_ticket("PRJ-001", status="in-progress", body_append="Started implementation.")
        assert moved.path.parent.name == "in-progress"
        assert moved.companion_dir is not None
        assert moved.companion_dir.is_dir()

        results = index.search(query="started implementation", limit=5, include_content=True)
        assert any(item["task_id"] == "PRJ-001" and item["status"] == "in-progress" for item in results)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_search_indexes_companion_folder_content() -> None:
    tmp_root = make_temp_root()
    try:
        settings = make_settings(tmp_root)
        repo = TaskRepository(settings)
        index = TicketIndex(settings.index_dir, repo)

        created = repo.create_ticket(
            title="Search companion notes",
            description="Primary ticket body.",
            prefix="PRJ",
            project="PRJ",
        )
        assert created.companion_dir is not None
        (created.companion_dir / "DESIGN.md").write_text(
            "# Notes\n\nGraph edge cache warmup sequence lives here.\n",
            encoding="utf-8",
        )

        results = index.search(query="graph edge cache", limit=10)
        assert any(item["task_id"] == created.ticket_id for item in results)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_index_refreshes_after_direct_file_edit() -> None:
    tmp_root = make_temp_root()
    try:
        settings = make_settings(tmp_root)
        repo = TaskRepository(settings)
        index = TicketIndex(settings.index_dir, repo)

        created = repo.create_ticket(
            title="Refresh after manual edit",
            description="Original text only.",
            prefix="TASK",
            project="TASK",
        )
        contents = created.path.read_text(encoding="utf-8")
        created.path.write_text(contents + "\nManual phrase: semaphore orchard.\n", encoding="utf-8")

        results = index.search(query="semaphore orchard", limit=5)
        assert any(item["task_id"] == created.ticket_id for item in results)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_scan_tolerates_simple_invalid_yaml_frontmatter() -> None:
    tmp_root = make_temp_root()
    try:
        settings = make_settings(tmp_root)
        repo = TaskRepository(settings)
        (settings.tasks_root / "backlog").mkdir(parents=True, exist_ok=True)
        path = settings.tasks_root / "backlog" / "TASK-001-disk-cleanup.md"
        path.write_text(
            """---
ticket: TASK-001
title: Disk cleanup on C: drive to reduce backup costs
status: backlog
project: TASK
---

Cleanup task details.
""",
            encoding="utf-8",
        )

        tickets = repo.scan_tickets()
        assert len(tickets) == 1
        assert tickets[0].ticket_id == "TASK-001"
        assert tickets[0].title == "Disk cleanup on C: drive to reduce backup costs"
        assert tickets[0].status == "backlog"
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_legacy_project_names_do_not_leak_into_task_output() -> None:
    tmp_root = make_temp_root()
    try:
        settings = make_settings(tmp_root)
        repo = TaskRepository(settings)
        index = TicketIndex(settings.index_dir, repo)
        (settings.tasks_root / "backlog").mkdir(parents=True, exist_ok=True)
        path = settings.tasks_root / "backlog" / "TASK_job-application-pipeline-design.md"
        path.write_text(
            """---
task: TASK_job-application-pipeline-design
title: Design end-to-end job application pipeline automation
status: backlog
project: open-moniker
---

Pipeline work.
""",
            encoding="utf-8",
        )

        ticket = repo.find_ticket("TASK_job-application-pipeline-design")
        data = repo.task_to_dict(ticket, include_content=False)
        assert ticket.prefix == "TASK"
        assert ticket.project == "TASK"
        assert data["project"] == "TASK"
        assert data["frontmatter"]["project"] == "TASK"

        results = index.search(query="pipeline automation", limit=5)
        assert any(
            item["task_id"] == "TASK_job-application-pipeline-design"
            and item["project"] == "TASK"
            and item["frontmatter"]["project"] == "TASK"
            for item in results
        )
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_migration_normalizes_legacy_forms() -> None:
    tmp_root = make_temp_root()
    try:
        settings = make_settings(tmp_root)
        repo = TaskRepository(settings)
        (settings.tasks_root / "backlog").mkdir(parents=True, exist_ok=True)
        (settings.tasks_root / "in-progress").mkdir(parents=True, exist_ok=True)
        (settings.tasks_root / "done").mkdir(parents=True, exist_ok=True)
        (settings.tasks_root / "blocked").mkdir(parents=True, exist_ok=True)

        (settings.tasks_root / "in-progress" / "MAIL-8_openclaw-lite-design.md").write_text(
            """---
title: OpenClaw Lite Design
status: in-progress
ticket: MAIL-8
project: mailrunner
---

Mail task content.
""",
            encoding="utf-8",
        )
        (settings.tasks_root / "blocked" / "001-title-keyword-matrix.md").write_text(
            """---
id: 001
title: Build title matrix
status: blocked
---

Blocked pending CV work.
""",
            encoding="utf-8",
        )
        (settings.tasks_root / "blocked" / "001-title-keyword-matrix").mkdir(parents=True, exist_ok=True)
        ((settings.tasks_root / "blocked" / "001-title-keyword-matrix") / "notes.md").write_text(
            "Legacy companion content.",
            encoding="utf-8",
        )
        (settings.tasks_root / "backlog" / "TASK_job-application-pipeline-design.md").write_text(
            """---
id: TASK_job-application-pipeline-design
title: Design end-to-end job application pipeline automation
status: backlog
project: TASK
---

Pipeline work.
""",
            encoding="utf-8",
        )
        (settings.tasks_root / "backlog" / "PROJ-5.10_import-saved-searches.md").write_text(
            """---
id: PROJ-5.10
title: Import saved searches
status: backlog
project: PROJ
parent: PROJ-5
---

Import work.
""",
            encoding="utf-8",
        )

        preview = repo.plan_migration()
        actions = {item.current_ticket_id: item for item in preview["actions"]}
        assert actions["MAIL-8"].target_ticket_id == "MAIL-008"
        assert actions["001"].target_ticket_id == "TASK-001"
        assert actions["TASK_job-application-pipeline-design"].target_ticket_id == "TASK-002"
        assert actions["PROJ-5.10"].target_ticket_id == "PROJ-510"

        applied = repo.apply_migration()
        assert applied["migrated_count"] == 4
        assert (settings.tasks_root / "in-progress" / "MAIL-008-openclaw-lite-design.md").exists()
        assert (settings.tasks_root / "blocked" / "TASK-001-build-title-matrix.md").exists()
        assert (settings.tasks_root / "blocked" / "TASK-001-build-title-matrix").is_dir()
        assert (settings.tasks_root / "backlog" / "TASK-002-design-end-to-end-job-application-pipeline-automation.md").exists()

        migrated = repo.find_ticket("PROJ-510")
        assert migrated.frontmatter["parent"] == "PROJ-005"
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_reopen_task_moves_done_to_backlog_with_history_note() -> None:
    tmp_root = make_temp_root()
    try:
        settings = make_settings(tmp_root)
        repo = TaskRepository(settings)

        repo.create_ticket(title="Close the loop", prefix="TEST", project="TEST")
        repo.move_ticket("TEST-001", "done")
        done_path = settings.tasks_root / "done" / "TEST-001-close-the-loop.md"
        assert done_path.exists()

        reopened = repo.reopen_ticket("TEST-001", status="backlog")
        assert reopened.status == "backlog"
        assert reopened.path.parent.name == "backlog"
        assert not done_path.exists()
        assert "## History" in reopened.body
        assert "Reopened to backlog." in reopened.body
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_reopen_rejects_non_done_task() -> None:
    tmp_root = make_temp_root()
    try:
        settings = make_settings(tmp_root)
        repo = TaskRepository(settings)
        repo.create_ticket(title="Still in flight", prefix="TEST", project="TEST")

        with pytest.raises(ValueError, match="Only done tickets can be reopened"):
            repo.reopen_ticket("TEST-001", status="backlog")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_reopen_rejects_invalid_target_status() -> None:
    tmp_root = make_temp_root()
    try:
        settings = make_settings(tmp_root)
        repo = TaskRepository(settings)
        repo.create_ticket(title="Finished work", prefix="TEST", project="TEST")
        repo.move_ticket("TEST-001", "done")

        with pytest.raises(ValueError, match="Reopen target must be"):
            repo.reopen_ticket("TEST-001", status="done")
        with pytest.raises(ValueError, match="Reopen target must be"):
            repo.reopen_ticket("TEST-001", status="blocked")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_move_with_replace_strategy_overwrites_stale_destination() -> None:
    """The MAIL-5 scenario: live copy in backlog with rich content, stale minimal
    copy in done, caller wants to close the live copy. Default strategy errors
    on the destination collision; strategy='replace' nukes the stale done file
    first and then moves the live copy in.
    """
    tmp_root = make_temp_root()
    try:
        settings = make_settings(tmp_root)
        repo = TaskRepository(settings)

        repo.create_ticket(
            title="Main copy",
            description="Live body with real history.",
            prefix="TEST",
            project="TEST",
        )

        (settings.tasks_root / "done").mkdir(parents=True, exist_ok=True)
        stale_path = settings.tasks_root / "done" / "TEST-001-main-copy.md"
        stale_path.write_text(
            """---
task: TEST-001
title: Main copy
status: backlog
project: TEST
---

Stale pre-existing copy.
""",
            encoding="utf-8",
        )

        with pytest.raises(FileExistsError):
            repo.move_ticket("TEST-001", "done")

        moved = repo.move_ticket("TEST-001", "done", strategy="replace")
        assert moved.status == "done"
        assert moved.path.parent.name == "done"
        assert "Live body with real history." in moved.body
        assert "Stale pre-existing copy." not in moved.body

        remaining = repo.find_all_tickets("TEST-001")
        assert len(remaining) == 1
        assert remaining[0].status == "done"
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_delete_task_removes_file_and_companion() -> None:
    tmp_root = make_temp_root()
    try:
        settings = make_settings(tmp_root)
        repo = TaskRepository(settings)

        created = repo.create_ticket(title="Short lived", prefix="TEST", project="TEST")
        companion = created.companion_dir
        assert companion and companion.is_dir()
        (companion / "notes.md").write_text("Discarded.", encoding="utf-8")

        result = repo.delete_ticket("TEST-001")
        assert not created.path.exists()
        assert not companion.exists()
        assert str(created.path) in result["removed"]
        assert str(companion) in result["removed"]
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_duplicate_detection_and_targeted_deletion() -> None:
    tmp_root = make_temp_root()
    try:
        settings = make_settings(tmp_root)
        repo = TaskRepository(settings)

        repo.create_ticket(title="Live copy", prefix="TEST", project="TEST")
        repo.move_ticket("TEST-001", "done")

        (settings.tasks_root / "backlog").mkdir(parents=True, exist_ok=True)
        stale_path = settings.tasks_root / "backlog" / "TEST-001-live-copy.md"
        stale_path.write_text(
            """---
task: TEST-001
title: Live copy
status: backlog
project: TEST
---

Stale copy.
""",
            encoding="utf-8",
        )

        duplicates = repo.list_duplicates()
        assert len(duplicates) == 1
        assert duplicates[0]["task_id"] == "TEST-001"
        assert duplicates[0]["count"] == 2

        with pytest.raises(DuplicateTicketError):
            repo.delete_ticket("TEST-001")

        result = repo.delete_ticket("TEST-001", path=str(stale_path))
        assert not stale_path.exists()
        assert len(repo.find_all_tickets("TEST-001")) == 1
        assert result["remaining"]
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
