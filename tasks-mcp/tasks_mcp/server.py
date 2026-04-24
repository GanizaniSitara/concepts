from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from threading import RLock
import argparse
import json

from mcp.server.fastmcp import FastMCP

from .config import Settings, normalize_prefix, normalize_status, STATUS_DIRECTORY_NAMES
from .indexer import TicketIndex
from .integrity import IntegrityService
from .markdown_store import TaskRepository, UNSET


class TaskService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.repo = TaskRepository(settings)
        self.repo.ensure_structure()
        self.index = TicketIndex(settings.index_dir, self.repo)
        self.integrity = IntegrityService(self.repo, self.index)
        self._lock = RLock()

    def summary(self) -> dict[str, object]:
        with self._lock:
            sync = self.index.sync()
            tickets = self.repo.scan_tickets()
            counts = {status: 0 for status in STATUS_DIRECTORY_NAMES}
            prefixes: dict[str, int] = {}
            for ticket in tickets:
                counts[ticket.status] = counts.get(ticket.status, 0) + 1
                if ticket.prefix:
                    prefixes[ticket.prefix] = prefixes.get(ticket.prefix, 0) + 1
            return {
                "tasks_root": str(self.settings.tasks_root),
                "index": self.index.describe(),
                "counts": counts,
                "prefix_counts": prefixes,
                "missing_status_dirs": [
                    status
                    for status in STATUS_DIRECTORY_NAMES
                    if not (self.settings.tasks_root / status).exists()
                ],
                "sync": sync,
            }

    def search(
        self,
        *,
        query: str = "",
        status: str | None = None,
        prefix: str | None = None,
        project: str | None = None,
        limit: int = 20,
        include_content: bool = False,
    ) -> dict[str, object]:
        with self._lock:
            normalized_status = normalize_status(status) if status else None
            normalized_prefix = normalize_prefix(prefix) if prefix else None
            normalized_project = normalize_prefix(project) if project else None
            if query.strip():
                results = self.index.search(
                    query=query,
                    status=normalized_status,
                    prefix=normalized_prefix,
                    project=normalized_project,
                    limit=limit,
                    include_content=include_content,
                )
                source = "whoosh"
            else:
                tickets = self.repo.scan_tickets()
                filtered = [
                    ticket
                    for ticket in tickets
                    if (not normalized_status or ticket.status == normalized_status)
                    and (not normalized_prefix or ticket.prefix == normalized_prefix)
                    and (not normalized_project or (ticket.project or "").upper() == normalized_project)
                ]
                filtered.sort(
                    key=lambda ticket: (
                        ticket.updated or ticket.created or "",
                        ticket.ticket_id,
                    ),
                    reverse=True,
                )
                results = [
                    self.repo.task_to_dict(ticket, include_content=include_content)
                    for ticket in filtered[:limit]
                ]
                source = "filesystem"
            return {
                "query": query,
                "status": normalized_status,
                "prefix": normalized_prefix,
                "project": normalized_project,
                "source": source,
                "results": results,
            }

    def get_task(self, task_id: str, *, include_content: bool = True) -> dict[str, object]:
        with self._lock:
            task = self.repo.find_ticket(task_id)
            return self.repo.task_to_dict(task, include_content=include_content)

    def create_task(
        self,
        *,
        title: str,
        description: str = "",
        prefix: str | None = None,
        project: str | None = None,
        status: str = "backlog",
        priority: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, object]:
        with self._lock:
            task = self.repo.create_ticket(
                title=title,
                description=description,
                prefix=prefix,
                project=project,
                status=status,
                priority=priority,
                tags=tags,
                create_companion=True,
            )
            sync = self.index.sync()
            return {
                "task": self.repo.task_to_dict(task, include_content=True),
                "sync": sync,
            }

    def update_task(
        self,
        *,
        task_id: str,
        title: str | None = None,
        description: str | None = None,
        status: str | None = None,
        priority: str | None = None,
        tags: list[str] | None = None,
        project: str | None = None,
    ) -> dict[str, object]:
        with self._lock:
            task = self.repo.update_ticket(
                task_id,
                title=title if title is not None else UNSET,
                body=description if description is not None else UNSET,
                status=status if status is not None else UNSET,
                priority=priority if priority is not None else UNSET,
                tags=tags if tags is not None else UNSET,
                project=project if project is not None else UNSET,
                create_companion=True,
            )
            sync = self.index.sync()
            return {
                "task": self.repo.task_to_dict(task, include_content=True),
                "sync": sync,
            }

    def move_task(
        self,
        task_id: str,
        status: str,
        *,
        strategy: str = "error",
    ) -> dict[str, object]:
        with self._lock:
            task = self.repo.move_ticket(task_id, status, strategy=strategy)
            sync = self.index.sync()
            return {
                "task": self.repo.task_to_dict(task, include_content=True),
                "sync": sync,
            }

    def reopen_task(
        self,
        task_id: str,
        status: str = "backlog",
    ) -> dict[str, object]:
        with self._lock:
            task = self.repo.reopen_ticket(task_id, status=status)
            sync = self.index.sync()
            return {
                "task": self.repo.task_to_dict(task, include_content=True),
                "sync": sync,
            }

    def delete_task(
        self,
        task_id: str,
        path: str | None = None,
    ) -> dict[str, object]:
        with self._lock:
            result = self.repo.delete_ticket(task_id, path=path)
            sync = self.index.sync()
            result["sync"] = sync
            return result

    def list_duplicates(self) -> dict[str, object]:
        with self._lock:
            duplicates = self.repo.list_duplicates()
            return {
                "count": len(duplicates),
                "duplicates": duplicates,
            }

    def append_task_note(self, task_id: str, note: str, heading: str = "Notes") -> dict[str, object]:
        with self._lock:
            task = self.repo.append_note(task_id, note, heading=heading)
            sync = self.index.sync()
            return {
                "task": self.repo.task_to_dict(task, include_content=True),
                "sync": sync,
            }

    def rebuild_index(self) -> dict[str, object]:
        with self._lock:
            return self.index.rebuild()

    def lint(self) -> dict[str, object]:
        with self._lock:
            self.index.sync()
            return self.integrity.lint()

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
        with self._lock:
            self.index.sync()
            return self.integrity.pivot(
                rows=rows,
                cols=cols,
                status=status,
                priorities=priorities,
                projects=projects,
                dedupe=dedupe,
                normalize_priority_enabled=normalize_priority_enabled,
            )

    def repair(
        self,
        *,
        fixes: list[str],
        dry_run: bool = True,
    ) -> dict[str, object]:
        with self._lock:
            result = self.integrity.repair(fixes=fixes, dry_run=dry_run)
            if not dry_run:
                result["sync"] = self.index.sync()
            return result

    def migrate_legacy_tasks(self, *, apply: bool = False) -> dict[str, object]:
        with self._lock:
            if apply:
                result = self.repo.apply_migration()
                result["sync"] = self.index.rebuild()
                return result
            plan = self.repo.plan_migration()
            return {
                "applied": False,
                "migrated_count": 0,
                "actions": [action.to_dict() for action in plan["actions"]],
                "skipped": plan["skipped"],
                "unchanged": plan["unchanged"],
                "can_apply": plan["can_apply"],
            }


def _parse_tags(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def create_app(settings: Settings) -> FastMCP:
    service = TaskService(settings)
    instructions = (
        "Tasks MCP exposes the markdown task corpus under the configured tasks root.\n"
        "Markdown frontmatter is the source of truth for task state.\n"
        "Whoosh is a derived search index for fast retrieval, not an authoritative store.\n"
        "Creating or moving tasks updates both the markdown file and matching companion folder."
    )

    mcp = FastMCP(
        name="tasks-mcp",
        instructions=instructions,
        host=settings.host,
        port=settings.port,
        streamable_http_path="/mcp",
    )

    @mcp.tool(description="Return counts and paths for the local task corpus.")
    def task_summary() -> dict[str, object]:
        return service.summary()

    @mcp.tool(description="Search or list tasks across backlog, in-progress, blocked, and done.")
    def search_tasks(
        query: str = "",
        status: str | None = None,
        prefix: str | None = None,
        project: str | None = None,
        limit: int = 20,
        include_content: bool = False,
    ) -> dict[str, object]:
        return service.search(
            query=query,
            status=status,
            prefix=prefix,
            project=project,
            limit=limit,
            include_content=include_content,
        )

    @mcp.tool(description="Fetch a single task, including its indexed content.")
    def get_task(task_id: str, include_content: bool = True) -> dict[str, object]:
        return service.get_task(task_id, include_content=include_content)

    @mcp.tool(description="Create a new markdown-backed task and companion folder.")
    def create_task(
        title: str,
        description: str = "",
        prefix: str | None = None,
        project: str | None = None,
        status: str = "backlog",
        priority: str | None = None,
        tags: str | None = None,
    ) -> dict[str, object]:
        return service.create_task(
            title=title,
            description=description,
            prefix=prefix,
            project=project,
            status=status,
            priority=priority,
            tags=_parse_tags(tags),
        )

    @mcp.tool(description="Update task metadata/body and keep the Whoosh index in sync.")
    def update_task(
        task_id: str,
        title: str | None = None,
        description: str | None = None,
        status: str | None = None,
        priority: str | None = None,
        tags: str | None = None,
        project: str | None = None,
    ) -> dict[str, object]:
        return service.update_task(
            task_id=task_id,
            title=title,
            description=description,
            status=status,
            priority=priority,
            tags=_parse_tags(tags),
            project=project,
        )

    @mcp.tool(
        description=(
            "Move a task between backlog, in-progress, blocked, and done. "
            "strategy='error' (default) raises when the destination file/companion already "
            "exists; strategy='replace' deletes the stale destination first."
        )
    )
    def move_task(
        task_id: str,
        status: str,
        strategy: str = "error",
    ) -> dict[str, object]:
        return service.move_task(task_id, status, strategy=strategy)

    @mcp.tool(
        description=(
            "Reopen a done task by moving it back to backlog or in-progress. "
            "Appends a timestamped entry under a 'History' heading."
        )
    )
    def reopen_task(task_id: str, status: str = "backlog") -> dict[str, object]:
        return service.reopen_task(task_id, status=status)

    @mcp.tool(
        description=(
            "Delete a task's markdown file and companion folder. When a task id has "
            "duplicates on disk, supply the exact path to remove a single copy; "
            "otherwise the call errors to prevent deleting the live copy by accident."
        )
    )
    def delete_task(task_id: str, path: str | None = None) -> dict[str, object]:
        return service.delete_task(task_id, path=path)

    @mcp.tool(
        description=(
            "List task ids that have more than one markdown file on disk. Useful for "
            "diagnosing the backlog/done stale-copy scenario."
        )
    )
    def list_task_duplicates() -> dict[str, object]:
        return service.list_duplicates()

    @mcp.tool(description="Append a timestamped note to a task body.")
    def append_task_note(task_id: str, note: str, heading: str = "Notes") -> dict[str, object]:
        return service.append_task_note(task_id, note, heading)

    @mcp.tool(description="Force a full rebuild of the Whoosh index from markdown files.")
    def rebuild_task_index() -> dict[str, object]:
        return service.rebuild_index()

    @mcp.tool(description="Preview or apply the in-place migration to canonical task IDs and stems.")
    def migrate_legacy_tasks(apply: bool = False) -> dict[str, object]:
        return service.migrate_legacy_tasks(apply=apply)

    @mcp.tool(
        description=(
            "Read-only integrity scan of the task corpus. Returns structured report "
            "covering duplicates, status mismatches, orphan companion dirs, split-brain "
            "content, non-canonical priorities, missing prefixes, and Whoosh index drift."
        )
    )
    def lint_tasks() -> dict[str, object]:
        return service.lint()

    @mcp.tool(
        description=(
            "Deterministic pivot over the task corpus. Dedupes by task_id by default "
            "and normalizes priorities to P0-P5. Accepts row/col axes from "
            "(project, prefix, status, priority). 'status' is a comma-separated filter "
            "(e.g. 'backlog,in-progress'). Every response includes integrity_warnings."
        )
    )
    def pivot_tasks(
        rows: str = "project",
        cols: str = "status",
        status: str | None = None,
        priorities: str | None = None,
        projects: str | None = None,
        dedupe: str = "task_id",
        normalize_priority: bool = True,
    ) -> dict[str, object]:
        return service.pivot(
            rows=rows,
            cols=cols,
            status=_parse_tags(status),
            priorities=_parse_tags(priorities),
            projects=_parse_tags(projects),
            dedupe=dedupe,
            normalize_priority_enabled=normalize_priority,
        )

    @mcp.tool(
        description=(
            "Safe auto-repair of corpus integrity issues. 'fixes' is a comma-separated "
            "list from: normalize_priority, resolve_status_mismatch, flag_split_brain, "
            "rebuild_whoosh. With dry_run=True returns the plan and any unresolved "
            "items; with dry_run=False applies the plan and returns applied steps. "
            "Never auto-merges duplicate bodies."
        )
    )
    def repair_tasks(
        fixes: str,
        dry_run: bool = True,
    ) -> dict[str, object]:
        fix_list = _parse_tags(fixes) or []
        return service.repair(fixes=fix_list, dry_run=dry_run)

    @mcp.resource("tasks://summary", mime_type="application/json")
    def summary_resource() -> str:
        return json.dumps(service.summary(), indent=2)

    @mcp.resource("tasks://backlog", mime_type="application/json")
    def backlog_resource() -> str:
        return json.dumps(service.search(status="backlog", limit=50), indent=2)

    return mcp


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the local markdown task corpus over FastMCP.")
    parser.add_argument("--tasks-root", help="Root task directory. Defaults to TASKS_ROOT or ~/tasks.")
    parser.add_argument("--index-dir", help="Directory for the Whoosh index. Defaults to TASKS_INDEX_DIR or <repo>/.data/whoosh.")
    parser.add_argument("--host", help="Bind host for network transports.")
    parser.add_argument("--port", type=int, help="Bind port for network transports.")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        help="FastMCP transport to run.",
    )
    parser.add_argument("--default-prefix", help="Default task prefix for new tasks.")
    return parser.parse_args()


def _settings_from_args(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env()
    tasks_root = settings.tasks_root
    index_dir = settings.index_dir

    if args.tasks_root:
        tasks_root = Path(args.tasks_root).expanduser().resolve()
    if args.index_dir:
        index_dir = Path(args.index_dir).expanduser().resolve()

    settings = replace(
        settings,
        tasks_root=tasks_root,
        index_dir=index_dir,
    )
    if args.host:
        settings = replace(settings, host=args.host)
    if args.port:
        settings = replace(settings, port=args.port)
    if args.transport:
        settings = replace(settings, transport=args.transport)
    if args.default_prefix:
        settings = replace(settings, default_prefix=args.default_prefix.upper())
    return settings


def main() -> None:
    args = _parse_args()
    settings = _settings_from_args(args)
    app = create_app(settings)
    app.run(transport=settings.transport)


if __name__ == "__main__":
    main()
