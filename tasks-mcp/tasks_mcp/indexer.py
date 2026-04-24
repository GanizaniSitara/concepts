from __future__ import annotations

from pathlib import Path
import json
from typing import Iterable

from whoosh import index
from whoosh.fields import ID, KEYWORD, NUMERIC, STORED, TEXT, Schema
from whoosh.highlight import ContextFragmenter
from whoosh.qparser import MultifieldParser, OrGroup

from .config import normalize_prefix, normalize_status
from .markdown_store import TaskRepository
from .models import Ticket


class TicketIndex:
    def __init__(self, index_dir: Path, repository: TaskRepository):
        self.index_dir = index_dir
        self.repository = repository
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.schema = Schema(
            ticket_id=ID(stored=True, unique=True),
            prefix=ID(stored=True),
            project=ID(stored=True),
            status=ID(stored=True),
            title=TEXT(stored=True),
            tags=KEYWORD(stored=True, commas=True, lowercase=True, scorable=True),
            path=ID(stored=True),
            companion_dir=ID(stored=True),
            content=TEXT(stored=True),
            frontmatter_json=STORED,
            asset_paths_json=STORED,
            content_hash=ID(stored=True),
            source_mtime=NUMERIC(stored=True, numtype=float),
        )
        self.ix = (
            index.open_dir(self.index_dir)
            if index.exists_in(self.index_dir)
            else index.create_in(self.index_dir, self.schema)
        )
        self.last_sync: str | None = None

    def rebuild(self) -> dict[str, int | str]:
        for child in self.index_dir.iterdir():
            if child.is_file():
                child.unlink()
        self.ix = index.create_in(self.index_dir, self.schema)
        return self.sync()

    def sync(self) -> dict[str, int | str]:
        tickets = self.repository.scan_tickets()
        updated = 0
        removed = 0
        live_ids = {ticket.ticket_id for ticket in tickets}

        with self.ix.searcher() as searcher:
            existing = {doc["ticket_id"]: doc for doc in searcher.all_stored_fields()}

        writer = self.ix.writer()
        dirty = False
        for ticket in tickets:
            current = existing.get(ticket.ticket_id)
            if current and current["content_hash"] == ticket.content_hash and current["path"] == str(ticket.path):
                continue
            writer.update_document(**self._document_from_ticket(ticket))
            updated += 1
            dirty = True

        for ticket_id, stored in existing.items():
            if ticket_id not in live_ids:
                writer.delete_by_term("ticket_id", ticket_id)
                removed += 1
                dirty = True

        if dirty:
            writer.commit()
        else:
            writer.cancel()

        self.last_sync = str(max((ticket.source_mtime for ticket in tickets), default=0.0))

        return {
            "scanned": len(tickets),
            "updated": updated,
            "removed": removed,
            "indexed_at": self.last_sync or "",
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
    ) -> list[dict[str, object]]:
        self.sync()
        normalized_project = normalize_prefix(project) if project else None
        parser = MultifieldParser(
            ["ticket_id", "title", "content", "tags", "project", "prefix"],
            schema=self.ix.schema,
            group=OrGroup.factory(0.9),
        )
        parsed = parser.parse(query.strip())

        results_out: list[dict[str, object]] = []
        with self.ix.searcher() as searcher:
            results = searcher.search(parsed, limit=max(limit * 5, limit))
            results.fragmenter = ContextFragmenter(maxchars=220, surround=40)

            for hit in results:
                item = self._hit_to_dict(hit, include_content=include_content)
                if status and item["status"] != normalize_status(status):
                    continue
                if prefix and (item.get("prefix") or "").upper() != prefix.upper():
                    continue
                if normalized_project and (item.get("project") or "").upper() != normalized_project:
                    continue
                results_out.append(item)
                if len(results_out) >= limit:
                    break

        return results_out

    def describe(self) -> dict[str, object]:
        self.sync()
        with self.ix.searcher() as searcher:
            return {
                "index_dir": str(self.index_dir),
                "document_count": searcher.doc_count_all(),
                "last_sync": self.last_sync,
            }

    def _document_from_ticket(self, ticket: Ticket) -> dict[str, object]:
        content = ticket.title.strip()
        if ticket.body.strip():
            content += "\n\n" + ticket.body.strip()
        if ticket.asset_blob.strip():
            content += "\n\n## Companion Files\n" + ticket.asset_blob.strip()
        return {
            "ticket_id": ticket.ticket_id,
            "prefix": ticket.prefix or "",
            "project": ticket.project or "",
            "status": ticket.status,
            "title": ticket.title,
            "tags": ",".join(ticket.tags),
            "path": str(ticket.path),
            "companion_dir": str(ticket.companion_dir) if ticket.companion_dir else "",
            "content": content,
            "frontmatter_json": json.dumps(ticket.frontmatter, default=str, sort_keys=True),
            "asset_paths_json": json.dumps(ticket.asset_paths),
            "content_hash": ticket.content_hash,
            "source_mtime": float(ticket.source_mtime),
        }

    def _hit_to_dict(self, hit, *, include_content: bool) -> dict[str, object]:
        item = dict(hit)
        frontmatter = json.loads(item.pop("frontmatter_json"))
        if "task" not in frontmatter:
            if "ticket" in frontmatter:
                frontmatter["task"] = frontmatter.pop("ticket")
            elif "id" in frontmatter:
                frontmatter["task"] = frontmatter.pop("id")
        frontmatter.pop("ticket", None)
        frontmatter.pop("id", None)
        project = item.get("project") or None
        if project:
            frontmatter["project"] = project
        else:
            frontmatter.pop("project", None)
        asset_paths = json.loads(item.pop("asset_paths_json"))
        snippet = hit.highlights("content")
        result = {
            "task_id": item["ticket_id"],
            "prefix": item.get("prefix") or None,
            "project": project,
            "status": item["status"],
            "title": item["title"],
            "tags": [part for part in item.get("tags", "").split(",") if part],
            "path": item["path"],
            "companion_dir": item.get("companion_dir") or None,
            "frontmatter": frontmatter,
            "asset_paths": asset_paths,
            "snippet": snippet,
            "source_mtime": item["source_mtime"],
        }
        if include_content:
            result["content"] = item["content"]
        return result
