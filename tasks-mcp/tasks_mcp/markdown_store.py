from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import hashlib
import json
import re
import shutil
import unicodedata
import yaml

from .config import (
    REFERENCE_FIELDS,
    READABLE_ASSET_SUFFIXES,
    STATUS_DIRECTORY_NAMES,
    Settings,
    canonical_ticket_id,
    normalize_status,
    normalize_prefix,
    status_dir,
)
from .models import MigrationAction, Ticket


UNSET = object()


class DuplicateTicketError(Exception):
    def __init__(self, ticket_id: str, paths: list[Path]):
        self.ticket_id = ticket_id
        self.paths = list(paths)
        joined = ", ".join(str(p) for p in self.paths)
        super().__init__(
            f"Ticket {ticket_id} has {len(self.paths)} copies on disk: {joined}"
        )

_FRONTMATTER_RE = re.compile(r"^---\s*\r?\n(.*?)\r?\n---\s*(?:\r?\n)?", re.S)
_CANONICAL_ID_RE = re.compile(r"^(?P<prefix>[A-Z][A-Z0-9]*)-(?P<number>\d+)$")
_DOTTED_ID_RE = re.compile(
    r"^(?P<prefix>[A-Z][A-Z0-9]*)-(?P<number>\d+(?:\.\d+)+)$"
)
_WORD_PREFIX_STEM_RE = re.compile(r"^(?P<prefix>[A-Z][A-Z0-9]*)(?:[_-](?P<rest>.+))$")
_LEGACY_UNDERSCORE_STEM_RE = re.compile(
    r"^(?P<identifier>(?:[A-Z][A-Z0-9]*-\d+(?:\.\d+)?|\d+))(?:_(?P<slug>.+))?$"
)
_CANONICAL_STEM_RE = re.compile(
    r"^(?P<identifier>[A-Z][A-Z0-9]*-\d+(?:\.\d+)?)(?:-(?P<slug>.+))?$"
)
_NUMERIC_STEM_RE = re.compile(r"^(?P<identifier>\d+)(?:-(?P<slug>.+))?$")
_NON_ALNUM_RE = re.compile(r"[^A-Z0-9]+")
_NON_SLUG_RE = re.compile(r"[^a-z0-9]+")
_CANONICAL_PROJECT_RE = re.compile(r"^[A-Z][A-Z0-9]*$")
_PREFERRED_FRONTMATTER_ORDER = (
    "task",
    "ticket",
    "title",
    "status",
    "created",
    "updated",
    "completed",
    "priority",
    "project",
    "depends_on",
    "blocked_by",
    "parallelizable",
    "tags",
)


class TaskRepository:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.tasks_root = settings.tasks_root

    def ensure_structure(self) -> None:
        self.settings.index_dir.mkdir(parents=True, exist_ok=True)

    def iter_markdown_files(self) -> list[tuple[str, Path]]:
        files: list[tuple[str, Path]] = []
        for canonical in STATUS_DIRECTORY_NAMES:
            folder = status_dir(self.tasks_root, canonical)
            if not folder.exists() or not folder.is_dir():
                continue
            for path in sorted(folder.iterdir()):
                if path.is_file() and path.suffix.lower() == ".md":
                    files.append((canonical, path))
        return files

    def scan_tickets(self) -> list[Ticket]:
        self.ensure_structure()
        tickets: list[Ticket] = []
        for canonical, path in self.iter_markdown_files():
            ticket = self.parse_ticket(path, canonical)
            if ticket is not None:
                tickets.append(ticket)
        return tickets

    def parse_ticket(self, path: Path, fallback_status: str | None = None) -> Ticket | None:
        if path.stem.upper() == "README":
            return None
        raw = path.read_text(encoding="utf-8", errors="replace")
        meta, body = self._split_frontmatter(raw)
        try:
            status = normalize_status(str(meta.get("status") or fallback_status or path.parent.name))
        except ValueError:
            if fallback_status:
                status = normalize_status(fallback_status)
            else:
                return None
        ticket_id = self._derive_ticket_id(meta, path.stem)
        prefix, number = self._parse_ticket_id(ticket_id)
        slug = self._derive_slug(path.stem, ticket_id, meta.get("title"))
        title = str(meta.get("title") or self._humanize_slug(slug or path.stem)).strip()
        tags = self._normalize_tags(meta.get("tags"))
        companion_dir = path.with_suffix("")
        asset_blob, asset_paths, asset_mtime = self._load_companion_assets(companion_dir)
        content_hash = hashlib.sha256((raw + "\n\n" + asset_blob).encode("utf-8")).hexdigest()

        return Ticket(
            ticket_id=ticket_id,
            title=title,
            status=status,
            path=path,
            stem=path.stem,
            body=body.rstrip() + ("\n" if body.strip() else ""),
            frontmatter=dict(meta),
            prefix=prefix,
            project=self._canonical_project_key(prefix, meta),
            number=number,
            slug=slug,
            priority=self._str_or_none(meta.get("priority")),
            created=self._str_or_none(meta.get("created")),
            updated=self._str_or_none(meta.get("updated")),
            tags=tags,
            companion_dir=companion_dir if companion_dir.is_dir() else None,
            asset_paths=asset_paths,
            asset_blob=asset_blob,
            source_mtime=max(path.stat().st_mtime, asset_mtime),
            content_hash=content_hash,
        )

    def find_ticket(self, ticket_id: str) -> Ticket:
        wanted = self._ticket_lookup_key(ticket_id)
        for ticket in self.scan_tickets():
            if self._ticket_lookup_key(ticket.ticket_id) == wanted:
                return ticket
        raise FileNotFoundError(f"Ticket not found: {ticket_id}")

    def find_all_tickets(self, ticket_id: str) -> list[Ticket]:
        wanted = self._ticket_lookup_key(ticket_id)
        return [
            ticket
            for ticket in self.scan_tickets()
            if self._ticket_lookup_key(ticket.ticket_id) == wanted
        ]

    def list_duplicates(self) -> list[dict[str, object]]:
        by_id: dict[str, list[Ticket]] = {}
        for ticket in self.scan_tickets():
            key = self._ticket_lookup_key(ticket.ticket_id)
            by_id.setdefault(key, []).append(ticket)
        return [
            {
                "task_id": tickets[0].ticket_id,
                "count": len(tickets),
                "paths": [str(t.path) for t in tickets],
                "statuses": [t.status for t in tickets],
            }
            for tickets in by_id.values()
            if len(tickets) > 1
        ]

    def delete_ticket(
        self,
        ticket_id: str,
        *,
        path: str | Path | None = None,
    ) -> dict[str, object]:
        matches = self.find_all_tickets(ticket_id)
        if not matches:
            raise FileNotFoundError(f"Ticket not found: {ticket_id}")

        if path is not None:
            target_path = Path(path).expanduser().resolve()
            target = next(
                (t for t in matches if t.path.resolve() == target_path),
                None,
            )
            if target is None:
                raise FileNotFoundError(
                    f"No ticket {ticket_id} found at path {target_path}"
                )
            targets = [target]
        else:
            if len(matches) > 1:
                raise DuplicateTicketError(ticket_id, [t.path for t in matches])
            targets = matches

        removed: list[str] = []
        for ticket in targets:
            if ticket.path.exists():
                ticket.path.unlink()
                removed.append(str(ticket.path))
            companion = ticket.companion_dir
            if companion and companion.exists():
                shutil.rmtree(companion)
                removed.append(str(companion))

        return {
            "task_id": ticket_id,
            "removed": removed,
            "remaining": [
                str(t.path) for t in matches if t not in targets
            ],
        }

    def reopen_ticket(
        self,
        ticket_id: str,
        *,
        status: str = "backlog",
    ) -> Ticket:
        target_status = normalize_status(status)
        if target_status not in ("backlog", "in-progress"):
            raise ValueError(
                f"Reopen target must be backlog or in-progress; got {target_status!r}"
            )

        matches = self.find_all_tickets(ticket_id)
        if not matches:
            raise FileNotFoundError(f"Ticket not found: {ticket_id}")
        if len(matches) > 1:
            raise DuplicateTicketError(ticket_id, [t.path for t in matches])

        ticket = matches[0]
        if ticket.status != "done":
            raise ValueError(
                f"Only done tickets can be reopened; {ticket_id} is currently {ticket.status}"
            )

        self.append_note(
            ticket_id,
            f"Reopened to {target_status}.",
            heading="History",
        )
        return self.update_ticket(ticket_id, status=target_status)

    def create_ticket(
        self,
        title: str,
        description: str = "",
        prefix: str | None = None,
        project: str | None = None,
        status: str = "backlog",
        priority: str | None = None,
        tags: list[str] | None = None,
        create_companion: bool = True,
    ) -> Ticket:
        self.ensure_structure()
        normalized_status = normalize_status(status)
        prefix_key = normalize_prefix(prefix or self.settings.default_prefix)
        ticket_id = canonical_ticket_id(prefix_key, self.next_ticket_number(prefix_key))
        slug = self.slugify(title)
        stem = f"{ticket_id}-{slug}"
        path = status_dir(self.tasks_root, normalized_status) / f"{stem}.md"
        if path.exists():
            raise FileExistsError(f"Ticket file already exists: {path}")

        today = date.today().isoformat()
        project_value = prefix_key
        meta: dict[str, object] = {
            "ticket": ticket_id,
            "title": title.strip(),
            "status": normalized_status,
            "created": today,
            "updated": today,
            "project": project_value,
        }
        if priority:
            meta["priority"] = priority
        if tags:
            meta["tags"] = list(tags)

        body = description.strip()
        if body:
            body = body + "\n"

        ticket = Ticket(
            ticket_id=ticket_id,
            title=title.strip(),
            status=normalized_status,
            path=path,
            stem=stem,
            body=body,
            frontmatter=meta,
            prefix=prefix_key,
            project=project_value,
            number=self._parse_ticket_id(ticket_id)[1],
            slug=slug,
            priority=priority,
            created=today,
            updated=today,
            tags=list(tags or []),
            companion_dir=path.with_suffix("") if create_companion else None,
        )
        self._write_ticket(ticket)
        if create_companion and ticket.companion_dir:
            ticket.companion_dir.mkdir(parents=True, exist_ok=True)
        return self.parse_ticket(path, normalized_status)

    def update_ticket(
        self,
        ticket_id: str,
        *,
        title: str | object = UNSET,
        body: str | object = UNSET,
        body_append: str | None = None,
        status: str | object = UNSET,
        priority: str | object = UNSET,
        tags: list[str] | object = UNSET,
        project: str | object = UNSET,
        create_companion: bool | None = None,
        strategy: str = "error",
    ) -> Ticket:
        if strategy not in ("error", "replace"):
            raise ValueError(
                f"Unknown strategy {strategy!r}; expected 'error' or 'replace'"
            )
        ticket = self.find_ticket(ticket_id)
        old_path = ticket.path
        old_companion = ticket.companion_dir if ticket.companion_dir and ticket.companion_dir.exists() else None

        if title is not UNSET:
            ticket.title = str(title).strip()
            ticket.slug = self.slugify(ticket.title)

        if body is not UNSET:
            text = str(body or "").rstrip()
            ticket.body = text + ("\n" if text else "")

        if body_append:
            extra = body_append.strip()
            if extra:
                if ticket.body.strip():
                    ticket.body = ticket.body.rstrip() + "\n\n" + extra + "\n"
                else:
                    ticket.body = extra + "\n"

        if status is not UNSET:
            ticket.status = normalize_status(str(status))

        if priority is not UNSET:
            ticket.priority = self._str_or_none(priority)

        if tags is not UNSET:
            ticket.tags = [str(tag).strip() for tag in list(tags or []) if str(tag).strip()]

        if project is not UNSET:
            ticket.project = self._canonical_project_value(project) or ticket.prefix

        if create_companion is None:
            create_companion = ticket.companion_dir is not None or old_companion is not None

        ticket.frontmatter = dict(ticket.frontmatter)
        ticket.updated = date.today().isoformat()

        new_stem = old_path.stem
        if title is not UNSET:
            new_stem = f"{ticket.ticket_id}-{ticket.slug}"

        new_path = status_dir(self.tasks_root, ticket.status) / f"{new_stem}.md"
        new_companion = new_path.with_suffix("")

        if old_path != new_path:
            if new_path.exists():
                if strategy == "replace":
                    new_path.unlink()
                else:
                    raise FileExistsError(
                        f"Destination ticket file already exists: {new_path}"
                    )
            new_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_path), str(new_path))
            if old_companion:
                if new_companion.exists():
                    if strategy == "replace":
                        shutil.rmtree(new_companion)
                    else:
                        raise FileExistsError(
                            f"Destination companion folder already exists: {new_companion}"
                        )
                shutil.move(str(old_companion), str(new_companion))
            ticket.path = new_path
            ticket.stem = new_stem

        ticket.companion_dir = new_companion if create_companion else None
        self._write_ticket(ticket)
        if create_companion and ticket.companion_dir:
            ticket.companion_dir.mkdir(parents=True, exist_ok=True)
        return self.parse_ticket(ticket.path, ticket.status)

    def move_ticket(
        self,
        ticket_id: str,
        status: str,
        *,
        strategy: str = "error",
    ) -> Ticket:
        return self.update_ticket(ticket_id, status=status, strategy=strategy)

    def append_note(self, ticket_id: str, note: str, heading: str = "Notes") -> Ticket:
        ticket = self.find_ticket(ticket_id)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        heading_line = f"## {heading.strip()}"
        entry = f"- {stamp}: {note.strip()}"

        body = ticket.body.rstrip()
        if heading_line not in body:
            if body:
                body += f"\n\n{heading_line}\n\n{entry}\n"
            else:
                body = f"{heading_line}\n\n{entry}\n"
        else:
            body += f"\n{entry}\n"

        return self.update_ticket(ticket_id, body=body)

    def next_ticket_number(self, prefix: str) -> int:
        target = normalize_prefix(prefix)
        highest = 0
        for ticket in self.scan_tickets():
            if ticket.prefix == target and ticket.number is not None:
                highest = max(highest, ticket.number)
        return highest + 1

    def slugify(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode(
            "ascii"
        )
        slug = _NON_SLUG_RE.sub("-", normalized.lower()).strip("-")
        return slug or "ticket"

    def _write_ticket(self, ticket: Ticket) -> None:
        meta = dict(ticket.frontmatter)
        meta["task"] = ticket.ticket_id
        meta.pop("ticket", None)
        meta.pop("id", None)

        meta["title"] = ticket.title
        meta["status"] = ticket.status
        if ticket.project:
            meta["project"] = ticket.project
        else:
            meta.pop("project", None)
        if ticket.created:
            meta["created"] = ticket.created
        if ticket.updated:
            meta["updated"] = ticket.updated
        if ticket.priority is not None:
            meta["priority"] = ticket.priority
        else:
            meta.pop("priority", None)
        if ticket.tags:
            meta["tags"] = ticket.tags
        else:
            meta.pop("tags", None)

        rendered = self._render_frontmatter(meta)
        body = ticket.body.rstrip()
        text = f"---\n{rendered}\n---\n"
        if body:
            text += f"\n{body}\n"
        else:
            text += "\n"

        ticket.path.parent.mkdir(parents=True, exist_ok=True)
        ticket.path.write_text(text, encoding="utf-8")

    def _render_frontmatter(self, meta: dict[str, object]) -> str:
        ordered: dict[str, object] = {}
        for key in _PREFERRED_FRONTMATTER_ORDER:
            if key in meta and meta[key] is not None:
                ordered[key] = meta[key]
        for key, value in meta.items():
            if key not in ordered and value is not None:
                ordered[key] = value
        return yaml.safe_dump(ordered, sort_keys=False, allow_unicode=False).strip()

    def _split_frontmatter(self, raw: str) -> tuple[dict[str, object], str]:
        match = _FRONTMATTER_RE.match(raw)
        if not match:
            return {}, raw.lstrip("\r\n")
        frontmatter_text = match.group(1)
        try:
            meta = yaml.safe_load(frontmatter_text) or {}
        except yaml.YAMLError:
            meta = self._parse_simple_frontmatter(frontmatter_text)
        body = raw[match.end() :].lstrip("\r\n")
        return dict(meta), body

    def _parse_simple_frontmatter(self, text: str) -> dict[str, object]:
        meta: dict[str, object] = {}
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            if not key:
                continue
            meta[key] = value.strip()
        return meta

    def _derive_ticket_id(self, meta: dict[str, object], stem: str) -> str:
        stem_identifier, _ = self._split_stem(stem)
        for key in ("task", "ticket", "id"):
            if key in meta and meta[key] is not None:
                raw = str(meta[key]).strip()
                if raw.isdigit() and stem_identifier.isdigit() and int(raw) == int(stem_identifier):
                    return stem_identifier
                return raw
        return stem_identifier

    def _derive_slug(self, stem: str, ticket_id: str, title: object) -> str:
        _, stem_slug = self._split_stem(stem)
        if stem_slug:
            return self.slugify(stem_slug)
        if title:
            return self.slugify(str(title))
        if stem.startswith(f"{ticket_id}-"):
            return stem[len(ticket_id) + 1 :]
        if stem.startswith(f"{ticket_id}_"):
            return stem[len(ticket_id) + 1 :]
        return self.slugify(stem)

    def _split_stem(self, stem: str) -> tuple[str, str | None]:
        for pattern in (_LEGACY_UNDERSCORE_STEM_RE, _CANONICAL_STEM_RE, _NUMERIC_STEM_RE):
            matched = pattern.match(stem)
            if matched:
                return matched.group("identifier"), matched.group("slug")
        return stem.strip(), None

    def _parse_ticket_id(self, ticket_id: str) -> tuple[str | None, int | None]:
        matched = _CANONICAL_ID_RE.match(ticket_id)
        if matched:
            return normalize_prefix(matched.group("prefix")), int(matched.group("number"))
        dotted = _DOTTED_ID_RE.match(ticket_id)
        if dotted:
            digits_only = int("".join(ch for ch in dotted.group("number") if ch.isdigit()))
            return normalize_prefix(dotted.group("prefix")), digits_only
        word_prefix = _WORD_PREFIX_STEM_RE.match(ticket_id)
        if word_prefix:
            return normalize_prefix(word_prefix.group("prefix")), None
        if ticket_id.isdigit():
            return None, int(ticket_id)
        return None, None

    def _canonical_project_key(self, prefix: str | None, meta: dict[str, object]) -> str | None:
        if prefix:
            return prefix
        return self._canonical_project_value(meta.get("project"))

    def _canonical_project_value(self, value: object) -> str | None:
        text = self._str_or_none(value)
        if text is None:
            return None
        if not _CANONICAL_PROJECT_RE.fullmatch(text):
            return None
        return normalize_prefix(text)

    def _normalize_tags(self, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            items = value.split(",")
        elif isinstance(value, list):
            items = value
        else:
            items = [value]
        return [str(item).strip() for item in items if str(item).strip()]

    def _load_companion_assets(self, companion_dir: Path) -> tuple[str, list[str], float]:
        if not companion_dir.is_dir():
            return "", [], 0.0

        chunks: list[str] = []
        asset_paths: list[str] = []
        total_chars = 0
        max_chars = 250_000
        max_file_size = 256 * 1024
        latest_mtime = 0.0

        for asset in sorted(companion_dir.rglob("*")):
            if not asset.is_file():
                continue
            if asset.suffix.lower() not in READABLE_ASSET_SUFFIXES:
                continue
            if asset.stat().st_size > max_file_size:
                continue

            content = asset.read_text(encoding="utf-8", errors="replace").strip()
            if not content:
                continue

            rel_path = asset.relative_to(companion_dir).as_posix()
            block = f"\n\n### {rel_path}\n\n{content}"
            if total_chars + len(block) > max_chars:
                break

            asset_paths.append(rel_path)
            chunks.append(block)
            total_chars += len(block)
            latest_mtime = max(latest_mtime, asset.stat().st_mtime)

        return "".join(chunks).strip(), asset_paths, latest_mtime

    def _ticket_lookup_key(self, value: str) -> str:
        return str(value).strip().upper()

    def _humanize_slug(self, value: str) -> str:
        text = value.replace("_", " ").replace("-", " ").strip()
        return text or "Untitled ticket"

    def _str_or_none(self, value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def task_to_dict(self, ticket: Ticket, *, include_content: bool = True) -> dict[str, object]:
        frontmatter = dict(ticket.frontmatter)
        if "task" not in frontmatter:
            if "ticket" in frontmatter:
                frontmatter["task"] = frontmatter.pop("ticket")
            elif "id" in frontmatter:
                frontmatter["task"] = frontmatter.pop("id")
        frontmatter.pop("ticket", None)
        frontmatter.pop("id", None)
        if ticket.project:
            frontmatter["project"] = ticket.project
        else:
            frontmatter.pop("project", None)
        data = {
            "task_id": ticket.ticket_id,
            "prefix": ticket.prefix,
            "project": ticket.project,
            "number": ticket.number,
            "title": ticket.title,
            "status": ticket.status,
            "priority": ticket.priority,
            "created": ticket.created,
            "updated": ticket.updated,
            "tags": list(ticket.tags),
            "path": str(ticket.path),
            "companion_dir": str(ticket.companion_dir) if ticket.companion_dir else None,
            "frontmatter": frontmatter,
            "asset_paths": list(ticket.asset_paths),
            "source_mtime": ticket.source_mtime,
        }
        if include_content:
            data["body"] = ticket.body
            data["asset_blob"] = ticket.asset_blob
        return data

    def ticket_to_dict(self, ticket: Ticket, *, include_content: bool = True) -> dict[str, object]:
        return self.task_to_dict(ticket, include_content=include_content)

    def plan_migration(self) -> dict[str, object]:
        self.ensure_structure()
        planned_specs: list[dict[str, object]] = []
        skipped: list[dict[str, str]] = []
        unchanged = 0

        for fallback_status, path in self.iter_markdown_files():
            if path.stem.upper() == "README":
                skipped.append({"path": str(path), "reason": "readme_skipped"})
                continue

            ticket = self.parse_ticket(path, fallback_status)
            if ticket is None:
                skipped.append({"path": str(path), "reason": "unparseable_ticket"})
                continue

            spec = self._migration_spec(ticket)
            if spec is None:
                skipped.append({"path": str(path), "reason": "unsupported_identifier"})
                continue
            planned_specs.append(spec)

        reserved: dict[str, set[int]] = {}
        for spec in planned_specs:
            if spec["explicit_number"] is None:
                continue
            reserved.setdefault(spec["prefix"], set()).add(spec["explicit_number"])

        next_numbers = {
            prefix: (max(numbers) + 1 if numbers else 1)
            for prefix, numbers in reserved.items()
        }

        actions: list[MigrationAction] = []
        for spec in sorted(planned_specs, key=lambda item: str(item["ticket"].path).lower()):
            prefix = str(spec["prefix"])
            number = spec["explicit_number"]
            if number is None:
                current = next_numbers.get(prefix, 1)
                used = reserved.setdefault(prefix, set())
                while current in used:
                    current += 1
                number = current
                used.add(number)
                next_numbers[prefix] = current + 1

            target_ticket_id = canonical_ticket_id(prefix, int(number))
            target_slug = str(spec["slug"])
            target_status = str(spec["status"])
            target_path = status_dir(self.tasks_root, target_status) / f"{target_ticket_id}-{target_slug}.md"
            current_companion = spec["ticket"].companion_dir
            target_companion = target_path.with_suffix("") if current_companion else None
            action = MigrationAction(
                current_ticket_id=spec["ticket"].ticket_id,
                current_path=spec["ticket"].path,
                current_status=spec["ticket"].status,
                target_ticket_id=target_ticket_id,
                target_path=target_path,
                target_status=target_status,
                target_prefix=prefix,
                target_number=int(number),
                target_slug=target_slug,
                current_companion_dir=current_companion,
                target_companion_dir=target_companion,
                reason=str(spec["reason"]),
            )

            if not self._migration_action_needed(spec["ticket"], action):
                unchanged += 1
                continue
            actions.append(action)

        filtered_actions, collision_skips = self._filter_collisions(actions)
        skipped.extend(collision_skips)

        return {
            "actions": filtered_actions,
            "skipped": skipped,
            "unchanged": unchanged,
            "can_apply": not any(item["reason"] == "collision" for item in skipped),
        }

    def apply_migration(self) -> dict[str, object]:
        plan = self.plan_migration()
        actions: list[MigrationAction] = plan["actions"]
        skipped = list(plan["skipped"])
        if not actions:
            return {
                "applied": True,
                "migrated_count": 0,
                "actions": [],
                "skipped": skipped,
                "unchanged": plan["unchanged"],
            }

        today = date.today().isoformat()
        id_map = {action.current_ticket_id: action.target_ticket_id for action in actions}
        temp_root = self.settings.index_dir / "_migration_tmp"
        if temp_root.exists():
            shutil.rmtree(temp_root)
        temp_root.mkdir(parents=True, exist_ok=True)

        file_moves: dict[str, Path] = {}
        dir_moves: dict[str, Path] = {}

        for index, action in enumerate(actions):
            if action.current_path != action.target_path and action.current_path.exists():
                temp_path = temp_root / f"{index:04d}.md"
                shutil.move(str(action.current_path), str(temp_path))
                file_moves[str(action.current_path)] = temp_path

            if (
                action.current_companion_dir
                and action.target_companion_dir
                and action.current_companion_dir != action.target_companion_dir
                and action.current_companion_dir.exists()
            ):
                temp_dir = temp_root / f"{index:04d}"
                shutil.move(str(action.current_companion_dir), str(temp_dir))
                dir_moves[str(action.current_companion_dir)] = temp_dir

        for action in actions:
            action.target_path.parent.mkdir(parents=True, exist_ok=True)
            temp_file = file_moves.get(str(action.current_path))
            if temp_file:
                shutil.move(str(temp_file), str(action.target_path))
            if action.target_companion_dir:
                temp_dir = dir_moves.get(str(action.current_companion_dir))
                if temp_dir:
                    shutil.move(str(temp_dir), str(action.target_companion_dir))

        for action in actions:
            ticket = self.parse_ticket(action.target_path, action.target_status)
            if ticket is None:
                continue

            meta = dict(ticket.frontmatter)
            meta.pop("id", None)
            meta["ticket"] = action.target_ticket_id
            meta["title"] = ticket.title
            meta["status"] = action.target_status
            meta["updated"] = today
            if not meta.get("created"):
                meta["created"] = today
            meta["project"] = action.target_prefix

            for key in REFERENCE_FIELDS:
                if key not in meta:
                    continue
                meta[key] = self._rewrite_reference_field(meta[key], id_map)

            ticket.ticket_id = action.target_ticket_id
            ticket.prefix = action.target_prefix
            ticket.number = action.target_number
            ticket.status = action.target_status
            ticket.path = action.target_path
            ticket.stem = action.target_path.stem
            ticket.slug = action.target_slug
            ticket.companion_dir = action.target_companion_dir
            ticket.updated = today
            ticket.project = self._canonical_project_key(action.target_prefix, meta)
            ticket.frontmatter = meta
            self._write_ticket(ticket)

        shutil.rmtree(temp_root, ignore_errors=True)

        return {
            "applied": True,
            "migrated_count": len(actions),
            "actions": [action.to_dict() for action in actions],
            "skipped": skipped,
            "unchanged": plan["unchanged"],
        }

    def _migration_spec(self, ticket: Ticket) -> dict[str, object] | None:
        raw_identifier = ticket.ticket_id.strip()
        dotted = _DOTTED_ID_RE.match(raw_identifier)
        canonical = _CANONICAL_ID_RE.match(raw_identifier)
        migration_slug = self.slugify(ticket.title) if ticket.title else ticket.slug

        if canonical:
            prefix = normalize_prefix(canonical.group("prefix"))
            return {
                "ticket": ticket,
                "prefix": prefix,
                "explicit_number": int(canonical.group("number")),
                "slug": migration_slug,
                "status": ticket.status,
                "reason": "normalize_padding_or_separator",
            }

        if dotted:
            prefix = normalize_prefix(dotted.group("prefix"))
            flattened = int("".join(ch for ch in dotted.group("number") if ch.isdigit()))
            return {
                "ticket": ticket,
                "prefix": prefix,
                "explicit_number": flattened,
                "slug": migration_slug,
                "status": ticket.status,
                "reason": "flatten_dotted_identifier",
            }

        if raw_identifier.isdigit():
            return {
                "ticket": ticket,
                "prefix": normalize_prefix(self.settings.default_prefix),
                "explicit_number": int(raw_identifier),
                "slug": migration_slug,
                "status": ticket.status,
                "reason": "prefixless_numeric_legacy",
            }

        prefix_match = _WORD_PREFIX_STEM_RE.match(ticket.stem)
        if prefix_match:
            return {
                "ticket": ticket,
                "prefix": normalize_prefix(prefix_match.group("prefix")),
                "explicit_number": None,
                "slug": migration_slug,
                "status": ticket.status,
                "reason": "allocate_number_for_prefix_word_ticket",
            }

        return None

    def _migration_action_needed(self, ticket: Ticket, action: MigrationAction) -> bool:
        ticket_key = "ticket" in ticket.frontmatter and "id" not in ticket.frontmatter
        return any(
            [
                ticket.ticket_id != action.target_ticket_id,
                ticket.path != action.target_path,
                ticket.status != action.target_status,
                not ticket_key,
                (ticket.companion_dir or None) != (action.target_companion_dir or None),
            ]
        )

    def _filter_collisions(
        self, actions: list[MigrationAction]
    ) -> tuple[list[MigrationAction], list[dict[str, str]]]:
        skips: list[dict[str, str]] = []
        valid: list[MigrationAction] = []
        by_target_path: dict[Path, MigrationAction] = {}
        by_target_id: dict[str, MigrationAction] = {}
        current_paths = {action.current_path for action in actions}
        current_dirs = {action.current_companion_dir for action in actions if action.current_companion_dir}

        for action in actions:
            conflict = None
            if action.target_ticket_id in by_target_id:
                conflict = f"ticket_id_collision:{action.target_ticket_id}"
            elif action.target_path in by_target_path:
                conflict = f"path_collision:{action.target_path}"
            elif (
                action.target_path.exists()
                and action.target_path not in current_paths
            ):
                conflict = f"existing_path_collision:{action.target_path}"
            elif (
                action.target_companion_dir
                and action.target_companion_dir.exists()
                and action.target_companion_dir not in current_dirs
            ):
                conflict = f"existing_companion_collision:{action.target_companion_dir}"

            if conflict:
                skips.append({"path": str(action.current_path), "reason": "collision", "detail": conflict})
                continue

            by_target_id[action.target_ticket_id] = action
            by_target_path[action.target_path] = action
            valid.append(action)

        return valid, skips

    def _rewrite_reference_field(
        self, value: object, id_map: dict[str, str]
    ) -> object:
        if isinstance(value, list):
            return [self._normalize_reference(item, id_map) for item in value]
        if isinstance(value, str):
            return self._normalize_reference(value, id_map)
        return value

    def _normalize_reference(self, value: object, id_map: dict[str, str]) -> object:
        text = str(value).strip()
        if text in id_map:
            return id_map[text]
        canonical = _CANONICAL_ID_RE.match(text)
        if canonical:
            return canonical_ticket_id(canonical.group("prefix"), int(canonical.group("number")))
        dotted = _DOTTED_ID_RE.match(text)
        if dotted:
            flattened = int("".join(ch for ch in dotted.group("number") if ch.isdigit()))
            return canonical_ticket_id(dotted.group("prefix"), flattened)
        if text.isdigit():
            return canonical_ticket_id(self.settings.default_prefix, int(text))
        return value
