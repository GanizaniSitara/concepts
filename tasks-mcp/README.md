# Tasks MCP

`tasks-mcp` is a FastMCP server for a markdown-backed task corpus.

It keeps two layers separate:

- Markdown files are the source of truth.
- `Whoosh` is a derived full-text index for fast task and companion-doc search.

The server is intended for local task workflows shared across coding agents and MCP clients.

## Storage Model

Tasks live under a configurable root with one folder per status:

```text
tasks-root/
  backlog/
  in-progress/
  blocked/
  done/
```

New tasks use a canonical basename such as `PROJ-008-example-task.md`.

Each task may have a companion folder with the same stem:

```text
PROJ-008-example-task.md
PROJ-008-example-task/
  DESIGN.md
  NOTES.md
```

Moving a task between statuses moves both the markdown file and its companion folder.

## Legacy Migration

The migration tool rewrites mixed legacy forms such as:

- `PROJ-8_example-old-form.md`
- `001-title-keyword-matrix.md`
- `TASK_legacy-task-name.md`
- `PROJ-5.10_import-saved-searches.md`

into a canonical `PREFIX-NNN-slug` scheme in place, preserving companion folders and normalizing references where possible.

## Tools

- `task_summary`
- `search_tasks`
- `get_task`
- `create_task`
- `update_task`
- `move_task`
- `reopen_task`
- `delete_task`
- `list_task_duplicates`
- `append_task_note`
- `rebuild_task_index`
- `migrate_legacy_tasks`
- `lint_tasks`
- `pivot_tasks`
- `repair_tasks`

## Defaults

- `TASKS_ROOT`: `~/tasks`
- `TASKS_INDEX_DIR`: `<repo>/.data/whoosh`
- `TASKS_MCP_HOST`: `127.0.0.1`
- `TASKS_MCP_PORT`: `8876`
- `TASKS_MCP_TRANSPORT`: `streamable-http`
- `TASKS_MCP_DEFAULT_PREFIX`: `TASK`

The repo-local `.data/` directory is derived state only and can be rebuilt.

## Install

```text
python -m venv .venv
.venv\Scripts\python -m pip install -U pip
.venv\Scripts\python -m pip install -e .
```

## Run HTTP Server

On Windows `cmd.exe`:

```text
set TASKS_ROOT=%USERPROFILE%\tasks
.venv\Scripts\python run_http.py
```

This serves MCP over:

```text
http://127.0.0.1:8876/mcp
```

## Run With Stdio

```text
.venv\Scripts\python -m tasks_mcp.server --transport stdio --tasks-root %USERPROFILE%\tasks
```

## Optional Windows Service Wrapper

The repo includes Windows-only helper scripts:

- `run_http.py` for foreground HTTP execution
- `tasks_mcp_service.py` for the service host
- `refresh_service.py` to reinstall and restart the service

If you want the Windows service mode, install `pywin32` into the same environment first.

## Client Registration

Example HTTP registration:

```text
claude mcp add --transport http --scope user tasks http://127.0.0.1:8876/mcp
```

Any MCP-capable client can use the same endpoint.

## Notes

- The server refreshes the derived index on demand using current markdown and companion-doc content.
- Companion-file content is indexed for search, subject to file-type and size limits.
- v1 has no MemPalace integration and no external database dependency.
