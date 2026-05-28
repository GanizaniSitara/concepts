from __future__ import annotations

import os


def apply_runtime_env() -> None:
    os.environ.setdefault("TASKS_ROOT", r"C:\Users\admin\tasks")
    os.environ.setdefault("TASKS_INDEX_DIR", r"C:\Users\admin\.codex\memories\tasks-mcp\whoosh")
    os.environ.setdefault("TASKS_MCP_HOST", "0.0.0.0")
    os.environ.setdefault("TASKS_MCP_PORT", "8876")
    os.environ.setdefault("TASKS_MCP_TRANSPORT", "streamable-http")
    os.environ.setdefault("TASKS_MCP_DEFAULT_PREFIX", "TASK")
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def main() -> None:
    apply_runtime_env()
    from tasks_mcp.server import main as server_main

    server_main()


if __name__ == "__main__":
    main()
