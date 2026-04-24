from __future__ import annotations

import os

from tasks_mcp.config import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_PREFIX,
    DEFAULT_TRANSPORT,
    default_index_dir,
    default_tasks_root,
)


def apply_runtime_env() -> None:
    os.environ.setdefault("TASKS_ROOT", str(default_tasks_root()))
    os.environ.setdefault("TASKS_INDEX_DIR", str(default_index_dir()))
    os.environ.setdefault("TASKS_MCP_HOST", DEFAULT_HOST)
    os.environ.setdefault("TASKS_MCP_PORT", str(DEFAULT_PORT))
    os.environ.setdefault("TASKS_MCP_TRANSPORT", DEFAULT_TRANSPORT)
    os.environ.setdefault("TASKS_MCP_DEFAULT_PREFIX", DEFAULT_PREFIX)
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def main() -> None:
    apply_runtime_env()
    from tasks_mcp.server import main as server_main

    server_main()


if __name__ == "__main__":
    main()
