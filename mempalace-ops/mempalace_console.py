"""Console-mode launcher for the mempalace MCP server (work-machine path).

Idempotent: if something is already listening on the bind port, we exit 0
rather than trying to bind a second instance. Otherwise we ``os.execv`` into
``python -m mempalace.mcp_server`` so signals (Ctrl+C) behave the way the
user expects in the console window.
"""

from __future__ import annotations

import os
import socket
import sys

PYTHON_EXE = r"C:\miniconda3\envs\python312\python.exe"
# All interfaces - see mempalace_service.py for the rationale / caveat.
HOST = "0.0.0.0"
PORT = 8765
# You can't ``connect()`` to 0.0.0.0 as a client, so the "is it already up?"
# probe targets loopback instead - a listener on 0.0.0.0 accepts on 127.0.0.1.
PROBE_HOST = "127.0.0.1"


def port_listening(host: str, port: int, timeout: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        try:
            s.connect((host, port))
        except OSError:
            return False
        return True


def main() -> int:
    if port_listening(PROBE_HOST, PORT):
        print(f"mempalace MCP: {PROBE_HOST}:{PORT} is already listening - nothing to do.")
        return 0

    if not os.path.exists(PYTHON_EXE):
        print(f"Python not found at {PYTHON_EXE}", file=sys.stderr)
        return 1

    print(f"Starting mempalace MCP on http://{HOST}:{PORT}/mcp")
    print(f"  Python:  {PYTHON_EXE}")
    print("  Ctrl+C stops the server.\n")

    args = [
        PYTHON_EXE,
        "-m", "mempalace.mcp_server",
        "--host", HOST,
        "--port", str(PORT),
    ]
    os.execv(PYTHON_EXE, args)
    return 0  # unreachable


if __name__ == "__main__":
    sys.exit(main())
