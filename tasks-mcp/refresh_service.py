from __future__ import annotations

from pathlib import Path
import subprocess
import sys


SERVICE_SCRIPT = Path(__file__).with_name("tasks_mcp_service.py")
PYTHON = Path(sys.executable)


def run_service_command(*args: str) -> None:
    subprocess.run(
        [str(PYTHON), str(SERVICE_SCRIPT), *args],
        check=False,
    )


def main() -> int:
    run_service_command("stop")
    run_service_command("remove")
    completed = subprocess.run(
        [str(PYTHON), str(SERVICE_SCRIPT), "--startup", "auto", "install"],
        check=False,
    )
    if completed.returncode != 0:
        return completed.returncode

    completed = subprocess.run(
        [str(PYTHON), str(SERVICE_SCRIPT), "start"],
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
