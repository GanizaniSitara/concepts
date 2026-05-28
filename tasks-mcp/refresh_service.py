from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time


SERVICE_SCRIPT = Path(__file__).with_name("tasks_mcp_service.py")
PYTHON = Path(sys.executable)
LEGACY_SERVICE_NAME = "TasksMcpHttp"


def run_service_command(*args: str) -> None:
    subprocess.run(
        [str(PYTHON), str(SERVICE_SCRIPT), *args],
        check=False,
    )


def cleanup_legacy() -> None:
    """One-shot rename cleanup: tear down pre-rename TasksMcpHttp if present."""
    probe = subprocess.run(
        ["sc.exe", "query", LEGACY_SERVICE_NAME],
        capture_output=True, check=False,
    )
    if probe.returncode != 0:
        return
    print(f"=== one-shot: removing legacy service {LEGACY_SERVICE_NAME} ===")
    subprocess.run(["sc.exe", "stop", LEGACY_SERVICE_NAME], check=False)
    time.sleep(2)
    subprocess.run(["sc.exe", "delete", LEGACY_SERVICE_NAME], check=False)


def main() -> int:
    cleanup_legacy()
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
