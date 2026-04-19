"""Stop and remove the MempalaceMCP Windows service. Self-elevates."""

from __future__ import annotations

import ctypes
import subprocess
import sys
from pathlib import Path

PYTHON_EXE = Path(r"C:\miniconda3\envs\python312\python.exe")
# Uninstall operates against the runtime replica (see ``deploy.py``).
RUNTIME_DIR = Path(r"C:\Tools\mempalace-ops")
SERVICE_PY = RUNTIME_DIR / "mempalace_service.py"
SERVICE_NAME = "MempalaceMCP"


def is_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def relaunch_elevated() -> None:
    params = " ".join(f'"{a}"' for a in [str(Path(__file__).resolve()), *sys.argv[1:]])
    rc = ctypes.windll.shell32.ShellExecuteW(
        None, "runas", str(PYTHON_EXE), params, None, 1
    )
    if rc <= 32:
        print(f"elevation failed (ShellExecuteW rc={rc})", file=sys.stderr)
        sys.exit(1)
    sys.exit(0)


def main() -> int:
    if not is_admin():
        print("Not elevated - relaunching as Administrator...")
        relaunch_elevated()
        return 0  # unreachable

    if subprocess.run(["sc.exe", "query", SERVICE_NAME],
                      capture_output=True).returncode != 0:
        print(f"{SERVICE_NAME} is not installed - nothing to do.")
        input("\nPress Enter to close ")
        return 0

    base = [str(PYTHON_EXE), str(SERVICE_PY)]
    subprocess.run(base + ["stop"])
    subprocess.run(base + ["remove"])

    print("\nDone.")
    input("\nPress Enter to close ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
