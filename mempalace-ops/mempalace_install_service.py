"""Install / refresh the MempalaceMCP Windows service.

LifeRunner-pattern: self-elevates to admin, then runs the standard
pywin32 ``stop / remove / install / start`` cycle against
``mempalace_service.py`` (the ``win32serviceutil.ServiceFramework`` subclass
that sits next to this script). Idempotent - if the service already exists,
we tear it down first so this script also doubles as the "refresh after code
change" driver (the LifeRunner ``refresh.cmd`` equivalent).

After ``start`` we poll ``/health`` so success means the server is actually
answering, not just that SCM accepted the start command.

Usage:

    python mempalace_install_service.py
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PYTHON_EXE = Path(r"C:\miniconda3\envs\python312\python.exe")
# The service is registered against the runtime replica, NOT the git working
# tree. See ``deploy.py``. Pulling / branching in git must not perturb what
# SCM points at.
RUNTIME_DIR = Path(r"C:\Tools\mempalace-ops")
SERVICE_PY = RUNTIME_DIR / "mempalace_service.py"
SERVICE_NAME = "MempalaceMCP"
HEALTH_URL = "http://127.0.0.1:8765/health"


def is_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def relaunch_elevated() -> None:
    """Re-exec this script via ShellExecuteW with the ``runas`` verb."""
    params = " ".join(f'"{a}"' for a in [str(Path(__file__).resolve()), *sys.argv[1:]])
    rc = ctypes.windll.shell32.ShellExecuteW(
        None, "runas", str(PYTHON_EXE), params, None, 1
    )
    if rc <= 32:
        print(f"elevation failed (ShellExecuteW rc={rc})", file=sys.stderr)
        sys.exit(1)
    sys.exit(0)


def run(label: str, args: list[str], *, check: bool) -> int:
    print(f"=== {label} ===")
    proc = subprocess.run(args)
    if check and proc.returncode != 0:
        print(f"{label} failed (exit {proc.returncode})", file=sys.stderr)
        sys.exit(proc.returncode)
    return proc.returncode


def sc_query_exists(name: str) -> bool:
    return subprocess.run(["sc.exe", "query", name],
                          capture_output=True).returncode == 0


def wait_for_health(url: str, *, attempts: int = 20, delay: float = 0.5) -> bool:
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                if r.status == 200:
                    print(f"  health OK: {r.read().decode('utf-8', 'replace').strip()}")
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(delay)
    return False


def deploy_to_runtime() -> None:
    """Mirror the sibling source files into ``RUNTIME_DIR`` before install.

    Imports ``deploy`` lazily — it sits next to this script in the source
    tree. In the unusual case that this script is being run from the runtime
    copy itself, there is no sibling ``deploy.py``; fall back to a minimal
    inline mirror that is still a no-op (src == dst).
    """
    here = Path(__file__).resolve().parent
    deploy_py = here / "deploy.py"
    if deploy_py.exists():
        sys.path.insert(0, str(here))
        import deploy  # type: ignore[import-not-found]
        deploy.mirror(src=here, dst=RUNTIME_DIR)
    else:
        # Running from the runtime copy without deploy.py present.
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        print(f"  (no deploy.py sibling; using existing runtime at {RUNTIME_DIR})")


def main() -> int:
    if not is_admin():
        print("Not elevated - relaunching as Administrator...")
        relaunch_elevated()
        return 0  # unreachable

    if not PYTHON_EXE.exists():
        print(f"missing: {PYTHON_EXE}", file=sys.stderr)
        return 1

    print(f"=== deploy -> {RUNTIME_DIR} ===")
    deploy_to_runtime()

    if not SERVICE_PY.exists():
        print(f"missing after deploy: {SERVICE_PY}", file=sys.stderr)
        return 1

    base = [str(PYTHON_EXE), str(SERVICE_PY)]

    if sc_query_exists(SERVICE_NAME):
        print(f"{SERVICE_NAME} already exists - stopping & removing first.")
        run("stop",   base + ["stop"],   check=False)
        run("remove", base + ["remove"], check=False)

    run("install", base + ["--startup", "auto", "install"], check=True)
    run("start",   base + ["start"],                        check=True)

    print("=== verify ===")
    if not wait_for_health(HEALTH_URL):
        print("  health check FAILED - see %USERPROFILE%\\.mempalace\\service_logs\\",
              file=sys.stderr)
        return 2

    print("\nDone. Listening on http://127.0.0.1:8765/mcp")
    print(f"sc query {SERVICE_NAME}  # to confirm STATE")
    input("\nPress Enter to close ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
