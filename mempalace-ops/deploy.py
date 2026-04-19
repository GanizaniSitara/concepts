"""Mirror the mempalace-ops source tree to ``C:\\Tools\\mempalace-ops``.

Rationale: git is the source of truth, but we don't want the Windows service
binary-path or the work-machine console launcher pointing inside a working
tree - pulling / switching branches would yank the running tool out from
under itself. So we keep a stable runtime copy in ``C:\\Tools\\`` and the
installers register / launch from there.

Usage::

    python deploy.py           # mirror src -> C:\\Tools\\mempalace-ops
    python -c "from deploy import mirror; mirror()"  # same, from code

Idempotent. Running it from within ``C:\\Tools\\mempalace-ops`` is a no-op.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

DEPLOY_DST = Path(r"C:\Tools\mempalace-ops")

# Files we ship to the runtime copy. Everything else (README extras, tests,
# deploy.py itself) stays in git.
_SHIPPED = (
    "mempalace_service.py",
    "mempalace_console.py",
    "mempalace_install_service.py",
    "mempalace_uninstall_service.py",
    "README.md",
)

_REFRESH_CMD_BODY = (
    "@echo off\r\n"
    "REM Local convenience wrapper (runtime copy, not under version control).\r\n"
    "REM Double-click to refresh the MempalaceMCP service.\r\n"
    '"C:\\miniconda3\\envs\\python312\\python.exe" "%~dp0mempalace_install_service.py"\r\n'
)


def mirror(src: Path | None = None, dst: Path = DEPLOY_DST) -> Path:
    """Copy the shipped files from ``src`` (defaults to this file's folder)
    to ``dst``. Creates ``dst`` and drops a double-click ``.cmd`` wrapper
    alongside the Python scripts.

    Returns ``dst``. If ``src`` and ``dst`` resolve to the same folder, skip
    the copy step and just ensure the ``.cmd`` wrapper is present.
    """
    if src is None:
        src = Path(__file__).resolve().parent
    src = src.resolve()
    dst = dst.resolve()

    dst.mkdir(parents=True, exist_ok=True)

    if src != dst:
        for name in _SHIPPED:
            s = src / name
            if not s.exists():
                print(f"  skip (missing): {s}")
                continue
            d = dst / name
            shutil.copy2(s, d)
            print(f"  copied: {name}")
    else:
        print(f"  src == dst ({src}); skipping copy")

    cmd_path = dst / "mempalace-refresh.cmd"
    if not cmd_path.exists():
        cmd_path.write_bytes(_REFRESH_CMD_BODY.encode("ascii"))
        print(f"  wrote:  mempalace-refresh.cmd")

    return dst


def main() -> int:
    print(f"mempalace-ops deploy -> {DEPLOY_DST}")
    dst = mirror()
    print(f"\nDone. Runtime copy at: {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
