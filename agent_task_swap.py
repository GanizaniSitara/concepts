#!/usr/bin/env python3
"""
Relaunch a task under Claude, Codex, Copilot, or OpenCode.

This is a Python replacement for local swap.ps1 / swap-ubuntu.sh workflows:
find a task, stop the previous recorded launch, then start the selected agent.
Use --bypass to add each tool's full-auto permission flag where one is known.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class ToolConfig:
    label: str
    prompt_style: str
    bypass_flag: str | None = None
    supports_name: bool = False
    extra_args: tuple[str, ...] = ()


TOOL_CONFIG: dict[str, ToolConfig] = {
    "claude": ToolConfig(
        label="Claude",
        prompt_style="positional",
        bypass_flag="--dangerously-skip-permissions",
        supports_name=True,
    ),
    "codex": ToolConfig(
        label="Codex",
        prompt_style="positional",
        bypass_flag="--dangerously-bypass-approvals-and-sandbox",
        extra_args=("-c", "tui.terminal_title=['thread']"),
    ),
    "copilot": ToolConfig(
        label="Copilot",
        prompt_style="interactive-flag",
        bypass_flag="--allow-all",
    ),
    "opencode": ToolConfig(
        label="OpenCode",
        prompt_style="prompt-flag",
    ),
}


def default_task_root() -> Path:
    return Path(os.environ.get("TASK_ROOT", Path.home() / "tasks")).expanduser()


def normalize_slug(raw: str) -> str:
    return raw[5:] if raw.upper().startswith("TASK_") else raw


def find_task_in_dir(directory: Path, needle: str) -> Path | None:
    if not directory.exists():
        return None

    exact = sorted(p for p in directory.glob(f"{needle}*.md") if p.is_file())
    if exact:
        return exact[0]

    matches = sorted(p for p in directory.glob(f"*{needle}*.md") if p.is_file())
    if len(matches) > 1:
        names = ", ".join(p.name for p in matches)
        raise SystemExit(f"Ambiguous task slug '{needle}': {names}")
    return matches[0] if matches else None


def move_task_to_progress(task: Path, target: Path) -> None:
    if target.exists():
        raise SystemExit(f"Refusing to overwrite existing task: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(task), str(target))

    companion = task.with_suffix("")
    target_companion = target.with_suffix("")
    if companion.is_dir():
        if target_companion.exists():
            raise SystemExit(f"Refusing to overwrite existing companion folder: {target_companion}")
        shutil.move(str(companion), str(target_companion))


def resolve_task(task_root: Path, slug: str, *, move_backlog: bool, dry_run: bool) -> tuple[Path, bool]:
    progress = task_root / "in-progress"
    backlog = task_root / "backlog"

    task = find_task_in_dir(progress, slug)
    if task:
        return task, False

    task = find_task_in_dir(backlog, slug)
    if not task:
        raise SystemExit(f"No task matching '{slug}' under {task_root}")

    target = progress / task.name
    if not move_backlog:
        raise SystemExit(f"Task is still in backlog: {task}. Re-run without --no-move-backlog to move it.")
    if not dry_run:
        move_task_to_progress(task, target)
    return target, True


def info_path_for(task_root: Path, task_file: Path) -> Path:
    return task_root / "in-progress" / f"{task_file.stem}.launch-info.json"


def read_launch_info(info_path: Path) -> dict[str, object]:
    if not info_path.exists():
        return {}
    try:
        return json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read previous launch info at {info_path}: {exc}") from exc


def wait_until_dead(pid: int, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not process_exists(pid):
            return True
        time.sleep(0.1)
    return not process_exists(pid)


def process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        handle = ctypes.windll.kernel32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
        if not handle:
            return False
        try:
            result = ctypes.windll.kernel32.WaitForSingleObject(handle, 0)
            return result == 0x00000102  # WAIT_TIMEOUT
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def terminate_windows_pid(pid: int, timeout_seconds: float) -> bool:
    process_terminate = 0x0001
    synchronize = 0x00100000
    handle = ctypes.windll.kernel32.OpenProcess(process_terminate | synchronize, False, pid)
    if not handle:
        return False
    try:
        ctypes.windll.kernel32.TerminateProcess(handle, 1)
        result = ctypes.windll.kernel32.WaitForSingleObject(handle, int(timeout_seconds * 1000))
        return result != 0x00000102  # WAIT_TIMEOUT
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def stop_previous_launch(info: dict[str, object], timeout_seconds: float) -> None:
    pid_value = info.get("PID") or info.get("Pid") or info.get("pid")
    if not pid_value:
        return

    try:
        pid = int(pid_value)
    except (TypeError, ValueError):
        print(f"Ignoring invalid recorded PID: {pid_value}", file=sys.stderr)
        return

    if not process_exists(pid):
        print(f"Previous PID {pid} is not running.")
        return

    print(f"Stopping previous {info.get('Tool', 'agent')} session PID {pid}...")
    if os.name == "nt":
        if not terminate_windows_pid(pid, timeout_seconds):
            print(f"PID {pid} did not exit cleanly after terminate request.", file=sys.stderr)
        return

    os.kill(pid, signal.SIGTERM)
    if wait_until_dead(pid, timeout_seconds):
        return
    os.kill(pid, signal.SIGKILL)
    wait_until_dead(pid, timeout_seconds)


def candidate_paths(tool: str) -> list[str]:
    home = Path.home()
    appdata = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming"))
    npm_bin = appdata / "npm"

    if os.name != "nt":
        return [tool]

    if tool == "claude":
        return [
            str(npm_bin / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"),
            str(npm_bin / "claude.cmd"),
            "claude",
        ]
    if tool == "codex":
        return [
            str(home / "bin" / "codex.cmd"),
            str(npm_bin / "codex.cmd"),
            "codex",
        ]
    if tool == "copilot":
        return [
            str(npm_bin / "copilot.cmd"),
            "copilot",
        ]
    if tool == "opencode":
        return [
            r"C:\npm\opencode.cmd",
            "opencode",
        ]
    return [tool]


def resolve_executable(tool: str, override: str | None) -> str:
    if override:
        return override

    env_key = f"AGENT_SWAP_{tool.upper()}_EXE"
    if os.environ.get(env_key):
        return os.environ[env_key]

    for candidate in candidate_paths(tool):
        path = Path(candidate)
        if path.exists():
            return str(path)
        found = shutil.which(candidate)
        if found:
            return found
    return tool


def build_command(tool: str, task_slug: str, task_file: Path, *, bypass: bool, exe: str) -> list[str]:
    cfg = TOOL_CONFIG[tool]
    brief = f"Read the task file at {task_file} and begin working on it."
    argv = [exe, *cfg.extra_args]

    if bypass and cfg.bypass_flag:
        argv.append(cfg.bypass_flag)
    if cfg.supports_name:
        argv.extend(["--name", task_slug])

    if cfg.prompt_style == "interactive-flag":
        argv.extend(["-i", brief])
    elif cfg.prompt_style == "prompt-flag":
        argv.extend(["--prompt", brief])
    else:
        argv.append(brief)
    return argv


def printable_command(argv: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(argv))
    import shlex

    return shlex.join(argv)


def launch(argv: Sequence[str], cwd: Path) -> subprocess.Popen[bytes]:
    if os.name == "nt":
        return subprocess.Popen(
            list(argv),
            cwd=str(cwd),
            creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
        )
    return subprocess.Popen(list(argv), cwd=str(cwd), start_new_session=True)


def write_launch_info(info_path: Path, task_slug: str, tool: str, process: subprocess.Popen[bytes]) -> None:
    info_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "TaskSlug": task_slug,
        "Tool": tool,
        "PID": process.pid,
        "LaunchedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Launcher": "agent_task_swap.py",
        "Platform": platform.platform(),
    }
    info_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slug", help="Task slug or partial slug, with or without TASK_ prefix.")
    parser.add_argument("tool", choices=sorted(TOOL_CONFIG), help="Agent CLI to launch.")
    parser.add_argument("--bypass", action="store_true", help="Add the tool's full-auto permission flag.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without stopping or launching.")
    parser.add_argument("--task-root", type=Path, default=default_task_root(), help="Task root; defaults to TASK_ROOT or ~/tasks.")
    parser.add_argument("--workdir", type=Path, default=Path.home(), help="Working directory for the launched agent.")
    parser.add_argument("--exe", help="Override the executable path for the selected tool.")
    parser.add_argument("--stop-timeout", type=float, default=2.0, help="Seconds to wait for a previous PID to exit.")
    parser.add_argument(
        "--no-move-backlog",
        action="store_true",
        help="Fail instead of moving a matching backlog task into in-progress.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    slug = normalize_slug(args.slug)
    task_root = args.task_root.expanduser().resolve()
    workdir = args.workdir.expanduser().resolve()

    task_file, would_move = resolve_task(
        task_root,
        slug,
        move_backlog=not args.no_move_backlog,
        dry_run=args.dry_run,
    )
    task_slug = task_file.stem
    info_path = info_path_for(task_root, task_file)
    previous_info = read_launch_info(info_path)
    exe = resolve_executable(args.tool, args.exe)
    command = build_command(args.tool, task_slug, task_file, bypass=args.bypass, exe=exe)

    if args.dry_run:
        print(f"Task: {task_file}")
        if would_move:
            print(f"Would move backlog task to: {task_file}")
        print(f"Previous launch info: {info_path}")
        if previous_info.get("PID"):
            print(f"Would stop previous PID: {previous_info['PID']}")
        print(f"Would run: {printable_command(command)}")
        return 0

    stop_previous_launch(previous_info, args.stop_timeout)
    process = launch(command, workdir)
    write_launch_info(info_path, task_slug, args.tool, process)
    print(f"Relaunched {task_slug} with {args.tool} (PID {process.pid}).")
    print(f"Launch info: {info_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
