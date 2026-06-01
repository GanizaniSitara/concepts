#!/usr/bin/env python3
"""Install, manage, and remove the local AI review git hook."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any


APP_NAME = "ai-review"
INSTALL_DIR_NAME = ".ai-review"
CONFIG_NAME = "config.json"
MANIFEST_NAME = "install-manifest.json"
HOOK_RELATIVE_PATH = Path("hooks") / "pre-push"
ASSET_NAMES = ("ai-review-server.py", "review-ui.html")
MANIFEST_VERSION = 1
CONFIG_VERSION = 1
HOOK_MARKER = "AI_REVIEW_MANAGED_HOOK"


class SetupError(Exception):
    """An expected setup failure with a user-facing message."""


class Runner:
    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run

    def note(self, message: str) -> None:
        prefix = "DRY RUN: " if self.dry_run else ""
        print(f"{prefix}{message}")

    def mkdir(self, path: Path) -> None:
        self.note(f"create directory {path}")
        if not self.dry_run:
            path.mkdir(parents=True, exist_ok=True)

    def write_text(self, path: Path, text: str) -> None:
        self.note(f"write {path}")
        if not self.dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")

    def copy_file(self, source: Path, target: Path) -> None:
        self.note(f"copy {source} -> {target}")
        if not self.dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    def chmod_executable(self, path: Path) -> None:
        self.note(f"mark executable {path}")
        if not self.dry_run:
            mode = path.stat().st_mode
            path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    def remove_path(self, path: Path) -> None:
        self.note(f"remove {path}")
        if self.dry_run:
            return
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)

    def move(self, source: Path, target: Path) -> None:
        self.note(f"move {source} -> {target}")
        if not self.dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))

    def symlink(self, target: Path, link: Path) -> None:
        self.note(f"symlink {link} -> {target}")
        if not self.dry_run:
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(target)


def install_dir() -> Path:
    override = os.environ.get("AI_REVIEW_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / INSTALL_DIR_NAME


def config_path(base: Path) -> Path:
    return base / CONFIG_NAME


def manifest_path(base: Path) -> Path:
    return base / MANIFEST_NAME


def hook_script_path(base: Path) -> Path:
    return base / HOOK_RELATIVE_PATH


def timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d%H%M%S")


def read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SetupError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SetupError(f"{path} must contain a JSON object")
    return data


def save_json(runner: Runner, path: Path, data: dict[str, Any]) -> None:
    text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    runner.write_text(path, text)


def source_dir() -> Path:
    return Path(__file__).resolve().parent


def asset_paths() -> list[Path]:
    return [source_dir() / name for name in ASSET_NAMES]


def ensure_assets_present() -> None:
    missing = [str(path) for path in asset_paths() if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {item}" for item in missing)
        raise SetupError(f"Required installer asset(s) are missing:\n{formatted}")


def run_git(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=False,
    )


def find_git_root(start: Path) -> Path | None:
    result = run_git(["rev-parse", "--show-toplevel"], start)
    if result.returncode != 0:
        return None
    root = result.stdout.strip()
    if not root:
        return None
    return Path(root).expanduser().resolve()


def require_git_repo(repo: Path) -> Path:
    if not repo.exists():
        raise SetupError(f"Repository path does not exist: {repo}")
    root = find_git_root(repo)
    if root is None:
        raise SetupError(f"Not a git repository: {repo}")
    return root


def git_hook_path(repo: Path) -> Path:
    result = run_git(["rev-parse", "--git-path", "hooks/pre-push"], repo)
    if result.returncode != 0:
        raise SetupError(f"Could not locate git hook path for {repo}: {result.stderr.strip()}")
    raw = result.stdout.strip()
    if not raw:
        raise SetupError(f"git returned an empty hook path for {repo}")
    hook_path = Path(raw)
    if not hook_path.is_absolute():
        hook_path = repo / hook_path
    return hook_path.parent.resolve() / hook_path.name


def empty_config(environment: str | None = None) -> dict[str, Any]:
    return {
        "version": CONFIG_VERSION,
        "environment": environment or detect_environment(),
        "repos": [],
    }


def empty_manifest() -> dict[str, Any]:
    return {
        "version": MANIFEST_VERSION,
        "entries": [],
    }


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    repos = config.get("repos", [])
    normalized: list[dict[str, str]] = []
    if not isinstance(repos, list):
        repos = []
    for item in repos:
        if isinstance(item, str):
            normalized.append({"path": item})
        elif isinstance(item, dict) and isinstance(item.get("path"), str):
            normalized.append({"path": item["path"]})
    config["version"] = CONFIG_VERSION
    config["repos"] = normalized
    return config


def repo_entries(config: dict[str, Any]) -> list[dict[str, str]]:
    return list(config.get("repos", []))


def repo_paths(config: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for entry in repo_entries(config):
        paths.append(Path(entry["path"]).expanduser().resolve())
    return paths


def add_repo_to_config(config: dict[str, Any], repo: Path) -> bool:
    repo = repo.expanduser().resolve()
    existing = {Path(entry["path"]).expanduser().resolve() for entry in repo_entries(config)}
    if repo in existing:
        return False
    config.setdefault("repos", []).append({"path": str(repo)})
    return True


def remove_repo_from_config(config: dict[str, Any], repo: Path) -> bool:
    repo = repo.expanduser().resolve()
    before = repo_entries(config)
    after = [
        entry
        for entry in before
        if Path(entry["path"]).expanduser().resolve() != repo
    ]
    config["repos"] = after
    return len(after) != len(before)


def manifest_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    entries = manifest.get("entries", [])
    if not isinstance(entries, list):
        entries = []
    manifest["version"] = MANIFEST_VERSION
    manifest["entries"] = entries
    return entries


def find_manifest_entry(manifest: dict[str, Any], repo: Path) -> dict[str, Any] | None:
    repo = repo.expanduser().resolve()
    for entry in manifest_entries(manifest):
        raw = entry.get("repo")
        if isinstance(raw, str) and Path(raw).expanduser().resolve() == repo:
            return entry
    return None


def remove_manifest_entry(manifest: dict[str, Any], repo: Path) -> bool:
    repo = repo.expanduser().resolve()
    entries = manifest_entries(manifest)
    kept = []
    removed = False
    for entry in entries:
        raw = entry.get("repo")
        if isinstance(raw, str) and Path(raw).expanduser().resolve() == repo:
            removed = True
        else:
            kept.append(entry)
    manifest["entries"] = kept
    return removed


def detect_environment() -> str:
    distro = os.environ.get("WSL_DISTRO_NAME")
    if distro:
        return "wsl"

    proc_version = Path("/proc/version")
    if proc_version.exists():
        try:
            text = proc_version.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            text = ""
        if "microsoft" in text or "wsl" in text:
            return "wsl"

    hostnamectl = shutil.which("hostnamectl")
    if hostnamectl:
        result = subprocess.run(
            [hostnamectl],
            text=True,
            capture_output=True,
            check=False,
        )
        output = f"{result.stdout}\n{result.stderr}".lower()
        if "ubuntu" in output:
            return "ubuntu"

    system = platform.system().lower()
    if "linux" in system:
        return "linux"
    if "darwin" in system:
        return "mac"
    return system or "unknown"


def check_dependencies() -> None:
    if not shutil.which("git"):
        raise SetupError("git is required to install repository hooks")

    missing_optional = [
        tool
        for tool in ("claude", "opencode", "copilot", "gh")
        if shutil.which(tool) is None
    ]
    if missing_optional:
        print(
            "Warning: optional review tools not found: "
            + ", ".join(missing_optional)
        )


def fail_on_windows_real_install(dry_run: bool) -> None:
    if os.name == "nt" and not dry_run:
        raise SetupError(
            "Windows install is out of scope. Run this from WSL, Ubuntu, Linux, or macOS."
        )


def hook_text() -> str:
    return f"""#!/usr/bin/env python3
# {HOOK_MARKER}
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def main() -> int:
    install_dir = Path(os.environ.get("AI_REVIEW_DIR", Path.home() / "{INSTALL_DIR_NAME}"))
    server = install_dir / "ai-review-server.py"
    if not server.exists():
        print("ai-review: server is missing; allowing push", file=sys.stderr)
        return 0

    stdin_text = sys.stdin.read()
    cmd = [sys.executable, str(server), "--hook", "--repo", os.getcwd()]
    result = subprocess.run(cmd, input=stdin_text, text=True, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
"""


def install_assets(runner: Runner, base: Path) -> None:
    ensure_assets_present()
    runner.mkdir(base)
    runner.mkdir(base / "hooks")
    for asset in asset_paths():
        runner.copy_file(asset, base / asset.name)
    hook = hook_script_path(base)
    runner.write_text(hook, hook_text())
    runner.chmod_executable(hook)


def backup_path_for(hook_path: Path) -> Path:
    base = hook_path.with_name(f"{hook_path.name}.backup.{timestamp()}")
    candidate = base
    counter = 1
    while candidate.exists():
        candidate = hook_path.with_name(f"{hook_path.name}.backup.{timestamp()}.{counter}")
        counter += 1
    return candidate


def paths_same(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except (OSError, RuntimeError):
        return False


def is_our_hook(hook_path: Path, target: Path) -> bool:
    if hook_path.is_symlink():
        return paths_same(hook_path, target)
    if hook_path.is_file():
        try:
            text = hook_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return False
        return HOOK_MARKER in text
    return False


def install_hook_for_repo(
    runner: Runner,
    repo: Path,
    manifest: dict[str, Any],
    hook_target: Path,
) -> None:
    repo = require_git_repo(repo)
    hook_path = git_hook_path(repo)
    runner.mkdir(hook_path.parent)

    entry = find_manifest_entry(manifest, repo)
    backup: str | None = None
    if entry and isinstance(entry.get("backup"), str):
        backup = entry["backup"]

    exists = hook_path.exists() or hook_path.is_symlink()
    if exists and is_our_hook(hook_path, hook_target):
        runner.remove_path(hook_path)
    elif exists:
        backup_candidate = backup_path_for(hook_path)
        runner.move(hook_path, backup_candidate)
        backup = str(backup_candidate)

    runner.symlink(hook_target, hook_path)

    new_entry = {
        "repo": str(repo),
        "hook_path": str(hook_path),
        "target": str(hook_target),
        "backup": backup,
        "installed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    if entry is None:
        manifest_entries(manifest).append(new_entry)
    else:
        entry.clear()
        entry.update(new_entry)


def uninstall_hook_entry(runner: Runner, entry: dict[str, Any]) -> bool:
    repo = entry.get("repo")
    hook_raw = entry.get("hook_path")
    target_raw = entry.get("target")
    if not isinstance(repo, str) or not isinstance(hook_raw, str):
        print("Warning: skipping malformed manifest entry")
        return False

    hook_path = Path(hook_raw).expanduser()
    target = Path(target_raw).expanduser() if isinstance(target_raw, str) else hook_script_path(install_dir())
    backup_raw = entry.get("backup")
    backup = Path(backup_raw).expanduser() if isinstance(backup_raw, str) else None

    exists = hook_path.exists() or hook_path.is_symlink()
    if exists and not is_our_hook(hook_path, target):
        print(f"Warning: leaving changed hook in place for {repo}: {hook_path}")
        return False

    if exists:
        runner.remove_path(hook_path)

    if backup is not None:
        if backup.exists() or runner.dry_run:
            if runner.dry_run or not hook_path.exists():
                runner.move(backup, hook_path)
            else:
                print(f"Warning: cannot restore backup because hook exists: {hook_path}")
                return False
        else:
            print(f"Warning: backup listed in manifest is missing: {backup}")

    return True


def load_config(base: Path, environment: str | None = None) -> dict[str, Any]:
    return normalize_config(read_json(config_path(base), empty_config(environment)))


def load_manifest(base: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path(base), empty_manifest())
    manifest_entries(manifest)
    return manifest


def command_install(args: argparse.Namespace) -> int:
    fail_on_windows_real_install(args.dry_run)
    check_dependencies()
    base = install_dir()
    environment = args.env or detect_environment()
    runner = Runner(args.dry_run)

    install_assets(runner, base)
    config = load_config(base, environment)
    config["environment"] = environment
    manifest = load_manifest(base)

    current_repo = find_git_root(Path.cwd())
    if current_repo is not None:
        if add_repo_to_config(config, current_repo):
            runner.note(f"register repository {current_repo}")
    elif not args.non_interactive:
        answer = input("No git repository found from the current directory. Add one now? [path/blank] ").strip()
        if answer:
            repo = require_git_repo(Path(answer).expanduser())
            if add_repo_to_config(config, repo):
                runner.note(f"register repository {repo}")

    hook_target = hook_script_path(base)
    for repo in repo_paths(config):
        install_hook_for_repo(runner, repo, manifest, hook_target)

    save_json(runner, config_path(base), config)
    save_json(runner, manifest_path(base), manifest)
    print(f"Installed {APP_NAME} in {base}")
    return 0


def command_uninstall(args: argparse.Namespace) -> int:
    base = install_dir()
    runner = Runner(args.dry_run)
    manifest_file = manifest_path(base)

    if not base.exists():
        print(f"{APP_NAME} is not installed; nothing to do")
        return 0

    manifest = load_manifest(base) if manifest_file.exists() else empty_manifest()
    all_removed = True
    for entry in list(manifest_entries(manifest)):
        if uninstall_hook_entry(runner, entry):
            continue
        all_removed = False

    if base.exists() or args.dry_run:
        runner.remove_path(base)

    if all_removed:
        print(f"Uninstalled {APP_NAME}")
    else:
        print(f"Uninstalled {APP_NAME} with warnings; review changed hooks manually")
    return 0


def command_list(args: argparse.Namespace) -> int:
    base = install_dir()
    config = load_config(base) if config_path(base).exists() else empty_config()
    repos = repo_paths(config)
    if not repos:
        print("No repositories registered.")
        return 0

    hook_target = hook_script_path(base)
    for repo in repos:
        if not repo.exists():
            status = "missing repo"
            hook_path = "(unknown)"
        else:
            try:
                hook_path_obj = git_hook_path(repo)
            except SetupError:
                status = "not a git repo"
                hook_path = "(unknown)"
            else:
                hook_path = str(hook_path_obj)
                if is_our_hook(hook_path_obj, hook_target):
                    status = "installed"
                elif hook_path_obj.exists() or hook_path_obj.is_symlink():
                    status = "different hook"
                else:
                    status = "missing hook"
        print(f"{repo} [{status}] {hook_path}")
    return 0


def command_add(args: argparse.Namespace) -> int:
    fail_on_windows_real_install(args.dry_run)
    check_dependencies()
    base = install_dir()
    if not hook_script_path(base).exists() and not args.dry_run:
        raise SetupError(f"{APP_NAME} is not installed; run install first")

    repo = require_git_repo(Path(args.repo).expanduser())
    runner = Runner(args.dry_run)
    config = load_config(base)
    manifest = load_manifest(base)
    added = add_repo_to_config(config, repo)
    install_hook_for_repo(runner, repo, manifest, hook_script_path(base))
    save_json(runner, config_path(base), config)
    save_json(runner, manifest_path(base), manifest)
    print(("Added" if added else "Updated") + f" {repo}")
    return 0


def command_remove(args: argparse.Namespace) -> int:
    base = install_dir()
    runner = Runner(args.dry_run)
    repo = Path(args.repo).expanduser().resolve()
    if not base.exists() and not args.dry_run:
        print(f"{APP_NAME} is not installed; nothing to do")
        return 0

    config = load_config(base) if config_path(base).exists() else empty_config()
    manifest = load_manifest(base) if manifest_path(base).exists() else empty_manifest()

    removed_config = remove_repo_from_config(config, repo)
    entry = find_manifest_entry(manifest, repo)
    removed_hook = True
    if entry is not None:
        removed_hook = uninstall_hook_entry(runner, entry)
        if removed_hook:
            remove_manifest_entry(manifest, repo)

    save_json(runner, config_path(base), config)
    save_json(runner, manifest_path(base), manifest)
    if removed_config or entry is not None:
        print(f"Removed {repo}")
    else:
        print(f"{repo} was not registered")
    if not removed_hook:
        print("Warning: hook was changed and left in place")
    return 0


def command_reinstall_hooks(args: argparse.Namespace) -> int:
    fail_on_windows_real_install(args.dry_run)
    check_dependencies()
    base = install_dir()
    if not hook_script_path(base).exists() and not args.dry_run:
        raise SetupError(f"{APP_NAME} is not installed; run install first")

    runner = Runner(args.dry_run)
    config = load_config(base)
    manifest = load_manifest(base)
    for repo in repo_paths(config):
        install_hook_for_repo(runner, repo, manifest, hook_script_path(base))
    save_json(runner, manifest_path(base), manifest)
    print("Reinstalled hooks")
    return 0


def extract_global_dry_run(argv: list[str]) -> tuple[bool, list[str]]:
    dry_run = False
    cleaned: list[str] = []
    for arg in argv:
        if arg == "--dry-run":
            dry_run = True
        else:
            cleaned.append(arg)
    return dry_run, cleaned


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Install and manage the AI review git hook.",
        epilog="Global --dry-run may appear before or after a subcommand.",
    )
    parser.add_argument("--dry-run", action="store_true", help="show changes without writing")
    subparsers = parser.add_subparsers(dest="command")

    install = subparsers.add_parser("install", help="install files and wire registered hooks")
    install.add_argument("--env", choices=("wsl", "ubuntu", "linux", "mac"), help="override environment detection")
    install.add_argument("--non-interactive", action="store_true", help="do not prompt for a repo path")
    install.set_defaults(func=command_install)

    uninstall = subparsers.add_parser("uninstall", help="remove install files and restore hooks")
    uninstall.set_defaults(func=command_uninstall)

    list_cmd = subparsers.add_parser("list", help="list registered repositories")
    list_cmd.set_defaults(func=command_list)

    add = subparsers.add_parser("add", help="register a repository and install its hook")
    add.add_argument("repo", help="path to a git repository")
    add.set_defaults(func=command_add)

    remove = subparsers.add_parser("remove", help="unregister a repository and restore its hook")
    remove.add_argument("repo", help="path to a registered repository")
    remove.set_defaults(func=command_remove)

    reinstall = subparsers.add_parser("reinstall-hooks", help="reinstall hooks for registered repositories")
    reinstall.set_defaults(func=command_reinstall_hooks)
    return parser


def parse_args(argv: list[str]) -> argparse.Namespace:
    dry_run, cleaned = extract_global_dry_run(argv)
    if not cleaned:
        cleaned = ["install"]
    parser = build_parser()
    args = parser.parse_args(cleaned)
    args.dry_run = bool(getattr(args, "dry_run", False) or dry_run)
    if not hasattr(args, "func"):
        parser.error("a subcommand is required")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        return int(args.func(args))
    except SetupError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
