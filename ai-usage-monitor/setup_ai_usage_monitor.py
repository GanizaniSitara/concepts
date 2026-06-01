#!/usr/bin/env python3
"""Check and launch the local AI usage monitor."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys


DEFAULT_PORT = "8876"
BROWSER_CANDIDATES = (
    "Brave Browser",
    "Google Chrome",
    "Arc",
    "Microsoft Edge",
    "Google Chrome Canary",
)


class SetupError(Exception):
    """Expected setup failure with a user-facing message."""


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def detect_browser(configured: str | None = None) -> str:
    if configured:
        return configured

    env_value = os.environ.get("AI_USAGE_MONITOR_BROWSER_APP")
    if env_value:
        return env_value

    roots = (Path("/Applications"), Path.home() / "Applications")
    for candidate in BROWSER_CANDIDATES:
        for root in roots:
            if (root / f"{candidate}.app").exists():
                return candidate

    raise SetupError(
        "No supported browser found. Set AI_USAGE_MONITOR_BROWSER_APP "
        "to a Chromium browser app name."
    )


def require_runtime() -> None:
    if shutil.which("osascript") is None:
        raise SetupError("osascript not found; this monitor is intended for macOS")

    scraper = script_dir() / "scraper.py"
    if not scraper.exists():
        raise SetupError(f"scraper.py not found next to {Path(__file__).name}")


def build_env(browser_app: str, port: str) -> dict[str, str]:
    env = os.environ.copy()
    env["AI_USAGE_MONITOR_BROWSER_APP"] = browser_app
    env["AI_USAGE_MONITOR_PORT"] = port
    return env


def print_summary(browser_app: str, port: str) -> None:
    scraper = script_dir() / "scraper.py"
    print(f"Browser: {browser_app}")
    print(f"Port: {port}")
    print(f"Dashboard: http://127.0.0.1:{port}/")
    print()
    print("To run now:")
    print(f'AI_USAGE_MONITOR_BROWSER_APP="{browser_app}" AI_USAGE_MONITOR_PORT={port} python3 {scraper}')


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check and launch the AI usage monitor.")
    parser.add_argument("--run", action="store_true", help="launch scraper.py immediately")
    parser.add_argument(
        "--browser",
        help="Chromium browser app name; defaults to AI_USAGE_MONITOR_BROWSER_APP or auto-detect",
    )
    parser.add_argument(
        "--port",
        default=os.environ.get("AI_USAGE_MONITOR_PORT", DEFAULT_PORT),
        help=f"dashboard port; defaults to AI_USAGE_MONITOR_PORT or {DEFAULT_PORT}",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        require_runtime()
        browser_app = detect_browser(args.browser)
    except SetupError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print_summary(browser_app, str(args.port))
    if args.run:
        scraper = script_dir() / "scraper.py"
        os.execvpe(sys.executable, [sys.executable, str(scraper)], build_env(browser_app, str(args.port)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
