#!/usr/bin/env python3
"""Tiny local server used by the AI review hook."""

from __future__ import annotations

import argparse
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
from urllib.parse import urlparse


INSTALL_DIR_NAME = ".ai-review"


def install_dir() -> Path:
    override = os.environ.get("AI_REVIEW_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / INSTALL_DIR_NAME


def read_config() -> dict:
    path = install_dir() / "config.json"
    if not path.exists():
        return {"repos": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"repos": []}


def run_hook(repo: str) -> int:
    config = read_config()
    repos = [item.get("path") for item in config.get("repos", []) if isinstance(item, dict)]
    if repo not in repos:
        print(f"ai-review: {repo} is not registered; allowing push", file=sys.stderr)
        return 0
    print(f"ai-review: hook active for {repo}")
    return 0


class Handler(BaseHTTPRequestHandler):
    server_version = "AIReviewHTTP/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self.send_file(install_dir() / "review-ui.html", "text/html; charset=utf-8")
            return
        if parsed.path == "/config.json":
            self.send_json(read_config())
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def send_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, data: dict) -> None:
        raw = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, format: str, *args: object) -> None:
        print(f"ai-review-server: {format % args}", file=sys.stderr)


def serve(port: int) -> int:
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"ai-review-server: http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local AI review helper.")
    parser.add_argument("--hook", action="store_true", help="run in git hook mode")
    parser.add_argument("--repo", default=os.getcwd(), help="repository path for hook mode")
    parser.add_argument("--port", type=int, default=8768, help="local server port")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.hook:
        return run_hook(str(Path(args.repo).expanduser().resolve()))
    return serve(args.port)


if __name__ == "__main__":
    raise SystemExit(main())
