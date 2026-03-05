#!/usr/bin/env python3
"""Build script for TinyMD editor prototypes.

Usage:
    python build.py              # dev build (all targets)
    python build.py --prod       # production build (optimized, no console)
    python build.py --dev        # dev build (console window kept for crash output)
    python build.py gdi d2d      # build specific targets only
    python build.py --prod wv    # production build, WebView2 only
"""

import argparse
import os
import subprocess
import sys
import time

TARGETS = {
    "wv":       {"dir": "webview",   "exe": "tinymd-webview.exe",   "label": "WebView2"},
    "gdi":      {"dir": "gdi",       "exe": "tinymd-gdi.exe",       "label": "GDI"},
    "d2d":      {"dir": "direct2d",  "exe": "tinymd-d2d.exe",       "label": "Direct2D"},
    "richedit": {"dir": "richedit",  "exe": "tinymd-richedit.exe",  "label": "RichEdit (deprecated)"},
}

ACTIVE_TARGETS = ["wv", "gdi", "d2d"]  # richedit excluded by default


def build(target_key, prod=False):
    t = TARGETS[target_key]
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), t["dir"])
    exe_path = os.path.join(src_dir, t["exe"])

    ldflags = "-s -w -H windowsgui" if prod else ""
    trimpath = "-trimpath" if prod else ""

    cmd = ["go", "build"]
    if ldflags:
        cmd += ["-ldflags", ldflags]
    if trimpath:
        cmd.append(trimpath)
    cmd += ["-o", exe_path, "."]

    label = t["label"]
    mode = "prod" if prod else "dev"
    print(f"  [{mode}] {label:20s} -> {t['exe']}", end="", flush=True)

    start = time.time()
    result = subprocess.run(cmd, cwd=src_dir, capture_output=True, text=True)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"  FAILED ({elapsed:.1f}s)")
        print(result.stderr)
        return False

    size_mb = os.path.getsize(exe_path) / (1024 * 1024)
    print(f"  {size_mb:.1f} MB  ({elapsed:.1f}s)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Build TinyMD editor prototypes")
    parser.add_argument("--prod", action="store_true",
                        help="Production build: strip symbols, hide console window")
    parser.add_argument("--dev", action="store_true",
                        help="Dev build: keep console window for crash diagnostics (default)")
    parser.add_argument("--all", action="store_true",
                        help="Include deprecated richedit target")
    parser.add_argument("targets", nargs="*",
                        help=f"Targets to build: {', '.join(TARGETS.keys())} (default: {' '.join(ACTIVE_TARGETS)})")
    args = parser.parse_args()

    prod = args.prod

    if args.targets:
        keys = []
        for t in args.targets:
            if t not in TARGETS:
                print(f"Unknown target '{t}'. Available: {', '.join(TARGETS.keys())}")
                sys.exit(1)
            keys.append(t)
    elif args.all:
        keys = list(TARGETS.keys())
    else:
        keys = ACTIVE_TARGETS

    mode = "PRODUCTION" if prod else "DEVELOPMENT"
    print(f"\nBuilding TinyMD ({mode})")
    if not prod:
        print("  (console window kept for crash output — use --prod for release builds)\n")
    else:
        print("  (symbols stripped, console hidden, trimpath enabled)\n")

    ok = 0
    fail = 0
    for k in keys:
        if build(k, prod):
            ok += 1
        else:
            fail += 1

    print(f"\nDone: {ok} succeeded, {fail} failed")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
