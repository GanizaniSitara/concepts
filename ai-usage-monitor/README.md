# AI usage monitor

Small local dashboard that polls open Chromium-browser tabs for usage data from:

- Claude usage settings
- Codex usage analytics
- GitHub Copilot settings

It serves a simple dashboard and JSON API on `http://127.0.0.1:8876/` by default.

## Features

- Sliding-window bars with a live pace marker
- Reset labels with short weekday codes such as `MON`, `TUE`, `FRI`
- JSON endpoint at `/usage`
- Opens missing canonical settings tabs automatically
- Browser auto-detection for Brave, Chrome, Arc, Edge, and Chrome Canary

## Setup

```bash
./setup.sh
```

To launch immediately:

```bash
./setup.sh --run
```

## Run

```bash
python3 scraper.py
```

Override the default port if needed:

```bash
AI_USAGE_MONITOR_PORT=9000 python3 scraper.py
```

Override the browser app if auto-detection picks the wrong one:

```bash
AI_USAGE_MONITOR_BROWSER_APP="Google Chrome" python3 scraper.py
```

## Notes

- This is intended for local use on macOS with a Chromium-based browser installed.
- No logs, caches, or user-specific data are included in this folder.
