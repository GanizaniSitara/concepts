# AI usage monitor

Small local dashboard that polls open Brave tabs for usage data from:

- Claude usage settings
- Codex usage analytics
- GitHub Copilot settings

It serves a simple dashboard and JSON API on `http://127.0.0.1:8876/` by default.

## Features

- Sliding-window bars with a live pace marker
- Reset labels with short weekday codes such as `MON`, `TUE`, `FRI`
- JSON endpoint at `/usage`
- Opens missing canonical settings tabs automatically in Brave

## Run

```bash
python3 scraper.py
```

Override the default port if needed:

```bash
AI_USAGE_MONITOR_PORT=9000 python3 scraper.py
```

## Notes

- This is intended for local use on a machine with Brave installed.
- No logs, caches, or user-specific data are included in this folder.
