#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${AI_USAGE_MONITOR_PORT:-8876}"

detect_browser() {
  local configured="${AI_USAGE_MONITOR_BROWSER_APP:-}"
  if [[ -n "$configured" ]]; then
    printf '%s\n' "$configured"
    return 0
  fi

  local candidates=(
    "Brave Browser"
    "Google Chrome"
    "Arc"
    "Microsoft Edge"
    "Google Chrome Canary"
  )

  local app
  for app in "${candidates[@]}"; do
    if [[ -d "/Applications/${app}.app" || -d "${HOME}/Applications/${app}.app" ]]; then
      printf '%s\n' "$app"
      return 0
    fi
  done

  return 1
}

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found" >&2
  exit 1
fi

if ! command -v osascript >/dev/null 2>&1; then
  echo "osascript not found; this script is for macOS" >&2
  exit 1
fi

BROWSER_APP="$(detect_browser)" || {
  echo "No supported browser found. Set AI_USAGE_MONITOR_BROWSER_APP to a Chromium browser app name." >&2
  exit 1
}

echo "Browser: ${BROWSER_APP}"
echo "Port: ${PORT}"
echo "Dashboard: http://127.0.0.1:${PORT}/"

if [[ "${1:-}" == "--run" ]]; then
  export AI_USAGE_MONITOR_BROWSER_APP="${BROWSER_APP}"
  export AI_USAGE_MONITOR_PORT="${PORT}"
  exec python3 "${SCRIPT_DIR}/scraper.py"
fi

echo
echo "To run now:"
echo "AI_USAGE_MONITOR_BROWSER_APP=\"${BROWSER_APP}\" AI_USAGE_MONITOR_PORT=${PORT} python3 ${SCRIPT_DIR}/scraper.py"
