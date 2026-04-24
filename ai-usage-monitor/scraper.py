#!/opt/homebrew/bin/python3
"""Poll browser tabs for AI-tool usage and serve as JSON on localhost:8876/usage."""
import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

POLL_SECONDS = 300
PORT = int(os.environ.get("AI_USAGE_MONITOR_PORT", "8876"))
BROWSER_CANDIDATES = [
    "Brave Browser",
    "Google Chrome",
    "Arc",
    "Microsoft Edge",
    "Google Chrome Canary",
]

TAB_MATCHERS = {
    "claude":  "claude.ai/settings/usage",
    "codex":   "chatgpt.com/codex",
    "copilot": "github.com/settings/copilot",
}

CANONICAL_URLS = {
    "claude":  "https://claude.ai/settings/usage",
    "codex":   "https://chatgpt.com/codex/cloud/settings/analytics#usage",
    "copilot": "https://github.com/settings/copilot/features",
}

_state_lock = threading.Lock()
_state = {"updated_at": None, "sources": {}, "errors": {}}


def osa(script: str) -> str:
    r = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True, text=True, timeout=15,
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or "osascript failed")
    return r.stdout


def resolve_browser_app() -> str:
    configured = os.environ.get("AI_USAGE_MONITOR_BROWSER_APP")
    if configured:
        return configured
    search_roots = [Path("/Applications"), Path.home() / "Applications"]
    for candidate in BROWSER_CANDIDATES:
        for root in search_roots:
            if (root / f"{candidate}.app").exists():
                return candidate
    raise RuntimeError(
        "No supported browser app found. Set AI_USAGE_MONITOR_BROWSER_APP to a Chromium browser app name."
    )


BROWSER_APP = resolve_browser_app()
BROWSER_APP_LITERAL = BROWSER_APP.replace("\\", "\\\\").replace('"', '\\"')


def locate_tabs() -> dict[str, tuple[int, int]]:
    script = f'''
    tell application "{BROWSER_APP_LITERAL}"
        set out to ""
        set wcount to count of windows
        repeat with w from 1 to wcount
            set tcount to count of tabs of window w
            repeat with t from 1 to tcount
                set out to out & w & "," & t & "," & (URL of tab t of window w) & linefeed
            end repeat
        end repeat
        return out
    end tell
    '''
    result: dict[str, tuple[int, int]] = {}
    for line in osa(script).splitlines():
        parts = line.split(",", 2)
        if len(parts) != 3:
            continue
        w, t, url = parts
        for name, needle in TAB_MATCHERS.items():
            if name not in result and needle in url:
                result[name] = (int(w), int(t))
    return result


def fetch_text(window: int, tab: int) -> str:
    script = (
        f'tell application "{BROWSER_APP_LITERAL}" to execute tab {tab} of window {window} '
        'javascript "document.body.innerText"'
    )
    return osa(script)


def pct(s: str) -> float | None:
    m = re.search(r"([\d.]+)\s*%", s)
    return float(m.group(1)) if m else None


def parse_claude(text: str) -> dict:
    out: dict = {}

    def block(header: str, span: int = 300) -> str:
        i = text.find(header)
        return text[i:i + span] if i >= 0 else ""

    def reset_text(b: str) -> str | None:
        m = re.search(r"Resets(?:\s+in)?\s+([^\n]+)", b)
        return m.group(1).strip() if m else None

    session = block("Current session")
    out["session"] = {"used_pct": pct(session), "resets_in": reset_text(session)}

    # Weekly section — two subsections: "All models" and "Sonnet only".
    all_models = block("All models")
    out["weekly_all"] = {"used_pct": pct(all_models), "resets_in": reset_text(all_models)}

    sonnet = block("Sonnet only")
    out["weekly_sonnet"] = {"used_pct": pct(sonnet), "resets_in": reset_text(sonnet)}

    extra = block("Extra usage", span=500)
    m_spent = re.search(r"£([\d.]+)\s+spent", extra)
    m_resets = re.search(r"Resets\s+([A-Z][a-z]+\s+\d+)", extra)
    out["extra"] = {
        "spent_gbp": float(m_spent.group(1)) if m_spent else None,
        "used_pct": pct(extra),
        "resets": m_resets.group(1) if m_resets else None,
    }

    m_bal = re.search(r"£([\d.]+)\s*\n?\s*Current balance", text)
    out["balance_gbp"] = float(m_bal.group(1)) if m_bal else None
    return out


def parse_codex(text: str) -> dict:
    out: dict = {}

    def block(header: str, span: int = 300) -> str:
        i = text.find(header)
        return text[i:i + span] if i >= 0 else ""

    def remaining_pct(b: str) -> float | None:
        m = re.search(r"([\d.]+)\s*%\s*\n?\s*remaining", b)
        return float(m.group(1)) if m else None

    def resets(b: str) -> str | None:
        m = re.search(r"Resets\s+([^\n]+)", b)
        return m.group(1).strip() if m else None

    hourly = block("5 hour usage limit")
    out["hourly"] = {"remaining_pct": remaining_pct(hourly), "resets": resets(hourly)}

    weekly = block("Weekly usage limit")
    out["weekly"] = {"remaining_pct": remaining_pct(weekly), "resets": resets(weekly)}

    m_credits = re.search(r"Credits remaining\s*\n+\s*(\d+)", text)
    out["credits_remaining"] = int(m_credits.group(1)) if m_credits else None
    return out


def parse_copilot(text: str) -> dict:
    i = text.find("Premium requests")
    region = text[i:i + 200] if i >= 0 else ""
    return {"premium_requests_used_pct": pct(region)}


PARSERS = {"claude": parse_claude, "codex": parse_codex, "copilot": parse_copilot}


def open_tab(url: str) -> None:
    """Open URL in a background tab without activating the configured browser."""
    script = f'''
    tell application "{BROWSER_APP_LITERAL}"
        if (count of windows) = 0 then
            make new window
        end if
        tell window 1
            make new tab with properties {{URL:"{url}"}}
        end tell
    end tell
    '''
    try:
        osa(script)
    except Exception:
        pass


def ensure_tabs(tabs: dict) -> dict:
    """Open any missing canonical tabs; return updated tabs map."""
    opened = False
    for name, url in CANONICAL_URLS.items():
        if name not in tabs:
            open_tab(url)
            opened = True
    if opened:
        time.sleep(4)  # let new tabs start loading
        try:
            tabs = locate_tabs()
        except Exception:
            pass
    return tabs


def poll_once() -> None:
    sources: dict = {}
    errors: dict = {}
    try:
        tabs = locate_tabs()
    except Exception as e:
        errors["_locate"] = str(e)
        tabs = {}

    tabs = ensure_tabs(tabs)

    for name in TAB_MATCHERS:
        if name not in tabs:
            errors[name] = "tab not found (may still be loading)"
            continue
        w, t = tabs[name]
        try:
            text = fetch_text(w, t)
            sources[name] = PARSERS[name](text)
        except Exception as e:
            errors[name] = str(e)

    with _state_lock:
        _state["updated_at"] = datetime.now(timezone.utc).isoformat()
        _state["sources"] = sources
        _state["errors"] = errors


def poll_loop() -> None:
    while True:
        try:
            poll_once()
        except Exception as e:
            with _state_lock:
                _state["errors"] = {"_loop": str(e)}
        time.sleep(POLL_SECONDS)


DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AI usage</title>
<style>
  :root { color-scheme: light; }
  body { font: 14px/1.4 -apple-system, system-ui, sans-serif;
         background:#fafafa; color:#222; margin:0; padding:24px; max-width:720px; }
  h1 { font-size:15px; font-weight:600; margin:0 0 4px; letter-spacing:.02em; }
  h2 { font-size:11px; font-weight:600; text-transform:uppercase;
       letter-spacing:.08em; color:#666; margin:28px 0 10px; }
  .meta { color:#888; font-size:12px; margin-bottom:4px; }
  .row { display:grid; grid-template-columns: 160px 1fr 100px;
         gap:12px; align-items:center; padding:8px 0;
         border-bottom:1px solid #e5e5e5; }
  .label { color:#222; }
  .sub { color:#888; font-size:11px; }
  .bar { position:relative; height:8px; background:#e8e8e8; border-radius:4px; overflow:visible; }
  .fill { height:100%; transition:width .4s ease; }
  .fill.ok   { background:#2a9d6a; }
  .fill.warn { background:#c97a13; }
  .fill.hot  { background:#b33224; }
  .marker { position:absolute; top:-3px; bottom:-3px; width:2px; background:#111; border-radius:999px; opacity:.7; }
  .marker::after { content:''; position:absolute; top:-3px; left:50%; width:6px; height:6px; transform:translateX(-50%); background:#111; border-radius:999px; }
  .right { text-align:right; color:#555; font-variant-numeric:tabular-nums; }
  .err { color:#b33224; font-size:12px; }
</style>
</head>
<body>
<h1>AI usage</h1>
<div class="meta" id="updated">loading…</div>
<div id="content"></div>
<script>
function bar(used, markerPct) {
  const pct = Math.max(0, Math.min(100, used ?? 0));
  const cls = pct >= 85 ? 'hot' : pct >= 60 ? 'warn' : 'ok';
  const marker = markerPct == null ? '' : `<div class="marker" style="left:calc(${Math.max(0, Math.min(100, markerPct))}% - 1px)"></div>`;
  return `<div class="bar"><div class="fill ${cls}" style="width:${pct}%"></div>${marker}</div>`;
}
function parseDurationMs(text) {
  if (!text) return null;
  const units = { w: 7 * 24 * 60, week: 7 * 24 * 60, weeks: 7 * 24 * 60,
                  d: 24 * 60, day: 24 * 60, days: 24 * 60,
                  h: 60, hr: 60, hrs: 60, hour: 60, hours: 60,
                  m: 1, min: 1, mins: 1, minute: 1, minutes: 1 };
  let totalMinutes = 0;
  let matched = false;
  for (const match of text.matchAll(/(\d+(?:\.\d+)?)\s*(weeks?|w|days?|d|hours?|hrs?|hr|h|minutes?|mins?|min|m)\b/gi)) {
    const value = Number(match[1]);
    const unit = match[2].toLowerCase();
    if (!Number.isFinite(value) || units[unit] == null) continue;
    totalMinutes += value * units[unit];
    matched = true;
  }
  return matched ? totalMinutes * 60 * 1000 : null;
}
function formatClock(date) {
  return date.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'});
}
function formatMonthDay(date) {
  return date.toLocaleDateString([], {month: 'short', day: 'numeric'});
}
function formatWeekdayCode(date) {
  return ['SUN', 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT'][date.getDay()];
}
function formatResetDay(date) {
  const weekday = formatWeekdayCode(date);
  const monthDay = date.toLocaleDateString([], {month: 'long', day: 'numeric'});
  return `${weekday}, ${monthDay}`;
}
function formatResetDateTime(date) {
  const weekday = formatWeekdayCode(date);
  const monthDay = date.toLocaleDateString([], {month: 'long', day: 'numeric'});
  const time = date.toLocaleTimeString([], {hour: 'numeric', minute: '2-digit'});
  return `${weekday}, ${monthDay}, ${time}`;
}
function formatDuration(ms) {
  const totalMinutes = Math.round(ms / 60000);
  if (totalMinutes <= 0) return '0m';
  const weeks = Math.floor(totalMinutes / (7 * 24 * 60));
  const days = Math.floor((totalMinutes % (7 * 24 * 60)) / (24 * 60));
  const hours = Math.floor((totalMinutes % (24 * 60)) / 60);
  const minutes = totalMinutes % 60;
  const parts = [];
  if (weeks) parts.push(`${weeks}w`);
  if (days) parts.push(`${days}d`);
  if (hours) parts.push(`${hours}h`);
  if (minutes || parts.length === 0) parts.push(`${minutes}m`);
  return parts.slice(0, 2).join(' ');
}
function parseClockParts(text) {
  const match = text.match(/^(\d{1,2}):(\d{2})\s*([AP]M)$/i);
  if (!match) return null;
  let hour = Number(match[1]) % 12;
  if (match[3].toUpperCase() === 'PM') hour += 12;
  return {hour, minute: Number(match[2])};
}
function resolveResetAt(resetText, now = new Date()) {
  if (!resetText) return null;
  const durationMs = parseDurationMs(resetText);
  if (durationMs != null) return new Date(now.getTime() + durationMs);

  const weekdayMatch = resetText.match(/^(Sun|Mon|Tue|Wed|Thu|Fri|Sat)\s+(\d{1,2}:\d{2}\s*[AP]M)$/i);
  if (weekdayMatch) {
    const days = {sun: 0, mon: 1, tue: 2, wed: 3, thu: 4, fri: 5, sat: 6};
    const parts = parseClockParts(weekdayMatch[2]);
    if (parts) {
      const candidate = new Date(now);
      candidate.setHours(parts.hour, parts.minute, 0, 0);
      let delta = (days[weekdayMatch[1].toLowerCase()] - now.getDay() + 7) % 7;
      if (delta === 0 && candidate <= now) delta = 7;
      candidate.setDate(candidate.getDate() + delta);
      return candidate;
    }
  }

  const timeOnly = parseClockParts(resetText.trim());
  if (timeOnly) {
    const candidate = new Date(now);
    candidate.setHours(timeOnly.hour, timeOnly.minute, 0, 0);
    if (candidate <= now) candidate.setDate(candidate.getDate() + 1);
    return candidate;
  }

  const parsed = new Date(resetText);
  if (!Number.isNaN(parsed.getTime())) {
    if (!/\b\d{4}\b/.test(resetText) && parsed <= now) parsed.setFullYear(parsed.getFullYear() + 1);
    return parsed;
  }
  return null;
}
function windowInfoFromReset(resetText, totalMs, now = new Date()) {
  if (totalMs == null) return null;
  const resetAt = resolveResetAt(resetText, now);
  if (!resetAt) return null;
  const remainingMs = resetAt.getTime() - now.getTime();
  const elapsedMs = Math.max(0, Math.min(totalMs, totalMs - remainingMs));
  return {resetAt, totalMs, elapsedMs};
}
function monthWindowInfo(now = new Date()) {
  const start = new Date(now.getFullYear(), now.getMonth(), 1);
  const resetAt = new Date(now.getFullYear(), now.getMonth() + 1, 1);
  const totalMs = resetAt.getTime() - start.getTime();
  const elapsedMs = now.getTime() - start.getTime();
  return {resetAt, totalMs, elapsedMs};
}
function formatResetLabel(resetText, info, options = {}) {
  if (!resetText) return '?';
  if (parseDurationMs(resetText) != null) return resetText;
  if (!info) return resetText;
  if (options.dateOnly) return formatResetDay(info.resetAt);
  return formatResetDateTime(info.resetAt);
}
function timingSubForWindow(base, info, usedPct) {
  if (!info || usedPct == null) return base;
  const now = new Date();
  const paceDeltaMs = (usedPct / 100 * info.totalMs) - info.elapsedMs;
  let pace = 'on pace';
  if (Math.abs(paceDeltaMs) >= 2 * 60 * 1000) {
    pace = `${formatDuration(Math.abs(paceDeltaMs))} ${paceDeltaMs > 0 ? 'ahead' : 'behind'}`;
  }
  return `${base} · now ${formatClock(now)} -> ${formatClock(info.resetAt)} · ${pace}`;
}
function timingSub(resetPrefix, resetText, totalMs, usedPct) {
  const info = windowInfoFromReset(resetText, totalMs);
  const base = `${resetPrefix}${formatResetLabel(resetText, info)}`;
  return timingSubForWindow(base, info, usedPct);
}
function paceMarkerPct(resetText, totalMs) {
  const info = windowInfoFromReset(resetText, totalMs);
  return info ? info.elapsedMs / info.totalMs * 100 : null;
}
function markerPctFromWindow(info) {
  return info ? info.elapsedMs / info.totalMs * 100 : null;
}
function row(label, sub, usedPct, rightText, markerPct) {
  const subHtml = sub ? `<div class="sub">${sub}</div>` : '';
  const pctText = usedPct == null ? '—' : `${usedPct.toFixed(0)}%`;
  return `<div class="row">
    <div><div class="label">${label}</div>${subHtml}</div>
    ${bar(usedPct, markerPct)}
    <div class="right">${rightText ?? pctText}</div>
  </div>`;
}
async function refresh() {
  try {
    const r = await fetch('/usage', {cache:'no-store'});
    const d = await r.json();
    const s = d.sources || {};
    document.getElementById('updated').textContent =
      'updated ' + new Date(d.updated_at).toLocaleTimeString();

    let html = '';
    html += '<h2>Sliding windows</h2>';
    const c = s.claude || {};
    if (c.session)       html += row('Claude · session (5h)', timingSub('resets in ', c.session.resets_in, 5 * 60 * 60 * 1000, c.session.used_pct), c.session.used_pct, null, paceMarkerPct(c.session.resets_in, 5 * 60 * 60 * 1000));
    if (c.weekly_all)    html += row('Claude · weekly', timingSub('resets ', c.weekly_all.resets_in, 7 * 24 * 60 * 60 * 1000, c.weekly_all.used_pct), c.weekly_all.used_pct, null, paceMarkerPct(c.weekly_all.resets_in, 7 * 24 * 60 * 60 * 1000));
    const x = s.codex || {};
    const inv = r => r == null ? null : 100 - r;
    if (x.hourly) html += row('Codex · 5h', timingSub('resets ', x.hourly.resets, 5 * 60 * 60 * 1000, inv(x.hourly.remaining_pct)), inv(x.hourly.remaining_pct), null, paceMarkerPct(x.hourly.resets, 5 * 60 * 60 * 1000));
    if (x.weekly) html += row('Codex · weekly', timingSub('resets ', x.weekly.resets, 7 * 24 * 60 * 60 * 1000, inv(x.weekly.remaining_pct)), inv(x.weekly.remaining_pct), null, paceMarkerPct(x.weekly.resets, 7 * 24 * 60 * 60 * 1000));

    html += '<h2>Monthly</h2>';
    const cp = s.copilot || {};
    const cpMonth = monthWindowInfo();
    html += row('Copilot · premium requests', timingSubForWindow(`resets ${formatResetLabel('month', cpMonth, {dateOnly: true})}`, cpMonth, cp.premium_requests_used_pct), cp.premium_requests_used_pct, null, markerPctFromWindow(cpMonth));

    html += '<h2>Balance &amp; credits</h2>';
    if (c.extra) {
      const right = c.extra.spent_gbp != null ? `£${c.extra.spent_gbp.toFixed(2)}` : '—';
      html += row('Claude · extra usage', 'resets ' + (c.extra.resets ?? '?'), c.extra.used_pct, right);
    }
    if (c.balance_gbp != null) html += row('Claude · balance', '', 0, `£${c.balance_gbp.toFixed(2)}`);
    if (x.credits_remaining != null) html += row('Codex · credits', '', 0, String(x.credits_remaining));

    const errs = d.errors || {};
    if (Object.keys(errs).length) {
      html += '<h2>Errors</h2>';
      for (const [k,v] of Object.entries(errs)) html += `<div class="err">${k}: ${v}</div>`;
    }
    document.getElementById('content').innerHTML = html;
  } catch (e) {
    document.getElementById('updated').textContent = 'fetch error: ' + e;
  }
}
refresh();
setInterval(refresh, 15000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/usage":
            with _state_lock:
                body = json.dumps(_state, indent=2).encode()
            ctype = "application/json"
        elif self.path in ("/", "/dashboard"):
            body = DASHBOARD_HTML.encode()
            ctype = "text/html; charset=utf-8"
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_a, **_k) -> None:  # quiet
        return


def main() -> None:
    poll_once()
    threading.Thread(target=poll_loop, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"serving http://0.0.0.0:{PORT}/usage  (poll every {POLL_SECONDS}s)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
