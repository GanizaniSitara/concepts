#!/usr/bin/env python3
"""3-monitor 2x2-per-monitor Windows window tiler (pure stdlib + ctypes).

Tiles all visible windows of a target process across three landscape monitors,
giving 12 zones total (4 per monitor, arranged 2 columns x 2 rows on each).
Zones are numbered left-to-right across the physical monitor layout regardless
of what Windows calls "Display 1, 2, 3":

    monitor-left    : zones 1-4
    monitor-middle  : zones 5-8
    monitor-right   : zones 9-12

Within each monitor:

    +-----+-----+
    |  1  |  2  |
    +-----+-----+
    |  3  |  4  |
    +-----+-----+

Windows beyond zone 12 wrap modulo 12 and overlap older ones in the same
zone (Alt+Tab to switch between overlapping windows).

Window ordering: read from the Windows taskbar left-to-right via the classic
IAccessible (oleacc.dll) COM interface. Drag a taskbar button to reorder,
rerun, the corresponding window moves to the new zone. Falls back to PID
order if oleacc enumeration fails or returns no matches.

Visible window frames sit flush edge-to-edge by compensating for the
invisible DWM extended-frame border (typically 7 px on Windows 10/11).

Requires Windows 10/11 and Python 3.8+. No third-party packages.

Usage:
    python 2026-05_window_tiler_3mon_2x2.py
    python 2026-05_window_tiler_3mon_2x2.py --process WindowsTerminal
    python 2026-05_window_tiler_3mon_2x2.py --by-pid

Notes:
- Works with 1, 2, 3, or more monitors; zone count scales with monitor count
  (4 zones per monitor). The "3-monitor" framing is just the most common
  triple-head workstation case.
- UI Automation (System.Windows.Automation) does not enumerate
  MSTaskListWClass children on Windows 10 22H2 (returns 0). The oleacc
  IAccessible interface is used instead -- same approach the 7+ Taskbar
  Tweaker and Windhawk projects rely on.
"""

import argparse
import ctypes
import re
import sys
from collections import deque
from ctypes import wintypes as wt

# ---------------------------------------------------------------------------
# Win32 / DWM / oleacc loading
# ---------------------------------------------------------------------------

user32 = ctypes.WinDLL("user32", use_last_error=True)
dwmapi = ctypes.WinDLL("dwmapi", use_last_error=True)
oleacc = ctypes.WinDLL("oleacc", use_last_error=True)
oleaut32 = ctypes.WinDLL("oleaut32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

SW_RESTORE = 9
SWP_NOZORDER = 0x0004
SWP_NOACTIVATE = 0x0010
DWMWA_EXTENDED_FRAME_BOUNDS = 9
OBJID_CLIENT = 0xFFFFFFFC
VT_I4 = 3
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

EnumWindowsProc = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
EnumMonitorsProc = ctypes.WINFUNCTYPE(
    wt.BOOL, wt.HMONITOR, wt.HDC, ctypes.POINTER(wt.RECT), wt.LPARAM
)


class MONITORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wt.DWORD),
        ("rcMonitor", wt.RECT),
        ("rcWork", wt.RECT),
        ("dwFlags", wt.DWORD),
    ]


user32.EnumWindows.argtypes = [EnumWindowsProc, wt.LPARAM]
user32.EnumWindows.restype = wt.BOOL
user32.GetWindowTextW.argtypes = [wt.HWND, wt.LPWSTR, ctypes.c_int]
user32.GetWindowTextW.restype = ctypes.c_int
user32.GetWindowTextLengthW.argtypes = [wt.HWND]
user32.GetWindowTextLengthW.restype = ctypes.c_int
user32.GetClassNameW.argtypes = [wt.HWND, wt.LPWSTR, ctypes.c_int]
user32.GetClassNameW.restype = ctypes.c_int
user32.IsWindowVisible.argtypes = [wt.HWND]
user32.IsWindowVisible.restype = wt.BOOL
user32.GetWindowThreadProcessId.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
user32.GetWindowThreadProcessId.restype = wt.DWORD
user32.FindWindowW.argtypes = [wt.LPCWSTR, wt.LPCWSTR]
user32.FindWindowW.restype = wt.HWND
user32.FindWindowExW.argtypes = [wt.HWND, wt.HWND, wt.LPCWSTR, wt.LPCWSTR]
user32.FindWindowExW.restype = wt.HWND
user32.SetWindowPos.argtypes = [
    wt.HWND, wt.HWND, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, wt.UINT,
]
user32.SetWindowPos.restype = wt.BOOL
user32.ShowWindow.argtypes = [wt.HWND, ctypes.c_int]
user32.ShowWindow.restype = wt.BOOL
user32.GetWindowRect.argtypes = [wt.HWND, ctypes.POINTER(wt.RECT)]
user32.GetWindowRect.restype = wt.BOOL
user32.EnumDisplayMonitors.argtypes = [
    wt.HDC, ctypes.POINTER(wt.RECT), EnumMonitorsProc, wt.LPARAM,
]
user32.EnumDisplayMonitors.restype = wt.BOOL
user32.GetMonitorInfoW.argtypes = [wt.HMONITOR, ctypes.POINTER(MONITORINFO)]
user32.GetMonitorInfoW.restype = wt.BOOL

dwmapi.DwmGetWindowAttribute.argtypes = [
    wt.HWND, wt.DWORD, ctypes.c_void_p, wt.DWORD,
]
dwmapi.DwmGetWindowAttribute.restype = ctypes.c_long

kernel32.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
kernel32.OpenProcess.restype = wt.HANDLE
kernel32.CloseHandle.argtypes = [wt.HANDLE]
kernel32.CloseHandle.restype = wt.BOOL
kernel32.QueryFullProcessImageNameW.argtypes = [
    wt.HANDLE, wt.DWORD, wt.LPWSTR, ctypes.POINTER(wt.DWORD),
]
kernel32.QueryFullProcessImageNameW.restype = wt.BOOL

oleaut32.SysFreeString.argtypes = [ctypes.c_void_p]
oleaut32.SysFreeString.restype = None


# ---------------------------------------------------------------------------
# VARIANT + IAccessible COM plumbing
# ---------------------------------------------------------------------------

class _VARIANT_VALUE(ctypes.Union):
    _fields_ = [
        ("lVal", ctypes.c_long),
        ("_pad", ctypes.c_int64 * 2),  # ensures 16-byte union on x64
    ]


class VARIANT(ctypes.Structure):
    _anonymous_ = ("_v",)
    _fields_ = [
        ("vt", ctypes.c_ushort),
        ("wReserved1", ctypes.c_ushort),
        ("wReserved2", ctypes.c_ushort),
        ("wReserved3", ctypes.c_ushort),
        ("_v", _VARIANT_VALUE),
    ]


class GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_uint8 * 8),
    ]


IID_IAccessible = GUID(
    0x618736E0, 0x3C3D, 0x11CF,
    (ctypes.c_uint8 * 8)(0x81, 0x0C, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71),
)

oleacc.AccessibleObjectFromWindow.argtypes = [
    wt.HWND, wt.DWORD, ctypes.POINTER(GUID),
    ctypes.POINTER(ctypes.c_void_p),
]
oleacc.AccessibleObjectFromWindow.restype = ctypes.c_long


def _vtable_call(p_acc: int, index: int, restype, *argtypes):
    """Bind vtable[index] of the COM object at p_acc as a callable."""
    vtable_ptr = ctypes.c_void_p.from_address(p_acc).value
    slot = vtable_ptr + index * ctypes.sizeof(ctypes.c_void_p)
    fn_addr = ctypes.c_void_p.from_address(slot).value
    prototype = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)
    return prototype(fn_addr)


def acc_release(p_acc: int) -> None:
    # IUnknown::Release at vtable index 2.
    release = _vtable_call(p_acc, 2, ctypes.c_ulong)
    release(p_acc)


def acc_child_count(p_acc: int) -> int:
    # IAccessible::get_accChildCount at vtable index 8.
    count = ctypes.c_long(0)
    fn = _vtable_call(p_acc, 8, ctypes.c_long, ctypes.POINTER(ctypes.c_long))
    hr = fn(p_acc, ctypes.byref(count))
    if hr != 0:
        raise OSError(f"get_accChildCount failed: 0x{hr & 0xFFFFFFFF:08X}")
    return count.value


def acc_name(p_acc: int, child_id: int) -> str:
    # IAccessible::get_accName at vtable index 10. Signature:
    #   HRESULT get_accName(VARIANT varChild, BSTR* pszName)
    var = VARIANT()
    var.vt = VT_I4
    var.lVal = child_id
    bstr = ctypes.c_void_p(0)
    fn = _vtable_call(
        p_acc, 10, ctypes.c_long, VARIANT, ctypes.POINTER(ctypes.c_void_p)
    )
    hr = fn(p_acc, var, ctypes.byref(bstr))
    if hr != 0 or not bstr.value:
        return ""
    try:
        return ctypes.wstring_at(bstr.value)
    finally:
        oleaut32.SysFreeString(bstr.value)


# ---------------------------------------------------------------------------
# Taskbar reading
# ---------------------------------------------------------------------------

def find_all_trays():
    """Primary Shell_TrayWnd followed by every Shell_SecondaryTrayWnd."""
    trays = []
    primary = user32.FindWindowW("Shell_TrayWnd", None)
    if primary:
        trays.append(primary)

    buf = ctypes.create_unicode_buffer(256)

    def cb(hwnd, _lp):
        user32.GetClassNameW(hwnd, buf, 256)
        if buf.value == "Shell_SecondaryTrayWnd":
            trays.append(hwnd)
        return True

    user32.EnumWindows(EnumWindowsProc(cb), 0)
    return trays


def find_mstasklist(parent: int) -> int:
    if not parent:
        return 0
    direct = user32.FindWindowExW(parent, 0, "MSTaskListWClass", None)
    if direct:
        return direct
    buf = ctypes.create_unicode_buffer(256)
    child = user32.FindWindowExW(parent, 0, None, None)
    while child:
        user32.GetClassNameW(child, buf, 256)
        if buf.value == "MSTaskListWClass":
            return child
        nested = find_mstasklist(child)
        if nested:
            return nested
        child = user32.FindWindowExW(parent, child, None, None)
    return 0


def get_taskbar_title_order():
    """Return taskbar button names in left-to-right order across all trays."""
    titles = []
    for tray in find_all_trays():
        tasklist = find_mstasklist(tray)
        if not tasklist:
            continue
        p_acc = ctypes.c_void_p(0)
        hr = oleacc.AccessibleObjectFromWindow(
            tasklist, OBJID_CLIENT,
            ctypes.byref(IID_IAccessible), ctypes.byref(p_acc),
        )
        if hr != 0 or not p_acc.value:
            continue
        try:
            count = acc_child_count(p_acc.value)
            for i in range(1, count + 1):
                name = acc_name(p_acc.value, i)
                if name:
                    titles.append(name)
        except OSError:
            pass
        finally:
            acc_release(p_acc.value)
    return titles


# ---------------------------------------------------------------------------
# Title normalization
# ---------------------------------------------------------------------------

_SUFFIX_RE = re.compile(r" \S \d+ running windows?$")
_LEADING_GLYPHS_RE = re.compile(r"^[^A-Za-z0-9/\\:.]+")


def strip_taskbar_suffix(s: str) -> str:
    # Each taskbar entry looks like:  '<title> <dash> N running window(s)'
    # with U+2013 / U+2014 / ASCII hyphen depending on locale. One non-space
    # char between two spaces handles all three.
    return _SUFFIX_RE.sub("", s)


def normalize_title(s: str) -> str:
    # Strip leading non-alphanumeric decorative glyphs. Some terminals inject
    # animated braille spinners ahead of the title; the spinner phase changes
    # between reads, so equality breaks unless both sides are normalised.
    if not s:
        return ""
    return _LEADING_GLYPHS_RE.sub("", s).strip()


# ---------------------------------------------------------------------------
# Window + process enumeration
# ---------------------------------------------------------------------------

def get_window_title(hwnd: int) -> str:
    length = user32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return ""
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value


def get_process_image(pid: int) -> str:
    handle = kernel32.OpenProcess(
        PROCESS_QUERY_LIMITED_INFORMATION, False, pid
    )
    if not handle:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(512)
        size = wt.DWORD(len(buf))
        if kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            return buf.value
    finally:
        kernel32.CloseHandle(handle)
    return ""


def collect_target_windows(process_name: str):
    """Visible windows of the target process as (hwnd, pid, title) tuples."""
    target = process_name.lower()
    if not target.endswith(".exe"):
        target += ".exe"

    results = []

    def cb(hwnd, _lp):
        if not user32.IsWindowVisible(hwnd):
            return True
        title = get_window_title(hwnd)
        if not title:
            return True
        pid = wt.DWORD(0)
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        image = get_process_image(pid.value).lower()
        if image.endswith("\\" + target) or image.endswith("/" + target):
            results.append((hwnd, pid.value, title))
        return True

    user32.EnumWindows(EnumWindowsProc(cb), 0)
    return results


# ---------------------------------------------------------------------------
# Monitor enumeration + zones
# ---------------------------------------------------------------------------

def enumerate_monitors_left_to_right():
    monitors = []

    def cb(hmon, _hdc, _rect_ptr, _lp):
        info = MONITORINFO()
        info.cbSize = ctypes.sizeof(MONITORINFO)
        if user32.GetMonitorInfoW(hmon, ctypes.byref(info)):
            r = info.rcWork
            monitors.append((r.left, r.top, r.right - r.left, r.bottom - r.top))
        return True

    user32.EnumDisplayMonitors(None, None, EnumMonitorsProc(cb), 0)
    monitors.sort(key=lambda m: m[0])
    return monitors


def build_zones(monitors):
    """Per monitor: 2x2 grid -> 4 zones in TL, TR, BL, BR order."""
    zones = []
    for mi, (x, y, w, h) in enumerate(monitors):
        cw = w // 2
        ch = h // 2
        for row in range(2):
            for col in range(2):
                zones.append({
                    "x": x + col * cw,
                    "y": y + row * ch,
                    "w": cw,
                    "h": ch,
                    "label": f"mon{mi + 1}[{row * 2 + col + 1}]",
                })
    return zones


# ---------------------------------------------------------------------------
# DWM frame compensation
# ---------------------------------------------------------------------------

def get_invisible_border(hwnd):
    """Pixels of invisible DWM border on (left, top, right, bottom)."""
    outer = wt.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(outer)):
        return (0, 0, 0, 0)
    frame = wt.RECT()
    hr = dwmapi.DwmGetWindowAttribute(
        hwnd, DWMWA_EXTENDED_FRAME_BOUNDS,
        ctypes.byref(frame), ctypes.sizeof(frame),
    )
    if hr != 0:
        return (0, 0, 0, 0)
    return (
        frame.left - outer.left,
        frame.top - outer.top,
        outer.right - frame.right,
        outer.bottom - frame.bottom,
    )


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------

def resolve_taskbar_order(windows):
    """Order target windows by taskbar left-to-right. None on failure."""
    try:
        raw = get_taskbar_title_order()
    except Exception as exc:
        print(f"  taskbar enumeration error: {exc} (falling back to PID order)",
              file=sys.stderr)
        return None
    if not raw:
        print("  taskbar returned no entries (falling back to PID order)",
              file=sys.stderr)
        return None

    by_title = {}
    for w in sorted(windows, key=lambda t: t[1]):
        key = normalize_title(w[2])
        by_title.setdefault(key, deque()).append(w)

    ordered = []
    used = set()
    for entry in raw:
        key = normalize_title(strip_taskbar_suffix(entry))
        bucket = by_title.get(key)
        if bucket:
            w = bucket.popleft()
            if w[1] not in used:
                used.add(w[1])
                ordered.append(w)

    if not ordered:
        print("  no taskbar entries matched any target windows "
              "(falling back to PID order)", file=sys.stderr)
        return None

    for w in sorted(windows, key=lambda t: t[1]):
        if w[1] not in used:
            ordered.append(w)
    return ordered


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Tile windows across landscape monitors in a 2x2 grid "
                    "per monitor. Designed for a 3-monitor setup (12 zones) "
                    "but scales to any monitor count.",
    )
    ap.add_argument(
        "--process", default="cmd",
        help="Process image name to target (default: cmd).",
    )
    ap.add_argument(
        "--by-pid", action="store_true",
        help="Force PID-order tiling (skip taskbar reading).",
    )
    args = ap.parse_args()

    monitors = enumerate_monitors_left_to_right()
    if not monitors:
        print("No monitors found.", file=sys.stderr)
        return 2
    zones = build_zones(monitors)

    windows = collect_target_windows(args.process)
    if not windows:
        print(f"No visible {args.process} windows found.")
        return 0

    if args.by_pid:
        ordered = sorted(windows, key=lambda t: t[1])
        source = "PID (forced)"
    else:
        resolved = resolve_taskbar_order(windows)
        if resolved:
            ordered = resolved
            source = "taskbar (oleacc)"
        else:
            ordered = sorted(windows, key=lambda t: t[1])
            source = "PID (fallback)"

    print(
        f"Tiling {len(ordered)} {args.process} window(s) across "
        f"{len(zones)} zone(s) over {len(monitors)} monitor(s); "
        f"ordering: {source}"
    )

    # Tile in reverse so window 1 ends up Z-top within its zone after the
    # cycling wraps overlap older windows.
    for i in range(len(ordered) - 1, -1, -1):
        hwnd = ordered[i][0]
        z = zones[i % len(zones)]
        user32.ShowWindow(hwnd, SW_RESTORE)
        user32.SetWindowPos(
            hwnd, 0, z["x"], z["y"], z["w"], z["h"], SWP_NOZORDER,
        )
        lpad, tpad, rpad, bpad = get_invisible_border(hwnd)
        user32.SetWindowPos(
            hwnd, 0,
            z["x"] - lpad, z["y"] - tpad,
            z["w"] + lpad + rpad, z["h"] + tpad + bpad,
            SWP_NOZORDER,
        )

    stdout_enc = (sys.stdout.encoding or "ascii").lower()
    for i, (_hwnd, _pid, title) in enumerate(ordered):
        z = zones[i % len(zones)]
        clip = title if len(title) <= 50 else title[:47] + "..."
        # Some terminal titles inject non-ASCII glyphs (spinners, sparkles).
        # Default Windows consoles are cp1252; round-trip with replacement
        # so the diagnostic print never blows up on the way out.
        try:
            clip.encode(stdout_enc)
        except UnicodeEncodeError:
            clip = clip.encode(stdout_enc, "replace").decode(stdout_enc)
        print(
            f"  win {i + 1:3d} -> zone {i % len(zones) + 1:2d} "
            f"({z['label']:<10}) '{clip}'"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
