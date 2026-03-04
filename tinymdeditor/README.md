# TinyMD

A fast, minimal split-pane Markdown editor for Windows. Single Go file, ~380 lines.

Left pane: raw Markdown editing. Right pane: live rendered preview. Draggable divider between them.

## Build

Requires Go 1.23+ and targets Windows (uses WebView2).

**Release** (2.6 MB, no console window):
```
GOOS=windows go build -ldflags="-s -w -H windowsgui" -trimpath -o tinymd.exe .
```

**Development** (with debug info and console output):
```
GOOS=windows go build -o tinymd.exe .
```

**Even smaller** (~1 MB, adds ~50ms startup):
```
GOOS=windows go build -ldflags="-s -w -H windowsgui" -trimpath -o tinymd.exe .
upx --best tinymd.exe
```

## Usage

```
tinymd.exe                  # empty editor, Ctrl+S opens Save As dialog
tinymd.exe path/to/file.md  # opens file for editing
```

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| Ctrl+S | Save (or Save As if no file) |
| Tab | Insert tab character |

## Requirements

- Windows 10/11 with [Edge WebView2 Runtime](https://developer.microsoft.com/en-us/microsoft-edge/webview2/) installed (ships with Windows 11, available as a standalone install for Windows 10).
