# Prototype 1: RichEdit + RTF

## Approach
Replace WebView2 entirely with native Win32 controls. The editor is a plain `EDIT`
control (multiline). The preview is a read-only `RichEdit` control. Markdown is
parsed in Go with `goldmark`, walked as an AST, and converted to RTF strings that
are streamed into the RichEdit control via `EM_STREAMIN`.

No browser engine. No CGo. Pure Go + Win32 syscalls.

## Why This Might Win
- **Instant startup** — RichEdit is loaded from `msftedit.dll`, already on every
  Windows machine. No runtime to bootstrap.
- **Scrolling, selection, copy/paste for free** — RichEdit is a mature control.
- **Smallest binary** — only goldmark added as a dependency.
- **Least code** — RTF generation is string concatenation, no graphics programming.

## Architecture

```
┌──────────────────────────────────────────────────┐
│  Native Win32 Window (CreateWindowExW)           │
│  ┌──────────────┬───┬──────────────────────────┐ │
│  │  EDIT control│ ▌ │  RichEdit control        │ │
│  │  (multiline) │ ▌ │  (read-only, EM_STREAMIN)│ │
│  │              │ ▌ │                          │ │
│  │  raw markdown│ ▌ │  rendered RTF output     │ │
│  │              │ ▌ │                          │ │
│  └──────────────┴───┴──────────────────────────┘ │
└──────────────────────────────────────────────────┘
         │                        ▲
         │  EN_CHANGE             │  EM_STREAMIN
         ▼                        │
    ┌──────────┐           ┌──────────────┐
    │ Get text │──────────▶│ goldmark AST │
    │ from EDIT│           │      │       │
    └──────────┘           │  RTF Walker  │
                           │      │       │
                           │  RTF string  │
                           └──────────────┘
```

## Implementation Steps

### Step 1: Scaffold the Win32 window
- `main.go` — register window class, create main window, message loop
- Reuse the existing syscall patterns from tinymdeditor (user32, gdi32, kernel32)
- Add `msftedit.dll` lazy load for RichEdit

### Step 2: Create the two-pane layout
- Left pane: `CreateWindowExW` with class `"EDIT"`, styles `ES_MULTILINE | ES_AUTOVSCROLL | WS_VSCROLL`
- Right pane: `CreateWindowExW` with class `"RICHEDIT50W"` (from msftedit.dll), styles `ES_READONLY | ES_MULTILINE`
- Divider: either a static control or manual hit-testing on WM_MOUSEMOVE
- Handle `WM_SIZE` to resize both panes proportionally

### Step 3: Wire up text change notifications
- Subclass or handle `WM_COMMAND` with `EN_CHANGE` from the EDIT control
- On change: read text from EDIT via `GetWindowTextW`
- Debounce (50ms timer via `SetTimer` / `WM_TIMER`) to avoid rendering on every keystroke

### Step 4: Markdown → RTF conversion
- Add `github.com/yuin/goldmark` dependency
- Write a custom `goldmark` renderer that outputs RTF instead of HTML
- RTF basics:
  - `{\rtf1\ansi\deff0 {\fonttbl {\f0 Segoe UI;}{\f1 Consolas;}}` — header
  - `{\b bold}`, `{\i italic}`, `{\f1\fs20 code}`
  - `\par` for paragraphs, `\line` for line breaks
  - `{\fs48 heading}` for headings (varying sizes)
  - `{\pntext\tab}` or `\fi-360\li720` for list items
  - `{\brdrl\brdrw20\brdrdb\brdrcf1 blockquote text}` for blockquotes
  - `\trowd ... \cell ... \row` for tables

### Step 5: Stream RTF into RichEdit
- Use `EM_STREAMIN` message with `SF_RTF` format
- Provide a callback via `EDITSTREAMCALLBACK` that feeds the RTF byte buffer
- This is more efficient than `SetWindowText` for formatted content

### Step 6: File operations
- Ctrl+S: `GetWindowTextW` from EDIT → `os.WriteFile`
- Save As: reuse existing `GetSaveFileNameW` dialog code
- File open: read file, `SetWindowTextW` into EDIT, trigger render

### Step 7: Polish
- Font setup: set EDIT font to Consolas/Cascadia Code via `WM_SETFONT`
- Toolbar: optional static controls for filename display and save indicator
- Tab key handling: intercept in EDIT's wndproc, insert `\t`

## Dependencies
```
github.com/yuin/goldmark  (markdown parser, pure Go)
```
No other dependencies. All Win32 via syscall.

## Expected Characteristics
| Property         | Value                              |
|------------------|------------------------------------|
| Startup time     | ~50ms (native controls only)       |
| Binary size      | ~3-4 MB                            |
| Memory usage     | ~5-10 MB                           |
| Rendering        | RTF in RichEdit — good for text    |
| Code blocks      | Monospace font, no syntax highlight|
| Tables           | RTF tables (basic but functional)  |
| Images           | Not supported (RTF can embed, but complex) |
| Scrolling        | Native, smooth, free               |
| Text selection   | Native, free                       |

## Risks
- RTF table generation is verbose and fiddly
- No syntax highlighting in code blocks (just monospace font)
- Complex markdown (nested lists, footnotes) needs careful RTF mapping
- RichEdit has quirks with certain RTF constructs
