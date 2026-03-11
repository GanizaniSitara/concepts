# Prototype 3: GDI DrawText (Extended Splash Approach)

## Approach
Take the splash screen rendering code that already works in `tinymdeditor/main.go`
and extend it into a full markdown preview. The editor is a standard EDIT control.
The preview is an owner-drawn window using GDI `DrawTextW`, `CreateFontW`,
`FillRect`, and `SetTextColor` — the same APIs already proven in the splash code.

No new dependencies beyond goldmark. Zero learning curve — it's the code you already
wrote, scaled up.

## Why This Might Win
- **Already works** — the splash screen proves GDI text rendering from Go via syscall.
  This just does more of it.
- **Absolute minimum dependencies** — goldmark + the Win32 DLLs already imported.
- **Fastest to build** — no COM interfaces, no RTF format strings, no new APIs.
- **Smallest, simplest binary**.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  Native Win32 Window                                 │
│  ┌──────────────┬───┬──────────────────────────────┐ │
│  │  EDIT control│   │  Owner-drawn child window    │ │
│  │  (multiline) │   │  (WM_PAINT + GDI calls)     │ │
│  │              │   │                              │ │
│  │  raw markdown│   │  DrawTextW per block         │ │
│  │              │   │  CreateFontW per style       │ │
│  │              │   │  FillRect for backgrounds    │ │
│  └──────────────┴───┴──────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
         │                        ▲
         │  EN_CHANGE             │  WM_PAINT
         ▼                        │
    ┌──────────┐           ┌──────────────────┐
    │ Get text │──────────▶│ goldmark AST     │
    │ from EDIT│           │       │          │
    └──────────┘           │ Block Walker     │
                           │       │          │
                           │ []DrawBlock      │
                           │ (text, font,     │
                           │  color, y-pos)   │
                           └──────────────────┘
```

## Implementation Steps

### Step 1: Scaffold (reuse existing patterns)
- Copy the Win32 boilerplate from `tinymdeditor/main.go` — window class registration,
  message loop, the syscall declarations for user32/gdi32/kernel32
- Create main window, EDIT child (left), preview child (right)
- The preview child window gets its own `WndProc` — just like `splashWndProc`
  but extended

### Step 2: Font cache
- Pre-create a set of GDI fonts at startup via `CreateFontW`:
  ```go
  var fonts = struct {
      h1, h2, h3     uintptr  // bold, decreasing sizes
      body           uintptr  // normal 14px
      bodyBold       uintptr  // bold 14px
      bodyItalic     uintptr  // italic 14px
      code           uintptr  // Consolas 13px
      codeBlock      uintptr  // Consolas 13px
  }{}
  ```
- These are created once and reused for every paint — `CreateFontW` is the expensive
  part, `SelectObject` is cheap

### Step 3: Markdown → DrawBlock list
- Add `github.com/yuin/goldmark` dependency
- Walk the AST and produce a flat `[]DrawBlock`:
  ```go
  type DrawBlock struct {
      Text      string
      FontKey   string    // "h1", "body", "code", etc.
      Color     uint32    // RGB for SetTextColor
      BgColor   *uint32   // optional background fill
      Indent    int       // left margin in pixels
      SpaceBefore int     // vertical gap before this block
      Flags     uint32    // DT_WORDBREAK, DT_NOPREFIX, etc.
  }
  ```
- Headings → large bold font, dark color
- Paragraphs → body font, word-wrap
- Code blocks → monospace font, gray background
- List items → body font, indented, with bullet prefix "• "
- Blockquotes → body font, indented, blue left bar (FillRect), gray text
- Horizontal rules → no text, draw a 1px line via `FillRect`

### Step 4: Measurement pass
- Before painting, measure each block to know its height:
  - Use `DrawTextW` with `DT_CALCRECT | DT_WORDBREAK` — this measures without drawing
  - Set the RECT width to the preview pane width minus padding
  - GDI fills in the RECT height → that's the block height
- Stack blocks vertically: each block's Y = previous block's Y + height + spacing
- Track total content height for scrolling

### Step 5: Paint (WM_PAINT handler)
- `BeginPaint` → get HDC
- Fill background white
- For each DrawBlock in visible range (checking against scroll offset):
  1. If BgColor set → `FillRect` the background area
  2. `SelectObject(hdc, fonts[block.FontKey])`
  3. `SetTextColor(hdc, block.Color)`
  4. `SetBkMode(hdc, TRANSPARENT)`
  5. Set RECT to `{left: padding + indent, top: block.Y - scrollY, right: width - padding, bottom: ...}`
  6. `DrawTextW(hdc, text, -1, &rect, DT_WORDBREAK | DT_NOPREFIX)`
- For horizontal rules: `FillRect` a 1px tall strip
- For blockquote left bars: `FillRect` a 3px wide blue strip
- `EndPaint`

### Step 6: Scrolling
- Handle `WM_MOUSEWHEEL` on the preview child window
- Adjust `scrollY`, clamp to `[0, totalContentHeight - viewHeight]`
- `InvalidateRect` to trigger repaint
- Set `WS_VSCROLL` style and maintain scrollbar position via `SetScrollInfo`

### Step 7: Inline formatting (the hard part)
- GDI `DrawTextW` does NOT support mixed fonts in a single call
- For paragraphs with bold/italic/code spans, two options:
  - **Option A (simple):** Treat each inline run as a separate draw call,
    advancing X position manually. Lose word-wrap across style boundaries.
  - **Option B (better):** Draw plain text with `DrawTextW` for word-wrap,
    then overlay bold/italic spans at measured positions using `GetTextExtentPoint32W`.
    Complex but doable.
  - **Option C (pragmatic):** For the prototype, just render paragraphs in
    body font. Bold/italic only works at the block level (headings).
    Acknowledge this limitation.
- Recommend: **Option C for prototype**, upgrade to B later if this approach wins.

### Step 8: File operations
- Same as other prototypes — Ctrl+S, Save As, file loading

## Dependencies
```
github.com/yuin/goldmark  (markdown parser, pure Go)
```
Nothing else. All rendering via existing user32.dll + gdi32.dll syscalls.

## Expected Characteristics
| Property         | Value                                    |
|------------------|------------------------------------------|
| Startup time     | ~30-50ms (fastest possible)              |
| Binary size      | ~3 MB (smallest)                         |
| Memory usage     | ~3-5 MB (lowest)                         |
| Rendering        | GDI — CPU-based, perfectly fine for text |
| Code blocks      | Monospace font + gray background         |
| Tables           | Hard — would need manual column layout   |
| Images           | Not practical with GDI DrawText          |
| Scrolling        | Manual, functional but not smooth        |
| Text selection   | NOT free — would need manual impl        |
| Inline bold/ital | Limited (block-level only in prototype)  |

## Risks
- **No inline mixed formatting** — GDI's DrawTextW is one-font-per-call. This is
  the biggest limitation. Paragraphs with `**bold**` mid-sentence won't render
  the bold inline without significant extra work.
- **No text selection/copy** from the preview (would need to implement hit-testing)
- **Tables are very hard** — manual column measurement and positioning
- **Scrolling is basic** — no smooth scroll, no momentum
- **Looks the most "homemade"** — no antialiased curves, no sub-pixel rendering
  (GDI does ClearType but it's coarser than DirectWrite)

## When to Pick This
- If you just want the absolute fastest, smallest, simplest thing
- If the preview is "good enough" — headings, code blocks, lists, basic paragraphs
- If you plan to iterate toward Direct2D later and want a stepping stone
