# Prototype 2: Direct2D + DirectWrite

## Approach
Full custom rendering pipeline. Parse markdown with `goldmark`, walk the AST, and
draw every element using Direct2D (shapes, backgrounds) and DirectWrite (text layout,
font styling). The preview pane is a custom-drawn surface with manual scrolling.

No browser. No RichEdit. Every pixel is ours.

## Why This Might Win
- **Total visual control** — pixel-perfect rendering, custom themes, smooth animations.
- **Hardware accelerated** — Direct2D uses the GPU. Buttery smooth even with large docs.
- **Closest to "how GitHub feels"** — GitHub renders to a custom layout engine, not RTF.
  This is the approach that can match that quality.
- **Syntax highlighting possible** — we control every glyph's color.

## Architecture

```
┌───────────────────────────────────────────────────────┐
│  Native Win32 Window                                  │
│  ┌──────────────┬───┬───────────────────────────────┐ │
│  │  EDIT control│   │  Direct2D Render Target       │ │
│  │  (multiline) │   │  ┌─────────────────────────┐  │ │
│  │              │   │  │ DirectWrite TextLayouts  │  │ │
│  │  raw markdown│   │  │ + D2D rectangles/lines   │  │ │
│  │              │   │  │ + scroll offset          │  │ │
│  │              │   │  └─────────────────────────┘  │ │
│  └──────────────┴───┴───────────────────────────────┘ │
└───────────────────────────────────────────────────────┘
         │                        ▲
         │  EN_CHANGE             │  WM_PAINT → D2D render
         ▼                        │
    ┌──────────┐           ┌──────────────────┐
    │ Get text │──────────▶│ goldmark AST     │
    │ from EDIT│           │       │          │
    └──────────┘           │ Layout Walker    │
                           │       │          │
                           │ LayoutNode tree  │
                           │ (position, size, │
                           │  style per block)│
                           └──────────────────┘
```

## Implementation Steps

### Step 1: Direct2D / DirectWrite bindings
- Use `github.com/bupjae/direct` (no CGo wrapper) OR hand-roll COM vtable calls
  via syscall (Direct2D and DirectWrite are COM interfaces)
- Key interfaces needed:
  - `ID2D1Factory` → create render targets
  - `ID2D1HwndRenderTarget` → draw to a window
  - `ID2D1SolidColorBrush` → text and shape colors
  - `IDWriteFactory` → create text formats and layouts
  - `IDWriteTextFormat` → font family, size, weight, style
  - `IDWriteTextLayout` → measure and render text with mixed formatting
- Evaluate `bupjae/direct` first; if incomplete, use `golang.org/x/sys/windows`
  to call COM methods directly through vtable pointers

### Step 2: Scaffold the window
- Main window with Win32 message loop
- Left pane: standard EDIT control (same as prototype 1)
- Right pane: a child window that we own and paint via Direct2D
- Create `ID2D1HwndRenderTarget` bound to the right pane's HWND
- Handle `WM_SIZE` → resize render target

### Step 3: Layout engine
- Define a `LayoutNode` struct:
  ```go
  type LayoutNode struct {
      Type     NodeType  // paragraph, heading, code, list, blockquote, hr, table
      Text     string
      Style    TextStyle // font size, weight, italic, color, font family
      X, Y     float32   // position
      W, H     float32   // measured size
      BgColor  *Color    // optional background (code blocks, blockquotes)
      Children []LayoutNode
  }
  ```
- Walk goldmark AST → produce a flat list of LayoutNodes
- Measure each node using `IDWriteTextLayout::GetMetrics` to get exact height
- Stack nodes vertically with appropriate margins
- Track total content height for scroll range

### Step 4: Rendering
- On `WM_PAINT` (or on-demand after text change):
  1. `BeginDraw()`
  2. Clear background (white)
  3. Apply scroll offset transform: `SetTransform(translate(0, -scrollY))`
  4. For each LayoutNode in visible range:
     - Draw background rect if needed (`FillRectangle`)
     - Draw left border for blockquotes (`FillRectangle` thin strip)
     - Draw horizontal rule (`DrawLine`)
     - Draw text via `DrawTextLayout` at node's position
  5. `EndDraw()`

### Step 5: Scrolling
- Track `scrollY` offset (float32)
- Handle `WM_MOUSEWHEEL` → adjust scrollY, clamp to [0, totalHeight - viewHeight]
- Handle `WM_VSCROLL` if we add a scrollbar
- Call `InvalidateRect` to trigger repaint
- Optional: smooth scrolling with `SetTimer` animation

### Step 6: Mixed-format text (inline styles)
- For paragraphs with bold/italic/code spans:
  - Use `IDWriteTextLayout` with the full paragraph text
  - Apply `IDWriteTextLayout::SetFontWeight` on character ranges for bold
  - Apply `IDWriteTextLayout::SetFontStyle` for italic
  - Apply `IDWriteTextLayout::SetFontFamilyName` for code spans (switch to monospace)
  - Apply `IDWriteTextLayout::SetDrawingEffect` for colored text
- This gives us rich inline formatting with proper word-wrap

### Step 7: File operations
- Same as prototype 1 — Ctrl+S, Save As dialog, file loading

### Step 8: Polish
- Code block backgrounds: rounded rectangles with `FillRoundedRectangle`
- Heading underlines: `DrawLine` below h1/h2
- List bullets: draw filled circles or numbers
- Link coloring: blue text via `SetDrawingEffect`
- Optional: syntax highlighting in code blocks using a tokenizer

## Dependencies
```
github.com/yuin/goldmark           (markdown parser)
github.com/bupjae/direct           (Direct2D/DirectWrite, no CGo) — evaluate first
  OR hand-rolled COM vtable calls via syscall
```

## Expected Characteristics
| Property         | Value                                    |
|------------------|------------------------------------------|
| Startup time     | ~50-80ms (D2D factory + render target)   |
| Binary size      | ~4-5 MB                                  |
| Memory usage     | ~15-25 MB (GPU render target + layouts)  |
| Rendering        | GPU-accelerated, pixel-perfect           |
| Code blocks      | Monospace + background rect, syntax HL possible |
| Tables           | Custom drawn (full control over styling) |
| Images           | Possible via WIC (Windows Imaging Component) |
| Scrolling        | Manual but can be butter-smooth          |
| Text selection   | NOT free — would need hit-testing impl   |

## Risks
- **Most code to write** — layout engine, scrolling, hit-testing are all manual
- **COM vtable calls from Go** are tedious if `bupjae/direct` doesn't cover
  everything we need
- **Text selection / copy** is a significant feature to implement from scratch
- **Edge cases in layout** — deeply nested lists, wide tables, long code blocks
  all need careful measurement
- Worth it if we want GitHub-quality rendering; overkill if "good enough" suffices
