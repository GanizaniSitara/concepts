// TinyMD GDI Prototype — native markdown preview using GDI DrawText.
//
// Build: GOOS=windows go build -ldflags="-s -w -H windowsgui" -trimpath -o tinymd-gdi.exe .

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"unsafe"

	"github.com/yuin/goldmark"
	"github.com/yuin/goldmark/ast"
	"github.com/yuin/goldmark/extension"
	extast "github.com/yuin/goldmark/extension/ast"
	"github.com/yuin/goldmark/text"
)

// Win32 DLLs and procs
var (
	user32 = syscall.NewLazyDLL("user32.dll")
	gdi32  = syscall.NewLazyDLL("gdi32.dll")
	kernel32 = syscall.NewLazyDLL("kernel32.dll")
	comdlg32 = syscall.NewLazyDLL("comdlg32.dll")

	// user32
	registerClassExW    = user32.NewProc("RegisterClassExW")
	createWindowExW     = user32.NewProc("CreateWindowExW")
	showWindowProc      = user32.NewProc("ShowWindow")
	updateWindowProc    = user32.NewProc("UpdateWindow")
	destroyWindowProc   = user32.NewProc("DestroyWindow")
	defWindowProcW      = user32.NewProc("DefWindowProcW")
	getSystemMetrics    = user32.NewProc("GetSystemMetrics")
	beginPaint          = user32.NewProc("BeginPaint")
	endPaint            = user32.NewProc("EndPaint")
	fillRect            = user32.NewProc("FillRect")
	drawTextW           = user32.NewProc("DrawTextW")
	loadCursorW         = user32.NewProc("LoadCursorW")
	getClientRect       = user32.NewProc("GetClientRect")
	getSysColorBrush    = user32.NewProc("GetSysColorBrush")
	getMessageW         = user32.NewProc("GetMessageW")
	translateMessage    = user32.NewProc("TranslateMessage")
	dispatchMessageW    = user32.NewProc("DispatchMessageW")
	postQuitMessage     = user32.NewProc("PostQuitMessage")
	sendMessageW        = user32.NewProc("SendMessageW")
	moveWindow          = user32.NewProc("MoveWindow")
	invalidateRect      = user32.NewProc("InvalidateRect")
	setFocus            = user32.NewProc("SetFocus")
	setTimer            = user32.NewProc("SetTimer")
	killTimer           = user32.NewProc("KillTimer")
	getWindowTextLengthW = user32.NewProc("GetWindowTextLengthW")
	getWindowTextW      = user32.NewProc("GetWindowTextW")
	setScrollInfo       = user32.NewProc("SetScrollInfo")
	getScrollInfo       = user32.NewProc("GetScrollInfo")
	getForegroundWin    = user32.NewProc("GetForegroundWindow")

	// gdi32
	createFontW      = gdi32.NewProc("CreateFontW")
	selectObject     = gdi32.NewProc("SelectObject")
	setBkMode        = gdi32.NewProc("SetBkMode")
	setTextColor     = gdi32.NewProc("SetTextColor")
	deleteObject     = gdi32.NewProc("DeleteObject")
	createSolidBrush = gdi32.NewProc("CreateSolidBrush")
	createPen        = gdi32.NewProc("CreatePen")
	moveToEx         = gdi32.NewProc("MoveToEx")
	lineTo           = gdi32.NewProc("LineTo")
	createCompatibleDC   = gdi32.NewProc("CreateCompatibleDC")
	createCompatibleBitmap = gdi32.NewProc("CreateCompatibleBitmap")
	bitBlt           = gdi32.NewProc("BitBlt")
	deleteDC         = gdi32.NewProc("DeleteDC")
	ellipseProc      = gdi32.NewProc("Ellipse")
	textOutW         = gdi32.NewProc("TextOutW")
	getTextExtentPoint32W = gdi32.NewProc("GetTextExtentPoint32W")

	// kernel32
	getModuleHandleW = kernel32.NewProc("GetModuleHandleW")

	// comdlg32
	getSaveFileNameW = comdlg32.NewProc("GetSaveFileNameW")
	printDlgW        = comdlg32.NewProc("PrintDlgW")

	// gdi32 — printing
	startDocW  = gdi32.NewProc("StartDocW")
	endDoc     = gdi32.NewProc("EndDoc")
	startPage  = gdi32.NewProc("StartPage")
	endPage    = gdi32.NewProc("EndPage")
	getDeviceCaps = gdi32.NewProc("GetDeviceCaps")
)

// Win32 constants
const (
	WS_OVERLAPPEDWINDOW = 0x00CF0000
	WS_VISIBLE          = 0x10000000
	WS_CHILD            = 0x40000000
	WS_VSCROLL          = 0x00200000
	WS_HSCROLL          = 0x00100000
	WS_BORDER           = 0x00800000
	WS_CLIPCHILDREN     = 0x02000000
	WS_EX_CLIENTEDGE    = 0x00000200
	ES_MULTILINE        = 0x0004
	ES_AUTOVSCROLL      = 0x0040
	ES_WANTRETURN       = 0x1000
	SM_CXSCREEN          = 0
	SM_CYSCREEN          = 1
	IDC_ARROW            = 32512
	CW_USEDEFAULT        = 0x80000000

	WM_CREATE    = 0x0001
	WM_DESTROY   = 0x0002
	WM_SIZE      = 0x0005
	WM_PAINT     = 0x000F
	WM_COMMAND   = 0x0111
	WM_TIMER     = 0x0113
	WM_KEYDOWN   = 0x0100
	WM_MOUSEWHEEL = 0x020A
	WM_VSCROLL   = 0x0115
	WM_SETFONT   = 0x0030
	WM_SETFOCUS  = 0x0007
	WM_ERASEBKGND = 0x0014

	EN_CHANGE = 0x0300

	VK_S = 0x53
	VK_P = 0x50
	VK_TAB = 0x09

	DT_WORDBREAK  = 0x0010
	DT_NOPREFIX   = 0x0800
	DT_CALCRECT   = 0x0400
	DT_EXPANDTABS = 0x0040
	DT_LEFT       = 0x0000
	DT_EDITCONTROL = 0x2000
	DT_SINGLELINE  = 0x0020
	DT_NOCLIP      = 0x0100

	TRANSPARENT = 1

	SB_VERT      = 1
	SIF_ALL      = 0x17
	SIF_POS      = 0x04
	SIF_RANGE    = 0x01
	SIF_PAGE     = 0x02
	SIF_DISABLENOSCROLL = 0x08

	SB_THUMBTRACK  = 5
	SB_THUMBPOSITION = 4
	SB_LINEUP      = 0
	SB_LINEDOWN    = 1
	SB_PAGEUP      = 2
	SB_PAGEDOWN    = 3

	SRCCOPY = 0x00CC0020

	MK_CONTROL = 0x0008

	TIMER_DEBOUNCE = 1

	// Printing
	PD_RETURNDC        = 0x00000100
	PD_USEDEVMODECOPIESANDCOLLATE = 0x00040000
	LOGPIXELSX         = 88
	LOGPIXELSY         = 90
	HORZRES            = 8
	VERTRES            = 10
	PHYSICALOFFSETX    = 112
	PHYSICALOFFSETY    = 113
)

// Helpers
func utf16From(s string) []uint16 {
	r, _ := syscall.UTF16FromString(s)
	return r
}

func utf16Ptr(s string) *uint16 {
	p, _ := syscall.UTF16PtrFromString(s)
	return p
}

func loword(v uintptr) uint16 { return uint16(v & 0xFFFF) }
func hiword(v uintptr) uint16 { return uint16((v >> 16) & 0xFFFF) }

// Font cache
var (
	fontH1         uintptr
	fontH2         uintptr
	fontH3         uintptr
	fontBody       uintptr
	fontBodyBold   uintptr
	fontBodyItalic uintptr
	fontCode       uintptr
)

func makeFont(height, weight int32, italic uint32, face string) uintptr {
	f, _, _ := createFontW.Call(
		uintptr(uint32(uint16(height))|0xFFFF0000), // negative height for pixel size
		0, 0, 0,
		uintptr(weight),
		uintptr(italic),
		0, 0, 0, 0, 0, 0, 0,
		uintptr(unsafe.Pointer(utf16Ptr(face))),
	)
	return f
}

func initFonts() {
	fontH1 = makeFont(-28, 700, 0, "Segoe UI")
	fontH2 = makeFont(-22, 700, 0, "Segoe UI")
	fontH3 = makeFont(-18, 700, 0, "Segoe UI")
	fontBody = makeFont(-14, 400, 0, "Segoe UI")
	fontBodyBold = makeFont(-14, 700, 0, "Segoe UI")
	fontBodyItalic = makeFont(-14, 400, 1, "Segoe UI")
	fontCode = makeFont(-14, 400, 0, "Consolas")
}

func destroyFonts() {
	for _, f := range []uintptr{fontH1, fontH2, fontH3, fontBody, fontBodyBold, fontBodyItalic, fontCode} {
		if f != 0 {
			deleteObject.Call(f)
		}
	}
}

// Table data types
type TableCell struct {
	Text string
	Bold bool
}
type TableData struct {
	Headers []TableCell
	Rows    [][]TableCell
}

// InlineSpan represents a run of text with inline formatting
type InlineSpan struct {
	Text   string
	Bold   bool
	Italic bool
	Code   bool
}

// DrawBlock represents a rendered markdown block
type DrawBlock struct {
	Text       string
	Font       uintptr
	Color      uint32
	BgColor    uint32
	HasBg      bool
	Indent     int32
	SpaceAbove int32
	IsHR       bool
	IsQuote    bool // draw blue left bar
	IsBullet   bool // draw filled circle bullet
	IsTable    bool
	Table      *TableData
	Runs       []InlineSpan
}

// Markdown → blocks
func markdownToBlocks(source []byte) []DrawBlock {
	md := goldmark.New(goldmark.WithExtensions(extension.Table))
	reader := text.NewReader(source)
	doc := md.Parser().Parse(reader)

	var blocks []DrawBlock
	ast.Walk(doc, func(n ast.Node, entering bool) (ast.WalkStatus, error) {
		if !entering {
			return ast.WalkContinue, nil
		}
		switch n.Kind() {
		case ast.KindHeading:
			h := n.(*ast.Heading)
			txt := extractText(n, source)
			var font uintptr
			var space int32
			switch h.Level {
			case 1:
				font = fontH1
				space = 24
			case 2:
				font = fontH2
				space = 20
			default:
				font = fontH3
				space = 16
			}
			blocks = append(blocks, DrawBlock{
				Text: txt, Font: font, Color: 0x00111111,
				SpaceAbove: space,
			})
			return ast.WalkSkipChildren, nil

		case ast.KindParagraph:
			if n.Parent() != nil && n.Parent().Kind() == ast.KindBlockquote {
				return ast.WalkContinue, nil
			}
			txt := extractInlineText(n, source)
			if txt != "" {
				runs := extractInlineRuns(n, source)
				blocks = append(blocks, DrawBlock{
					Text: txt, Font: fontBody, Color: 0x00222222,
					SpaceAbove: 12, Runs: runs,
				})
			}
			return ast.WalkSkipChildren, nil

		case ast.KindFencedCodeBlock, ast.KindCodeBlock:
			var buf strings.Builder
			lines := n.Lines()
			for i := 0; i < lines.Len(); i++ {
				seg := lines.At(i)
				buf.Write(seg.Value(source))
			}
			blocks = append(blocks, DrawBlock{
				Text: strings.TrimRight(buf.String(), "\n"),
				Font: fontCode, Color: 0x00333333,
				BgColor: 0x00FAF8F6, HasBg: true,
				SpaceAbove: 12,
			})
			return ast.WalkSkipChildren, nil

		case ast.KindBlockquote:
			txt := extractInlineText(n, source)
			runs := extractInlineRuns(n, source)
			blocks = append(blocks, DrawBlock{
				Text: txt, Font: fontBody, Color: 0x00666666,
				Indent: 20, SpaceAbove: 12, IsQuote: true, Runs: runs,
			})
			return ast.WalkSkipChildren, nil

		case ast.KindListItem:
			txt := extractInlineText(n, source)
			runs := extractInlineRuns(n, source)
			blocks = append(blocks, DrawBlock{
				Text: txt, Font: fontBody, Color: 0x00222222,
				Indent: 30, SpaceAbove: 6, Runs: runs, IsBullet: true,
			})
			return ast.WalkSkipChildren, nil

		case ast.KindThematicBreak:
			blocks = append(blocks, DrawBlock{
				IsHR: true, SpaceAbove: 16,
			})

		default:
			// Handle table nodes from the extension
			if n.Kind() == extast.KindTable {
				td := &TableData{}
				for child := n.FirstChild(); child != nil; child = child.NextSibling() {
					if child.Kind() == extast.KindTableHeader {
						// Extract header cells from the header row
						for cell := child.FirstChild(); cell != nil; cell = cell.NextSibling() {
							if cell.Kind() == extast.KindTableCell {
								td.Headers = append(td.Headers, TableCell{
									Text: extractInlineText(cell, source),
									Bold: true,
								})
							}
						}
					} else if child.Kind() == extast.KindTableRow {
						// Body rows are direct children of Table (no TableBody wrapper)
						var cells []TableCell
						for cell := child.FirstChild(); cell != nil; cell = cell.NextSibling() {
							if cell.Kind() == extast.KindTableCell {
								cells = append(cells, TableCell{
									Text: extractInlineText(cell, source),
									Bold: false,
								})
							}
						}
						td.Rows = append(td.Rows, cells)
					}
				}
				blocks = append(blocks, DrawBlock{
					IsTable:    true,
					Table:      td,
					SpaceAbove: 12,
				})
				return ast.WalkSkipChildren, nil
			}
		}
		return ast.WalkContinue, nil
	})
	return blocks
}

func extractText(n ast.Node, source []byte) string {
	var buf strings.Builder
	for c := n.FirstChild(); c != nil; c = c.NextSibling() {
		if c.Kind() == ast.KindText {
			t := c.(*ast.Text)
			buf.Write(t.Segment.Value(source))
			if t.SoftLineBreak() {
				buf.WriteByte(' ')
			}
		} else {
			buf.WriteString(extractText(c, source))
		}
	}
	return buf.String()
}

func extractInlineText(n ast.Node, source []byte) string {
	var buf strings.Builder
	ast.Walk(n, func(c ast.Node, entering bool) (ast.WalkStatus, error) {
		if !entering {
			// When leaving an emphasis node, check whether a space is needed
			// between it and its next sibling (goldmark may strip the leading
			// space from the following text segment inside blockquotes).
			if c.Kind() == ast.KindEmphasis && c.NextSibling() != nil {
				needsSpace := false
				next := c.NextSibling()
				emphEnd := 0
				for last := c.LastChild(); last != nil; last = last.LastChild() {
					if last.Kind() == ast.KindText {
						emphEnd = last.(*ast.Text).Segment.Stop
						break
					}
				}
				nextStart := 0
				if next.Kind() == ast.KindText {
					nextStart = next.(*ast.Text).Segment.Start
				} else if next.Kind() == ast.KindEmphasis {
					for fc := next.FirstChild(); fc != nil; fc = fc.FirstChild() {
						if fc.Kind() == ast.KindText {
							nextStart = fc.(*ast.Text).Segment.Start
							break
						}
					}
				}
				if emphEnd > 0 && nextStart > emphEnd {
					between := source[emphEnd:nextStart]
					for _, b := range between {
						if b == ' ' || b == '\t' {
							needsSpace = true
							break
						}
					}
				} else if emphEnd > 0 {
					needsSpace = true
				}
				if needsSpace && buf.Len() > 0 {
					s := buf.String()
					if s[len(s)-1] != ' ' {
						buf.WriteByte(' ')
					}
				}
			}
			return ast.WalkContinue, nil
		}
		switch c.Kind() {
		case ast.KindText:
			t := c.(*ast.Text)
			txt := t.Segment.Value(source)
			// Fix missing space after emphasis in blockquotes: if previous
			// sibling is an emphasis node and this text doesn't start with
			// a space, insert one.
			if buf.Len() > 0 && len(txt) > 0 && txt[0] != ' ' {
				prev := c.PreviousSibling()
				if prev != nil && prev.Kind() == ast.KindEmphasis {
					s := buf.String()
					if s[len(s)-1] != ' ' {
						buf.WriteByte(' ')
					}
				}
			}
			buf.Write(txt)
			if t.SoftLineBreak() {
				buf.WriteByte(' ')
			}
		case ast.KindCodeSpan:
			// extract code span text
			for gc := c.FirstChild(); gc != nil; gc = gc.NextSibling() {
				if gc.Kind() == ast.KindText {
					buf.Write(gc.(*ast.Text).Segment.Value(source))
				}
			}
			return ast.WalkSkipChildren, nil
		case ast.KindString:
			buf.Write(c.(*ast.String).Value)
		}
		return ast.WalkContinue, nil
	})
	return buf.String()
}

func extractInlineRuns(n ast.Node, source []byte) []InlineSpan {
	var runs []InlineSpan
	ast.Walk(n, func(c ast.Node, entering bool) (ast.WalkStatus, error) {
		if !entering {
			// When leaving an emphasis node, check if a space separator is
			// needed between the last run and the next sibling.  Goldmark
			// sometimes strips leading whitespace from text segments in
			// blockquotes and other containers, so the space between e.g.
			// "*italic* text." can be lost.  Detect this by looking at the
			// raw source bytes between the emphasis closing delimiter and
			// the start of the next sibling node.
			if c.Kind() == ast.KindEmphasis && c.NextSibling() != nil {
				needsSpace := false
				next := c.NextSibling()

				// Find the end offset of the emphasis node (last child's segment end)
				emphEnd := 0
				for last := c.LastChild(); last != nil; last = last.LastChild() {
					if last.Kind() == ast.KindText {
						emphEnd = last.(*ast.Text).Segment.Stop
						break
					}
				}

				// Find the start offset of the next sibling
				nextStart := 0
				if next.Kind() == ast.KindText {
					nextStart = next.(*ast.Text).Segment.Start
				} else if next.Kind() == ast.KindEmphasis {
					for fc := next.FirstChild(); fc != nil; fc = fc.FirstChild() {
						if fc.Kind() == ast.KindText {
							nextStart = fc.(*ast.Text).Segment.Start
							break
						}
					}
				}

				if emphEnd > 0 && nextStart > emphEnd {
					// Check source bytes between emphasis end and next node start
					between := source[emphEnd:nextStart]
					for _, b := range between {
						if b == ' ' || b == '\t' {
							needsSpace = true
							break
						}
					}
				} else if emphEnd > 0 {
					// Fallback: if offsets don't help (e.g. blockquote remapping),
					// check whether the last run ends without a space and the
					// next sibling likely needs one.
					if len(runs) > 0 {
						last := runs[len(runs)-1].Text
						if len(last) > 0 && last[len(last)-1] != ' ' {
							needsSpace = true
						}
					}
				}

				if needsSpace {
					// Only inject if the last run doesn't already end with space
					if len(runs) > 0 {
						last := runs[len(runs)-1].Text
						if len(last) > 0 && last[len(last)-1] != ' ' {
							runs = append(runs, InlineSpan{Text: " "})
						}
					}
				}
			}
			return ast.WalkContinue, nil
		}
		switch c.Kind() {
		case ast.KindText:
			t := c.(*ast.Text)
			text := string(t.Segment.Value(source))
			if t.SoftLineBreak() {
				text += " "
			}
			bold := false
			italic := false
			for p := c.Parent(); p != nil && p != n; p = p.Parent() {
				if p.Kind() == ast.KindEmphasis {
					em := p.(*ast.Emphasis)
					if em.Level == 2 {
						bold = true
					} else {
						italic = true
					}
				}
			}
			// If this text node follows an emphasis node (as a sibling or
			// the emphasis is the previous sibling of an ancestor), the
			// leading space may be lost due to blockquote segment remapping.
			// Detect and fix: if the previous run doesn't end with a space
			// and this text doesn't start with a space, insert one.
			if len(runs) > 0 && len(text) > 0 && text[0] != ' ' {
				prev := c.PreviousSibling()
				if prev != nil && prev.Kind() == ast.KindEmphasis {
					lastRun := runs[len(runs)-1].Text
					if len(lastRun) > 0 && lastRun[len(lastRun)-1] != ' ' {
						runs = append(runs, InlineSpan{Text: " "})
					}
				}
			}
			runs = append(runs, InlineSpan{Text: text, Bold: bold, Italic: italic})
		case ast.KindCodeSpan:
			var buf strings.Builder
			for gc := c.FirstChild(); gc != nil; gc = gc.NextSibling() {
				if gc.Kind() == ast.KindText {
					buf.Write(gc.(*ast.Text).Segment.Value(source))
				}
			}
			runs = append(runs, InlineSpan{Text: buf.String(), Code: true})
			return ast.WalkSkipChildren, nil
		case ast.KindString:
			runs = append(runs, InlineSpan{Text: string(c.(*ast.String).Value)})
		}
		return ast.WalkContinue, nil
	})
	return runs
}

// Global state
var (
	hInstance    uintptr
	mainHwnd    uintptr
	editorHwnd  uintptr
	previewHwnd uintptr

	currentFile string
	currentBlocks []DrawBlock
	scrollY     int32
	totalHeight int32

	grayBrush uintptr
	blueBrush uintptr
	codeBrush uintptr
)

// Main window proc
func mainWndProc(hwnd, msg, wParam, lParam uintptr) uintptr {
	switch msg {
	case WM_CREATE:
		editorHwnd, _, _ = createWindowExW.Call(
			WS_EX_CLIENTEDGE,
			uintptr(unsafe.Pointer(utf16Ptr("EDIT"))),
			0,
			WS_CHILD|WS_VISIBLE|WS_VSCROLL|ES_MULTILINE|ES_AUTOVSCROLL|ES_WANTRETURN,
			0, 0, 0, 0,
			hwnd, 1, hInstance, 0,
		)
		sendMessageW.Call(editorHwnd, WM_SETFONT, fontCode, 1)

		// Set editor margins so text doesn't touch the edges
		const EM_SETMARGINS = 0x00D3
		margins := uintptr((10 << 16) | 10) // 10px left and right
		sendMessageW.Call(editorHwnd, EM_SETMARGINS, 3, margins) // EC_LEFTMARGIN|EC_RIGHTMARGIN = 3

		// Register preview window class
		previewClass := utf16From("TinyMDGDIPreview")
		previewBgBrush, _, _ := createSolidBrush.Call(0x00FFFFFF) // pure white background
		var wc [80]byte
		*(*uint32)(unsafe.Pointer(&wc[0])) = 80
		*(*uintptr)(unsafe.Pointer(&wc[8])) = syscall.NewCallback(previewWndProc)
		*(*uintptr)(unsafe.Pointer(&wc[24])) = hInstance
		cursor, _, _ := loadCursorW.Call(0, IDC_ARROW)
		*(*uintptr)(unsafe.Pointer(&wc[40])) = cursor
		*(*uintptr)(unsafe.Pointer(&wc[48])) = previewBgBrush // pure white
		*(*uintptr)(unsafe.Pointer(&wc[64])) = uintptr(unsafe.Pointer(&previewClass[0]))
		registerClassExW.Call(uintptr(unsafe.Pointer(&wc[0])))

		previewHwnd, _, _ = createWindowExW.Call(
			WS_EX_CLIENTEDGE,
			uintptr(unsafe.Pointer(&previewClass[0])),
			0,
			WS_CHILD|WS_VISIBLE|WS_VSCROLL,
			0, 0, 0, 0,
			hwnd, 2, hInstance, 0,
		)

		// Create brushes
		grayBrush, _, _ = createSolidBrush.Call(0x00DDDDDD)
		blueBrush, _, _ = createSolidBrush.Call(0x00D66603) // BGR for #0366d6
		codeBrush, _, _ = createSolidBrush.Call(0x00FAF8F6) // light gray bg

		return 0

	case WM_SIZE:
		w := int32(loword(lParam))
		h := int32(hiword(lParam))
		half := w / 2
		divider := int32(6)
		moveWindow.Call(editorHwnd, 0, 0, uintptr(half-divider/2), uintptr(h), 1)
		moveWindow.Call(previewHwnd, uintptr(half+divider/2), 0, uintptr(w-half-divider/2), uintptr(h), 1)
		return 0

	case WM_COMMAND:
		if hiword(wParam) == EN_CHANGE && loword(wParam) == 1 {
			// Debounce: set 50ms timer
			setTimer.Call(hwnd, TIMER_DEBOUNCE, 50, 0)
		}
		return 0

	case WM_TIMER:
		if wParam == TIMER_DEBOUNCE {
			killTimer.Call(hwnd, TIMER_DEBOUNCE)
			updatePreview()
		}
		return 0

	case WM_KEYDOWN:
		// Ctrl+S
		state, _, _ := user32.NewProc("GetKeyState").Call(0x11) // VK_CONTROL
		if int16(state) < 0 && wParam == VK_S {
			saveFile()
			return 0
		}
		if int16(state) < 0 && wParam == VK_P {
			printFormatted()
			return 0
		}

	case WM_SETFOCUS:
		setFocus.Call(editorHwnd)
		return 0

	case WM_DESTROY:
		destroyFonts()
		deleteObject.Call(grayBrush)
		deleteObject.Call(blueBrush)
		deleteObject.Call(codeBrush)
		postQuitMessage.Call(0)
		return 0
	}

	ret, _, _ := defWindowProcW.Call(hwnd, msg, wParam, lParam)
	return ret
}

func updatePreview() {
	length, _, _ := getWindowTextLengthW.Call(editorHwnd)
	if length == 0 {
		currentBlocks = nil
		scrollY = 0
		invalidateRect.Call(previewHwnd, 0, 1)
		return
	}
	buf := make([]uint16, length+1)
	getWindowTextW.Call(editorHwnd, uintptr(unsafe.Pointer(&buf[0])), length+1)
	mdText := syscall.UTF16ToString(buf)

	currentBlocks = markdownToBlocks([]byte(mdText))
	invalidateRect.Call(previewHwnd, 0, 1)
}

// Preview window proc
func previewWndProc(hwnd, msg, wParam, lParam uintptr) uintptr {
	switch msg {
	case WM_ERASEBKGND:
		return 1 // we handle background in WM_PAINT (double-buffered)

	case WM_PAINT:
		var ps [72]byte
		hdc, _, _ := beginPaint.Call(hwnd, uintptr(unsafe.Pointer(&ps[0])))
		if hdc == 0 {
			break
		}

		var rc [16]byte
		getClientRect.Call(hwnd, uintptr(unsafe.Pointer(&rc[0])))
		clientW := *(*int32)(unsafe.Pointer(&rc[8]))
		clientH := *(*int32)(unsafe.Pointer(&rc[12]))

		// Double-buffer: create offscreen DC
		memDC, _, _ := createCompatibleDC.Call(hdc)
		memBmp, _, _ := createCompatibleBitmap.Call(hdc, uintptr(clientW), uintptr(clientH))
		oldBmp, _, _ := selectObject.Call(memDC, memBmp)

		// Fill white background (pure white, not system color)
		whiteBrush, _, _ := createSolidBrush.Call(0x00FFFFFF) // RGB(255,255,255)
		fillRect.Call(memDC, uintptr(unsafe.Pointer(&rc[0])), whiteBrush)
		deleteObject.Call(whiteBrush)
		setBkMode.Call(memDC, TRANSPARENT)

		padding := int32(30)
		drawWidth := clientW - padding*2
		y := padding - scrollY

		for i := range currentBlocks {
			b := &currentBlocks[i]
			y += b.SpaceAbove

			if b.IsHR {
				// Draw a 2px gray horizontal rule using a pen
				hrPen, _, _ := createPen.Call(0, 1, 0x00BBBBBB) // PS_SOLID, 1px, gray
				oldHRPen, _, _ := selectObject.Call(memDC, hrPen)
				hrY := y + 8
				moveToEx.Call(memDC, uintptr(padding), uintptr(hrY), 0)
				lineTo.Call(memDC, uintptr(clientW-padding), uintptr(hrY))
				selectObject.Call(memDC, oldHRPen)
				deleteObject.Call(hrPen)
				y += 16
				continue
			}

			// Table rendering
			if b.IsTable && b.Table != nil {
				td := b.Table
				numCols := len(td.Headers)
				if numCols == 0 {
					continue
				}
				cellPad := int32(8)

				// --- Measure column widths based on content ---
				colWidths := make([]int32, numCols)

				// Measure header text widths (using bold header font)
				oldFont, _, _ := selectObject.Call(memDC, fontH3)
				for ci, cell := range td.Headers {
					cellTxt := utf16From(cell.Text)
					var mRC [16]byte
					*(*int32)(unsafe.Pointer(&mRC[8])) = 10000
					*(*int32)(unsafe.Pointer(&mRC[12])) = 10000
					drawTextW.Call(memDC, uintptr(unsafe.Pointer(&cellTxt[0])),
						uintptr(len(cellTxt)-1),
						uintptr(unsafe.Pointer(&mRC[0])),
						DT_CALCRECT|DT_NOPREFIX|DT_SINGLELINE)
					w := *(*int32)(unsafe.Pointer(&mRC[8]))
					if w > colWidths[ci] {
						colWidths[ci] = w
					}
				}
				selectObject.Call(memDC, oldFont)

				// Measure body cell text widths (using body font)
				oldFont, _, _ = selectObject.Call(memDC, fontBody)
				for _, row := range td.Rows {
					for ci, cell := range row {
						if ci >= numCols {
							break
						}
						cellTxt := utf16From(cell.Text)
						var mRC [16]byte
						*(*int32)(unsafe.Pointer(&mRC[8])) = 10000
						*(*int32)(unsafe.Pointer(&mRC[12])) = 10000
						drawTextW.Call(memDC, uintptr(unsafe.Pointer(&cellTxt[0])),
							uintptr(len(cellTxt)-1),
							uintptr(unsafe.Pointer(&mRC[0])),
							DT_CALCRECT|DT_NOPREFIX|DT_SINGLELINE)
						w := *(*int32)(unsafe.Pointer(&mRC[8]))
						if w > colWidths[ci] {
							colWidths[ci] = w
						}
					}
				}

				// Add padding to each column width
				for ci := range colWidths {
					colWidths[ci] += cellPad * 2
				}

				// Measure row height
				sampleTxt := utf16From("Ay")
				var measureRC [16]byte
				*(*int32)(unsafe.Pointer(&measureRC[8])) = 10000
				*(*int32)(unsafe.Pointer(&measureRC[12])) = 10000
				drawTextW.Call(memDC, uintptr(unsafe.Pointer(&sampleTxt[0])),
					uintptr(len(sampleTxt)-1),
					uintptr(unsafe.Pointer(&measureRC[0])),
					DT_CALCRECT|DT_NOPREFIX|DT_SINGLELINE)
				rowH := *(*int32)(unsafe.Pointer(&measureRC[12])) + cellPad*2
				selectObject.Call(memDC, oldFont)

				// Total rows = 1 header + N body rows
				totalRows := 1 + int32(len(td.Rows))
				tableH := totalRows * rowH

				// Calculate column left positions
				colLeft := make([]int32, numCols+1) // +1 for right edge
				colLeft[0] = padding
				for ci := 0; ci < numCols; ci++ {
					colLeft[ci+1] = colLeft[ci] + colWidths[ci]
				}
				tableRight := colLeft[numCols]

				// Create thin gray pen for borders
				tablePen, _, _ := createPen.Call(0, 1, 0x00DDDDDD)
				oldPen, _, _ := selectObject.Call(memDC, tablePen)

				tableTop := y

				// Only draw if visible
				if tableTop+tableH >= 0 && tableTop < clientH {
					setTextColor.Call(memDC, uintptr(uint32(0x00222222)))

					// Draw header row
					{
						oldFont, _, _ := selectObject.Call(memDC, fontH3)
						for ci, cell := range td.Headers {
							cl := colLeft[ci]
							cr := colLeft[ci+1]
							cellTop := tableTop
							var cellRC [16]byte
							*(*int32)(unsafe.Pointer(&cellRC[0])) = cl + cellPad
							*(*int32)(unsafe.Pointer(&cellRC[4])) = cellTop + cellPad
							*(*int32)(unsafe.Pointer(&cellRC[8])) = cr - cellPad
							*(*int32)(unsafe.Pointer(&cellRC[12])) = cellTop + rowH - cellPad
							cellTxt := utf16From(cell.Text)
							drawTextW.Call(memDC, uintptr(unsafe.Pointer(&cellTxt[0])),
								uintptr(len(cellTxt)-1),
								uintptr(unsafe.Pointer(&cellRC[0])),
								DT_NOPREFIX|DT_SINGLELINE|DT_NOCLIP)
							// Cell borders
							moveToEx.Call(memDC, uintptr(cl), uintptr(cellTop))
							lineTo.Call(memDC, uintptr(cr), uintptr(cellTop))
							moveToEx.Call(memDC, uintptr(cl), uintptr(cellTop))
							lineTo.Call(memDC, uintptr(cl), uintptr(cellTop+rowH))
							moveToEx.Call(memDC, uintptr(cr), uintptr(cellTop))
							lineTo.Call(memDC, uintptr(cr), uintptr(cellTop+rowH))
							moveToEx.Call(memDC, uintptr(cl), uintptr(cellTop+rowH))
							lineTo.Call(memDC, uintptr(cr), uintptr(cellTop+rowH))
						}
						// Draw right edge of table at header bottom
						moveToEx.Call(memDC, uintptr(tableRight), uintptr(tableTop))
						lineTo.Call(memDC, uintptr(tableRight), uintptr(tableTop+rowH))
						selectObject.Call(memDC, oldFont)
					}

					// Draw body rows
					{
						oldFont, _, _ := selectObject.Call(memDC, fontBody)
						for ri, row := range td.Rows {
							rowTop := tableTop + (int32(ri)+1)*rowH
							for ci, cell := range row {
								if ci >= numCols {
									break
								}
								cl := colLeft[ci]
								cr := colLeft[ci+1]
								var cellRC [16]byte
								*(*int32)(unsafe.Pointer(&cellRC[0])) = cl + cellPad
								*(*int32)(unsafe.Pointer(&cellRC[4])) = rowTop + cellPad
								*(*int32)(unsafe.Pointer(&cellRC[8])) = cr - cellPad
								*(*int32)(unsafe.Pointer(&cellRC[12])) = rowTop + rowH - cellPad
								cellTxt := utf16From(cell.Text)
								drawTextW.Call(memDC, uintptr(unsafe.Pointer(&cellTxt[0])),
									uintptr(len(cellTxt)-1),
									uintptr(unsafe.Pointer(&cellRC[0])),
									DT_NOPREFIX|DT_SINGLELINE|DT_NOCLIP)
								// Cell borders
								moveToEx.Call(memDC, uintptr(cl), uintptr(rowTop))
								lineTo.Call(memDC, uintptr(cr), uintptr(rowTop))
								moveToEx.Call(memDC, uintptr(cl), uintptr(rowTop))
								lineTo.Call(memDC, uintptr(cl), uintptr(rowTop+rowH))
								moveToEx.Call(memDC, uintptr(cr), uintptr(rowTop))
								lineTo.Call(memDC, uintptr(cr), uintptr(rowTop+rowH))
								moveToEx.Call(memDC, uintptr(cl), uintptr(rowTop+rowH))
								lineTo.Call(memDC, uintptr(cr), uintptr(rowTop+rowH))
							}
						}
						selectObject.Call(memDC, oldFont)
					}
				}

				selectObject.Call(memDC, oldPen)
				deleteObject.Call(tablePen)
				y += tableH
				continue
			}

			// Code block internal padding
			codePad := int32(0)
			if b.HasBg {
				codePad = 10
			}

			// Measure text height
			var measureRC [16]byte
			*(*int32)(unsafe.Pointer(&measureRC[0])) = 0
			*(*int32)(unsafe.Pointer(&measureRC[4])) = 0
			*(*int32)(unsafe.Pointer(&measureRC[8])) = drawWidth - b.Indent - codePad*2
			*(*int32)(unsafe.Pointer(&measureRC[12])) = 10000

			txt := utf16From(b.Text)
			oldFont, _, _ := selectObject.Call(memDC, b.Font)
			drawTextW.Call(memDC, uintptr(unsafe.Pointer(&txt[0])),
				uintptr(len(txt)-1),
				uintptr(unsafe.Pointer(&measureRC[0])),
				DT_CALCRECT|DT_WORDBREAK|DT_NOPREFIX|DT_EXPANDTABS|DT_EDITCONTROL)
			textH := *(*int32)(unsafe.Pointer(&measureRC[12]))

			// Only draw if visible
			if y+textH+codePad*2 >= 0 && y < clientH {
				// Background for code blocks
				if b.HasBg {
					var bgRC [16]byte
					*(*int32)(unsafe.Pointer(&bgRC[0])) = padding + b.Indent - 4
					*(*int32)(unsafe.Pointer(&bgRC[4])) = y
					*(*int32)(unsafe.Pointer(&bgRC[8])) = clientW - padding + 4
					*(*int32)(unsafe.Pointer(&bgRC[12])) = y + textH + codePad*2
					fillRect.Call(memDC, uintptr(unsafe.Pointer(&bgRC[0])), codeBrush)
				}

				// Blockquote left bar
				if b.IsQuote {
					var barRC [16]byte
					*(*int32)(unsafe.Pointer(&barRC[0])) = padding
					*(*int32)(unsafe.Pointer(&barRC[4])) = y - 2
					*(*int32)(unsafe.Pointer(&barRC[8])) = padding + 3
					*(*int32)(unsafe.Pointer(&barRC[12])) = y + textH + 2
					fillRect.Call(memDC, uintptr(unsafe.Pointer(&barRC[0])), blueBrush)
				}

				// Draw bullet character for list items
				if b.IsBullet {
					oldColor, _, _ := setTextColor.Call(memDC, 0x00333333)
					selectObject.Call(memDC, fontBody)
					bullet := utf16From("\u2022") // bullet character •
					textOutW.Call(memDC,
						uintptr(padding+8), uintptr(y+codePad),
						uintptr(unsafe.Pointer(&bullet[0])),
						uintptr(len(bullet)-1))
					setTextColor.Call(memDC, oldColor)
				}

				setTextColor.Call(memDC, uintptr(b.Color))

				// Draw with inline runs if available (for bold/italic/code spans)
				if len(b.Runs) > 1 || (len(b.Runs) == 1 && (b.Runs[0].Bold || b.Runs[0].Italic || b.Runs[0].Code)) {
					// Draw each run sequentially with appropriate fonts
					curX := padding + b.Indent + codePad
					curY := y + codePad
					maxX := clientW - padding - codePad
					lineH := int32(0)

					// Measure line height with body font
					{
						hSample := utf16From("Ay")
						var hRC [16]byte
						*(*int32)(unsafe.Pointer(&hRC[8])) = 10000
						*(*int32)(unsafe.Pointer(&hRC[12])) = 10000
						drawTextW.Call(memDC, uintptr(unsafe.Pointer(&hSample[0])),
							uintptr(len(hSample)-1),
							uintptr(unsafe.Pointer(&hRC[0])),
							DT_CALCRECT|DT_NOPREFIX)
						lineH = *(*int32)(unsafe.Pointer(&hRC[12]))
					}

					for _, run := range b.Runs {
						if run.Text == "" {
							continue
						}
						// Select appropriate font for this run
						runFont := b.Font
						if run.Bold {
							runFont = fontBodyBold
						} else if run.Italic {
							runFont = fontBodyItalic
						} else if run.Code {
							runFont = fontCode
						}
						selectObject.Call(memDC, runFont)

						runTxt := utf16From(run.Text)
						nChars := len(runTxt) - 1 // exclude null terminator

						// Measure run width using GetTextExtentPoint32 (preserves spaces)
						var sz [8]byte // SIZE struct: cx, cy (int32 each)
						getTextExtentPoint32W.Call(memDC,
							uintptr(unsafe.Pointer(&runTxt[0])),
							uintptr(nChars),
							uintptr(unsafe.Pointer(&sz[0])))
						runW := *(*int32)(unsafe.Pointer(&sz[0]))

						// Wrap to next line if needed
						if curX+runW > maxX && curX > padding+b.Indent+codePad {
							curX = padding + b.Indent + codePad
							curY += lineH
						}

						// Draw with TextOut (preserves leading spaces exactly)
						textOutW.Call(memDC,
							uintptr(curX), uintptr(curY),
							uintptr(unsafe.Pointer(&runTxt[0])),
							uintptr(nChars))

						curX += runW
						// Italic fonts overhang past their measured width; add
						// a small gap so the next run's text isn't overlapped.
						if run.Italic {
							curX += 2
						}
					}
				} else {
					// Draw text as single block (offset by codePad for code blocks)
					var drawRC [16]byte
					*(*int32)(unsafe.Pointer(&drawRC[0])) = padding + b.Indent + codePad
					*(*int32)(unsafe.Pointer(&drawRC[4])) = y + codePad
					*(*int32)(unsafe.Pointer(&drawRC[8])) = clientW - padding - codePad
					*(*int32)(unsafe.Pointer(&drawRC[12])) = y + codePad + textH

					drawTextW.Call(memDC, uintptr(unsafe.Pointer(&txt[0])),
						uintptr(len(txt)-1),
						uintptr(unsafe.Pointer(&drawRC[0])),
						DT_WORDBREAK|DT_NOPREFIX|DT_EXPANDTABS|DT_EDITCONTROL)
				}
			}

			selectObject.Call(memDC, oldFont)
			y += textH + codePad*2
		}

		totalHeight = y + scrollY + padding

		// Blit to screen
		bitBlt.Call(hdc, 0, 0, uintptr(clientW), uintptr(clientH), memDC, 0, 0, SRCCOPY)

		// Cleanup
		selectObject.Call(memDC, oldBmp)
		deleteObject.Call(memBmp)
		deleteDC.Call(memDC)

		endPaint.Call(hwnd, uintptr(unsafe.Pointer(&ps[0])))

		// Update scrollbar
		updateScrollbar(hwnd, clientH)

		return 0

	case WM_MOUSEWHEEL:
		delta := int16(hiword(wParam))
		scrollY -= int32(delta) / 3
		if scrollY < 0 {
			scrollY = 0
		}
		maxScroll := totalHeight - getClientHeight(hwnd)
		if maxScroll < 0 {
			maxScroll = 0
		}
		if scrollY > maxScroll {
			scrollY = maxScroll
		}
		invalidateRect.Call(hwnd, 0, 0)
		updateScrollbar(hwnd, getClientHeight(hwnd))
		return 0

	case WM_VSCROLL:
		code := loword(wParam)
		var rc [16]byte
		getClientRect.Call(hwnd, uintptr(unsafe.Pointer(&rc[0])))
		clientH := *(*int32)(unsafe.Pointer(&rc[12]))
		maxScroll := totalHeight - clientH
		if maxScroll < 0 {
			maxScroll = 0
		}
		switch code {
		case SB_LINEUP:
			scrollY -= 30
		case SB_LINEDOWN:
			scrollY += 30
		case SB_PAGEUP:
			scrollY -= clientH
		case SB_PAGEDOWN:
			scrollY += clientH
		case SB_THUMBTRACK, SB_THUMBPOSITION:
			pos := int32(hiword(wParam))
			scrollY = pos
		}
		if scrollY < 0 {
			scrollY = 0
		}
		if scrollY > maxScroll {
			scrollY = maxScroll
		}
		invalidateRect.Call(hwnd, 0, 0)
		updateScrollbar(hwnd, clientH)
		return 0
	}

	ret, _, _ := defWindowProcW.Call(hwnd, msg, wParam, lParam)
	return ret
}

func getClientHeight(hwnd uintptr) int32 {
	var rc [16]byte
	getClientRect.Call(hwnd, uintptr(unsafe.Pointer(&rc[0])))
	return *(*int32)(unsafe.Pointer(&rc[12]))
}

func updateScrollbar(hwnd uintptr, clientH int32) {
	// SCROLLINFO: cbSize(4) + fMask(4) + nMin(4) + nMax(4) + nPage(4) + nPos(4) + nTrackPos(4) = 28 bytes
	var si [28]byte
	*(*uint32)(unsafe.Pointer(&si[0])) = 28
	*(*uint32)(unsafe.Pointer(&si[4])) = SIF_RANGE | SIF_PAGE | SIF_POS
	*(*int32)(unsafe.Pointer(&si[8])) = 0           // nMin
	*(*int32)(unsafe.Pointer(&si[12])) = totalHeight // nMax
	*(*uint32)(unsafe.Pointer(&si[16])) = uint32(clientH) // nPage
	*(*int32)(unsafe.Pointer(&si[20])) = scrollY    // nPos
	setScrollInfo.Call(hwnd, SB_VERT, uintptr(unsafe.Pointer(&si[0])), 1)
}

// File operations
func windowTitle() string {
	if currentFile != "" {
		return "TinyMD GDI — " + currentFile
	}
	return "TinyMD GDI Prototype"
}

func saveFile() {
	if currentFile == "" {
		currentFile = showSaveDialog(mainHwnd)
		if currentFile == "" {
			return
		}
	}
	// Get text from editor
	length, _, _ := getWindowTextLengthW.Call(editorHwnd)
	buf := make([]uint16, length+1)
	getWindowTextW.Call(editorHwnd, uintptr(unsafe.Pointer(&buf[0])), length+1)
	text := syscall.UTF16ToString(buf)
	os.WriteFile(currentFile, []byte(text), 0644)
}

func showSaveDialog(hwnd uintptr) string {
	buf := make([]uint16, 260)
	filter := append(utf16From("Markdown Files (*.md)"), 0)
	filter = append(filter, utf16From("*.md")...)
	filter = append(filter, 0)
	filter = append(filter, utf16From("All Files (*.*)")...)
	filter = append(filter, 0)
	filter = append(filter, utf16From("*.*")...)
	filter = append(filter, 0, 0)
	title := utf16From("Save As")
	title = append(title, 0)
	defExt := utf16From("md")
	defExt = append(defExt, 0)

	const structSize = 152
	var ofn [structSize]byte
	*(*uint32)(unsafe.Pointer(&ofn[0])) = structSize
	*(*uintptr)(unsafe.Pointer(&ofn[8])) = hwnd
	*(*uintptr)(unsafe.Pointer(&ofn[24])) = uintptr(unsafe.Pointer(&filter[0]))
	*(*uint32)(unsafe.Pointer(&ofn[44])) = 1
	*(*uintptr)(unsafe.Pointer(&ofn[48])) = uintptr(unsafe.Pointer(&buf[0]))
	*(*uint32)(unsafe.Pointer(&ofn[56])) = uint32(len(buf))
	*(*uintptr)(unsafe.Pointer(&ofn[80])) = uintptr(unsafe.Pointer(&title[0]))
	*(*uint32)(unsafe.Pointer(&ofn[88])) = 0x00000002 | 0x00000800
	*(*uintptr)(unsafe.Pointer(&ofn[96])) = uintptr(unsafe.Pointer(&defExt[0]))

	ret, _, _ := getSaveFileNameW.Call(uintptr(unsafe.Pointer(&ofn[0])))
	if ret == 0 {
		return ""
	}
	return syscall.UTF16ToString(buf)
}

func printFormatted() {
	if len(currentBlocks) == 0 {
		return
	}

	// PRINTDLGW is 120 bytes on 64-bit
	const pdSize = 120
	var pd [pdSize]byte
	*(*uint32)(unsafe.Pointer(&pd[0])) = pdSize
	*(*uintptr)(unsafe.Pointer(&pd[8])) = mainHwnd // hwndOwner
	*(*uint32)(unsafe.Pointer(&pd[40])) = PD_RETURNDC | PD_USEDEVMODECOPIESANDCOLLATE

	ret, _, _ := printDlgW.Call(uintptr(unsafe.Pointer(&pd[0])))
	if ret == 0 {
		return
	}

	printerDC := *(*uintptr)(unsafe.Pointer(&pd[32])) // hDC
	if printerDC == 0 {
		return
	}
	defer deleteDC.Call(printerDC)

	// Get printer metrics
	dpiX, _, _ := getDeviceCaps.Call(printerDC, LOGPIXELSX)
	dpiY, _, _ := getDeviceCaps.Call(printerDC, LOGPIXELSY)
	pageW, _, _ := getDeviceCaps.Call(printerDC, HORZRES)
	pageH, _, _ := getDeviceCaps.Call(printerDC, VERTRES)
	scaleX := float64(dpiX) / 96.0
	scaleY := float64(dpiY) / 96.0

	// Create scaled fonts for the printer DC
	pFontH1 := makePrintFont(-28, 700, 0, "Segoe UI", scaleY)
	pFontH2 := makePrintFont(-22, 700, 0, "Segoe UI", scaleY)
	pFontH3 := makePrintFont(-18, 700, 0, "Segoe UI", scaleY)
	pFontBody := makePrintFont(-14, 400, 0, "Segoe UI", scaleY)
	pFontBodyBold := makePrintFont(-14, 700, 0, "Segoe UI", scaleY)
	pFontBodyItalic := makePrintFont(-14, 400, 1, "Segoe UI", scaleY)
	pFontCode := makePrintFont(-14, 400, 0, "Consolas", scaleY)
	defer func() {
		for _, f := range []uintptr{pFontH1, pFontH2, pFontH3, pFontBody, pFontBodyBold, pFontBodyItalic, pFontCode} {
			deleteObject.Call(f)
		}
	}()

	// Map screen fonts to printer fonts
	fontMap := map[uintptr]uintptr{
		fontH1: pFontH1, fontH2: pFontH2, fontH3: pFontH3,
		fontBody: pFontBody, fontBodyBold: pFontBodyBold,
		fontBodyItalic: pFontBodyItalic, fontCode: pFontCode,
	}

	// StartDoc
	// DOCINFOW on 64-bit: cbSize(4) + pad(4) + lpszDocName(8) + lpszOutput(8) + lpszDatatype(8) + fwType(4) + pad(4) = 40
	docName := utf16From("TinyMD Print")
	var di [40]byte
	*(*int32)(unsafe.Pointer(&di[0])) = 40
	*(*uintptr)(unsafe.Pointer(&di[8])) = uintptr(unsafe.Pointer(&docName[0]))
	r, _, _ := startDocW.Call(printerDC, uintptr(unsafe.Pointer(&di[0])))
	if int32(r) <= 0 {
		return
	}

	padding := int32(float64(30) * scaleX)
	drawWidth := int32(pageW) - padding*2

	// Create printer brushes
	pGrayBrush, _, _ := createSolidBrush.Call(0x00DDDDDD)
	pBlueBrush, _, _ := createSolidBrush.Call(0x00D66603)
	pCodeBrush, _, _ := createSolidBrush.Call(0x00FAF8F6)
	defer func() {
		deleteObject.Call(pGrayBrush)
		deleteObject.Call(pBlueBrush)
		deleteObject.Call(pCodeBrush)
	}()

	startPage.Call(printerDC)
	setBkMode.Call(printerDC, TRANSPARENT)

	y := padding
	maxY := int32(pageH) - padding
	for i := range currentBlocks {
		b := &currentBlocks[i]
		spaceAbove := int32(float64(b.SpaceAbove) * scaleY)
		indent := int32(float64(b.Indent) * scaleX)

		// Map font
		pFont := fontMap[b.Font]
		if pFont == 0 {
			pFont = pFontBody
		}

		if b.IsHR {
			y += spaceAbove
			hrPen, _, _ := createPen.Call(0, 1, 0x00BBBBBB)
			oldP, _, _ := selectObject.Call(printerDC, hrPen)
			hrY := y + int32(8*scaleY)
			moveToEx.Call(printerDC, uintptr(padding), uintptr(hrY), 0)
			lineTo.Call(printerDC, uintptr(int32(pageW)-padding), uintptr(hrY))
			selectObject.Call(printerDC, oldP)
			deleteObject.Call(hrPen)
			y += int32(16 * scaleY)
			continue
		}

		if b.IsTable && b.Table != nil {
			y += spaceAbove
			td := b.Table
			numCols := len(td.Headers)
			if numCols == 0 {
				continue
			}
			cellPad := int32(float64(8) * scaleX)

			// Measure columns
			colWidths := make([]int32, numCols)
			oldF, _, _ := selectObject.Call(printerDC, pFontH3)
			for ci, cell := range td.Headers {
				cellTxt := utf16From(cell.Text)
				var mRC [16]byte
				*(*int32)(unsafe.Pointer(&mRC[8])) = 10000
				*(*int32)(unsafe.Pointer(&mRC[12])) = 10000
				drawTextW.Call(printerDC, uintptr(unsafe.Pointer(&cellTxt[0])),
					uintptr(len(cellTxt)-1), uintptr(unsafe.Pointer(&mRC[0])),
					DT_CALCRECT|DT_NOPREFIX|DT_SINGLELINE)
				w := *(*int32)(unsafe.Pointer(&mRC[8]))
				if w > colWidths[ci] {
					colWidths[ci] = w
				}
			}
			selectObject.Call(printerDC, pFontBody)
			for _, row := range td.Rows {
				for ci, cell := range row {
					if ci >= numCols {
						break
					}
					cellTxt := utf16From(cell.Text)
					var mRC [16]byte
					*(*int32)(unsafe.Pointer(&mRC[8])) = 10000
					*(*int32)(unsafe.Pointer(&mRC[12])) = 10000
					drawTextW.Call(printerDC, uintptr(unsafe.Pointer(&cellTxt[0])),
						uintptr(len(cellTxt)-1), uintptr(unsafe.Pointer(&mRC[0])),
						DT_CALCRECT|DT_NOPREFIX|DT_SINGLELINE)
					w := *(*int32)(unsafe.Pointer(&mRC[8]))
					if w > colWidths[ci] {
						colWidths[ci] = w
					}
				}
			}
			for ci := range colWidths {
				colWidths[ci] += cellPad * 2
			}

			// Measure row height
			sampleTxt := utf16From("Ay")
			var measureRC [16]byte
			*(*int32)(unsafe.Pointer(&measureRC[8])) = 10000
			*(*int32)(unsafe.Pointer(&measureRC[12])) = 10000
			drawTextW.Call(printerDC, uintptr(unsafe.Pointer(&sampleTxt[0])),
				uintptr(len(sampleTxt)-1), uintptr(unsafe.Pointer(&measureRC[0])),
				DT_CALCRECT|DT_NOPREFIX|DT_SINGLELINE)
			rowH := *(*int32)(unsafe.Pointer(&measureRC[12])) + cellPad*2
			selectObject.Call(printerDC, oldF)

			colLeft := make([]int32, numCols+1)
			colLeft[0] = padding
			for ci := 0; ci < numCols; ci++ {
				colLeft[ci+1] = colLeft[ci] + colWidths[ci]
			}

			tablePen, _, _ := createPen.Call(0, 1, 0x00DDDDDD)
			oldPen, _, _ := selectObject.Call(printerDC, tablePen)
			setTextColor.Call(printerDC, 0x00222222)

			// Header row
			oldF2, _, _ := selectObject.Call(printerDC, pFontH3)
			for ci, cell := range td.Headers {
				cl := colLeft[ci]
				cr := colLeft[ci+1]
				var cellRC [16]byte
				*(*int32)(unsafe.Pointer(&cellRC[0])) = cl + cellPad
				*(*int32)(unsafe.Pointer(&cellRC[4])) = y + cellPad
				*(*int32)(unsafe.Pointer(&cellRC[8])) = cr - cellPad
				*(*int32)(unsafe.Pointer(&cellRC[12])) = y + rowH - cellPad
				cellTxt := utf16From(cell.Text)
				drawTextW.Call(printerDC, uintptr(unsafe.Pointer(&cellTxt[0])),
					uintptr(len(cellTxt)-1), uintptr(unsafe.Pointer(&cellRC[0])),
					DT_NOPREFIX|DT_SINGLELINE|DT_NOCLIP)
				moveToEx.Call(printerDC, uintptr(cl), uintptr(y), 0)
				lineTo.Call(printerDC, uintptr(cr), uintptr(y))
				moveToEx.Call(printerDC, uintptr(cl), uintptr(y), 0)
				lineTo.Call(printerDC, uintptr(cl), uintptr(y+rowH))
				moveToEx.Call(printerDC, uintptr(cr), uintptr(y), 0)
				lineTo.Call(printerDC, uintptr(cr), uintptr(y+rowH))
				moveToEx.Call(printerDC, uintptr(cl), uintptr(y+rowH), 0)
				lineTo.Call(printerDC, uintptr(cr), uintptr(y+rowH))
			}
			selectObject.Call(printerDC, oldF2)
			y += rowH

			// Body rows
			oldF2, _, _ = selectObject.Call(printerDC, pFontBody)
			for _, row := range td.Rows {
				if y+rowH > maxY {
					endPage.Call(printerDC)
					startPage.Call(printerDC)
					setBkMode.Call(printerDC, TRANSPARENT)
					y = padding
				}
				for ci, cell := range row {
					if ci >= numCols {
						break
					}
					cl := colLeft[ci]
					cr := colLeft[ci+1]
					var cellRC [16]byte
					*(*int32)(unsafe.Pointer(&cellRC[0])) = cl + cellPad
					*(*int32)(unsafe.Pointer(&cellRC[4])) = y + cellPad
					*(*int32)(unsafe.Pointer(&cellRC[8])) = cr - cellPad
					*(*int32)(unsafe.Pointer(&cellRC[12])) = y + rowH - cellPad
					cellTxt := utf16From(cell.Text)
					drawTextW.Call(printerDC, uintptr(unsafe.Pointer(&cellTxt[0])),
						uintptr(len(cellTxt)-1), uintptr(unsafe.Pointer(&cellRC[0])),
						DT_NOPREFIX|DT_SINGLELINE|DT_NOCLIP)
					moveToEx.Call(printerDC, uintptr(cl), uintptr(y), 0)
					lineTo.Call(printerDC, uintptr(cr), uintptr(y))
					moveToEx.Call(printerDC, uintptr(cl), uintptr(y), 0)
					lineTo.Call(printerDC, uintptr(cl), uintptr(y+rowH))
					moveToEx.Call(printerDC, uintptr(cr), uintptr(y), 0)
					lineTo.Call(printerDC, uintptr(cr), uintptr(y+rowH))
					moveToEx.Call(printerDC, uintptr(cl), uintptr(y+rowH), 0)
					lineTo.Call(printerDC, uintptr(cr), uintptr(y+rowH))
				}
				y += rowH
			}
			selectObject.Call(printerDC, oldF2)
			selectObject.Call(printerDC, oldPen)
			deleteObject.Call(tablePen)
			continue
		}

		// Regular text block
		codePad := int32(0)
		if b.HasBg {
			codePad = int32(float64(10) * scaleX)
		}

		// Measure
		var measureRC [16]byte
		*(*int32)(unsafe.Pointer(&measureRC[8])) = drawWidth - indent - codePad*2
		*(*int32)(unsafe.Pointer(&measureRC[12])) = 10000
		txt := utf16From(b.Text)
		oldF, _, _ := selectObject.Call(printerDC, pFont)
		drawTextW.Call(printerDC, uintptr(unsafe.Pointer(&txt[0])),
			uintptr(len(txt)-1), uintptr(unsafe.Pointer(&measureRC[0])),
			DT_CALCRECT|DT_WORDBREAK|DT_NOPREFIX|DT_EXPANDTABS|DT_EDITCONTROL)
		textH := *(*int32)(unsafe.Pointer(&measureRC[12]))

		blockH := textH + codePad*2 + spaceAbove

		// Page break if needed
		if y+blockH > maxY && y > padding {
			endPage.Call(printerDC)
			startPage.Call(printerDC)
			setBkMode.Call(printerDC, TRANSPARENT)
			y = padding
		}

		y += spaceAbove

		// Background for code blocks
		if b.HasBg {
			var bgRC [16]byte
			*(*int32)(unsafe.Pointer(&bgRC[0])) = padding + indent - int32(4*scaleX)
			*(*int32)(unsafe.Pointer(&bgRC[4])) = y
			*(*int32)(unsafe.Pointer(&bgRC[8])) = int32(pageW) - padding + int32(4*scaleX)
			*(*int32)(unsafe.Pointer(&bgRC[12])) = y + textH + codePad*2
			fillRect.Call(printerDC, uintptr(unsafe.Pointer(&bgRC[0])), pCodeBrush)
		}

		// Blockquote bar
		if b.IsQuote {
			var barRC [16]byte
			*(*int32)(unsafe.Pointer(&barRC[0])) = padding
			*(*int32)(unsafe.Pointer(&barRC[4])) = y - int32(2*scaleY)
			*(*int32)(unsafe.Pointer(&barRC[8])) = padding + int32(3*scaleX)
			*(*int32)(unsafe.Pointer(&barRC[12])) = y + textH + int32(2*scaleY)
			fillRect.Call(printerDC, uintptr(unsafe.Pointer(&barRC[0])), pBlueBrush)
		}

		// Bullet
		if b.IsBullet {
			selectObject.Call(printerDC, pFontBody)
			setTextColor.Call(printerDC, 0x00333333)
			bullet := utf16From("\u2022")
			textOutW.Call(printerDC,
				uintptr(padding+int32(8*scaleX)), uintptr(y+codePad),
				uintptr(unsafe.Pointer(&bullet[0])), uintptr(len(bullet)-1))
		}

		setTextColor.Call(printerDC, uintptr(b.Color))

		// Draw text with inline runs
		if len(b.Runs) > 1 || (len(b.Runs) == 1 && (b.Runs[0].Bold || b.Runs[0].Italic || b.Runs[0].Code)) {
			curX := padding + indent + codePad
			curY := y + codePad
			maxX := int32(pageW) - padding - codePad
			lineH := int32(0)
			hSample := utf16From("Ay")
			var hRC [16]byte
			*(*int32)(unsafe.Pointer(&hRC[8])) = 10000
			*(*int32)(unsafe.Pointer(&hRC[12])) = 10000
			drawTextW.Call(printerDC, uintptr(unsafe.Pointer(&hSample[0])),
				uintptr(len(hSample)-1), uintptr(unsafe.Pointer(&hRC[0])),
				DT_CALCRECT|DT_NOPREFIX)
			lineH = *(*int32)(unsafe.Pointer(&hRC[12]))

			for _, run := range b.Runs {
				if run.Text == "" {
					continue
				}
				runFont := pFont
				if run.Bold {
					runFont = pFontBodyBold
				} else if run.Italic {
					runFont = pFontBodyItalic
				} else if run.Code {
					runFont = pFontCode
				}
				selectObject.Call(printerDC, runFont)

				runTxt := utf16From(run.Text)
				nChars := len(runTxt) - 1
				var sz [8]byte
				getTextExtentPoint32W.Call(printerDC,
					uintptr(unsafe.Pointer(&runTxt[0])), uintptr(nChars),
					uintptr(unsafe.Pointer(&sz[0])))
				runW := *(*int32)(unsafe.Pointer(&sz[0]))

				if curX+runW > maxX && curX > padding+indent+codePad {
					curX = padding + indent + codePad
					curY += lineH
				}
				textOutW.Call(printerDC, uintptr(curX), uintptr(curY),
					uintptr(unsafe.Pointer(&runTxt[0])), uintptr(nChars))
				curX += runW
				if run.Italic {
					curX += int32(2 * scaleX)
				}
			}
		} else {
			var drawRC [16]byte
			*(*int32)(unsafe.Pointer(&drawRC[0])) = padding + indent + codePad
			*(*int32)(unsafe.Pointer(&drawRC[4])) = y + codePad
			*(*int32)(unsafe.Pointer(&drawRC[8])) = int32(pageW) - padding - codePad
			*(*int32)(unsafe.Pointer(&drawRC[12])) = y + codePad + textH
			drawTextW.Call(printerDC, uintptr(unsafe.Pointer(&txt[0])),
				uintptr(len(txt)-1), uintptr(unsafe.Pointer(&drawRC[0])),
				DT_WORDBREAK|DT_NOPREFIX|DT_EXPANDTABS|DT_EDITCONTROL)
		}

		selectObject.Call(printerDC, oldF)
		y += textH + codePad*2
	}

	endPage.Call(printerDC)
	endDoc.Call(printerDC)
}

func makePrintFont(height, weight int32, italic uint32, face string, scale float64) uintptr {
	scaledH := int32(float64(height) * scale)
	f, _, _ := createFontW.Call(
		uintptr(uint32(uint16(scaledH))|0xFFFF0000),
		0, 0, 0,
		uintptr(weight),
		uintptr(italic),
		0, 0, 0, 0, 0, 0, 0,
		uintptr(unsafe.Pointer(utf16Ptr(face))),
	)
	return f
}

func main() {
	runtime.LockOSThread()

	var initialContent string
	autoPrint := false
	for _, arg := range os.Args[1:] {
		if arg == "--print" {
			autoPrint = true
		} else if currentFile == "" {
			currentFile = arg
			data, err := os.ReadFile(currentFile)
			if err == nil {
				initialContent = string(data)
			}
		}
	}

	hInstance, _, _ = getModuleHandleW.Call(0)
	initFonts()

	// Register main window class
	className := utf16From("TinyMDGDI")
	cursor, _, _ := loadCursorW.Call(0, IDC_ARROW)

	var wc [80]byte
	*(*uint32)(unsafe.Pointer(&wc[0])) = 80
	*(*uintptr)(unsafe.Pointer(&wc[8])) = syscall.NewCallback(mainWndProc)
	*(*uintptr)(unsafe.Pointer(&wc[24])) = hInstance
	*(*uintptr)(unsafe.Pointer(&wc[40])) = cursor
	*(*uintptr)(unsafe.Pointer(&wc[48])) = 6 // COLOR_WINDOW+1
	*(*uintptr)(unsafe.Pointer(&wc[64])) = uintptr(unsafe.Pointer(&className[0]))
	registerClassExW.Call(uintptr(unsafe.Pointer(&wc[0])))

	// Center on screen
	w, h := uintptr(1400), uintptr(900)
	screenW, _, _ := getSystemMetrics.Call(SM_CXSCREEN)
	screenH, _, _ := getSystemMetrics.Call(SM_CYSCREEN)
	x := (screenW - w) / 2
	y := (screenH - h) / 2

	title := utf16From(windowTitle())
	mainHwnd, _, _ = createWindowExW.Call(
		0,
		uintptr(unsafe.Pointer(&className[0])),
		uintptr(unsafe.Pointer(&title[0])),
		WS_OVERLAPPEDWINDOW|WS_VISIBLE|WS_CLIPCHILDREN,
		x, y, w, h,
		0, 0, hInstance, 0,
	)

	showWindowProc.Call(mainHwnd, 5) // SW_SHOW
	updateWindowProc.Call(mainHwnd)

	// Set initial content
	if initialContent != "" {
		txt := utf16From(initialContent)
		sendMessageW.Call(editorHwnd, 0x000C, 0, uintptr(unsafe.Pointer(&txt[0]))) // WM_SETTEXT
		updatePreview()
	}

	setFocus.Call(editorHwnd)

	if autoPrint {
		fmt.Println("[GDI] --print flag detected, auto-printing...")
		printFormatted()
		fmt.Println("[GDI] print done, exiting")
		return
	}

	// Message loop — intercept Ctrl+S/Ctrl+P before dispatch since
	// the EDIT control has focus and mainWndProc never sees WM_KEYDOWN.
	var msgBuf [48]byte
	for {
		ret, _, _ := getMessageW.Call(uintptr(unsafe.Pointer(&msgBuf[0])), 0, 0, 0)
		if ret == 0 || int32(ret) == -1 {
			break
		}
		// MSG struct: hwnd(8), message(4+pad4), wParam(8), lParam(8)
		msgID := *(*uint32)(unsafe.Pointer(&msgBuf[8]))
		wParam := *(*uintptr)(unsafe.Pointer(&msgBuf[16]))
		if msgID == WM_KEYDOWN {
			state, _, _ := user32.NewProc("GetKeyState").Call(0x11)
			if int16(state) < 0 {
				if wParam == VK_S {
					saveFile()
					continue
				}
				if wParam == VK_P {
					printFormatted()
					continue
				}
			}
		}
		translateMessage.Call(uintptr(unsafe.Pointer(&msgBuf[0])))
		dispatchMessageW.Call(uintptr(unsafe.Pointer(&msgBuf[0])))
	}

	_ = filepath.Base // keep import
}
