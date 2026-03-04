// TinyMD GDI Prototype — native markdown preview using GDI DrawText.
//
// Build: GOOS=windows go build -ldflags="-s -w -H windowsgui" -trimpath -o tinymd-gdi.exe .

package main

import (
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"unsafe"

	"github.com/yuin/goldmark"
	"github.com/yuin/goldmark/ast"
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

	// kernel32
	getModuleHandleW = kernel32.NewProc("GetModuleHandleW")

	// comdlg32
	getSaveFileNameW = comdlg32.NewProc("GetSaveFileNameW")
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
	VK_TAB = 0x09

	DT_WORDBREAK  = 0x0010
	DT_NOPREFIX   = 0x0800
	DT_CALCRECT   = 0x0400
	DT_EXPANDTABS = 0x0040
	DT_LEFT       = 0x0000
	DT_EDITCONTROL = 0x2000

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
	fontH1   uintptr
	fontH2   uintptr
	fontH3   uintptr
	fontBody uintptr
	fontCode uintptr
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
	fontCode = makeFont(-13, 400, 0, "Consolas")
}

func destroyFonts() {
	for _, f := range []uintptr{fontH1, fontH2, fontH3, fontBody, fontCode} {
		if f != 0 {
			deleteObject.Call(f)
		}
	}
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
}

// Markdown → blocks
func markdownToBlocks(source []byte) []DrawBlock {
	md := goldmark.New()
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
			switch h.Level {
			case 1:
				font = fontH1
			case 2:
				font = fontH2
			default:
				font = fontH3
			}
			blocks = append(blocks, DrawBlock{
				Text: txt, Font: font, Color: 0x00111111,
				SpaceAbove: 16,
			})
			return ast.WalkSkipChildren, nil

		case ast.KindParagraph:
			if n.Parent() != nil && n.Parent().Kind() == ast.KindBlockquote {
				return ast.WalkContinue, nil
			}
			txt := extractInlineText(n, source)
			if txt != "" {
				blocks = append(blocks, DrawBlock{
					Text: txt, Font: fontBody, Color: 0x00222222,
					SpaceAbove: 8,
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
				SpaceAbove: 8,
			})
			return ast.WalkSkipChildren, nil

		case ast.KindBlockquote:
			txt := extractInlineText(n, source)
			blocks = append(blocks, DrawBlock{
				Text: txt, Font: fontBody, Color: 0x00666666,
				Indent: 20, SpaceAbove: 8, IsQuote: true,
			})
			return ast.WalkSkipChildren, nil

		case ast.KindListItem:
			txt := "•  " + extractInlineText(n, source)
			blocks = append(blocks, DrawBlock{
				Text: txt, Font: fontBody, Color: 0x00222222,
				Indent: 20, SpaceAbove: 4,
			})
			return ast.WalkSkipChildren, nil

		case ast.KindThematicBreak:
			blocks = append(blocks, DrawBlock{
				IsHR: true, SpaceAbove: 12,
			})
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
			return ast.WalkContinue, nil
		}
		switch c.Kind() {
		case ast.KindText:
			t := c.(*ast.Text)
			buf.Write(t.Segment.Value(source))
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

		// Register preview window class
		previewClass := utf16From("TinyMDGDIPreview")
		var wc [80]byte
		*(*uint32)(unsafe.Pointer(&wc[0])) = 80
		*(*uintptr)(unsafe.Pointer(&wc[8])) = syscall.NewCallback(previewWndProc)
		*(*uintptr)(unsafe.Pointer(&wc[48])) = hInstance
		cursor, _, _ := loadCursorW.Call(0, IDC_ARROW)
		*(*uintptr)(unsafe.Pointer(&wc[56])) = cursor
		*(*uintptr)(unsafe.Pointer(&wc[64])) = 6 // COLOR_WINDOW+1
		*(*uintptr)(unsafe.Pointer(&wc[72])) = uintptr(unsafe.Pointer(&previewClass[0]))
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

		// Fill white background
		whiteBrush, _, _ := getSysColorBrush.Call(0)
		fillRect.Call(memDC, uintptr(unsafe.Pointer(&rc[0])), whiteBrush)
		setBkMode.Call(memDC, TRANSPARENT)

		padding := int32(20)
		drawWidth := clientW - padding*2
		y := padding - scrollY

		for i := range currentBlocks {
			b := &currentBlocks[i]
			y += b.SpaceAbove

			if b.IsHR {
				if y >= -10 && y < clientH+10 {
					pen, _, _ := createPen.Call(0, 1, 0x00DDDDDD)
					oldPen, _, _ := selectObject.Call(memDC, pen)
					moveToEx.Call(memDC, uintptr(padding), uintptr(y))
					lineTo.Call(memDC, uintptr(clientW-padding), uintptr(y))
					selectObject.Call(memDC, oldPen)
					deleteObject.Call(pen)
				}
				y += 12
				continue
			}

			// Measure text height
			var measureRC [16]byte
			*(*int32)(unsafe.Pointer(&measureRC[0])) = 0
			*(*int32)(unsafe.Pointer(&measureRC[4])) = 0
			*(*int32)(unsafe.Pointer(&measureRC[8])) = drawWidth - b.Indent
			*(*int32)(unsafe.Pointer(&measureRC[12])) = 10000

			txt := utf16From(b.Text)
			oldFont, _, _ := selectObject.Call(memDC, b.Font)
			drawTextW.Call(memDC, uintptr(unsafe.Pointer(&txt[0])),
				uintptr(len(txt)-1),
				uintptr(unsafe.Pointer(&measureRC[0])),
				DT_CALCRECT|DT_WORDBREAK|DT_NOPREFIX|DT_EXPANDTABS|DT_EDITCONTROL)
			textH := *(*int32)(unsafe.Pointer(&measureRC[12]))

			// Only draw if visible
			if y+textH >= 0 && y < clientH {
				// Background for code blocks
				if b.HasBg {
					var bgRC [16]byte
					*(*int32)(unsafe.Pointer(&bgRC[0])) = padding + b.Indent - 8
					*(*int32)(unsafe.Pointer(&bgRC[4])) = y - 4
					*(*int32)(unsafe.Pointer(&bgRC[8])) = clientW - padding + 8
					*(*int32)(unsafe.Pointer(&bgRC[12])) = y + textH + 4
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

				// Draw text
				var drawRC [16]byte
				*(*int32)(unsafe.Pointer(&drawRC[0])) = padding + b.Indent
				*(*int32)(unsafe.Pointer(&drawRC[4])) = y
				*(*int32)(unsafe.Pointer(&drawRC[8])) = clientW - padding
				*(*int32)(unsafe.Pointer(&drawRC[12])) = y + textH

				setTextColor.Call(memDC, uintptr(b.Color))
				drawTextW.Call(memDC, uintptr(unsafe.Pointer(&txt[0])),
					uintptr(len(txt)-1),
					uintptr(unsafe.Pointer(&drawRC[0])),
					DT_WORDBREAK|DT_NOPREFIX|DT_EXPANDTABS|DT_EDITCONTROL)
			}

			selectObject.Call(memDC, oldFont)
			y += textH
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

func main() {
	var initialContent string
	if len(os.Args) > 1 {
		currentFile = os.Args[1]
		data, err := os.ReadFile(currentFile)
		if err == nil {
			initialContent = string(data)
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
	*(*uintptr)(unsafe.Pointer(&wc[48])) = hInstance
	*(*uintptr)(unsafe.Pointer(&wc[56])) = cursor
	*(*uintptr)(unsafe.Pointer(&wc[64])) = 6 // COLOR_WINDOW+1
	*(*uintptr)(unsafe.Pointer(&wc[72])) = uintptr(unsafe.Pointer(&className[0]))
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

	// Message loop
	var msgBuf [48]byte
	for {
		ret, _, _ := getMessageW.Call(uintptr(unsafe.Pointer(&msgBuf[0])), 0, 0, 0)
		if ret == 0 || int32(ret) == -1 {
			break
		}
		translateMessage.Call(uintptr(unsafe.Pointer(&msgBuf[0])))
		dispatchMessageW.Call(uintptr(unsafe.Pointer(&msgBuf[0])))
	}

	_ = filepath.Base // keep import
}
