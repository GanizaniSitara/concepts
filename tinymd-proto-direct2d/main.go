// TinyMD Direct2D Prototype — native markdown preview using Direct2D + DirectWrite.
//
// Build: GOOS=windows go build -ldflags="-s -w -H windowsgui" -trimpath -o tinymd-d2d.exe .

package main

import (
	"math"
	"os"
	"strings"
	"syscall"
	"unsafe"

	"github.com/yuin/goldmark"
	"github.com/yuin/goldmark/ast"
	"github.com/yuin/goldmark/text"
)

// Win32 DLLs and procs
var (
	user32   = syscall.NewLazyDLL("user32.dll")
	gdi32    = syscall.NewLazyDLL("gdi32.dll")
	kernel32 = syscall.NewLazyDLL("kernel32.dll")
	comdlg32 = syscall.NewLazyDLL("comdlg32.dll")
	ole32    = syscall.NewLazyDLL("ole32.dll")
	d2d1     = syscall.NewLazyDLL("d2d1.dll")
	dwrite   = syscall.NewLazyDLL("dwrite.dll")

	// user32
	registerClassExW     = user32.NewProc("RegisterClassExW")
	createWindowExW      = user32.NewProc("CreateWindowExW")
	showWindowProc       = user32.NewProc("ShowWindow")
	updateWindowProc     = user32.NewProc("UpdateWindow")
	defWindowProcW       = user32.NewProc("DefWindowProcW")
	getSystemMetrics     = user32.NewProc("GetSystemMetrics")
	loadCursorW          = user32.NewProc("LoadCursorW")
	getClientRect        = user32.NewProc("GetClientRect")
	getMessageW          = user32.NewProc("GetMessageW")
	translateMessage     = user32.NewProc("TranslateMessage")
	dispatchMessageW     = user32.NewProc("DispatchMessageW")
	postQuitMessage      = user32.NewProc("PostQuitMessage")
	sendMessageW         = user32.NewProc("SendMessageW")
	moveWindow           = user32.NewProc("MoveWindow")
	invalidateRect       = user32.NewProc("InvalidateRect")
	setFocus             = user32.NewProc("SetFocus")
	setTimer             = user32.NewProc("SetTimer")
	killTimer            = user32.NewProc("KillTimer")
	getWindowTextLengthW = user32.NewProc("GetWindowTextLengthW")
	getWindowTextW       = user32.NewProc("GetWindowTextW")
	beginPaint           = user32.NewProc("BeginPaint")
	endPaint             = user32.NewProc("EndPaint")
	setScrollInfo        = user32.NewProc("SetScrollInfo")
	getKeyState          = user32.NewProc("GetKeyState")

	// gdi32
	createFontW  = gdi32.NewProc("CreateFontW")
	deleteObject = gdi32.NewProc("DeleteObject")

	// kernel32
	getModuleHandleW = kernel32.NewProc("GetModuleHandleW")

	// comdlg32
	getSaveFileNameW = comdlg32.NewProc("GetSaveFileNameW")

	// ole32
	coInitializeEx = ole32.NewProc("CoInitializeEx")

	// d2d1
	d2d1CreateFactory = d2d1.NewProc("D2D1CreateFactory")

	// dwrite
	dwriteCreateFactory = dwrite.NewProc("DWriteCreateFactory")
)

// Win32 constants
const (
	WS_OVERLAPPEDWINDOW = 0x00CF0000
	WS_VISIBLE          = 0x10000000
	WS_CHILD            = 0x40000000
	WS_VSCROLL          = 0x00200000
	WS_CLIPCHILDREN     = 0x02000000
	WS_EX_CLIENTEDGE    = 0x00000200
	ES_MULTILINE        = 0x0004
	ES_AUTOVSCROLL      = 0x0040
	ES_WANTRETURN       = 0x1000
	SM_CXSCREEN          = 0
	SM_CYSCREEN          = 1
	IDC_ARROW            = 32512

	WM_CREATE     = 0x0001
	WM_DESTROY    = 0x0002
	WM_SIZE       = 0x0005
	WM_PAINT      = 0x000F
	WM_COMMAND    = 0x0111
	WM_TIMER      = 0x0113
	WM_KEYDOWN    = 0x0100
	WM_MOUSEWHEEL = 0x020A
	WM_VSCROLL    = 0x0115
	WM_SETFONT    = 0x0030
	WM_SETFOCUS   = 0x0007
	WM_ERASEBKGND = 0x0014
	WM_SETTEXT    = 0x000C

	EN_CHANGE = 0x0300
	VK_S      = 0x53

	SB_VERT      = 1
	SIF_RANGE    = 0x01
	SIF_PAGE     = 0x02
	SIF_POS      = 0x04
	SB_LINEUP    = 0
	SB_LINEDOWN  = 1
	SB_PAGEUP    = 2
	SB_PAGEDOWN  = 3
	SB_THUMBTRACK    = 5
	SB_THUMBPOSITION = 4

	TIMER_DEBOUNCE = 1

	// COM
	COINIT_APARTMENTTHREADED = 0x2

	// D2D1
	D2D1_FACTORY_TYPE_SINGLE_THREADED = 0

	// DWRITE
	DWRITE_FACTORY_TYPE_SHARED = 0
	DWRITE_FONT_WEIGHT_NORMAL  = 400
	DWRITE_FONT_WEIGHT_BOLD    = 700
	DWRITE_FONT_STYLE_NORMAL   = 0
	DWRITE_FONT_STRETCH_NORMAL = 5
	DWRITE_TEXT_ALIGNMENT_LEADING       = 0
	DWRITE_PARAGRAPH_ALIGNMENT_NEAR     = 0
	DWRITE_WORD_WRAPPING_WRAP           = 0
)

// GUIDs
var (
	IID_ID2D1Factory    = guid(0x06152247, 0x6f50, 0x465a, [8]byte{0x92, 0x45, 0x11, 0x8b, 0xfd, 0x3b, 0x60, 0x07})
	IID_IDWriteFactory  = guid(0xb859ee5a, 0xd838, 0x4b5b, [8]byte{0xa2, 0xe8, 0x1a, 0xdc, 0x7d, 0x93, 0xdb, 0x48})
)

func guid(d1 uint32, d2, d3 uint16, d4 [8]byte) [16]byte {
	var g [16]byte
	*(*uint32)(unsafe.Pointer(&g[0])) = d1
	*(*uint16)(unsafe.Pointer(&g[4])) = d2
	*(*uint16)(unsafe.Pointer(&g[6])) = d3
	copy(g[8:], d4[:])
	return g
}

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

// COM vtable call helper
func comCall(obj uintptr, methodIndex int, args ...uintptr) uintptr {
	vtable := *(*uintptr)(unsafe.Pointer(obj))
	method := *(*uintptr)(unsafe.Pointer(vtable + uintptr(methodIndex)*unsafe.Sizeof(uintptr(0))))

	allArgs := make([]uintptr, 0, 1+len(args))
	allArgs = append(allArgs, obj) // this pointer
	allArgs = append(allArgs, args...)

	var ret uintptr
	switch len(allArgs) {
	case 1:
		ret, _, _ = syscall.Syscall(method, 1, allArgs[0], 0, 0)
	case 2:
		ret, _, _ = syscall.Syscall(method, 2, allArgs[0], allArgs[1], 0)
	case 3:
		ret, _, _ = syscall.Syscall(method, 3, allArgs[0], allArgs[1], allArgs[2])
	case 4:
		ret, _, _ = syscall.Syscall6(method, 4, allArgs[0], allArgs[1], allArgs[2], allArgs[3], 0, 0)
	case 5:
		ret, _, _ = syscall.Syscall6(method, 5, allArgs[0], allArgs[1], allArgs[2], allArgs[3], allArgs[4], 0)
	case 6:
		ret, _, _ = syscall.Syscall6(method, 6, allArgs[0], allArgs[1], allArgs[2], allArgs[3], allArgs[4], allArgs[5])
	case 7:
		ret, _, _ = syscall.Syscall9(method, 7, allArgs[0], allArgs[1], allArgs[2], allArgs[3], allArgs[4], allArgs[5], allArgs[6], 0, 0)
	case 8:
		ret, _, _ = syscall.Syscall9(method, 8, allArgs[0], allArgs[1], allArgs[2], allArgs[3], allArgs[4], allArgs[5], allArgs[6], allArgs[7], 0)
	case 9:
		ret, _, _ = syscall.Syscall9(method, 9, allArgs[0], allArgs[1], allArgs[2], allArgs[3], allArgs[4], allArgs[5], allArgs[6], allArgs[7], allArgs[8])
	case 10:
		ret, _, _ = syscall.Syscall12(method, 10, allArgs[0], allArgs[1], allArgs[2], allArgs[3], allArgs[4], allArgs[5], allArgs[6], allArgs[7], allArgs[8], allArgs[9], 0, 0)
	case 11:
		ret, _, _ = syscall.Syscall12(method, 11, allArgs[0], allArgs[1], allArgs[2], allArgs[3], allArgs[4], allArgs[5], allArgs[6], allArgs[7], allArgs[8], allArgs[9], allArgs[10], 0)
	case 12:
		ret, _, _ = syscall.Syscall12(method, 12, allArgs[0], allArgs[1], allArgs[2], allArgs[3], allArgs[4], allArgs[5], allArgs[6], allArgs[7], allArgs[8], allArgs[9], allArgs[10], allArgs[11])
	}
	return ret
}

func comRelease(obj uintptr) {
	if obj != 0 {
		comCall(obj, 2) // IUnknown::Release is vtable index 2
	}
}

// Layout types for markdown rendering
const (
	blockHeading = iota
	blockParagraph
	blockCode
	blockQuote
	blockListItem
	blockHR
)

type LayoutBlock struct {
	Type       int
	Text       string
	FontSize   float32
	Bold       bool
	Indent     float32
	SpaceAbove float32
	Color      uint32   // ARGB
	BgColor    uint32   // ARGB, 0 = none
	BarColor   uint32   // ARGB, 0 = none
	Y          float32  // computed during layout
	Height     float32  // measured height
}

// D2D resources (created when render target exists)
type d2dResources struct {
	factory      uintptr // ID2D1Factory
	renderTarget uintptr // ID2D1HwndRenderTarget
	dwFactory    uintptr // IDWriteFactory

	// Text formats
	fmtH1   uintptr // IDWriteTextFormat
	fmtH2   uintptr
	fmtH3   uintptr
	fmtBody uintptr
	fmtCode uintptr

	// Brushes
	brushText  uintptr // ID2D1SolidColorBrush
	brushGray  uintptr
	brushBlue  uintptr
	brushCodeBg uintptr
	brushHR    uintptr
	brushWhite uintptr
}

var res d2dResources

func floatBits(f float32) uintptr {
	return uintptr(math.Float32bits(f))
}

func createTextFormat(dwFactory uintptr, family string, weight, style, stretch uint32, size float32) uintptr {
	familyU := utf16From(family)
	localeU := utf16From("en-us")
	var fmt uintptr
	comCall(dwFactory, 15, // IDWriteFactory::CreateTextFormat
		uintptr(unsafe.Pointer(&familyU[0])),
		0, // font collection (nil = system)
		uintptr(weight),
		uintptr(style),
		uintptr(stretch),
		floatBits(size),
		uintptr(unsafe.Pointer(&localeU[0])),
		uintptr(unsafe.Pointer(&fmt)),
	)
	if fmt != 0 {
		// Set word wrapping
		comCall(fmt, 26, uintptr(DWRITE_WORD_WRAPPING_WRAP)) // SetWordWrapping
	}
	return fmt
}

func createBrush(rt uintptr, r, g, b, a float32) uintptr {
	color := [4]float32{r, g, b, a}
	var brush uintptr
	comCall(rt, 8, // ID2D1RenderTarget::CreateSolidColorBrush
		uintptr(unsafe.Pointer(&color[0])),
		0,
		uintptr(unsafe.Pointer(&brush)),
	)
	return brush
}

func initD2D(hwnd uintptr) bool {
	// Initialize COM
	coInitializeEx.Call(0, COINIT_APARTMENTTHREADED)

	// Create D2D1 factory
	iid := IID_ID2D1Factory
	hr, _, _ := d2d1CreateFactory.Call(
		D2D1_FACTORY_TYPE_SINGLE_THREADED,
		uintptr(unsafe.Pointer(&iid[0])),
		0,
		uintptr(unsafe.Pointer(&res.factory)),
	)
	if hr != 0 || res.factory == 0 {
		return false
	}

	// Create DWrite factory
	iid2 := IID_IDWriteFactory
	hr, _, _ = dwriteCreateFactory.Call(
		DWRITE_FACTORY_TYPE_SHARED,
		uintptr(unsafe.Pointer(&iid2[0])),
		uintptr(unsafe.Pointer(&res.dwFactory)),
	)
	if hr != 0 || res.dwFactory == 0 {
		return false
	}

	return createDeviceResources(hwnd)
}

func createDeviceResources(hwnd uintptr) bool {
	if res.renderTarget != 0 {
		return true
	}

	var rc [16]byte
	getClientRect.Call(hwnd, uintptr(unsafe.Pointer(&rc[0])))
	w := *(*int32)(unsafe.Pointer(&rc[8]))
	h := *(*int32)(unsafe.Pointer(&rc[12]))

	// D2D1_RENDER_TARGET_PROPERTIES (default)
	var rtProps [28]byte // type(4)+pixelFormat(8)+dpiX(4)+dpiY(4)+usage(4)+minLevel(4)

	// D2D1_HWND_RENDER_TARGET_PROPERTIES
	var hwndProps [20]byte // hwnd(8)+size(8)+presentOptions(4)
	*(*uintptr)(unsafe.Pointer(&hwndProps[0])) = hwnd
	*(*uint32)(unsafe.Pointer(&hwndProps[8])) = uint32(w)
	*(*uint32)(unsafe.Pointer(&hwndProps[12])) = uint32(h)

	// ID2D1Factory::CreateHwndRenderTarget is vtable index 14
	hr := comCall(res.factory, 14,
		uintptr(unsafe.Pointer(&rtProps[0])),
		uintptr(unsafe.Pointer(&hwndProps[0])),
		uintptr(unsafe.Pointer(&res.renderTarget)),
	)
	if hr != 0 || res.renderTarget == 0 {
		return false
	}

	// Create text formats
	res.fmtH1 = createTextFormat(res.dwFactory, "Segoe UI", DWRITE_FONT_WEIGHT_BOLD, DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL, 28)
	res.fmtH2 = createTextFormat(res.dwFactory, "Segoe UI", DWRITE_FONT_WEIGHT_BOLD, DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL, 22)
	res.fmtH3 = createTextFormat(res.dwFactory, "Segoe UI", DWRITE_FONT_WEIGHT_BOLD, DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL, 18)
	res.fmtBody = createTextFormat(res.dwFactory, "Segoe UI", DWRITE_FONT_WEIGHT_NORMAL, DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL, 14)
	res.fmtCode = createTextFormat(res.dwFactory, "Consolas", DWRITE_FONT_WEIGHT_NORMAL, DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL, 13)

	// Create brushes
	res.brushText = createBrush(res.renderTarget, 0.13, 0.13, 0.13, 1) // dark text
	res.brushGray = createBrush(res.renderTarget, 0.4, 0.4, 0.4, 1)    // gray
	res.brushBlue = createBrush(res.renderTarget, 0.01, 0.4, 0.84, 1)  // link blue
	res.brushCodeBg = createBrush(res.renderTarget, 0.94, 0.94, 0.94, 1) // code bg
	res.brushHR = createBrush(res.renderTarget, 0.85, 0.85, 0.85, 1)     // hr line
	res.brushWhite = createBrush(res.renderTarget, 1, 1, 1, 1)           // white bg

	return true
}

func discardDeviceResources() {
	for _, p := range []*uintptr{
		&res.brushText, &res.brushGray, &res.brushBlue,
		&res.brushCodeBg, &res.brushHR, &res.brushWhite,
		&res.fmtH1, &res.fmtH2, &res.fmtH3, &res.fmtBody, &res.fmtCode,
		&res.renderTarget,
	} {
		if *p != 0 {
			comRelease(*p)
			*p = 0
		}
	}
}

// Markdown → layout blocks
func markdownToLayout(source []byte) []LayoutBlock {
	md := goldmark.New()
	reader := text.NewReader(source)
	doc := md.Parser().Parse(reader)

	var blocks []LayoutBlock
	ast.Walk(doc, func(n ast.Node, entering bool) (ast.WalkStatus, error) {
		if !entering {
			return ast.WalkContinue, nil
		}
		switch n.Kind() {
		case ast.KindHeading:
			h := n.(*ast.Heading)
			txt := extractInlineText(n, source)
			var fs float32
			switch h.Level {
			case 1:
				fs = 28
			case 2:
				fs = 22
			default:
				fs = 18
			}
			blocks = append(blocks, LayoutBlock{
				Type: blockHeading, Text: txt, FontSize: fs, Bold: true,
				SpaceAbove: 16, Color: 0xFF222222,
			})
			return ast.WalkSkipChildren, nil

		case ast.KindParagraph:
			if n.Parent() != nil && n.Parent().Kind() == ast.KindBlockquote {
				return ast.WalkContinue, nil
			}
			txt := extractInlineText(n, source)
			if txt != "" {
				blocks = append(blocks, LayoutBlock{
					Type: blockParagraph, Text: txt, FontSize: 14,
					SpaceAbove: 8, Color: 0xFF222222,
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
			blocks = append(blocks, LayoutBlock{
				Type: blockCode, Text: strings.TrimRight(buf.String(), "\n"),
				FontSize: 13, SpaceAbove: 8,
				Color: 0xFF333333, BgColor: 0xFFF0F0F0,
			})
			return ast.WalkSkipChildren, nil

		case ast.KindBlockquote:
			txt := extractInlineText(n, source)
			blocks = append(blocks, LayoutBlock{
				Type: blockQuote, Text: txt, FontSize: 14,
				Indent: 20, SpaceAbove: 8,
				Color: 0xFF666666, BarColor: 0xFF0366D6,
			})
			return ast.WalkSkipChildren, nil

		case ast.KindListItem:
			txt := "•  " + extractInlineText(n, source)
			blocks = append(blocks, LayoutBlock{
				Type: blockListItem, Text: txt, FontSize: 14,
				Indent: 20, SpaceAbove: 4, Color: 0xFF222222,
			})
			return ast.WalkSkipChildren, nil

		case ast.KindThematicBreak:
			blocks = append(blocks, LayoutBlock{
				Type: blockHR, SpaceAbove: 12,
			})
		}
		return ast.WalkContinue, nil
	})
	return blocks
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
	editorFont  uintptr
	currentFile string

	currentBlocks []LayoutBlock
	scrollY       float32
	totalHeight   float32
)

func getTextFormat(b *LayoutBlock) uintptr {
	if b.Type == blockCode {
		return res.fmtCode
	}
	if b.Bold {
		switch {
		case b.FontSize >= 28:
			return res.fmtH1
		case b.FontSize >= 22:
			return res.fmtH2
		default:
			return res.fmtH3
		}
	}
	return res.fmtBody
}

func getBrush(b *LayoutBlock) uintptr {
	if b.Color == 0xFF666666 {
		return res.brushGray
	}
	return res.brushText
}

func renderPreview(hwnd uintptr) {
	if res.renderTarget == 0 {
		if !createDeviceResources(hwnd) {
			return
		}
	}

	var rc [16]byte
	getClientRect.Call(hwnd, uintptr(unsafe.Pointer(&rc[0])))
	clientW := float32(*(*int32)(unsafe.Pointer(&rc[8])))
	clientH := float32(*(*int32)(unsafe.Pointer(&rc[12])))

	padding := float32(20)
	drawWidth := clientW - padding*2

	// Begin draw — ID2D1RenderTarget::BeginDraw is vtable index 48
	comCall(res.renderTarget, 48)

	// Clear to white — ID2D1RenderTarget::Clear is vtable index 47
	white := [4]float32{1, 1, 1, 1}
	comCall(res.renderTarget, 47, uintptr(unsafe.Pointer(&white[0])))

	// Set transform for scrolling — ID2D1RenderTarget::SetTransform is vtable index 30
	// Matrix3x2F: [m11, m12, m21, m22, dx, dy]
	transform := [6]float32{1, 0, 0, 1, 0, -scrollY}
	comCall(res.renderTarget, 30, uintptr(unsafe.Pointer(&transform[0])))

	y := padding
	for i := range currentBlocks {
		b := &currentBlocks[i]
		y += b.SpaceAbove
		b.Y = y

		if b.Type == blockHR {
			// Draw horizontal line
			if y-scrollY >= -10 && y-scrollY < clientH+10 {
				p1 := [2]float32{padding, y}
				p2 := [2]float32{clientW - padding, y}
				// ID2D1RenderTarget::DrawLine is vtable index 15
				comCall(res.renderTarget, 15,
					uintptr(unsafe.Pointer(&p1[0])),
					uintptr(unsafe.Pointer(&p2[0])),
					res.brushHR,
					floatBits(1), // stroke width
					0,
				)
			}
			y += 12
			continue
		}

		// Create text layout for measurement
		textU := utf16From(b.Text)
		textLen := uint32(len(textU) - 1)
		fmt := getTextFormat(b)
		if fmt == 0 {
			continue
		}

		var layout uintptr
		// IDWriteFactory::CreateTextLayout is vtable index 18
		comCall(res.dwFactory, 18,
			uintptr(unsafe.Pointer(&textU[0])),
			uintptr(textLen),
			fmt,
			floatBits(drawWidth-b.Indent),
			floatBits(10000),
			uintptr(unsafe.Pointer(&layout)),
		)
		if layout == 0 {
			continue
		}

		// Get metrics to measure height
		// IDWriteTextLayout inherits from IDWriteTextFormat
		// GetMetrics is at vtable index 60 for IDWriteTextLayout
		var metrics [36]byte // DWRITE_TEXT_METRICS struct
		comCall(layout, 60, uintptr(unsafe.Pointer(&metrics[0])))
		textH := *(*float32)(unsafe.Pointer(&metrics[16])) // height field

		b.Height = textH

		// Only draw if visible
		if y-scrollY+textH >= 0 && y-scrollY < clientH {
			// Draw code block background
			if b.BgColor != 0 {
				bgRect := [4]float32{
					padding + b.Indent - 8, y - 4,
					clientW - padding + 8, y + textH + 4,
				}
				// ID2D1RenderTarget::FillRectangle is vtable index 16
				comCall(res.renderTarget, 16,
					uintptr(unsafe.Pointer(&bgRect[0])),
					res.brushCodeBg,
				)
			}

			// Draw blockquote left bar
			if b.BarColor != 0 {
				barRect := [4]float32{
					padding, y - 2,
					padding + 3, y + textH + 2,
				}
				comCall(res.renderTarget, 16,
					uintptr(unsafe.Pointer(&barRect[0])),
					res.brushBlue,
				)
			}

			// Draw text
			origin := [2]float32{padding + b.Indent, y}
			brush := getBrush(b)
			// ID2D1RenderTarget::DrawTextLayout is vtable index 27
			comCall(res.renderTarget, 27,
				uintptr(unsafe.Pointer(&origin[0])),
				layout,
				brush,
				0, // D2D1_DRAW_TEXT_OPTIONS_NONE
			)
		}

		comRelease(layout)
		y += textH
	}

	totalHeight = y + padding

	// End draw — ID2D1RenderTarget::EndDraw is vtable index 49
	var tag1, tag2 uint64
	hr := comCall(res.renderTarget, 49,
		uintptr(unsafe.Pointer(&tag1)),
		uintptr(unsafe.Pointer(&tag2)),
	)

	// Check for device loss (D2DERR_RECREATE_TARGET = 0x8899000C)
	if hr == 0x8899000C {
		discardDeviceResources()
	}

	// Update scrollbar
	updateScrollbar(hwnd, int32(clientH))
}

func updateScrollbar(hwnd uintptr, clientH int32) {
	var si [28]byte
	*(*uint32)(unsafe.Pointer(&si[0])) = 28
	*(*uint32)(unsafe.Pointer(&si[4])) = SIF_RANGE | SIF_PAGE | SIF_POS
	*(*int32)(unsafe.Pointer(&si[8])) = 0
	*(*int32)(unsafe.Pointer(&si[12])) = int32(totalHeight)
	*(*uint32)(unsafe.Pointer(&si[16])) = uint32(clientH)
	*(*int32)(unsafe.Pointer(&si[20])) = int32(scrollY)
	setScrollInfo.Call(hwnd, SB_VERT, uintptr(unsafe.Pointer(&si[0])), 1)
}

// Main window proc
func mainWndProc(hwnd, msg, wParam, lParam uintptr) uintptr {
	switch msg {
	case WM_CREATE:
		// Editor (left)
		editorHwnd, _, _ = createWindowExW.Call(
			WS_EX_CLIENTEDGE,
			uintptr(unsafe.Pointer(utf16Ptr("EDIT"))),
			0,
			WS_CHILD|WS_VISIBLE|WS_VSCROLL|ES_MULTILINE|ES_AUTOVSCROLL|ES_WANTRETURN,
			0, 0, 0, 0,
			hwnd, 1, hInstance, 0,
		)
		editorFont, _, _ = createFontW.Call(
			uintptr(uint32(uint16(14))|0xFFFF0000), 0, 0, 0,
			400, 0, 0, 0, 0, 0, 0, 0, 0,
			uintptr(unsafe.Pointer(utf16Ptr("Consolas"))),
		)
		sendMessageW.Call(editorHwnd, WM_SETFONT, editorFont, 1)

		// Preview (right) — custom D2D window
		previewClass := utf16From("TinyMDD2DPreview")
		var wc [80]byte
		*(*uint32)(unsafe.Pointer(&wc[0])) = 80
		*(*uintptr)(unsafe.Pointer(&wc[8])) = syscall.NewCallback(previewWndProc)
		*(*uintptr)(unsafe.Pointer(&wc[48])) = hInstance
		cursor, _, _ := loadCursorW.Call(0, IDC_ARROW)
		*(*uintptr)(unsafe.Pointer(&wc[56])) = cursor
		*(*uintptr)(unsafe.Pointer(&wc[72])) = uintptr(unsafe.Pointer(&previewClass[0]))
		registerClassExW.Call(uintptr(unsafe.Pointer(&wc[0])))

		previewHwnd, _, _ = createWindowExW.Call(
			0,
			uintptr(unsafe.Pointer(&previewClass[0])),
			0,
			WS_CHILD|WS_VISIBLE|WS_VSCROLL,
			0, 0, 0, 0,
			hwnd, 2, hInstance, 0,
		)

		// Initialize D2D with preview window
		initD2D(previewHwnd)

		return 0

	case WM_SIZE:
		w := int32(loword(lParam))
		h := int32(hiword(lParam))
		half := w / 2
		divider := int32(6)
		moveWindow.Call(editorHwnd, 0, 0, uintptr(half-divider/2), uintptr(h), 1)
		moveWindow.Call(previewHwnd, uintptr(half+divider/2), 0, uintptr(w-half-divider/2), uintptr(h), 1)

		// Resize D2D render target
		if res.renderTarget != 0 {
			newW := uint32(w - int32(half) - divider/2)
			newH := uint32(h)
			size := [2]uint32{newW, newH}
			// ID2D1HwndRenderTarget::Resize is vtable index 53 (after ID2D1RenderTarget methods)
			comCall(res.renderTarget, 53, uintptr(unsafe.Pointer(&size[0])))
		}
		return 0

	case WM_COMMAND:
		if hiword(wParam) == EN_CHANGE && loword(wParam) == 1 {
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
		state, _, _ := getKeyState.Call(0x11)
		if int16(state) < 0 && wParam == VK_S {
			saveFile()
			return 0
		}

	case WM_SETFOCUS:
		setFocus.Call(editorHwnd)
		return 0

	case WM_DESTROY:
		discardDeviceResources()
		comRelease(res.dwFactory)
		comRelease(res.factory)
		deleteObject.Call(editorFont)
		postQuitMessage.Call(0)
		return 0
	}

	ret, _, _ := defWindowProcW.Call(hwnd, msg, wParam, lParam)
	return ret
}

func previewWndProc(hwnd, msg, wParam, lParam uintptr) uintptr {
	switch msg {
	case WM_PAINT:
		var ps [72]byte
		beginPaint.Call(hwnd, uintptr(unsafe.Pointer(&ps[0])))
		renderPreview(hwnd)
		endPaint.Call(hwnd, uintptr(unsafe.Pointer(&ps[0])))
		return 0

	case WM_ERASEBKGND:
		return 1

	case WM_SIZE:
		if res.renderTarget != 0 {
			w := uint32(loword(lParam))
			h := uint32(hiword(lParam))
			size := [2]uint32{w, h}
			comCall(res.renderTarget, 53, uintptr(unsafe.Pointer(&size[0])))
			invalidateRect.Call(hwnd, 0, 0)
		}
		return 0

	case WM_MOUSEWHEEL:
		delta := int16(hiword(wParam))
		scrollY -= float32(delta) / 3
		if scrollY < 0 {
			scrollY = 0
		}
		var rc [16]byte
		getClientRect.Call(hwnd, uintptr(unsafe.Pointer(&rc[0])))
		clientH := float32(*(*int32)(unsafe.Pointer(&rc[12])))
		maxScroll := totalHeight - clientH
		if maxScroll < 0 {
			maxScroll = 0
		}
		if scrollY > maxScroll {
			scrollY = maxScroll
		}
		invalidateRect.Call(hwnd, 0, 0)
		return 0

	case WM_VSCROLL:
		code := loword(wParam)
		var rc [16]byte
		getClientRect.Call(hwnd, uintptr(unsafe.Pointer(&rc[0])))
		clientH := float32(*(*int32)(unsafe.Pointer(&rc[12])))
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
			scrollY = float32(int16(hiword(wParam)))
		}
		if scrollY < 0 {
			scrollY = 0
		}
		if scrollY > maxScroll {
			scrollY = maxScroll
		}
		invalidateRect.Call(hwnd, 0, 0)
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
		invalidateRect.Call(previewHwnd, 0, 0)
		return
	}
	buf := make([]uint16, length+1)
	getWindowTextW.Call(editorHwnd, uintptr(unsafe.Pointer(&buf[0])), length+1)
	mdText := syscall.UTF16ToString(buf)

	currentBlocks = markdownToLayout([]byte(mdText))
	invalidateRect.Call(previewHwnd, 0, 0)
}

// File operations
func saveFile() {
	if currentFile == "" {
		currentFile = showSaveDialog(mainHwnd)
		if currentFile == "" {
			return
		}
	}
	length, _, _ := getWindowTextLengthW.Call(editorHwnd)
	buf := make([]uint16, length+1)
	getWindowTextW.Call(editorHwnd, uintptr(unsafe.Pointer(&buf[0])), length+1)
	content := syscall.UTF16ToString(buf)
	os.WriteFile(currentFile, []byte(content), 0644)
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
	defExt := utf16From("md")

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

	// Register main window class
	className := utf16From("TinyMDD2D")
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

	title := "TinyMD Direct2D Prototype"
	if currentFile != "" {
		title = "TinyMD D2D — " + currentFile
	}
	titleU := utf16From(title)

	mainHwnd, _, _ = createWindowExW.Call(
		0,
		uintptr(unsafe.Pointer(&className[0])),
		uintptr(unsafe.Pointer(&titleU[0])),
		WS_OVERLAPPEDWINDOW|WS_VISIBLE|WS_CLIPCHILDREN,
		x, y, w, h,
		0, 0, hInstance, 0,
	)

	showWindowProc.Call(mainHwnd, 5)
	updateWindowProc.Call(mainHwnd)

	// Set initial content
	if initialContent != "" {
		txt := utf16From(initialContent)
		sendMessageW.Call(editorHwnd, WM_SETTEXT, 0, uintptr(unsafe.Pointer(&txt[0])))
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
}
