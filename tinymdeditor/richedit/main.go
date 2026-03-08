// TinyMD RichEdit Prototype — native markdown preview using RichEdit + RTF.
//
// Build: GOOS=windows go build -ldflags="-s -w -H windowsgui" -trimpath -o tinymd-richedit.exe .

package main

import (
	"fmt"
	"os"
	"runtime"
	"strings"
	"syscall"
	"unsafe"

	"github.com/yuin/goldmark"
	"github.com/yuin/goldmark/ast"
	"github.com/yuin/goldmark/extension"
	east "github.com/yuin/goldmark/extension/ast"
	"github.com/yuin/goldmark/text"
)

// Win32 DLLs and procs
var (
	user32   = syscall.NewLazyDLL("user32.dll")
	gdi32    = syscall.NewLazyDLL("gdi32.dll")
	kernel32 = syscall.NewLazyDLL("kernel32.dll")
	comdlg32 = syscall.NewLazyDLL("comdlg32.dll")
	msftedit = syscall.NewLazyDLL("msftedit.dll") // registers RICHEDIT50W

	// user32
	registerClassExW     = user32.NewProc("RegisterClassExW")
	createWindowExW      = user32.NewProc("CreateWindowExW")
	showWindowProc       = user32.NewProc("ShowWindow")
	updateWindowProc     = user32.NewProc("UpdateWindow")
	defWindowProcW       = user32.NewProc("DefWindowProcW")
	getSystemMetrics     = user32.NewProc("GetSystemMetrics")
	loadCursorW          = user32.NewProc("LoadCursorW")
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
	getKeyState          = user32.NewProc("GetKeyState")

	// gdi32
	createFontW  = gdi32.NewProc("CreateFontW")
	deleteObject = gdi32.NewProc("DeleteObject")

	// kernel32
	getModuleHandleW = kernel32.NewProc("GetModuleHandleW")

	// comdlg32
	getSaveFileNameW = comdlg32.NewProc("GetSaveFileNameW")
	printDlgW        = comdlg32.NewProc("PrintDlgW")

	// gdi32 — printing
	startDocW     = gdi32.NewProc("StartDocW")
	endDoc        = gdi32.NewProc("EndDoc")
	startPage     = gdi32.NewProc("StartPage")
	endPage       = gdi32.NewProc("EndPage")
	getDeviceCaps = gdi32.NewProc("GetDeviceCaps")
	deleteDC      = gdi32.NewProc("DeleteDC")
)

// Win32 constants
const (
	WS_OVERLAPPEDWINDOW = 0x00CF0000
	WS_VISIBLE          = 0x10000000
	WS_CHILD            = 0x40000000
	WS_VSCROLL          = 0x00200000
	WS_HSCROLL          = 0x00100000
	WS_CLIPCHILDREN     = 0x02000000
	WS_EX_CLIENTEDGE    = 0x00000200
	ES_MULTILINE        = 0x0004
	ES_AUTOVSCROLL      = 0x0040
	ES_WANTRETURN       = 0x1000
	ES_READONLY          = 0x0800
	SM_CXSCREEN          = 0
	SM_CYSCREEN          = 1
	IDC_ARROW            = 32512

	WM_CREATE    = 0x0001
	WM_DESTROY   = 0x0002
	WM_SIZE      = 0x0005
	WM_COMMAND   = 0x0111
	WM_TIMER     = 0x0113
	WM_KEYDOWN   = 0x0100
	WM_SETFONT   = 0x0030
	WM_SETFOCUS  = 0x0007
	WM_SETTEXT   = 0x000C

	EN_CHANGE = 0x0300
	VK_S      = 0x53

	EM_SETBKGNDCOLOR = 0x0443
	EM_SETMARGINS    = 0x00D3
	EC_LEFTMARGIN    = 0x0001
	EC_RIGHTMARGIN   = 0x0002

	TIMER_DEBOUNCE = 1

	VK_P = 0x50

	// Printing
	PD_RETURNDC        = 0x00000100
	PD_USEDEVMODECOPIESANDCOLLATE = 0x00040000
	EM_FORMATRANGE     = 0x0439
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

// Use EM_SETTEXTEX to load RTF directly (no callback needed)
const EM_SETTEXTEX = 0x0461

func setRTFContent(hwnd uintptr, rtfData string) {
	// SETTEXTEX struct: flags(4) + codepage(4)
	var st [8]byte
	*(*uint32)(unsafe.Pointer(&st[0])) = 0 // ST_DEFAULT
	*(*uint32)(unsafe.Pointer(&st[4])) = 0 // CP_ACP for ANSI RTF
	// RTF must be null-terminated ANSI bytes
	buf := append([]byte(rtfData), 0)
	sendMessageW.Call(hwnd, EM_SETTEXTEX, uintptr(unsafe.Pointer(&st[0])), uintptr(unsafe.Pointer(&buf[0])))
}

// RTF generation from goldmark AST
// Color table indices (1-based):
//   1 = dark text (#222222)
//   2 = blue for blockquote border (#0366D6)
//   3 = gray text for blockquotes (#666666)
//   4 = code bg gray (#D0D0D0)
//   5 = hr border gray (#B4B4B4)
//   6 = blockquote bg (#F6F8FA)
const rtfHeader = `{\rtf1\ansi\deff0` +
	`{\fonttbl{\f0\fswiss Segoe UI;}{\f1\fmodern\fcharset0 Consolas;}}` +
	`{\colortbl;\red34\green34\blue34;\red3\green102\blue214;\red102\green102\blue102;\red208\green208\blue208;\red180\green180\blue180;\red246\green248\blue250;}` +
	`\viewkind4\uc1\pard\cf1\f0\fs28 `

func countTableColumns(table ast.Node) int {
	// Count columns from the first row (header)
	for row := table.FirstChild(); row != nil; row = row.NextSibling() {
		cols := 0
		for cell := row.FirstChild(); cell != nil; cell = cell.NextSibling() {
			cols++
		}
		if cols > 0 {
			return cols
		}
	}
	return 1
}

func getCellTextLength(cell ast.Node, source []byte) int {
	length := 0
	for child := cell.FirstChild(); child != nil; child = child.NextSibling() {
		length += getNodeTextLength(child, source)
	}
	return length
}

func getNodeTextLength(n ast.Node, source []byte) int {
	switch n.Kind() {
	case ast.KindText:
		return len(n.(*ast.Text).Segment.Value(source))
	case ast.KindString:
		return len(n.(*ast.String).Value)
	default:
		length := 0
		for child := n.FirstChild(); child != nil; child = child.NextSibling() {
			length += getNodeTextLength(child, source)
		}
		return length
	}
}

// measureTableColumnWidths returns per-column twip widths based on content.
// It measures the max text length per column across all rows and converts
// to twips using an approximate character width (150 twips per char at fs28).
func measureTableColumnWidths(table ast.Node, source []byte, numCols int) []int {
	maxChars := make([]int, numCols)
	for row := table.FirstChild(); row != nil; row = row.NextSibling() {
		col := 0
		for cell := row.FirstChild(); cell != nil; cell = cell.NextSibling() {
			if col < numCols {
				n := getCellTextLength(cell, source)
				if n < 3 {
					n = 3 // minimum width
				}
				if n > maxChars[col] {
					maxChars[col] = n
				}
			}
			col++
		}
	}
	// Convert to twips: ~150 twips per character at fs28 Segoe UI, plus 500 twips padding
	widths := make([]int, numCols)
	for i, ch := range maxChars {
		w := ch*150 + 500
		if w < 1500 {
			w = 1500 // minimum column width
		}
		widths[i] = w
	}
	return widths
}

func writeTableRow(buf *strings.Builder, row ast.Node, source []byte, numCols int, isHeader bool, colWidths []int) {
	buf.WriteString(`\trowd\trgaph108`)
	// Define cell borders and positions using content-based widths
	cumulative := 0
	for i := 0; i < numCols; i++ {
		cumulative += colWidths[i]
		buf.WriteString(`\clbrdrt\brdrs\brdrw10`)
		buf.WriteString(`\clbrdrl\brdrs\brdrw10`)
		buf.WriteString(`\clbrdrb\brdrs\brdrw10`)
		buf.WriteString(`\clbrdrr\brdrs\brdrw10`)
		buf.WriteString(fmt.Sprintf(`\cellx%d`, cumulative))
	}
	buf.WriteByte('\n')

	// Write each cell
	for cell := row.FirstChild(); cell != nil; cell = cell.NextSibling() {
		if isHeader {
			buf.WriteString(`\pard\intbl\b\f0\fs28 `)
		} else {
			buf.WriteString(`\pard\intbl\f0\fs28 `)
		}
		// Walk cell children to get text content
		for child := cell.FirstChild(); child != nil; child = child.NextSibling() {
			renderInlineNode(buf, child, source)
		}
		buf.WriteString(`\cell`)
	}
	if isHeader {
		buf.WriteString(`\b0`)
	}
	buf.WriteString(`\row`)
	buf.WriteByte('\n')
}

func renderInlineNode(buf *strings.Builder, n ast.Node, source []byte) {
	switch n.Kind() {
	case ast.KindText:
		t := n.(*ast.Text)
		buf.WriteString(escapeRTF(string(t.Segment.Value(source))))
		if t.SoftLineBreak() {
			buf.WriteByte(' ')
		}
	case ast.KindEmphasis:
		e := n.(*ast.Emphasis)
		if e.Level == 2 {
			buf.WriteString(`{\b `)
		} else {
			buf.WriteString(`{\i `)
		}
		for child := n.FirstChild(); child != nil; child = child.NextSibling() {
			renderInlineNode(buf, child, source)
		}
		buf.WriteString(`}`)
	case ast.KindCodeSpan:
		buf.WriteString(`{\f1\fs22\highlight4 `)
		for child := n.FirstChild(); child != nil; child = child.NextSibling() {
			if child.Kind() == ast.KindText {
				buf.WriteString(escapeRTF(string(child.(*ast.Text).Segment.Value(source))))
			}
		}
		buf.WriteString(`}`)
	case ast.KindString:
		buf.WriteString(escapeRTF(string(n.(*ast.String).Value)))
	case ast.KindParagraph:
		for child := n.FirstChild(); child != nil; child = child.NextSibling() {
			renderInlineNode(buf, child, source)
		}
	default:
		// For any other inline node, try to render children
		for child := n.FirstChild(); child != nil; child = child.NextSibling() {
			renderInlineNode(buf, child, source)
		}
	}
}

func markdownToRTF(source []byte) string {
	md := goldmark.New(goldmark.WithExtensions(extension.Table))
	reader := text.NewReader(source)
	doc := md.Parser().Parse(reader)

	var buf strings.Builder
	buf.WriteString(rtfHeader)

	ast.Walk(doc, func(n ast.Node, entering bool) (ast.WalkStatus, error) {
		switch n.Kind() {
		case ast.KindDocument:
			return ast.WalkContinue, nil

		case ast.KindHeading:
			if entering {
				h := n.(*ast.Heading)
				var fs, sbefore, safter int
				switch h.Level {
				case 1:
					fs = 56 // 28pt
					sbefore = 360
					safter = 120
				case 2:
					fs = 44 // 22pt
					sbefore = 300
					safter = 100
				case 3:
					fs = 36 // 18pt
					sbefore = 240
					safter = 80
				default:
					fs = 32
					sbefore = 200
					safter = 80
				}
				buf.WriteString(fmt.Sprintf(`\pard\sb%d\sa%d\cf1\b\f0\fs%d `, sbefore, safter, fs))
			} else {
				buf.WriteString(`\b0\par `)
			}
			return ast.WalkContinue, nil

		case ast.KindParagraph:
			if entering {
				if n.Parent() != nil && n.Parent().Kind() == ast.KindBlockquote {
					// handled by blockquote
					return ast.WalkContinue, nil
				}
				buf.WriteString(`\pard\sb120\sa60\cf1\f0\fs28 `)
			} else {
				if n.Parent() != nil && n.Parent().Kind() == ast.KindBlockquote {
					return ast.WalkContinue, nil
				}
				buf.WriteString(`\par `)
			}
			return ast.WalkContinue, nil

		case ast.KindText:
			if entering {
				t := n.(*ast.Text)
				buf.WriteString(escapeRTF(string(t.Segment.Value(source))))
				if t.SoftLineBreak() {
					buf.WriteByte(' ')
				}
				if t.HardLineBreak() {
					buf.WriteString(`\line `)
				}
			}

		case ast.KindEmphasis:
			e := n.(*ast.Emphasis)
			if entering {
				if e.Level == 2 {
					buf.WriteString(`{\b `)
				} else {
					buf.WriteString(`{\i `)
				}
			} else {
				buf.WriteString(`}`)
			}

		case ast.KindCodeSpan:
			if entering {
				buf.WriteString(`{\f1\fs22\highlight4 `)
				// Extract code span text
				for c := n.FirstChild(); c != nil; c = c.NextSibling() {
					if c.Kind() == ast.KindText {
						buf.WriteString(escapeRTF(string(c.(*ast.Text).Segment.Value(source))))
					}
				}
				buf.WriteString(`}`)
				return ast.WalkSkipChildren, nil
			}

		case ast.KindFencedCodeBlock, ast.KindCodeBlock:
			if entering {
				buf.WriteString(`\pard\sb180\sa180\li120\ri120\highlight4\cbpat4\f1\fs22\cf1 `)
				lines := n.Lines()
				for i := 0; i < lines.Len(); i++ {
					seg := lines.At(i)
					line := string(seg.Value(source))
					line = strings.TrimRight(line, "\n")
					buf.WriteString(escapeRTF(line))
					if i < lines.Len()-1 {
						buf.WriteString(`\line `)
					}
				}
				buf.WriteString(`\highlight0\f0\fs28\par `)
				return ast.WalkSkipChildren, nil
			}

		case ast.KindBlockquote:
			if entering {
				buf.WriteString(`\pard\sb120\sa120\li360\cf3\f0\fs28 {\cf2\u9612?} `)
			} else {
				buf.WriteString(`\cf1\par `)
			}
			return ast.WalkContinue, nil

		case ast.KindList:
			return ast.WalkContinue, nil

		case ast.KindListItem:
			if entering {
				buf.WriteString(`\pard\sb60\sa60\fi-360\li720\cf1\f0\fs28 \bullet\tab `)
			} else {
				buf.WriteString(`\par `)
			}
			return ast.WalkContinue, nil

		case east.KindTable:
			if entering {
				numCols := countTableColumns(n)
				colWidths := measureTableColumnWidths(n, source, numCols)
				for row := n.FirstChild(); row != nil; row = row.NextSibling() {
					isHeader := row.Kind() == east.KindTableHeader
					if row.Kind() == east.KindTableHeader || row.Kind() == east.KindTableRow {
						writeTableRow(&buf, row, source, numCols, isHeader, colWidths)
					}
				}
			}
			return ast.WalkSkipChildren, nil

		case east.KindTableHeader, east.KindTableRow, east.KindTableCell:
			// Handled by KindTable above
			return ast.WalkSkipChildren, nil

		case ast.KindThematicBreak:
			if entering {
				// Use a paragraph with a bottom border for a clean horizontal rule
				buf.WriteString(`\pard\sb120\sa120\brdrb\brdrs\brdrw10\brdrcf5 \par `)
			}

		case ast.KindHTMLBlock, ast.KindRawHTML:
			return ast.WalkSkipChildren, nil

		case ast.KindString:
			if entering {
				buf.WriteString(escapeRTF(string(n.(*ast.String).Value)))
			}
		}

		return ast.WalkContinue, nil
	})

	buf.WriteString(`}`)
	return buf.String()
}

func escapeRTF(s string) string {
	var buf strings.Builder
	for _, c := range s {
		switch {
		case c == '\\':
			buf.WriteString(`\\`)
		case c == '{':
			buf.WriteString(`\{`)
		case c == '}':
			buf.WriteString(`\}`)
		case c > 127:
			buf.WriteString(fmt.Sprintf(`\u%d?`, c))
		default:
			buf.WriteRune(c)
		}
	}
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
)

func mainWndProc(hwnd, msg, wParam, lParam uintptr) uintptr {
	switch msg {
	case WM_CREATE:
		// Load msftedit.dll to register RICHEDIT50W class
		msftedit.Load()

		// Create editor (left pane)
		editorHwnd, _, _ = createWindowExW.Call(
			WS_EX_CLIENTEDGE,
			uintptr(unsafe.Pointer(utf16Ptr("EDIT"))),
			0,
			WS_CHILD|WS_VISIBLE|WS_VSCROLL|ES_MULTILINE|ES_AUTOVSCROLL|ES_WANTRETURN,
			0, 0, 0, 0,
			hwnd, 1, hInstance, 0,
		)
		editorFont, _, _ = createFontW.Call(
			uintptr(0xFFFFFFF2), 0, 0, 0, // -14 pixel height
			400, 0, 0, 0, 0, 0, 0, 0, 0,
			uintptr(unsafe.Pointer(utf16Ptr("Consolas"))),
		)
		sendMessageW.Call(editorHwnd, WM_SETFONT, editorFont, 1)

		// Set left/right margins (~10px each) on editor pane
		editorMargins := uintptr((10 << 16) | 10) // HIWORD=right, LOWORD=left
		sendMessageW.Call(editorHwnd, EM_SETMARGINS, 3, editorMargins)

		// Create preview (right pane) — RichEdit
		richeditClass := utf16From("RICHEDIT50W")
		previewHwnd, _, _ = createWindowExW.Call(
			WS_EX_CLIENTEDGE,
			uintptr(unsafe.Pointer(&richeditClass[0])),
			0,
			WS_CHILD|WS_VISIBLE|WS_VSCROLL|ES_MULTILINE|ES_READONLY,
			0, 0, 0, 0,
			hwnd, 2, hInstance, 0,
		)

		// Set white background for preview
		sendMessageW.Call(previewHwnd, EM_SETBKGNDCOLOR, 0, 0x00FFFFFF)

		// Set left/right margins (~30px each) on preview pane
		margins := uintptr((30 << 16) | 30) // HIWORD=right, LOWORD=left
		sendMessageW.Call(previewHwnd, EM_SETMARGINS, EC_LEFTMARGIN|EC_RIGHTMARGIN, margins)

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
		state, _, _ := getKeyState.Call(0x11) // VK_CONTROL
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
		deleteObject.Call(editorFont)
		postQuitMessage.Call(0)
		return 0
	}

	ret, _, _ := defWindowProcW.Call(hwnd, msg, wParam, lParam)
	return ret
}

func updatePreview() {
	length, _, _ := getWindowTextLengthW.Call(editorHwnd)
	if length == 0 {
		setRTFContent(previewHwnd, `{\rtf1\ansi }`)
		return
	}
	buf := make([]uint16, length+1)
	getWindowTextW.Call(editorHwnd, uintptr(unsafe.Pointer(&buf[0])), length+1)
	mdText := syscall.UTF16ToString(buf)

	rtf := markdownToRTF([]byte(mdText))
	setRTFContent(previewHwnd, rtf)
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

func printFormatted() {
	// PRINTDLGW is 120 bytes on 64-bit
	const pdSize = 120
	var pd [pdSize]byte
	*(*uint32)(unsafe.Pointer(&pd[0])) = pdSize
	*(*uintptr)(unsafe.Pointer(&pd[8])) = mainHwnd
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

	// Get printer resolution
	dpiX, _, _ := getDeviceCaps.Call(printerDC, LOGPIXELSX)
	dpiY, _, _ := getDeviceCaps.Call(printerDC, LOGPIXELSY)
	pageW, _, _ := getDeviceCaps.Call(printerDC, HORZRES)
	pageH, _, _ := getDeviceCaps.Call(printerDC, VERTRES)

	// FORMATRANGE struct layout (64-bit):
	// hdc(8) + hdcTarget(8) + rc(16: RECT in twips) + rcPage(16: RECT in twips) + CHARRANGE(8: cpMin+cpMax)
	// Total: 56 bytes
	// rc and rcPage are in twips (1 twip = 1/1440 inch)
	rcRight := int32(pageW) * 1440 / int32(dpiX)
	rcBottom := int32(pageH) * 1440 / int32(dpiY)

	// Get total text length
	textLen, _, _ := getWindowTextLengthW.Call(previewHwnd)

	// StartDoc
	docName := utf16From("TinyMD Print")
	var di [40]byte
	*(*int32)(unsafe.Pointer(&di[0])) = 40
	*(*uintptr)(unsafe.Pointer(&di[8])) = uintptr(unsafe.Pointer(&docName[0]))
	r, _, _ := startDocW.Call(printerDC, uintptr(unsafe.Pointer(&di[0])))
	if int32(r) <= 0 {
		return
	}

	cpMin := int32(0)
	for cpMin < int32(textLen) {
		startPage.Call(printerDC)

		var fr [56]byte
		*(*uintptr)(unsafe.Pointer(&fr[0])) = printerDC  // hdc
		*(*uintptr)(unsafe.Pointer(&fr[8])) = printerDC  // hdcTarget
		// rc (rendering area)
		*(*int32)(unsafe.Pointer(&fr[16])) = 0        // left
		*(*int32)(unsafe.Pointer(&fr[20])) = 0        // top
		*(*int32)(unsafe.Pointer(&fr[24])) = rcRight   // right
		*(*int32)(unsafe.Pointer(&fr[28])) = rcBottom  // bottom
		// rcPage (physical page)
		*(*int32)(unsafe.Pointer(&fr[32])) = 0
		*(*int32)(unsafe.Pointer(&fr[36])) = 0
		*(*int32)(unsafe.Pointer(&fr[40])) = rcRight
		*(*int32)(unsafe.Pointer(&fr[44])) = rcBottom
		// CHARRANGE
		*(*int32)(unsafe.Pointer(&fr[48])) = cpMin
		*(*int32)(unsafe.Pointer(&fr[52])) = int32(textLen)

		result, _, _ := sendMessageW.Call(previewHwnd, EM_FORMATRANGE, 1, uintptr(unsafe.Pointer(&fr[0])))
		endPage.Call(printerDC)

		if int32(result) <= cpMin {
			break // no progress
		}
		cpMin = int32(result)
	}

	// Clear EM_FORMATRANGE cache
	sendMessageW.Call(previewHwnd, EM_FORMATRANGE, 0, 0)

	endDoc.Call(printerDC)
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

	// Register main window class
	className := utf16From("TinyMDRichEdit")
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

	title := "TinyMD RichEdit Prototype"
	if currentFile != "" {
		title = "TinyMD RichEdit — " + currentFile
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

	showWindowProc.Call(mainHwnd, 5) // SW_SHOW
	updateWindowProc.Call(mainHwnd)

	// Set initial content
	if initialContent != "" {
		txt := utf16From(initialContent)
		sendMessageW.Call(editorHwnd, WM_SETTEXT, 0, uintptr(unsafe.Pointer(&txt[0])))
		updatePreview()
	}

	setFocus.Call(editorHwnd)

	if autoPrint {
		fmt.Println("[RichEdit] --print flag detected, auto-printing...")
		printFormatted()
		fmt.Println("[RichEdit] print done, exiting")
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
		msgID := *(*uint32)(unsafe.Pointer(&msgBuf[8]))
		wParam := *(*uintptr)(unsafe.Pointer(&msgBuf[16]))
		if msgID == WM_KEYDOWN {
			state, _, _ := getKeyState.Call(0x11)
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
}
