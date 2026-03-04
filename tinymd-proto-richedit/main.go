// TinyMD RichEdit Prototype — native markdown preview using RichEdit + RTF.
//
// Build: GOOS=windows go build -ldflags="-s -w -H windowsgui" -trimpath -o tinymd-richedit.exe .

package main

import (
	"fmt"
	"os"
	"strings"
	"sync"
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

	EM_STREAMIN      = 0x0449
	EM_SETBKGNDCOLOR = 0x0443
	SF_RTF           = 0x0002

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

// EDITSTREAM for EM_STREAMIN
// Layout on 64-bit: dwCookie(8) + dwError(4) + pad(4) + pfnCallback(8) = 24 bytes
type editStreamData struct {
	data   []byte
	offset int
}

var (
	streamMu       sync.Mutex
	activeStreamData *editStreamData
)

// The callback is called by RichEdit to read chunks of RTF data
func editStreamCallback(dwCookie, pbBuff, cb, pcb uintptr) uintptr {
	streamMu.Lock()
	sd := activeStreamData
	streamMu.Unlock()

	if sd == nil {
		return 1 // error
	}

	remaining := len(sd.data) - sd.offset
	toRead := int(cb)
	if toRead > remaining {
		toRead = remaining
	}

	if toRead > 0 {
		// Copy data to the buffer provided by RichEdit
		dst := unsafe.Slice((*byte)(unsafe.Pointer(pbBuff)), toRead)
		copy(dst, sd.data[sd.offset:sd.offset+toRead])
		sd.offset += toRead
	}

	// Write number of bytes read to pcb
	*(*uint32)(unsafe.Pointer(pcb)) = uint32(toRead)
	return 0
}

var editStreamCallbackPtr = syscall.NewCallback(editStreamCallback)

func streamRTFInto(hwnd uintptr, rtfData []byte) {
	sd := &editStreamData{data: rtfData}
	streamMu.Lock()
	activeStreamData = sd
	streamMu.Unlock()

	// Build EDITSTREAM struct (24 bytes on 64-bit)
	var es [24]byte
	*(*uintptr)(unsafe.Pointer(&es[0])) = 0                     // dwCookie (unused, we use global)
	*(*uint32)(unsafe.Pointer(&es[8])) = 0                      // dwError
	*(*uintptr)(unsafe.Pointer(&es[16])) = editStreamCallbackPtr // pfnCallback

	sendMessageW.Call(hwnd, EM_STREAMIN, SF_RTF, uintptr(unsafe.Pointer(&es[0])))

	streamMu.Lock()
	activeStreamData = nil
	streamMu.Unlock()
}

// RTF generation from goldmark AST
// Color table indices (1-based): 1=dark text, 2=blue, 3=gray, 4=code bg
const rtfHeader = `{\rtf1\ansi\deff0` +
	`{\fonttbl{\f0\fswiss Segoe UI;}{\f1\fmodern Consolas;}}` +
	`{\colortbl;\red34\green34\blue34;\red3\green102\blue214;\red102\green102\blue102;\red240\green240\blue240;}` +
	`\viewkind4\uc1\pard\cf1\f0\fs28 `

func markdownToRTF(source []byte) string {
	md := goldmark.New()
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
				var fs int
				switch h.Level {
				case 1:
					fs = 56 // 28pt
				case 2:
					fs = 44 // 22pt
				case 3:
					fs = 36 // 18pt
				default:
					fs = 32
				}
				buf.WriteString(fmt.Sprintf(`\pard\sb200\sa100\cf1\b\f0\fs%d `, fs))
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
				buf.WriteString(`\pard\sb60\sa60\cf1\f0\fs28 `)
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
				buf.WriteString(`\pard\sb60\sa60\cbpat4\f1\fs22\cf1 `)
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
				buf.WriteString(`\f0\fs28\par `)
				return ast.WalkSkipChildren, nil
			}

		case ast.KindBlockquote:
			if entering {
				buf.WriteString(`\pard\sb60\sa60\li720\cf3\f0\fs28 `)
			} else {
				buf.WriteString(`\cf1\par `)
			}
			return ast.WalkContinue, nil

		case ast.KindList:
			return ast.WalkContinue, nil

		case ast.KindListItem:
			if entering {
				buf.WriteString(`\pard\sb30\sa30\fi-360\li720\cf1\f0\fs28 \bullet\tab `)
			} else {
				buf.WriteString(`\par `)
			}
			return ast.WalkContinue, nil

		case ast.KindThematicBreak:
			if entering {
				buf.WriteString(`\pard\sb100\sa100\brdrb\brdrs\brdrw10\brsp20 \par `)
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
			uintptr(uint32(uint16(14))|0xFFFF0000), 0, 0, 0,
			400, 0, 0, 0, 0, 0, 0, 0, 0,
			uintptr(unsafe.Pointer(utf16Ptr("Consolas"))),
		)
		sendMessageW.Call(editorHwnd, WM_SETFONT, editorFont, 1)

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
		// Clear preview
		empty := []byte(`{\rtf1\ansi }`)
		streamRTFInto(previewHwnd, empty)
		return
	}
	buf := make([]uint16, length+1)
	getWindowTextW.Call(editorHwnd, uintptr(unsafe.Pointer(&buf[0])), length+1)
	mdText := syscall.UTF16ToString(buf)

	rtf := markdownToRTF([]byte(mdText))
	streamRTFInto(previewHwnd, []byte(rtf))
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
	className := utf16From("TinyMDRichEdit")
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
