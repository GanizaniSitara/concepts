// TinyMD — a fast, minimal Markdown editor for Windows.
//
// Build for release (2.6 MB, no console window):
//   GOOS=windows go build -ldflags="-s -w -H windowsgui" -trimpath -o tinymd.exe .
//
// Build for development (with debug info and console):
//   GOOS=windows go build -o tinymd.exe .

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"unsafe"

	"github.com/jchv/go-webview2"
)

var currentFile string

var (
	comdlg32         = syscall.NewLazyDLL("comdlg32.dll")
	getSaveFileNameW = comdlg32.NewProc("GetSaveFileNameW")

	user32              = syscall.NewLazyDLL("user32.dll")
	getForegroundWin    = user32.NewProc("GetForegroundWindow")
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

	gdi32          = syscall.NewLazyDLL("gdi32.dll")
	createFontW    = gdi32.NewProc("CreateFontW")
	selectObject   = gdi32.NewProc("SelectObject")
	setBkMode      = gdi32.NewProc("SetBkMode")
	setTextColor   = gdi32.NewProc("SetTextColor")
	deleteObject   = gdi32.NewProc("DeleteObject")
	getSysColorBrush = user32.NewProc("GetSysColorBrush")

	kernel32          = syscall.NewLazyDLL("kernel32.dll")
	getModuleHandleW  = kernel32.NewProc("GetModuleHandleW")
)

func getHWND() uintptr {
	hwnd, _, _ := getForegroundWin.Call()
	return hwnd
}

func showSaveDialog(hwnd uintptr) string {
	buf := make([]uint16, 260)

	// Double-null separated filter string
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

	// Use a flat byte buffer to avoid Go struct padding issues with OPENFILENAMEW.
	// OPENFILENAMEW on 64-bit is 152 bytes.
	const structSize = 152
	var ofn [structSize]byte

	// lStructSize (offset 0, uint32)
	*(*uint32)(unsafe.Pointer(&ofn[0])) = structSize
	// hwndOwner (offset 8, uintptr — after 4 bytes padding on 64-bit)
	*(*uintptr)(unsafe.Pointer(&ofn[8])) = hwnd
	// lpstrFilter (offset 24 on 64-bit: 8+8+8)
	*(*uintptr)(unsafe.Pointer(&ofn[24])) = uintptr(unsafe.Pointer(&filter[0]))
	// nFilterIndex (offset 44: 24+8+8+4)
	*(*uint32)(unsafe.Pointer(&ofn[44])) = 1
	// lpstrFile (offset 48)
	*(*uintptr)(unsafe.Pointer(&ofn[48])) = uintptr(unsafe.Pointer(&buf[0]))
	// nMaxFile (offset 56)
	*(*uint32)(unsafe.Pointer(&ofn[56])) = uint32(len(buf))
	// lpstrTitle (offset 80: after lpstrFileTitle(72)+nMaxFileTitle(76) region)
	*(*uintptr)(unsafe.Pointer(&ofn[80])) = uintptr(unsafe.Pointer(&title[0]))
	// flags (offset 88)
	*(*uint32)(unsafe.Pointer(&ofn[88])) = 0x00000002 | 0x00000800 // OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST
	// lpstrDefExt (offset 96: after nFileOffset(92)+nFileExtension(94)+padding)
	*(*uintptr)(unsafe.Pointer(&ofn[96])) = uintptr(unsafe.Pointer(&defExt[0]))

	ret, _, _ := getSaveFileNameW.Call(uintptr(unsafe.Pointer(&ofn[0])))
	if ret == 0 {
		return ""
	}
	return syscall.UTF16ToString(buf)
}

func utf16From(s string) []uint16 {
	r, _ := syscall.UTF16FromString(s)
	return r
}

// splashTitle is set before showSplash and used by the WndProc to paint text.
var splashTitle string

func splashWndProc(hwnd, msg, wParam, lParam uintptr) uintptr {
	const (
		WM_PAINT   = 0x000F
		WM_DESTROY = 0x0002
		TRANSPARENT = 1
		DT_CENTER   = 0x01
		DT_VCENTER  = 0x04
		DT_SINGLELINE = 0x20
		DT_NOPREFIX   = 0x0800
	)

	switch msg {
	case WM_PAINT:
		// PAINTSTRUCT is 72 bytes on 64-bit
		var ps [72]byte
		hdc, _, _ := beginPaint.Call(hwnd, uintptr(unsafe.Pointer(&ps[0])))
		if hdc == 0 {
			break
		}

		var rc [16]byte // RECT: 4x int32
		getClientRect.Call(hwnd, uintptr(unsafe.Pointer(&rc[0])))

		// Fill white background
		whiteBrush, _, _ := getSysColorBrush.Call(0) // COLOR_WINDOW
		fillRect.Call(hdc, uintptr(unsafe.Pointer(&rc[0])), whiteBrush)

		setBkMode.Call(hdc, TRANSPARENT)

		// Large title font
		titleFont, _, _ := createFontW.Call(
			uintptr(uint32(0xFFFFFFD8)), // -40 (40px)
			0, 0, 0,
			700, // bold
			0, 0, 0, 0, 0, 0, 0, 0,
			uintptr(unsafe.Pointer(utf16Ptr("Segoe UI"))),
		)
		oldFont, _, _ := selectObject.Call(hdc, titleFont)
		setTextColor.Call(hdc, 0x00222222) // dark gray

		// Draw "TinyMD" centered, slightly above middle
		titleRC := rc
		// Shift up by 30px: reduce bottom by 60
		bottom := *(*int32)(unsafe.Pointer(&titleRC[12]))
		*(*int32)(unsafe.Pointer(&titleRC[12])) = bottom - 60
		titleText := utf16From("TinyMD")
		drawTextW.Call(hdc, uintptr(unsafe.Pointer(&titleText[0])),
			uintptr(len(titleText)-1), // exclude null terminator
			uintptr(unsafe.Pointer(&titleRC[0])),
			DT_CENTER|DT_VCENTER|DT_SINGLELINE|DT_NOPREFIX)

		// Small subtitle font
		subFont, _, _ := createFontW.Call(
			uintptr(uint32(0xFFFFFFF2)), // -14 (14px)
			0, 0, 0,
			400, // normal
			0, 0, 0, 0, 0, 0, 0, 0,
			uintptr(unsafe.Pointer(utf16Ptr("Segoe UI"))),
		)
		selectObject.Call(hdc, subFont)
		setTextColor.Call(hdc, 0x00999999) // light gray

		// Draw subtitle centered, slightly below middle
		subRC := rc
		*(*int32)(unsafe.Pointer(&subRC[4])) = *(*int32)(unsafe.Pointer(&subRC[4])) + 30 // shift top down
		subText := utf16From(splashTitle)
		drawTextW.Call(hdc, uintptr(unsafe.Pointer(&subText[0])),
			uintptr(len(subText)-1),
			uintptr(unsafe.Pointer(&subRC[0])),
			DT_CENTER|DT_VCENTER|DT_SINGLELINE|DT_NOPREFIX)

		selectObject.Call(hdc, oldFont)
		deleteObject.Call(titleFont)
		deleteObject.Call(subFont)
		endPaint.Call(hwnd, uintptr(unsafe.Pointer(&ps[0])))
		return 0
	}

	ret, _, _ := defWindowProcW.Call(hwnd, msg, wParam, lParam)
	return ret
}

func utf16Ptr(s string) *uint16 {
	p, _ := syscall.UTF16PtrFromString(s)
	return p
}

func showSplash(title string) uintptr {
	splashTitle = title

	const (
		WS_OVERLAPPEDWINDOW = 0x00CF0000
		WS_VISIBLE          = 0x10000000
		CW_USEDEFAULT       = 0x80000000
		SM_CXSCREEN         = 0
		SM_CYSCREEN         = 1
		IDC_ARROW           = 32512
	)

	hInstance, _, _ := getModuleHandleW.Call(0)
	cursor, _, _ := loadCursorW.Call(0, IDC_ARROW)

	className := utf16From("TinyMDSplash")

	// WNDCLASSEXW struct (80 bytes on 64-bit)
	var wc [80]byte
	*(*uint32)(unsafe.Pointer(&wc[0])) = 80                                                 // cbSize
	*(*uintptr)(unsafe.Pointer(&wc[8])) = syscall.NewCallback(splashWndProc)                 // lpfnWndProc
	*(*uintptr)(unsafe.Pointer(&wc[48])) = hInstance                                         // hInstance
	*(*uintptr)(unsafe.Pointer(&wc[56])) = cursor                                            // hCursor
	*(*uintptr)(unsafe.Pointer(&wc[64])) = 6                                                 // hbrBackground = COLOR_WINDOW+1
	*(*uintptr)(unsafe.Pointer(&wc[72])) = uintptr(unsafe.Pointer(&className[0]))            // lpszClassName

	registerClassExW.Call(uintptr(unsafe.Pointer(&wc[0])))

	// Center on screen
	w, h := uintptr(1400), uintptr(900)
	screenW, _, _ := getSystemMetrics.Call(SM_CXSCREEN)
	screenH, _, _ := getSystemMetrics.Call(SM_CYSCREEN)
	x := (screenW - w) / 2
	y := (screenH - h) / 2

	windowTitle := utf16From(title)
	hwnd, _, _ := createWindowExW.Call(
		0, // dwExStyle
		uintptr(unsafe.Pointer(&className[0])),
		uintptr(unsafe.Pointer(&windowTitle[0])),
		WS_OVERLAPPEDWINDOW|WS_VISIBLE,
		x, y, w, h,
		0, 0, hInstance, 0,
	)

	showWindowProc.Call(hwnd, 5) // SW_SHOW
	updateWindowProc.Call(hwnd)

	return hwnd
}

func destroySplash(hwnd uintptr) {
	if hwnd != 0 {
		destroyWindowProc.Call(hwnd)
	}
}

func main() {
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

	// Show a native splash window immediately (~50ms) while WebView2 loads (~2-3s).
	splash := showSplash(windowTitle())

	// Use a fixed data path so WebView2 reuses its cached browser profile
	// instead of recreating it every launch (~1s saving on cold start).
	dataPath := filepath.Join(os.Getenv("LOCALAPPDATA"), "TinyMD", "webview2")

	w := webview2.NewWithOptions(webview2.WebViewOptions{
		Debug:     false,
		AutoFocus: true,
		DataPath:  dataPath,
		WindowOptions: webview2.WindowOptions{
			Title:  windowTitle(),
			Width:  1400,
			Height: 900,
			Center: true,
		},
	})
	if w == nil {
		destroySplash(splash)
		fmt.Fprintln(os.Stderr, "Failed to create webview2 — is Edge WebView2 Runtime installed?")
		os.Exit(1)
	}
	defer w.Destroy()

	// WebView2 is ready — destroy splash so the real window takes over.
	destroySplash(splash)

	// Bind Go functions for JS to call
	w.Bind("goSaveFile", func(content string) string {
		if currentFile == "" {
			return "no file"
		}
		err := os.WriteFile(currentFile, []byte(content), 0644)
		if err != nil {
			return "error: " + err.Error()
		}
		return "ok"
	})

	w.Bind("goShowSaveDialog", func() string {
		path := showSaveDialog(getHWND())
		if path == "" {
			return ""
		}
		currentFile = path
		w.SetTitle(windowTitle())
		return filepath.Base(path)
	})

	w.SetHtml(htmlPage(initialContent, currentFile))

	if autoPrint {
		// Trigger print after a short delay to let marked.js load
		w.Dispatch(func() {
			w.Eval("setTimeout(function(){ window.print(); }, 500);")
		})
	}

	w.Run()
}

func windowTitle() string {
	if currentFile != "" {
		return "TinyMD — " + currentFile
	}
	return "TinyMD Editor"
}

func jsEscape(s string) string {
	r := strings.NewReplacer(
		`\`, `\\`,
		`"`, `\"`,
		"\n", `\n`,
		"\r", `\r`,
		"\t", `\t`,
		"</", `<\/`,
	)
	return r.Replace(s)
}

func htmlPage(initialContent, fileName string) string {
	return `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
* { margin:0; padding:0; box-sizing:border-box; }
html, body { height:100%; overflow:hidden; font-family: -apple-system, 'Segoe UI', sans-serif; background:#fff; color:#222; }

.toolbar {
  height: 36px; background:#f5f5f5; display:flex; align-items:center;
  padding: 0 12px; border-bottom: 1px solid #ddd; font-size:12px; gap:12px;
}
.toolbar .filename { color:#0366d6; font-weight:600; }
.toolbar .saved { color:#28a745; opacity:0; transition: opacity 0.3s; }
.toolbar .saved.show { opacity:1; }
.toolbar .hint { color:#999; margin-left:auto; }

.container { display:flex; height:calc(100% - 36px); }

.editor-pane {
  flex:1 1 50%; height:100%; display:flex; flex-direction:column;
  overflow:hidden;
}
.editor-pane textarea {
  flex:1; min-width:50vw; resize:none; border:none; outline:none;
  background:#fff; color:#222; padding:16px; font-size:14px;
  font-family: 'Cascadia Code','Consolas','Courier New', monospace;
  line-height:1.6; tab-size:4;
}

.divider {
  width:6px; flex-shrink:0; cursor:col-resize;
  background:#e0e0e0; transition:background 0.15s;
}
.divider:hover, .divider.active { background:#0366d6; }

.preview-pane {
  flex:1 1 50%; height:100%; overflow-y:auto; padding:20px 28px;
  background:#fff;
}

/* Markdown rendered styles */
.preview-pane h1 { font-size:2em; margin:0.5em 0 0.3em; color:#111; border-bottom:1px solid #ddd; padding-bottom:0.2em; }
.preview-pane h2 { font-size:1.5em; margin:0.5em 0 0.3em; color:#111; border-bottom:1px solid #eee; padding-bottom:0.2em; }
.preview-pane h3 { font-size:1.25em; margin:0.5em 0 0.3em; color:#111; }
.preview-pane h4,h5,h6 { margin:0.4em 0; color:#111; }
.preview-pane p { margin:0.5em 0; line-height:1.7; }
.preview-pane a { color:#0366d6; }
.preview-pane code {
  background:#f0f0f0; padding:2px 6px; border-radius:3px; font-size:0.9em;
  font-family: 'Cascadia Code','Consolas', monospace;
}
.preview-pane pre {
  background:#f6f8fa; padding:12px 16px; border-radius:6px; overflow-x:auto;
  margin:0.6em 0; border:1px solid #e1e4e8;
}
.preview-pane pre code { background:none; padding:0; }
.preview-pane blockquote {
  border-left:4px solid #0366d6; padding:4px 16px; margin:0.6em 0;
  color:#666; background:#f9f9f9;
}
.preview-pane ul, .preview-pane ol { padding-left:1.8em; margin:0.4em 0; }
.preview-pane li { margin:0.2em 0; line-height:1.6; }
.preview-pane table { border-collapse:collapse; margin:0.6em 0; }
.preview-pane th, .preview-pane td { border:1px solid #ddd; padding:6px 12px; }
.preview-pane th { background:#f6f8fa; }
.preview-pane img { max-width:100%; }
.preview-pane hr { border:none; border-top:1px solid #ddd; margin:1em 0; }

/* Scrollbar */
::-webkit-scrollbar { width:10px; }
::-webkit-scrollbar-track { background:#fff; }
::-webkit-scrollbar-thumb { background:#ccc; border-radius:5px; }
::-webkit-scrollbar-thumb:hover { background:#aaa; }

/* Print: show only the formatted preview */
@media print {
  .toolbar, .editor-pane, .divider { display:none !important; }
  .container { display:block !important; height:auto !important; }
  .preview-pane {
    flex:none !important; width:100% !important; height:auto !important;
    overflow:visible !important; padding:0 !important;
  }
}
</style>
</head>
<body>

<div class="toolbar">
  <span class="filename" id="fname">TinyMD</span>
  <span class="saved" id="saved">Saved!</span>
  <span class="hint">Ctrl+S save · Ctrl+P print</span>
</div>

<div class="container">
  <div class="editor-pane">
    <textarea id="editor" spellcheck="false" placeholder="Type your Markdown here..."></textarea>
  </div>
  <div class="divider" id="divider"></div>
  <div class="preview-pane" id="preview"></div>
</div>

<script>
var _initContent = "` + jsEscape(initialContent) + `";
var _initFname = "` + jsEscape(fileName) + `";

// Lightweight inline markdown renderer for instant startup (no CDN wait)
function quickMd(s) {
  var h = s
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/^### (.+)$/gm,'<h3>$1</h3>')
    .replace(/^## (.+)$/gm,'<h2>$1</h2>')
    .replace(/^# (.+)$/gm,'<h1>$1</h1>')
    .replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>')
    .replace(/\*(.+?)\*/g,'<em>$1</em>')
    .replace(/\x60([^\x60]+)\x60/g,'<code>$1</code>')
    .replace(/^[-*] (.+)$/gm,'<li>$1</li>')
    .replace(/^---$/gm,'<hr>')
    .replace(/\n\n/g,'</p><p>')
    .replace(/\n/g,'<br>');
  return '<p>' + h + '</p>';
}

var useMarked = false;

function render() {
  var text = document.getElementById('editor').value;
  document.getElementById('preview').innerHTML = useMarked ? marked.parse(text) : quickMd(text);
}

// Load marked.js async — UI is usable immediately
var sc = document.createElement('script');
sc.src = 'https://cdn.jsdelivr.net/npm/marked/marked.min.js';
sc.async = true;
sc.onload = function() { marked.setOptions({breaks:true,gfm:true}); useMarked=true; render(); };
document.head.appendChild(sc);

// Draggable divider between editor and preview panes
(function() {
  var divider = document.getElementById('divider');
  var container = document.querySelector('.container');
  var editorPane = document.querySelector('.editor-pane');
  var previewPane = document.querySelector('.preview-pane');
  var dragging = false;

  divider.addEventListener('mousedown', function(e) {
    e.preventDefault();
    dragging = true;
    divider.classList.add('active');
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  });

  document.addEventListener('mousemove', function(e) {
    if (!dragging) return;
    var rect = container.getBoundingClientRect();
    var offset = e.clientX - rect.left;
    var pct = (offset / rect.width) * 100;
    pct = Math.max(15, Math.min(85, pct));
    editorPane.style.flexBasis = pct + '%';
    previewPane.style.flexBasis = (100 - pct) + '%';
  });

  document.addEventListener('mouseup', function() {
    if (!dragging) return;
    dragging = false;
    divider.classList.remove('active');
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
  });
})();

// Synchronous init — no async bridge calls, everything is instant
(function() {
  var editor = document.getElementById('editor');
  var savedEl = document.getElementById('saved');
  var fnameEl = document.getElementById('fname');

  if (_initContent) editor.value = _initContent;
  if (_initFname) fnameEl.textContent = _initFname.replace(/.*[\\\/]/, '');

  editor.addEventListener('input', render);

  editor.addEventListener('keydown', function(e) {
    if (e.key === 'Tab') {
      e.preventDefault();
      var start = this.selectionStart;
      var end = this.selectionEnd;
      this.value = this.value.substring(0, start) + '\t' + this.value.substring(end);
      this.selectionStart = this.selectionEnd = start + 1;
      render();
    }
  });

  // Ctrl+S to save — shows native Save As dialog if no file was specified
  document.addEventListener('keydown', async function(e) {
    if (e.ctrlKey && e.key === 'p') {
      e.preventDefault();
      window.print();
      return;
    }
    if (e.ctrlKey && e.key === 's') {
      e.preventDefault();
      var result = await goSaveFile(editor.value);
      if (result === 'ok') {
        savedEl.classList.add('show');
        setTimeout(function() { savedEl.classList.remove('show'); }, 1500);
      } else if (result === 'no file') {
        var name = await goShowSaveDialog();
        if (name) {
          fnameEl.textContent = name;
          var r2 = await goSaveFile(editor.value);
          if (r2 === 'ok') {
            savedEl.classList.add('show');
            setTimeout(function() { savedEl.classList.remove('show'); }, 1500);
          }
        }
      }
    }
  });

  render();
  editor.focus();
})();
</script>
</body>
</html>`
}
