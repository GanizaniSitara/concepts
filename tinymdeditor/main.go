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
	user32           = syscall.NewLazyDLL("user32.dll")
	getForegroundWin = user32.NewProc("GetForegroundWindow")
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

func main() {
	// If a file was passed as argument, load it
	var initialContent string
	if len(os.Args) > 1 {
		currentFile = os.Args[1]
		data, err := os.ReadFile(currentFile)
		if err == nil {
			initialContent = string(data)
		}
	}

	w := webview2.NewWithOptions(webview2.WebViewOptions{
		Debug:     false,
		AutoFocus: true,
		WindowOptions: webview2.WindowOptions{
			Title:  windowTitle(),
			Width:  1400,
			Height: 900,
			Center: true,
		},
	})
	if w == nil {
		fmt.Fprintln(os.Stderr, "Failed to create webview2 — is Edge WebView2 Runtime installed?")
		os.Exit(1)
	}
	defer w.Destroy()

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

	w.Navigate("data:text/html," + dataURI(htmlPage(initialContent, currentFile)))
	w.Run()
}

func windowTitle() string {
	if currentFile != "" {
		return "TinyMD — " + currentFile
	}
	return "TinyMD Editor"
}

func dataURI(html string) string {
	// Minimal percent-encoding for data URI
	r := strings.NewReplacer(
		"%", "%25",
		" ", "%20",
		"#", "%23",
		"<", "%3C",
		">", "%3E",
		"\"", "%22",
		"{", "%7B",
		"}", "%7D",
		"|", "%7C",
		"^", "%5E",
		"`", "%60",
		"\n", "%0A",
		"\r", "%0D",
		"\t", "%09",
	)
	return r.Replace(html)
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
</style>
</head>
<body>

<div class="toolbar">
  <span class="filename" id="fname">TinyMD</span>
  <span class="saved" id="saved">Saved!</span>
  <span class="hint">Ctrl+S save</span>
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
