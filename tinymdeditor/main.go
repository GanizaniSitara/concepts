package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/jchv/go-webview2"
)

var currentFile string

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

	w.Bind("goGetInitialContent", func() string {
		return initialContent
	})

	w.Bind("goGetFileName", func() string {
		return currentFile
	})

	w.Navigate("data:text/html," + dataURI(htmlPage()))
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

func htmlPage() string {
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
  width:50%; height:100%; display:flex; flex-direction:column;
  border-right: 2px solid #e0e0e0;
}
.editor-pane textarea {
  flex:1; width:100%; resize:none; border:none; outline:none;
  background:#fff; color:#222; padding:16px; font-size:14px;
  font-family: 'Cascadia Code','Consolas','Courier New', monospace;
  line-height:1.6; tab-size:4;
}

.preview-pane {
  width:50%; height:100%; overflow-y:auto; padding:20px 28px;
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
  <div class="preview-pane" id="preview"></div>
</div>

<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script>
(async function() {
  const editor = document.getElementById('editor');
  const preview = document.getElementById('preview');
  const savedEl = document.getElementById('saved');
  const fnameEl = document.getElementById('fname');

  // Configure marked
  marked.setOptions({ breaks: true, gfm: true });

  // Load initial content from Go
  const initial = await goGetInitialContent();
  const fname = await goGetFileName();
  if (initial) editor.value = initial;
  if (fname) fnameEl.textContent = fname.replace(/.*[\\\/]/, '');

  function render() {
    preview.innerHTML = marked.parse(editor.value);
  }

  // Live preview on every keystroke
  editor.addEventListener('input', render);

  // Tab key inserts a tab instead of moving focus
  editor.addEventListener('keydown', function(e) {
    if (e.key === 'Tab') {
      e.preventDefault();
      const start = this.selectionStart;
      const end = this.selectionEnd;
      this.value = this.value.substring(0, start) + '\t' + this.value.substring(end);
      this.selectionStart = this.selectionEnd = start + 1;
      render();
    }
  });

  // Ctrl+S to save
  document.addEventListener('keydown', async function(e) {
    if (e.ctrlKey && e.key === 's') {
      e.preventDefault();
      const result = await goSaveFile(editor.value);
      if (result === 'ok') {
        savedEl.classList.add('show');
        setTimeout(() => savedEl.classList.remove('show'), 1500);
      } else if (result === 'no file') {
        // No file specified — could add Save As dialog later
      }
    }
  });

  // Initial render
  render();
  editor.focus();
})();
</script>
</body>
</html>`
}
