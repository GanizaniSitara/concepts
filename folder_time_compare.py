import os
import re
import hashlib
import platform
import subprocess
import webbrowser
from datetime import datetime
from pathlib import Path
import pandas as pd
import sys
import difflib
from urllib.parse import unquote, parse_qs
import html as html_module

def open_html(abs_path):
    """Open HTML file using appropriate method for the platform."""
    if "microsoft" in platform.uname().release.lower():  # crude WSL check
        # Translate to Windows-style path if needed
        win_path = abs_path.replace('/mnt/c/', 'C:\\').replace('/', '\\')
        subprocess.run(["/mnt/c/Windows/explorer.exe", win_path])
    else:
        webbrowser.open_new_tab(f'file://{abs_path}')

def show_delta():
    """Show delta between two files when script is called with file parameters."""
    # Parse query parameters
    if len(sys.argv) > 1:
        query_string = sys.argv[1]
        if query_string.startswith('?'):
            query_string = query_string[1:]
        
        params = parse_qs(query_string)
        file1 = params.get('file1', [''])[0]
        file2 = params.get('file2', [''])[0]
        
        if file1 and file2:
            # Decode URL-encoded paths
            file1 = unquote(file1)
            file2 = unquote(file2)
            
            # Read files
            try:
                with open(file1, 'r', encoding='utf-8', errors='replace') as f:
                    lines1 = f.readlines()
            except Exception as e:
                lines1 = [f"Error reading file: {e}\n"]
                
            try:
                with open(file2, 'r', encoding='utf-8', errors='replace') as f:
                    lines2 = f.readlines()
            except Exception as e:
                lines2 = [f"Error reading file: {e}\n"]
            
            # Generate diff
            diff = difflib.unified_diff(
                lines1, lines2,
                fromfile=f"Previous: {file1}",
                tofile=f"Current: {file2}",
                n=3
            )
            
            # Create HTML output
            html = f"""<!DOCTYPE html>
<html>
<head>
    <title>File Delta - {os.path.basename(file1)} vs {os.path.basename(file2)}</title>
    <style>
        body {{ font-family: monospace; margin: 20px; background-color: #f5f5f5; }}
        h1 {{ font-family: sans-serif; font-size: 1.5em; }}
        .file-info {{ font-family: sans-serif; background-color: #e0e0e0; padding: 10px; margin: 10px 0; border-radius: 5px; }}
        .diff-container {{ background-color: white; border: 1px solid #ddd; padding: 10px; overflow-x: auto; }}
        .line {{ white-space: pre; font-family: 'Consolas', 'Monaco', monospace; line-height: 1.4; }}
        .added {{ background-color: #c6efce; }}
        .removed {{ background-color: #ffcccc; }}
        .context {{ color: #666; }}
    </style>
</head>
<body>
    <h1>File Delta Viewer</h1>
    <div class="file-info">
        <strong>Previous:</strong> {file1}<br>
        <strong>Current:</strong> {file2}
    </div>
    <div class="diff-container">
"""
            
            for line in diff:
                if line.startswith('+') and not line.startswith('+++'):
                    html += f'<div class="line added">{html_module.escape(line.rstrip())}</div>\n'
                elif line.startswith('-') and not line.startswith('---'):
                    html += f'<div class="line removed">{html_module.escape(line.rstrip())}</div>\n'
                elif line.startswith('@@'):
                    html += f'<div class="line context">{html_module.escape(line.rstrip())}</div>\n'
                elif not line.startswith('+++') and not line.startswith('---'):
                    html += f'<div class="line">{html_module.escape(line.rstrip())}</div>\n'
            
            html += """
    </div>
</body>
</html>"""
            
            # Save and open
            output_dir = Path('comparison_output')
            output_dir.mkdir(exist_ok=True)
            delta_path = output_dir / 'delta_view.html'
            with open(delta_path, 'w', encoding='utf-8') as f:
                f.write(html)
            
            open_html(str(delta_path))
            sys.exit(0)

# Check if running in delta mode
if len(sys.argv) > 1 and ('file1=' in sys.argv[1] or 'file2=' in sys.argv[1]):
    show_delta()

# Check for debug mode
DEBUG_MODE = '--debug' in sys.argv
if DEBUG_MODE:
    print("DEBUG MODE: Processing only first file and exiting...")

# === CONFIGURATION ===
base_dir = r"..\evidence"  # ← Change this to your folder containing the control run subfolders
pattern = re.compile(r'.*CTRL-(\d+)_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2})')

# === DISCOVER RUNS AND PICK ONE PER CONTROL PER HOUR ===
runs = {}
for d in os.listdir(base_dir):
    p = os.path.join(base_dir, d)
    if not os.path.isdir(p):
        continue
    m = pattern.match(d)
    if not m:
        continue
    ctrl = f'CTRL-{m.group(1)}'
    dt = datetime.strptime(
        f'{m.group(2)} {m.group(3).replace("-", ":")}',
        '%Y-%m-%d %H:%M'
    )
    # Create 2-hour time slots (00:00, 02:00, 04:00, 06:00, 08:00, 10:00, etc.)
    hour_2_slot = dt.replace(minute=0, second=0, microsecond=0)
    hour_2_slot = hour_2_slot.replace(hour=(dt.hour // 2) * 2)
    runs.setdefault((ctrl, hour_2_slot), []).append((dt, p))

# keep the latest run within each 2-hour slot for each control
representatives = {
    (ctrl, hour_2_slot): max(entries, key=lambda x: x[0])[1]
    for (ctrl, hour_2_slot), entries in runs.items()
}

# In debug mode, only keep the first time slot
if DEBUG_MODE:
    hour_2_slots = sorted({hour_2_slot for (_, hour_2_slot) in representatives.keys()})
    first_slot = hour_2_slots[0]
    representatives = {
        (ctrl, hour_2_slot): path
        for (ctrl, hour_2_slot), path in representatives.items()
        if hour_2_slot == first_slot
    }
    print(f"DEBUG: Processing only first time slot: {first_slot.strftime('%Y-%m-%d %H:%M')}")

# === PREPARE ROWS, COLUMNS, AND FILE LIST ===
hour_2_slots = sorted({hour_2_slot for (_, hour_2_slot) in representatives.keys()})
# map each 2-hour slot to a header label with date on top line, time range below
hour_2_labels = {
    hour_2_slot: f"{hour_2_slot.strftime('%Y-%m-%d')}\n{hour_2_slot.strftime('%H:%M')}-{(hour_2_slot.hour + 2) % 24:02d}:00"
    for hour_2_slot in hour_2_slots
}
columns = [hour_2_labels[h] for h in hour_2_slots]

controls = sorted({ctrl for (ctrl, _) in representatives.keys()})

# Build control-specific file lists
control_files = {}
for ctrl in controls:
    # Get all paths for this control across all time slots
    ctrl_paths = [path for (c, _), path in representatives.items() if c == ctrl]
    # Get unique files across all time slots for this control
    ctrl_files = set()
    for path in ctrl_paths:
        for fname in os.listdir(path):
            if (os.path.splitext(fname)[1].lower() in ('.csv', '.json', '.txt') and 
                fname != 'test_summary.csv'):
                ctrl_files.add(fname)
    control_files[ctrl] = sorted(ctrl_files)

# build DataFrame with proper control-file mapping
index_tuples = []
for ctrl in controls:
    for fname in control_files[ctrl]:
        index_tuples.append((ctrl, fname))

index = pd.MultiIndex.from_tuples(index_tuples, names=['CONTROL', 'STEP'])
df = pd.DataFrame(index=index, columns=columns, dtype=object)

# === FILL IN SHA1 HASHES FOR EACH CELL ===
# Open log file for hash debugging
with open('hash_debug.log', 'w') as hash_log:
    hash_log.write("Control,TimeSlot,FileName,FilePath,FileSize,SHA1Hash\n")
    
    for (ctrl, hour_2_slot), path in representatives.items():
        label = hour_2_labels[hour_2_slot]
        # Only process files that exist for this control
        for fname in control_files[ctrl]:
            fp = os.path.join(path, fname)
            if os.path.exists(fp):
                with open(fp, 'rb') as fh:
                    content = fh.read()
                    file_hash = hashlib.sha1(content).hexdigest()
                    file_size = len(content)
                    df.at[(ctrl, fname), label] = file_hash
                    hash_log.write(f"{ctrl},{hour_2_slot.strftime('%Y-%m-%d %H:%M')},{fname},{fp},{file_size},{file_hash}\n")
            else:
                df.at[(ctrl, fname), label] = None
                hash_log.write(f"{ctrl},{hour_2_slot.strftime('%Y-%m-%d %H:%M')},{fname},{fp},0,FILE_NOT_FOUND\n")

# === COMPUTE STATUS (added, missing, unchanged, changed) ===
status = pd.DataFrame(index=df.index, columns=df.columns, dtype=object)
cols = df.columns.tolist()

# Create debug log for status computation
with open('status_debug.log', 'w') as status_log:
    status_log.write("Control,FileName,TimeSlot,CurrentHash,PreviousHash,Status\n")
    
    for i, col in enumerate(cols):
        prev = cols[i-1] if i > 0 else None
        for idx in df.index:
            cur = df.at[idx, col]
            if i == 0:
                status.at[idx, col] = 'present' if cur else 'missing'
                status_log.write(f"{idx[0]},{idx[1]},{col},{cur or 'None'},None,{status.at[idx, col]}\n")
            else:
                prv = df.at[idx, prev]
                if not cur:
                    status.at[idx, col] = 'missing'
                elif not prv:
                    status.at[idx, col] = 'added'
                elif cur == prv:
                    status.at[idx, col] = 'unchanged'
                else:
                    status.at[idx, col] = 'changed'
                status_log.write(f"{idx[0]},{idx[1]},{col},{cur or 'None'},{prv or 'None'},{status.at[idx, col]}\n")

# === STYLE MAP FOR COLORS ===
def color_map(val):
    return {
        'missing':    'background-color:#f2f2f2',
        'present':    'background-color:#dddddd',
        'added':      'background-color:#c6efce',
        'unchanged':  '',
        'changed':    'background-color:#ffeb9c'
    }.get(val, '')

# === APPLY STYLING ===
styles = [
    # column headers without rotation, positioned one cell down
    {
        'selector': 'th.col_heading',
        'props': [
            ('white-space', 'pre'),          # respect the "\n" in header
            ('width', '80px'),
            ('padding', '5px 2px'),
            ('vertical-align', 'top'),
            ('height', '60px'),
            ('font-family', 'sans-serif'),
            ('box-sizing', 'border-box'),
            ('text-align', 'center'),
            ('font-size', '0.75em')
        ]
    },
    # right-align the row labels (control/step) and remove bold, smaller font
    {
        'selector': 'th.row_heading',
        'props': [
            ('text-align', 'right'),
            ('padding', '3px 10px 3px 5px'),
            ('font-weight', 'normal'),
            ('font-family', 'sans-serif'),
            ('font-size', '0.8em'),
            ('box-sizing', 'border-box')
        ]
    },
    # left-align STEP column specifically
    {
        'selector': 'th.row_heading.level1',
        'props': [
            ('text-align', 'left'),
            ('padding', '3px 5px 3px 10px')
        ]
    },
    # add faint table outline with sans-serif font
    {
        'selector': 'table',
        'props': [
            ('border', '1px solid #e0e0e0'),
            ('border-collapse', 'collapse'),
            ('font-family', 'sans-serif')
        ]
    },
    {
        'selector': 'th, td',
        'props': [
            ('border', '1px solid #f0f0f0'),
            ('font-family', 'sans-serif')
        ]
    },
    # smaller font for data cells with left alignment
    {
        'selector': 'td',
        'props': [
            ('font-size', '0.8em'),
            ('text-align', 'left'),
            ('padding', '3px 5px'),
            ('box-sizing', 'border-box')
        ]
    }
]

styled = (
    status.style
          .applymap(color_map)            # color the data cells
          .set_table_styles(styles, overwrite=False)
)

def create_comparison_html(file1, file2, filename, status1, status2):
    """Create a side-by-side comparison HTML."""
    if file1 is None:
        # First occurrence - no previous file
        content1 = ""
        lines1 = []
        file1_exists = False
        hash1 = "NONE"
        file1 = "(none)"
    else:
        try:
            with open(file1, 'rb') as f:
                binary1 = f.read()
            with open(file1, 'r', encoding='utf-8', errors='replace') as f:
                content1 = f.read()
                lines1 = content1.splitlines(keepends=True)
            file1_exists = True
            hash1 = hashlib.sha1(binary1).hexdigest()
        except Exception as e:
            content1 = f"Error reading file: {e}"
            lines1 = [content1]
            file1_exists = False
            hash1 = "ERROR"
        
    try:
        with open(file2, 'rb') as f:
            binary2 = f.read()
        with open(file2, 'r', encoding='utf-8', errors='replace') as f:
            content2 = f.read()
            lines2 = content2.splitlines(keepends=True)
        file2_exists = True
        hash2 = hashlib.sha1(binary2).hexdigest()
    except Exception as e:
        content2 = f"Error reading file: {e}"
        lines2 = [content2]
        file2_exists = False
        hash2 = "ERROR"
    
    # Check if files are identical using the same logic as status computation
    files_identical_by_hash = hash1 == hash2
    files_identical_by_content = content1 == content2
    
    # The status should be based on hash comparison, not content comparison
    actual_status = 'unchanged' if files_identical_by_hash else 'changed'
    
    # Generate unified diff
    diff = list(difflib.unified_diff(lines1, lines2, n=3))
    
    # Helper function to make whitespace visible
    def show_whitespace(text):
        return (text
            .replace(' ', '·')  # Middle dot for spaces
            .replace('\t', '→   ')  # Arrow for tabs
            .replace('\r\n', '↵\n')  # Show Windows line endings
            .replace('\r', '↵')  # Show old Mac line endings
            .replace('\n', '↵\n'))  # Show line endings
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Delta: {filename}</title>
    <style>
        body {{ font-family: sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        h1 {{ font-size: 1.5em; margin-bottom: 10px; }}
        .status-info {{ 
            background-color: {'#c6efce' if files_identical_by_hash else '#ffeb9c'}; 
            padding: 10px; 
            margin-bottom: 10px; 
            border-radius: 5px; 
            font-weight: bold;
        }}
        .file-info {{ background-color: #e0e0e0; padding: 10px; margin-bottom: 20px; border-radius: 5px; font-size: 0.9em; }}
        .comparison {{ display: flex; gap: 20px; margin-bottom: 20px; }}
        .pane {{ flex: 1; background: white; border: 1px solid #ddd; border-radius: 5px; overflow: hidden; }}
        .pane-header {{ background: #333; color: white; padding: 10px; font-weight: bold; }}
        .pane-content {{ padding: 10px; overflow-x: auto; }}
        pre {{ margin: 0; font-family: 'Consolas', 'Monaco', monospace; font-size: 0.9em; line-height: 1.4; }}
        .whitespace {{ background-color: #f0f0f0; color: #999; }}
        .added {{ background-color: #c6efce; }}
        .removed {{ background-color: #ffcccc; }}
        .diff-view {{ background: white; border: 1px solid #ddd; border-radius: 5px; padding: 10px; }}
        .diff-line {{ font-family: 'Consolas', 'Monaco', monospace; white-space: pre; }}
        .show-whitespace {{ margin: 10px 0; }}
        .show-whitespace label {{ cursor: pointer; }}
    </style>
    <script>
        function toggleWhitespace() {{
            const checkbox = document.getElementById('showWhitespace');
            const elements = document.querySelectorAll('.content');
            elements.forEach(el => {{
                if (checkbox.checked) {{
                    el.classList.add('whitespace');
                }} else {{
                    el.classList.remove('whitespace');
                }}
            }});
        }}
    </script>
</head>
<body>
    <h1>File Comparison: {filename}</h1>
    
    <div class="status-info">
        Status: {status1} → {status2} ({actual_status} by hash)
    </div>
    
    <div class="file-info">
        <strong>Previous:</strong> {file1}<br>
        <strong>Current:</strong> {file2}<br>
        <strong>File sizes:</strong> {len(content1) if file1_exists else 'N/A'} → {len(content2) if file2_exists else 'N/A'} bytes<br>
        <strong>SHA1 hashes:</strong> {hash1[:16]}... → {hash2[:16]}...<br>
        <strong>Hash match:</strong> {files_identical_by_hash}<br>
        <strong>Content match:</strong> {files_identical_by_content}
    </div>
    
    <div class="show-whitespace">
        <label><input type="checkbox" id="showWhitespace" onchange="toggleWhitespace()"> Show whitespace characters</label>
    </div>
    
    <div class="comparison">
        <div class="pane">
            <div class="pane-header">Previous Version</div>
            <div class="pane-content">
                <pre class="content" data-normal="{html_module.escape(content1)}" data-whitespace="{html_module.escape(show_whitespace(content1))}">{html_module.escape(content1)}</pre>
            </div>
        </div>
        <div class="pane">
            <div class="pane-header">Current Version</div>
            <div class="pane-content">
                <pre class="content" data-normal="{html_module.escape(content2)}" data-whitespace="{html_module.escape(show_whitespace(content2))}">{html_module.escape(content2)}</pre>
            </div>
        </div>
    </div>
    
    <div class="diff-view">
        <h2 style="font-size: 1.2em;">Unified Diff {f"(No differences found)" if not diff else ""}</h2>
        <div>
"""
    
    if not diff:
        if files_identical_by_hash:
            html += '<div class="diff-line">Files are identical (same SHA1 hash)</div>'
        else:
            html += '<div class="diff-line">No text differences found, but SHA1 hashes differ - possible encoding issue?</div>'
    else:
        for line in diff:
            if line.startswith('+') and not line.startswith('+++'):
                html += f'<div class="diff-line added">{html_module.escape(line.rstrip())}</div>'
            elif line.startswith('-') and not line.startswith('---'):
                html += f'<div class="diff-line removed">{html_module.escape(line.rstrip())}</div>'
            else:
                html += f'<div class="diff-line">{html_module.escape(line.rstrip())}</div>'
    
    html += """
        </div>
    </div>
    
    <script>
        // Update content based on checkbox state
        document.querySelectorAll('.content').forEach(el => {
            el.textContent = el.dataset.normal;
        });
        
        function toggleWhitespace() {
            const checkbox = document.getElementById('showWhitespace');
            document.querySelectorAll('.content').forEach(el => {
                el.textContent = checkbox.checked ? el.dataset.whitespace : el.dataset.normal;
                if (checkbox.checked) {
                    el.classList.add('whitespace');
                } else {
                    el.classList.remove('whitespace');
                }
            });
        }
    </script>
</body>
</html>"""
    
    return html

# === OUTPUT ===
# Create output directory for delta HTML files
delta_output_dir = Path('comparison_output')
delta_output_dir.mkdir(exist_ok=True)

# Main comparison stays in current directory
html_path = Path('control_comparison.html').resolve()

# First, we need to track file paths for each cell
file_paths = {}
for (ctrl, hour_2_slot), path in representatives.items():
    label = hour_2_labels[hour_2_slot]
    for fname in control_files[ctrl]:
        fp = os.path.join(path, fname)
        if os.path.exists(fp):
            file_paths[(ctrl, fname, label)] = fp

# Create a modified status DataFrame with hyperlinks for "changed" cells
status_with_links = status.copy()

# Generate comparison HTML files and update status cells
comparison_count = 0
for row_idx in range(len(status.index)):
    ctrl, fname = status.index[row_idx]
    for col_idx in range(len(status.columns)):
        curr_status = status.iloc[row_idx, col_idx]
        
        # Skip missing files
        if curr_status == 'missing':
            continue
            
        # Find the current file path
        col_label = status.columns[col_idx]
        curr_path = file_paths.get((ctrl, fname, col_label))
        
        if not curr_path:
            continue
            
        # Find previous file path and status
        prev_path = None
        prev_status = 'missing'
        for i in range(col_idx - 1, -1, -1):
            prev_label = status.columns[i]
            if (ctrl, fname, prev_label) in file_paths:
                prev_path = file_paths[(ctrl, fname, prev_label)]
                prev_status = status.iloc[row_idx, i]
                break
        
        # Only create links for files that have differences or are new
        # Skip "unchanged" and "missing" files - they don't need comparison links
        if curr_status in ['unchanged', 'missing']:
            continue
            
        # Create links for: present (first occurrence), added, changed
        if curr_status in ['present', 'added', 'changed']:
            # Create a unique filename for this comparison
            comparison_id = f"delta_{comparison_count:04d}.html"
            comparison_count += 1
            
            # For first occurrence, compare against empty file
            if col_idx == 0:
                prev_path = None  # Will be handled in comparison function
                prev_status = "none"
            
            # Generate the comparison HTML
            comparison_html = create_comparison_html(prev_path, curr_path, fname, prev_status, curr_status)
            
            # Save the comparison HTML in the delta output directory
            delta_path = delta_output_dir / comparison_id
            with open(delta_path, 'w', encoding='utf-8') as f:
                f.write(comparison_html)
            
            # Update the status cell with a hyperlink (relative to main HTML file)
            status_with_links.iloc[row_idx, col_idx] = f'<a href="comparison_output/{comparison_id}" target="_blank">{curr_status}</a>'

# Apply styling but render HTML content (not escape it)
styled = (
    status_with_links.style
          .applymap(color_map)
          .set_table_styles(styles, overwrite=False)
)

# Generate HTML and tell pandas not to escape our HTML links
html_content = styled.to_html(notebook=False, escape=False)

# Write the HTML
with open(html_path, 'w') as f:
    f.write(html_content)

styled.to_excel('control_comparison.xlsx', merge_cells=False)

print(f"Rendered HTML → {html_path}")
print("Rendered Excel → control_comparison.xlsx")

# In debug mode, exit after processing
if DEBUG_MODE:
    print("DEBUG MODE: Processing complete. Check debug logs:")
    print("  - hash_debug.log: File hash computation")
    print("  - status_debug.log: Status comparison logic")
    print(f"  - control_comparison.html: Main comparison table")
    print("  - comparison_output/delta_*.html: Individual file comparisons")
    sys.exit(0)

# Open the HTML report
open_html(str(html_path))
