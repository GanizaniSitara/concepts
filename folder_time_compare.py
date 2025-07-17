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
            delta_path = Path('delta_view.html').resolve()
            with open(delta_path, 'w', encoding='utf-8') as f:
                f.write(html)
            
            open_html(str(delta_path))
            sys.exit(0)

# Check if running in delta mode
if len(sys.argv) > 1 and ('file1=' in sys.argv[1] or 'file2=' in sys.argv[1]):
    show_delta()

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
    hash_log.write("Control,TimeSlot,FileName,FilePath,SHA1Hash\n")
    
    for (ctrl, hour_2_slot), path in representatives.items():
        label = hour_2_labels[hour_2_slot]
        # Only process files that exist for this control
        for fname in control_files[ctrl]:
            fp = os.path.join(path, fname)
            if os.path.exists(fp):
                with open(fp, 'rb') as fh:
                    file_hash = hashlib.sha1(fh.read()).hexdigest()
                    df.at[(ctrl, fname), label] = file_hash
                    hash_log.write(f"{ctrl},{hour_2_slot.strftime('%Y-%m-%d %H:%M')},{fname},{fp},{file_hash}\n")
            else:
                df.at[(ctrl, fname), label] = None
                hash_log.write(f"{ctrl},{hour_2_slot.strftime('%Y-%m-%d %H:%M')},{fname},{fp},FILE_NOT_FOUND\n")

# === COMPUTE STATUS (added, missing, unchanged, changed) ===
status = pd.DataFrame(index=df.index, columns=df.columns, dtype=object)
cols = df.columns.tolist()

for i, col in enumerate(cols):
    prev = cols[i-1] if i > 0 else None
    for idx in df.index:
        cur = df.at[idx, col]
        if i == 0:
            status.at[idx, col] = 'present' if cur else 'missing'
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

# === OUTPUT ===
html_path = Path('control_comparison.html').resolve()

# First, we need to track file paths for each cell
file_paths = {}
for (ctrl, hour_2_slot), path in representatives.items():
    label = hour_2_labels[hour_2_slot]
    for fname in control_files[ctrl]:
        fp = os.path.join(path, fname)
        if os.path.exists(fp):
            file_paths[(ctrl, fname, label)] = fp

# Generate the HTML
html_content = styled.to_html(html_path, notebook=False)

# Post-process HTML to add links to "changed" cells
import html as html_module
lines = html_content.split('\n')
new_lines = []

# Track row/col for mapping to dataframe indices
data_row = -1
for line in lines:
    if '<tr>' in line and '<th' in line and 'level0' in line:
        # This is a data row
        data_row += 1
        data_col = -1
    
    if '<td' in line and 'background-color:#ffeb9c' in line and '>changed<' in line:
        # This is a "changed" cell - find which column it's in
        data_col = line.count('<td', 0, line.find('>changed<'))
        
        # Get the control and filename from the current row
        if data_row < len(status.index):
            ctrl, fname = status.index[data_row]
            col_label = status.columns[data_col - 1] if data_col > 0 else None
            
            # Find previous file path
            prev_path = None
            curr_path = None
            
            # Get current file path
            if (ctrl, fname, col_label) in file_paths:
                curr_path = file_paths[(ctrl, fname, col_label)]
                
                # Find previous file path
                for i in range(data_col - 1, -1, -1):
                    prev_label = status.columns[i]
                    if (ctrl, fname, prev_label) in file_paths:
                        prev_path = file_paths[(ctrl, fname, prev_label)]
                        break
            
            if curr_path and prev_path:
                # Create onclick handler
                onclick = f"navigator.clipboard.writeText('python3 folder_time_compare.py \\\"?file1={html_module.escape(prev_path)}&file2={html_module.escape(curr_path)}\\\"').then(() => alert('Command copied to clipboard! Run it in your terminal to see the delta.'));"
                # Replace the cell content with a link
                line = line.replace('>changed<', f' style="cursor:pointer;text-decoration:underline;" onclick="{onclick}" title="Click to copy delta command">changed<')
    
    new_lines.append(line)

html_content = '\n'.join(new_lines)

# Write the modified HTML
with open(html_path, 'w') as f:
    f.write(html_content)

styled.to_excel('control_comparison.xlsx', merge_cells=False)

print("Rendered HTML → control_comparison.html")
print("Rendered Excel → control_comparison.xlsx")

# Open the HTML report
open_html(str(html_path))
