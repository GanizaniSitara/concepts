import os
import re
import hashlib
import platform
import subprocess
import webbrowser
from datetime import datetime
from pathlib import Path
import pandas as pd

def open_html(abs_path):
    """Open HTML file using appropriate method for the platform."""
    if "microsoft" in platform.uname().release.lower():  # crude WSL check
        # Translate to Windows-style path if needed
        win_path = abs_path.replace('/mnt/c/', 'C:\\').replace('/', '\\')
        subprocess.run(["/mnt/c/Windows/explorer.exe", win_path])
    else:
        webbrowser.open_new_tab(f'file://{abs_path}')

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
    # Create 4-hour time slots (00:00, 04:00, 08:00, 12:00, 16:00, 20:00)
    hour_4_slot = dt.replace(minute=0, second=0, microsecond=0)
    hour_4_slot = hour_4_slot.replace(hour=(dt.hour // 4) * 4)
    runs.setdefault((ctrl, hour_4_slot), []).append((dt, p))

# keep the latest run within each 4-hour slot for each control
representatives = {
    (ctrl, hour_4_slot): max(entries, key=lambda x: x[0])[1]
    for (ctrl, hour_4_slot), entries in runs.items()
}

# === PREPARE ROWS, COLUMNS, AND FILE LIST ===
hour_4_slots = sorted({hour_4_slot for (_, hour_4_slot) in representatives.keys()})
# map each 4-hour slot to a header label with date on top line, time range below
hour_4_labels = {
    hour_4_slot: f"{hour_4_slot.strftime('%Y-%m-%d')}\n{hour_4_slot.strftime('%H:%M')}-{(hour_4_slot.hour + 4) % 24:02d}:00"
    for hour_4_slot in hour_4_slots
}
columns = [hour_4_labels[h] for h in hour_4_slots]

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
            if os.path.splitext(fname)[1].lower() in ('.csv', '.json', '.txt'):
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
for (ctrl, hour_4_slot), path in representatives.items():
    label = hour_4_labels[hour_4_slot]
    # Only process files that exist for this control
    for fname in control_files[ctrl]:
        fp = os.path.join(path, fname)
        if os.path.exists(fp):
            with open(fp, 'rb') as fh:
                df.at[(ctrl, fname), label] = hashlib.sha1(fh.read()).hexdigest()
        else:
            df.at[(ctrl, fname), label] = None

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
    # rotate column headers 90° CCW, make columns narrow, preserve line break
    {
        'selector': 'th.col_heading',
        'props': [
            ('transform', 'rotate(-90deg)'),
            ('transform-origin', 'center center'),
            ('white-space', 'pre'),          # respect the "\n" in header
            ('width', '30px'),
            ('padding', '2px'),
            ('vertical-align', 'middle'),
            ('height', '80px'),              # prevent bleeding off page
            ('max-height', '80px'),
            ('font-family', 'sans-serif'),
            ('box-sizing', 'border-box')
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
styled.to_html(html_path, notebook=False)
styled.to_excel('control_comparison.xlsx', merge_cells=False)

print("Rendered HTML → control_comparison.html")
print("Rendered Excel → control_comparison.xlsx")

# Open the HTML report
open_html(str(html_path))
