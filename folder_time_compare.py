import os
import re
import hashlib
from datetime import datetime
import pandas as pd

# === CONFIGURATION ===
base_dir = '/path/to/base_directory'  # ← Change this to your folder containing the control run subfolders
pattern = re.compile(r'.*EOD_CTRL-(\d+)_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2})')

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
    hour = dt.replace(minute=0, second=0, microsecond=0)
    runs.setdefault((ctrl, hour), []).append((dt, p))

# keep the latest run within each hour for each control
representatives = {
    (ctrl, hour): max(entries, key=lambda x: x[0])[1]
    for (ctrl, hour), entries in runs.items()
}

# === PREPARE ROWS, COLUMNS, AND FILE LIST ===
hours = sorted({hour for (_, hour) in representatives.keys()})
# map each hour to a header label with date on top line, time below
hour_labels = {
    hour: f"{hour.strftime('%Y-%m-%d')}\n{hour.strftime('%H:%M')}"
    for hour in hours
}
columns = [hour_labels[h] for h in hours]

controls = sorted({ctrl for (ctrl, _) in representatives.keys()})
files = sorted({
    fname
    for path in representatives.values()
    for fname in os.listdir(path)
    if os.path.splitext(fname)[1].lower() in ('.csv', '.json', '.txt')
})

# build an empty DataFrame indexed by (control, file) and columns by our hour labels
index = pd.MultiIndex.from_product(
    [controls, files],
    names=['control', 'file']
)
df = pd.DataFrame(index=index, columns=columns, dtype=object)

# === FILL IN SHA1 HASHES FOR EACH CELL ===
for (ctrl, hour), path in representatives.items():
    label = hour_labels[hour]
    for fname in files:
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
            ('white-space', 'pre'),          # respect the "\n" in header
            ('width', '30px'),
            ('padding', '2px'),
            ('vertical-align', 'bottom')
        ]
    },
    # left-align the row labels (control/file)
    {
        'selector': 'th.row_heading',
        'props': [
            ('text-align', 'left'),
            ('padding-right', '10px')
        ]
    }
]

styled = (
    status.style
          .applymap(color_map)            # color the data cells
          .set_table_styles(styles, overwrite=False)
)

# === OUTPUT ===
styled.to_html('control_comparison.html', notebook=False)
styled.to_excel('control_comparison.xlsx', merge_cells=False)

print("Rendered HTML → control_comparison.html")
print("Rendered Excel → control_comparison.xlsx")
