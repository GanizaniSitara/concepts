#!/usr/bin/env python3
# graph_hex_translate.py
#
# Builds a graph from an edge list (sample or Excel), lays it out,
# converts the 2‑D layout to hex‑grid axial coordinates, detects
# communities → dynamic “APPS n” clusters, assigns unique colours
# (RAG reserved for status only), and writes HexMap‑compatible JSON
# to a fixed path:  C:\Solutions\JavaScript\HexMap\src\data.json
#
# Example
#   python graph_hex_translate.py
#   python graph_hex_translate.py --excel flows.xlsx --sheet Flows
#
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------
JSON_OUT_PATH = Path(r"C:\Solutions\JavaScript\HexMap\src\data.json")
PNG_OUT_PATH = Path("graph.png")

# ⬤ RAG palette reserved for future per‑node status use (don’t touch)
RAG_PALETTE = ["#e31a1c",  # red
               "#ff7f00",  # amber
               "#33a02c"]  # green

# Base colours for clusters (exclude the three RAG colours)
CLUSTER_BASE_PALETTE = [
    "#1E90FF",
    "#4169E1",
    "#8A2BE2",
    "#FFD700",  # Gold
    "#FF69B4",  # Hot Pink
    "#40E0D0"   # Turquoise
    "#9932CC",
    "#008080",
]

SQRT3 = math.sqrt(3)
HEX_DIRS = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]


# ------------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------------
def load_edges_from_excel(path, sheet,
                          source_col="Source Application",
                          target_col="Target Application",
                          weight_col="Duplication Count"):
    df = pd.read_excel(path, sheet_name=sheet)
    df = df[[source_col, target_col, weight_col]].dropna()
    df[weight_col] = df[weight_col].astype(float)
    return list(df.itertuples(index=False, name=None))


def load_edges_from_sample():
    from sample_data import sample_edges
    return sample_edges


# ------------------------------------------------------------------
# GRAPH + LAYOUT
# ------------------------------------------------------------------
def build_graph(edges):
    g = nx.DiGraph()
    for s, t, w in edges:
        g.add_edge(s, t, weight=float(w))
    return g


def compute_layout(g, layout="spring", seed=42):
    if layout == "kamada_kawai":
        return nx.kamada_kawai_layout(g, weight="weight")
    if layout == "shell":
        return nx.shell_layout(g)
    return nx.spring_layout(g, weight="weight", seed=seed)


# ------------------------------------------------------------------
# HEX HELPERS
# ------------------------------------------------------------------
def to_axial(x, y, size):
    q = (SQRT3 / 3 * x - y / 3) / size
    r = (2 * y / 3) / size
    return q, r


def cube_round(qf, rf):
    x = qf
    z = rf
    y = -x - z
    rx, ry, rz = map(round, (x, y, z))
    dx, dy, dz = map(abs, (rx - x, ry - y, rz - z))
    if dx > dy and dx > dz:
        rx = -ry - rz
    elif dy > dz:
        ry = -rx - rz
    else:
        rz = -rx - ry
    return int(rx), int(rz)


def axial_add(a, b):
    return a[0] + b[0], a[1] + b[1]


def hex_ring(center, radius):
    if radius == 0:
        yield center
        return
    q, r = axial_add(center, (HEX_DIRS[4][0] * radius,
                              HEX_DIRS[4][1] * radius))
    for i in range(6):
        dq, dr = HEX_DIRS[i]
        for _ in range(radius):
            yield (q, r)
            q, r = q + dq, r + dr


def resolve_collisions(coords):
    occupied, dup = {}, {}
    for n, pos in coords.items():
        if pos in occupied:
            dup.setdefault(pos, []).append(n)
        else:
            occupied[pos] = n
    cache = {}
    for pos, nodes in dup.items():
        for n in nodes:
            r = 1
            while True:
                cache.setdefault(r, list(hex_ring((0, 0), r)))
                for dq, dr in cache[r]:
                    cand = (pos[0] + dq, pos[1] + dr)
                    if cand not in occupied:
                        occupied[cand] = n
                        coords[n] = cand
                        break
                else:
                    r += 1
                    continue
                break
    return coords


def layout_to_hex(pos, base_size=0.1, max_iter=25):
    xs, ys = zip(*pos.values())
    xs = np.array(xs) - np.mean(xs)
    ys = np.array(ys) - np.mean(ys)
    size = base_size
    for _ in range(max_iter):
        coords = {}
        for n, (x, y) in pos.items():
            qf, rf = to_axial(x - np.mean(xs), y - np.mean(ys), size)
            coords[n] = cube_round(qf, rf)
        if max(Counter(coords.values()).values()) == 1:
            return coords
        size *= 0.8
    return resolve_collisions(coords)


# ------------------------------------------------------------------
# CLUSTERING & COLOURS
# ------------------------------------------------------------------
def detect_clusters(g):
    """Greedy modularity communities -> list of node lists."""
    comms = nx.algorithms.community.greedy_modularity_communities(
        g.to_undirected(), weight="weight")
    return [sorted(c) for c in comms]


def extend_palette(n):
    if n <= len(CLUSTER_BASE_PALETTE):
        return CLUSTER_BASE_PALETTE[:n]
    cmap = cm.get_cmap("tab20", n)
    return [cm.colors.rgb2hex(cmap(i)) for i in range(n)]


# ------------------------------------------------------------------
# JSON BUILD
# ------------------------------------------------------------------
def strength_from_weight(w):
    if w >= 500:
        return "high"
    if w >= 250:
        return "medium"
    return "low"


def cluster_label_position(cluster_coords, all_coords):
    """Place label slightly outside cluster towards perimeter."""
    cq = int(round(np.mean([q for q, _ in cluster_coords])))
    cr = int(round(np.mean([r for _, r in cluster_coords])))
    gq = np.mean([q for q, _ in all_coords])
    gr = np.mean([r for _, r in all_coords])
    dq = int(math.copysign(2, cq - gq)) if cq != gq else 0
    dr = int(math.copysign(2, cr - gr)) if cr != gr else 3  # push down if centred
    return {"q": cq + dq, "r": cr + dr}


def build_hexmap_json(coords, edges, clusters):
    palette = extend_palette(len(clusters))
    conn_map = {n: [] for n in coords}
    for s, t, w in edges:
        conn_map[s].append({"to": t, "type": "link",
                            "strength": strength_from_weight(w)})

    all_coords = list(coords.values())
    clusters_json = []
    for idx, nodes in enumerate(clusters, start=1):
        col = palette[idx - 1]
        apps = [{
            "id": n,
            "name": n,
            "description": "",
            "color": col,
            "gridPosition": {"q": coords[n][0], "r": coords[n][1]},
            "showPositionIndicator": True,
            "connections": conn_map[n],
            "status": 100
        } for n in nodes]

        label_pos = cluster_label_position(
            [coords[n] for n in nodes], all_coords)

        clusters_json.append({
            "id": f"cluster_{idx}",
            "name": f"APPS {idx}",
            "color": col,
            "hexCount": len(apps),
            "gridPosition": label_pos,
            "priority": "Normal",
            "lastUpdated": "",
            "description": "",
            "applications": apps
        })
    return {"clusters": clusters_json}


# ------------------------------------------------------------------
# PNG DEBUG VISUAL
# ------------------------------------------------------------------
def save_png(g, pos):
    w = nx.get_edge_attributes(g, "weight")
    max_w = max(w.values())
    widths = [0.5 + 4 * wt / max_w for wt in w.values()]
    plt.figure(figsize=(12, 8))
    nx.draw_networkx_edges(g, pos, width=widths, alpha=0.6, edge_color="grey")
    nx.draw_networkx_nodes(g, pos, node_size=220, node_color="#cccccc",
                           edgecolors="white", linewidths=0.8)
    nx.draw_networkx_labels(g, pos, font_size=8)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(PNG_OUT_PATH, dpi=150)
    plt.close()


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", help="Path to Excel file")
    parser.add_argument("--sheet", help="Sheet name")
    parser.add_argument("--layout", default="spring",
                        choices=["spring", "kamada_kawai", "shell"])
    parser.add_argument("--hex-size", type=float, default=0.1)
    parser.add_argument("--no-png", action="store_true")
    args = parser.parse_args()

    edges = (load_edges_from_excel(args.excel, args.sheet)
             if args.excel and args.sheet else load_edges_from_sample())

    g = build_graph(edges)
    pos = compute_layout(g, args.layout)

    if not args.no_png:
        save_png(g, pos)

    coords = layout_to_hex(pos, base_size=args.hex_size)
    clusters = detect_clusters(g)

    data = build_hexmap_json(coords, edges, clusters)

    JSON_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"HexMap JSON saved -> {JSON_OUT_PATH}")


if __name__ == "__main__":
    main()
