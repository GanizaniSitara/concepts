#!/usr/bin/env python3
# graph_hex_translate.py
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
# CONSTANTS (overridable via CLI)
# ------------------------------------------------------------------
DEFAULT_JSON_OUT = Path(r"C:\Solutions\JavaScript\HexMap\src\data.json")
DEFAULT_PNG_OUT = Path("graph.png")

# reserved for future status colouring
RAG_PALETTE = ["#e31a1c", "#ff7f00", "#33a02c"]

CLUSTER_BASE_PALETTE = [
    "#1f78b4", "#6a3d9a", "#a6cee3", "#b2df8a",
    "#fb9a99", "#fdbf6f", "#cab2d6"
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


def compute_layout(g, layout="spring", k=0.15, seed=42):
    if layout == "kamada_kawai":
        return nx.kamada_kawai_layout(g, weight="weight")
    if layout == "shell":
        return nx.shell_layout(g)
    # spring layout: k ↓  -> tighter cluster
    return nx.spring_layout(g, weight="weight", seed=seed, k=k, scale=1.0)


# ---------------------------------------------------------------
# HIERARCHICAL LAYOUT  (cluster centres far apart, internals tight)
# ---------------------------------------------------------------
def hierarchical_layout(G, layout="kamada_kawai",
                        internal_k=0.05,        # tighter inside cluster
                        cluster_k=1.5,           # looser between clusters
                        seed=42):
    """
    Return a dict {node: (x, y)} where clusters are clearly separated.
    """
    # 1️⃣  detect communities
    communities = nx.algorithms.community.greedy_modularity_communities(
        G.to_undirected(), weight="weight")
    clusters = [list(c) for c in communities]

    # 2️⃣  build meta‑graph
    meta = nx.Graph()
    for idx, nodes in enumerate(clusters):
        meta.add_node(idx, members=nodes)

    # add weighted edges between clusters
    for u, v, w in G.edges(data="weight"):
        cu = next(i for i, c in enumerate(clusters) if u in c)
        cv = next(i for i, c in enumerate(clusters) if v in c)
        if cu == cv:
            continue
        meta.add_edge(cu, cv, weight=w + meta.get_edge_data(cu, cv, {}).get("weight", 0))

    # 3️⃣  layout meta‑graph (cluster centres)
    if layout == "spring":
        centre_pos = nx.spring_layout(meta, weight="weight",
                                      seed=seed, k=cluster_k, scale=1.0)
    else:  # kamada_kawai default
        centre_pos = nx.kamada_kawai_layout(meta, weight="weight", scale=1.0)

    # 4️⃣  layout each cluster internally, then translate
    pos = {}
    rng = np.random.default_rng(seed)
    for idx, nodes in enumerate(clusters):
        sub = G.subgraph(nodes)
        if layout == "spring":
            local = nx.spring_layout(sub, weight="weight",
                                     seed=rng.integers(0, 1000000),
                                     k=internal_k, scale=0.3)
        else:
            local = nx.kamada_kawai_layout(sub, weight="weight", scale=0.3)

        cx, cy = centre_pos[idx]
        for n, (x, y) in local.items():
            pos[n] = (cx + x, cy + y)

    return pos


# ------------------------------------------------------------------
# HEX HELPERS
# ------------------------------------------------------------------
def to_axial(x, y, size):
    q = (SQRT3 / 3 * x - y / 3) / size
    r = (2 * y / 3) / size
    return q, r


def cube_round(qf, rf):
    x, z = qf, rf
    y = -x - z
    rx, ry, rz = map(round, (x, y, z))
    dx, dy, dz = abs(rx - x), abs(ry - y), abs(rz - z)
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


def layout_to_hex(pos, base_size=1.5, max_iter=25):
    """
    base_size ↑  ->  tighter hex map
    """
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
        size *= 0.8  # shrink if collisions remain
    return resolve_collisions(coords)


# ------------------------------------------------------------------
# CLUSTERING & COLOURS
# ------------------------------------------------------------------
def detect_clusters(g):
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
    cq = int(round(np.mean([q for q, _ in cluster_coords])))
    cr = int(round(np.mean([r for _, r in cluster_coords])))
    gq = np.mean([q for q, _ in all_coords])
    gr = np.mean([r for _, r in all_coords])
    dq = int(math.copysign(2, cq - gq)) if cq != gq else 0
    dr = int(math.copysign(2, cr - gr)) if cr != gr else 3
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
def save_png(g, pos, png_path):
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
    plt.savefig(png_path, dpi=150)
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
    parser.add_argument("--spring-k", type=float, default=0.15,
                        help="Lower value → tighter initial spring layout")
    parser.add_argument("--hex-size", type=float, default=20,
                        help="Higher value → tighter hex‑grid projection")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT,
                        help="Output path for HexMap JSON")
    parser.add_argument("--png-out", type=Path, default=DEFAULT_PNG_OUT,
                        help="Debug PNG path")
    parser.add_argument("--no-png", action="store_true")
    args = parser.parse_args()

    edges = (load_edges_from_excel(args.excel, args.sheet)
             if args.excel and args.sheet else load_edges_from_sample())

    g = build_graph(edges)
    # pos = compute_layout(g, layout=args.layout, k=args.spring_k)
    pos = hierarchical_layout(
        g,
        layout="kamada_kawai",  # or "spring"
        internal_k=0.05,  # tight inside clusters
        cluster_k=1.5  # spacious between clusters
    )

    if not args.no_png:
        save_png(g, pos, args.png_out)

    coords = layout_to_hex(pos, base_size=args.hex_size)
    clusters = detect_clusters(g)
    data = build_hexmap_json(coords, edges, clusters)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"HexMap JSON saved -> {args.json_out}")


if __name__ == "__main__":
    main()
