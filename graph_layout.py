#!/usr/bin/env python3
# graph_layout_packed.py

import argparse
import json
import math
from collections import Counter, deque
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import numpy as np
import pandas as pd

# ───────────────────────────────────────────────────────────────
# CONSTANTS
# ───────────────────────────────────────────────────────────────
DEFAULT_JSON_OUT = Path(r"C:\Solutions\JavaScript\HexMap\src\data.json")
DEFAULT_PNG_OUT = Path("graph_layout_final.png")
MAX_PER_ROW = 10
RAG_PALETTE = ["#C7273A", "#FFE05A", "#3F7F37"]  # BrightClaret, LightOrange, Green
CLUSTER_BASE_PALETTE = ["#0076B6", "#4C3D6C", "#007481", "#CDF5E8", "#C3F5BA", "#FFCB05", "#E1C0E2"]  # LightBlue, DarkPurple, LightTeal, Mint, Lime, Orange, LightPurple
SQRT3 = math.sqrt(3)
DIRS = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]


# ───────────────────────────────────────────────────────────────
# HEX GRID UTILITIES
# ───────────────────────────────────────────────────────────────
def axial_to_cube(q, r): x = q; z = r; y = -x - z; return x, y, z


def cube_to_axial(x, y, z): q = x; r = z; return q, r


def cube_round(x_cube, y_cube, z_cube):
    rx = round(x_cube);
    ry = round(y_cube);
    rz = round(z_cube)
    x_diff = abs(rx - x_cube);
    y_diff = abs(ry - y_cube);
    z_diff = abs(rz - z_cube)
    if x_diff > y_diff and x_diff > z_diff:
        rx = -ry - rz
    elif y_diff > z_diff:
        ry = -rx - rz
    else:
        rz = -rx - ry
    return int(rx), int(ry), int(rz)


def cartesian_to_axial(x_cart, y_cart, hex_pixel_size=1.0):
    if hex_pixel_size == 0: return (0, 0)
    q_frac = (2 / 3 * x_cart) / hex_pixel_size;
    r_frac = (-1 / 3 * x_cart + SQRT3 / 3 * y_cart) / hex_pixel_size
    x_cube_frac, y_cube_frac, z_cube_frac = axial_to_cube(q_frac, r_frac)
    rx_cube, ry_cube, rz_cube = cube_round(x_cube_frac, y_cube_frac, z_cube_frac)
    final_q, final_r = cube_to_axial(rx_cube, ry_cube, rz_cube)
    return int(final_q), int(final_r)


# ───────────────────────────────────────────────────────────────
# DATA LOADING
# ───────────────────────────────────────────────────────────────
def load_edges_from_excel(path, sheet, sc="Source Application", tc="Target Application", wc="Duplication Count"):
    df = pd.read_excel(path, sheet_name=sheet)[[sc, tc, wc]].dropna()
    df[wc] = df[wc].astype(float)
    return list(df.itertuples(index=False, name=None))


def load_edges_from_sample():
    try:
        from sample_data import sample_edges; return sample_edges
    except ImportError:
        print("sample_data.py not found, using inline sample edges.")
        return [("AppA", "AppB", 10.0), ("AppB", "AppC", 20.0), ("AppA", "AppC", 5.0), ("AppD", "AppE", 15.0),
                ("AppE", "AppF", 25.0), ("AppD", "AppF", 8.0), ("AppG", "AppH", 12.0), ("AppA", "AppD", 30.0),
                ("AppB", "AppE", 20.0), ("AppX", "AppY", 50.0), ("AppY", "AppZ", 60.0), ("AppX", "AppZ", 40.0),
                ("AppA", "AppX", 70.0), *[(f"C1N{i}", f"C1N{i + 1}", 5.0) for i in range(12)],
                *[(f"C2N{i}", f"C2N{i + 1}", 5.0) for i in range(8)], ("AppC", f"C1N0", 15.0), ("AppF", f"C2N0", 15.0),
                (f"C1N6", f"C2N4", 25.0)]


# ───────────────────────────────────────────────────────────────
# GRAPH BUILD + LAYOUT
# ───────────────────────────────────────────────────────────────
def build_graph(edges):
    g = nx.DiGraph()
    for s, t, w in edges:
        g.add_edge(str(s), str(t), weight=float(w))
    return g


def hierarchical_layout(g, internal_k=0.05, meta_kk_scale=10.0, seed=42):
    graph_for_community = g.to_undirected() if g.is_directed() else g
    if graph_for_community.number_of_edges() == 0 and graph_for_community.number_of_nodes() > 0:
        comms_temp = [[node] for node in graph_for_community.nodes()]
    elif graph_for_community.number_of_nodes() == 0:
        comms_temp = []
    else:
        try:
            comms_gen = nx.algorithms.community.greedy_modularity_communities(graph_for_community, weight="weight")
            comms_temp = [list(c) for c in comms_gen if c]
            if not comms_temp and graph_for_community.number_of_nodes() > 0: comms_temp = [[node] for node in
                                                                                           graph_for_community.nodes()]
        except Exception as e:
            print(f"Community detection failed: {e}. Treating all nodes as separate communities.")
            comms_temp = [[node] for node in graph_for_community.nodes()]

    degrees_in_g = dict(g.degree())
    comms = []
    for community_nodes in comms_temp:
        sorted_community_nodes = sorted(
            community_nodes,
            key=lambda node: degrees_in_g.get(node, 0),
            reverse=True
        )
        comms.append(sorted_community_nodes)

    meta = nx.Graph()
    node_to_comm_idx = {}
    for i, nodes_in_comm in enumerate(comms):
        meta.add_node(i)
        for node in nodes_in_comm: node_to_comm_idx[node] = i

    for u, v, data in g.edges(data=True):
        w = data.get("weight", 1.0)
        cu, cv = node_to_comm_idx.get(u), node_to_comm_idx.get(v)
        if cu is not None and cv is not None and cu != cv:
            existing_weight = meta.get_edge_data(cu, cv, {}).get("weight", 0)
            meta.add_edge(cu, cv, weight=w + existing_weight)

    if meta.number_of_nodes() > 1 and meta.number_of_edges() > 0:
        meta_centers = nx.kamada_kawai_layout(meta, weight="weight", scale=meta_kk_scale)
    elif meta.number_of_nodes() > 0:
        meta_node_list = list(meta.nodes())
        meta_centers = {
            node_id: (idx * meta_kk_scale / max(1, len(meta_node_list) - 1) if len(meta_node_list) > 1 else 0, 0)
            for idx, node_id in enumerate(meta_node_list)}
        if not meta_centers and meta_node_list: meta_centers = {meta_node_list[0]: (0, 0)}
    else:
        meta_centers = {}

    pos = {};
    rng = np.random.default_rng(seed)
    for i, nodes_in_comm in enumerate(comms):
        sub = g.subgraph(nodes_in_comm)
        if not sub.nodes(): continue
        current_internal_k = internal_k * 5 if sub.number_of_nodes() <= 2 else internal_k
        seed_val = int(rng.integers(0, 1_000_000))
        if sub.number_of_nodes() > 1 and sub.number_of_edges() > 0:
            local_pos = nx.spring_layout(sub, weight="weight", seed=seed_val, k=current_internal_k, scale=0.3)
        elif sub.number_of_nodes() > 0:
            local_pos = {node: (0, 0) for node in sub.nodes()}
        else:
            local_pos = {}

        cx, cy = meta_centers.get(i, (0.0, 0.0))
        for n, (x, y) in local_pos.items(): pos[n] = (cx + x, cy + y)
    return pos, comms, meta_centers


# ───────────────────────────────────────────────────────────────
# HEX PACKING AND CLUSTER PLACEMENT
# ───────────────────────────────────────────────────────────────
def pack_rows(cluster_anchor_qr, nodes_in_cluster):
    q0, r0 = cluster_anchor_qr
    packed_coords = {}
    for idx, node_id in enumerate(nodes_in_cluster):
        # Logical row and column for filling the cluster
        row_idx = idx // MAX_PER_ROW
        col_idx = idx % MAX_PER_ROW

        # Axial coordinates for a standard flat-topped honeycomb packing
        # Changed from 'col_idx + (row_idx - (row_idx & 1)) // 2' to 'col_idx - (row_idx // 2)'
        # to achieve a consistent leftward slant for clusters as row_idx increases.
        q_local = col_idx - (row_idx // 2)
        r_local = row_idx
        
        current_q = q0 + q_local
        current_r = r0 + r_local
        packed_coords[node_id] = (current_q, current_r)
    return packed_coords


def place_clusters_on_hex_grid(cluster_node_lists, meta_cartesian_centers):
    cluster_local_shapes = {}
    for i, nodes in enumerate(cluster_node_lists):
        if not nodes: cluster_local_shapes[i] = {"nodes": [], "local_cells": []}; continue
        packed_at_origin_map = pack_rows((0, 0), nodes)
        local_cells_for_shape = list(packed_at_origin_map.values())
        cluster_local_shapes[i] = {"nodes": nodes, "local_cells": local_cells_for_shape}

    initial_hex_anchors = {}
    for i, (x, y) in meta_cartesian_centers.items():
        q_ideal, r_ideal = cartesian_to_axial(x, y, hex_pixel_size=1.0)
        initial_hex_anchors[i] = (q_ideal, r_ideal)

    # Sort clusters by size (number of nodes) in descending order (largest first)
    sorted_cluster_indices = sorted(
        meta_cartesian_centers.keys(),
        key=lambda idx: len(cluster_node_lists[idx]),
        reverse=True
    )
    
    occupied_global_cells = set();
    final_cluster_placements = {}
    max_search_attempts_bfs = 5000

    for cluster_idx in sorted_cluster_indices:
        if cluster_idx not in cluster_local_shapes or not cluster_local_shapes[cluster_idx]["nodes"]:
            final_cluster_placements[cluster_idx] = {"anchor": initial_hex_anchors.get(cluster_idx, (0, 0)),
                                                     "app_positions": {}, "cluster_nodes": []}
            continue

        shape_info = cluster_local_shapes[cluster_idx]
        q_ideal, r_ideal = initial_hex_anchors.get(cluster_idx, (0, 0))
        found_spot = False;
        search_q = deque([(q_ideal, r_ideal)]);
        visited_anchors = set([(q_ideal, r_ideal)])
        attempts = 0;
        chosen_anchor = (q_ideal, r_ideal)

        while search_q and attempts < max_search_attempts_bfs:
            attempts += 1;
            cand_anchor_q, cand_anchor_r = search_q.popleft()
            collision = False
            for q_local, r_local in shape_info["local_cells"]:
                g_q, g_r = cand_anchor_q + q_local, cand_anchor_r + r_local
                if (g_q, g_r) in occupied_global_cells: collision = True; break
            if not collision:
                chosen_anchor = (cand_anchor_q, cand_anchor_r); found_spot = True; break
            else:
                # Explore neighbors, sorted by distance to the original ideal anchor
                neighbors = []
                for dq, dr in DIRS:
                    next_q, next_r = cand_anchor_q + dq, cand_anchor_r + dr
                    if (next_q, next_r) not in visited_anchors:
                        neighbors.append((next_q, next_r))
                
                # Sort neighbors by Euclidean distance to the ideal anchor (q_ideal, r_ideal), closer first
                # This is a heuristic to guide the BFS towards the ideal if possible
                neighbors.sort(key=lambda coord: math.sqrt((coord[0] - q_ideal)**2 + (coord[1] - r_ideal)**2 + (coord[0] - q_ideal + coord[1] - r_ideal)**2)) # Cube distance for hex grid

                for next_q_sorted, next_r_sorted in neighbors:
                    search_q.append((next_q_sorted, next_r_sorted))
                    visited_anchors.add((next_q_sorted, next_r_sorted))

        if not found_spot: print(
            f"Warning: Could not find non-colliding buffered spot for cluster {cluster_idx} after {attempts} attempts. Using ideal/last tried anchor {chosen_anchor}.")

        app_hex_positions = pack_rows(chosen_anchor, shape_info["nodes"])
        final_cluster_placements[cluster_idx] = {"anchor": chosen_anchor, "app_positions": app_hex_positions,
                                                 "cluster_nodes": shape_info["nodes"]}

        cells_to_mark_occupied = set()
        for actual_cell_q, actual_cell_r in app_hex_positions.values():
            cells_to_mark_occupied.add((actual_cell_q, actual_cell_r))
            for dq, dr in DIRS: cells_to_mark_occupied.add((actual_cell_q + dq, actual_cell_r + dr))
        occupied_global_cells.update(cells_to_mark_occupied)
    return final_cluster_placements


# ───────────────────────────────────────────────────────────────
# JSON OUTPUT AND VISUALIZATION
# ───────────────────────────────────────────────────────────────
def extend_palette(n):
    if n <= 0: return []
    if n <= len(CLUSTER_BASE_PALETTE): return CLUSTER_BASE_PALETTE[:n]
    try:
        cmap = cm.get_cmap("tab20", n)
    except ValueError:
        cmap = cm.get_cmap("viridis", n)
    return [cm.colors.rgb2hex(cmap(i)) for i in range(n)]


def strength_category(w):
    if w >= 500: return "high";
    if w >= 250: return "medium";
    return "low"


def calculate_cluster_label_anchor(app_hex_positions_map):
    if not app_hex_positions_map: return (0, 0)
    min_r = min(pos[1] for pos in app_hex_positions_map.values())
    top_row_apps_pos = [pos for pos in app_hex_positions_map.values() if pos[1] == min_r]
    if not top_row_apps_pos:
        any_app_pos = next(iter(app_hex_positions_map.values()))
        return (any_app_pos[0], any_app_pos[1])
    top_row_apps_pos.sort(key=lambda pos: pos[0])
    median_top_app_pos = top_row_apps_pos[len(top_row_apps_pos) // 2]
    return (median_top_app_pos[0], median_top_app_pos[1])


def build_json(graph, cluster_node_lists, final_placements, indicator_threshold):
    num_clusters = len(cluster_node_lists);
    palette = extend_palette(num_clusters)
    conn_map = {n: [] for n in graph.nodes()}
    for s, t, w_data in graph.edges(data=True):
        weight = w_data.get('weight', 1.0)
        conn_map[s].append({"to": str(t), "type": "link", "strength": strength_category(weight)})

    deg_map = dict(graph.degree())
    output_data = {"clusters": []}
    for original_cluster_idx in range(num_clusters):
        if original_cluster_idx not in final_placements:
            print(f"Warning: Cluster index {original_cluster_idx} not found in final_placements. Skipping.");
            continue

        placement_info = final_placements[original_cluster_idx]
        json_cluster_id_num = original_cluster_idx + 1
        cluster_color = palette[original_cluster_idx % len(palette)] if palette else "#808080"

        app_hex_positions = placement_info["app_positions"]
        nodes_in_this_cluster = placement_info["cluster_nodes"]

        if nodes_in_this_cluster and app_hex_positions:
            cluster_label_anchor_q, cluster_label_anchor_r = calculate_cluster_label_anchor(app_hex_positions)
        else:
            cluster_label_anchor_q, cluster_label_anchor_r = placement_info["anchor"]

        apps_json_list = []
        for node_id in nodes_in_this_cluster:
            if node_id not in app_hex_positions:
                print(
                    f"Warning: Node {node_id} from cluster {original_cluster_idx} not in app_hex_positions map. Assigning default relative to label anchor.")
                app_q, app_r = cluster_label_anchor_q, cluster_label_anchor_r + 1
            else:
                app_q, app_r = app_hex_positions[node_id]

            app_data = {"id": str(node_id), "name": str(node_id), "color": cluster_color,
                        "connections": conn_map.get(node_id, []), "status": 100,
                        "gridPosition": {"q": app_q, "r": app_r}}
            
            current_degree = deg_map.get(node_id, 0)
            # Debug print statement added here:
            print(f"Node: {node_id}, Degree: {current_degree}, Threshold: {indicator_threshold}, ShowIndicator: {current_degree >= indicator_threshold}")
            if current_degree >= indicator_threshold:
                app_data["showPositionIndicator"] = True
            apps_json_list.append(app_data)

        output_data["clusters"].append({
            "id": f"cluster_{json_cluster_id_num}", "name": f"APPS {json_cluster_id_num}",
            "color": cluster_color, "hexCount": len(apps_json_list),
            "gridPosition": {"q": cluster_label_anchor_q, "r": cluster_label_anchor_r},
            "priority": "Normal", "applications": apps_json_list
        })
    return output_data


# ───────────────────────────────────────────────────────────────
# DEBUG VISUAL
# ───────────────────────────────────────────────────────────────
def save_png_debug(graph_obj, node_positions_cartesian, output_path):
    if not node_positions_cartesian: print(f"Skipping PNG debug: no node positions for {output_path}."); return
    edge_weights = nx.get_edge_attributes(graph_obj, "weight")
    max_w = max(edge_weights.values()) if edge_weights else 1.0;
    max_w = max(max_w, 1.0)
    widths = [0.5 + 4 * v / max_w for v in edge_weights.values()]
    plt.figure(figsize=(16, 12))
    nx.draw_networkx_edges(graph_obj, node_positions_cartesian, width=widths, alpha=0.5, edge_color="grey")
    nx.draw_networkx_nodes(graph_obj, node_positions_cartesian, node_size=120, node_color="#333333", edgecolors="white",
                           linewidths=0.5)
    nx.draw_networkx_labels(graph_obj, node_positions_cartesian, font_size=7, font_color="#000000")
    plt.title("Debug View: Kamada-Kawai Cluster Centers & Spring Internal Layouts (Cartesian)");
    plt.axis("off");
    plt.tight_layout()
    plt.savefig(output_path, dpi=200);
    plt.close();
    print(f"Debug PNG saved to {output_path}")


# ───────────────────────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate HexMap JSON with degree-sorted internal packing and custom row staggering.")
    parser.add_argument("--excel", help="Path to Excel file");
    parser.add_argument("--sheet", help="Sheet name")
    parser.add_argument("--internal-k", type=float, default=0.1, help="Spring k for PNG debug.")
    parser.add_argument("--meta-kk-scale", type=float, default=30.0,
                        help="Kamada-Kawai scale for cluster centers. Controls spread.")
    parser.add_argument("--indicator-threshold", type=int, default=15, help="Degree threshold for indicator.")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT, help="Output JSON path.")
    parser.add_argument("--png-out", type=Path, default=DEFAULT_PNG_OUT, help="Output PNG path.")
    parser.add_argument("--no-png", action="store_true", help="Disable PNG saving.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    edges = (load_edges_from_excel(args.excel, args.sheet) if args.excel and args.sheet else load_edges_from_sample())
    if not edges: print("No edges loaded. Exiting."); return
    g = build_graph(edges)
    if not g.nodes(): print("Graph has no nodes. Exiting."); return

    pos_cartesian, cluster_node_lists, meta_cartesian_centers = hierarchical_layout(g, args.internal_k,
                                                                                    args.meta_kk_scale, args.seed)

    if not args.no_png:
        if pos_cartesian:
            save_png_debug(g, pos_cartesian, args.png_out)
        else:
            print("Skipping PNG: No Cartesian positions generated.")
    if not cluster_node_lists: print("No clusters detected. Cannot proceed. Exiting."); return

    if cluster_node_lists and (not meta_cartesian_centers or
                               len(meta_cartesian_centers) != len(cluster_node_lists) or
                               any(i not in meta_cartesian_centers for i in range(len(cluster_node_lists)))):
        print(
            f"Warning: Mismatch/incompleteness in meta_cartesian_centers ({len(meta_cartesian_centers)}) vs communities ({len(cluster_node_lists)}). Rebuilding to align.")
        new_meta_centers = {}
        current_x_offset = 0
        default_spacing = args.meta_kk_scale / max(1, len(cluster_node_lists) - 1 if len(cluster_node_lists) > 1 else 1)

        for i in range(len(cluster_node_lists)):
            if i in meta_cartesian_centers:
                new_meta_centers[i] = meta_cartesian_centers[i]
                current_x_offset = meta_cartesian_centers[i][0] + default_spacing
            else:
                new_meta_centers[i] = (current_x_offset, 0)
                current_x_offset += default_spacing
        meta_cartesian_centers = new_meta_centers

    final_hex_placements = place_clusters_on_hex_grid(cluster_node_lists, meta_cartesian_centers)

    output_json_data = build_json(g, cluster_node_lists, final_hex_placements,
                                  args.indicator_threshold)  # Corrected variable name

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(output_json_data, indent=2), encoding="utf-8")
    print(f"HexMap JSON saved to {args.json_out}")


if __name__ == "__main__":
    main()