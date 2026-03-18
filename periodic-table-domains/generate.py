"""
Periodic Table of Data Domains — generates SVG and Draw.io outputs
from a YAML config file.

Usage:
    python generate.py [--yaml data.yaml] [--output-dir .]
"""

import argparse
from collections import defaultdict
import sys
import os
import xml.etree.ElementTree as ET

import yaml

# ---------------------------------------------------------------------------
# Draw.io helpers — reuse from C:\git\drawio
# ---------------------------------------------------------------------------
sys.path.insert(0, r"C:\git\drawio")
from lxml import etree
import drawio_tools
import drawio_shared_functions


# ---------------------------------------------------------------------------
# Text wrapping
# ---------------------------------------------------------------------------
def text_wrap(text, max_chars=14):
    """Word-wrap *text* into lines of at most *max_chars* characters."""
    words = text.split()
    lines, current = [], ""
    for w in words:
        if current and len(current) + 1 + len(w) > max_chars:
            lines.append(current)
            current = w
        else:
            current = f"{current} {w}".strip() if current else w
    if current:
        lines.append(current)
    return lines


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------
def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Position computation
# ---------------------------------------------------------------------------
def compute_positions(config):
    """Return a flat list of tile dicts with absolute x/y and resolved color."""
    layout = config["layout"]
    tw = layout["tile_width"]
    th = layout["tile_height"]
    gap = layout["gap"]
    cat_gap = layout.get("category_gap", 12)
    ml = layout["margin_left"]
    mt = layout["margin_top"]
    scheme_name = config.get("default_scheme", "cobalt-reef")
    palette = config["schemes"][scheme_name]

    # find global max row across all categories for bottom-alignment
    global_max_row = max(t["row"] for cat in config["categories"]
                        for t in cat["tiles"])

    # Repack tiles bottom-left → top-right: full rows at bottom,
    # partial (incomplete) row at top.  Preserves tile ordering.
    for cat in config["categories"]:
        sorted_tiles = sorted(cat["tiles"], key=lambda t: (t["row"], t["col"]))
        n_cols = max(t["col"] for t in cat["tiles"]) + 1
        n_tiles = len(sorted_tiles)
        total_rows = (n_tiles + n_cols - 1) // n_cols
        remainder = n_tiles % n_cols  # tiles in the partial top row (0 = all full)

        for idx, t in enumerate(sorted_tiles):
            if remainder > 0 and idx < remainder:
                # partial top row
                t["row"] = 0
                t["col"] = idx
            else:
                adj = idx - remainder if remainder > 0 else idx
                row_off = 1 if remainder > 0 else 0
                t["row"] = adj // n_cols + row_off
                t["col"] = adj % n_cols

    tiles = []
    cat_labels = []
    cum_x = ml  # cumulative x for category columns

    for cat in config["categories"]:
        color = palette[cat["color"]]
        max_col = max(t["col"] for t in cat["tiles"])
        cat_max_row = max(t["row"] for t in cat["tiles"])
        row_shift = global_max_row - cat_max_row  # push tiles to bottom

        for t in cat["tiles"]:
            x = cum_x + t["col"] * (tw + gap)
            y = mt + (t["row"] + row_shift) * (th + gap)
            tiles.append({
                "code": t["code"],
                "desc": t["desc"],
                "x": x,
                "y": y,
                "color": color,
                "category": cat["name"],
            })

        cat_width = (max_col + 1) * (tw + gap) - gap
        cat_labels.append({
            "name": cat["name"],
            "x": cum_x,
            "width": cat_width,
            "color": color,
        })
        cum_x += (max_col + 1) * (tw + gap) + cat_gap

    return tiles, cat_labels


# ---------------------------------------------------------------------------
# SVG renderer
# ---------------------------------------------------------------------------
def render_svg(tiles, cat_labels, config, output_path):
    layout = config["layout"]
    tw = layout["tile_width"]
    th = layout["tile_height"]
    font = layout["font_family"]
    mt = layout["margin_top"]
    gap = layout["gap"]
    meta = config["meta"]
    sidebar_w = layout.get("sidebar_width", 40)
    bar_color = layout.get("bar_color", "#53565A")

    # compute canvas size
    max_x = max(t["x"] + tw for t in tiles)
    grid_bottom = max(t["y"] + th for t in tiles)
    svg_w = max_x + 20
    svg_h = grid_bottom + 35  # space for bottom labels

    svg = ET.Element("svg", xmlns="http://www.w3.org/2000/svg",
                      width=str(svg_w), height=str(svg_h),
                      viewBox=f"0 0 {svg_w} {svg_h}")

    # white background
    ET.SubElement(svg, "rect", width=str(svg_w), height=str(svg_h), fill="white")

    # gray sidebar (left) — spans grid area only
    ET.SubElement(svg, "rect",
                  x="0", y="0",
                  width=str(sidebar_w), height=str(grid_bottom),
                  fill=bar_color)
    y_label = ET.SubElement(svg, "text",
                            x=str(sidebar_w / 2),
                            y=str((mt + grid_bottom) / 2),
                            fill="white",
                            transform=f"rotate(-90, {sidebar_w / 2}, {(mt + grid_bottom) / 2})")
    y_label.set("font-family", font)
    y_label.set("font-size", "14")
    y_label.set("font-weight", "bold")
    y_label.set("text-anchor", "middle")
    y_label.text = meta["y_label"]

    # tiles
    for t in tiles:
        # square rect with dark stroke
        tile_rect = ET.SubElement(svg, "rect",
                      x=str(t["x"]), y=str(t["y"]),
                      width=str(tw), height=str(th),
                      fill=t["color"],
                      stroke="#000000")
        tile_rect.set("stroke-width", "2")

        # code (bold)
        code_el = ET.SubElement(svg, "text",
                                x=str(t["x"] + 8), y=str(t["y"] + 18),
                                fill="white")
        code_el.set("font-family", font)
        code_el.set("font-size", "14")
        code_el.set("font-weight", "bold")
        code_el.text = t["code"]

        # description (word-wrapped)
        lines = text_wrap(t["desc"], max_chars=14)
        for i, line in enumerate(lines):
            desc_el = ET.SubElement(svg, "text",
                                    x=str(t["x"] + 8),
                                    y=str(t["y"] + 33 + i * 12),
                                    fill="white")
            desc_el.set("font-family", font)
            desc_el.set("font-size", "9")
            # no italic — matches reference style
            desc_el.text = line

    # thin separator line below grid
    ET.SubElement(svg, "line",
                  x1=str(sidebar_w), y1=str(grid_bottom + 4),
                  x2=str(svg_w), y2=str(grid_bottom + 4),
                  stroke="#999999")
    svg[-1].set("stroke-width", "1")

    # bottom labels
    cat_bar_w = layout.get("cat_bar_width", 120)
    bar_h = 25
    bar_y = grid_bottom + 8
    label_y = bar_y + bar_h / 2 + 4  # text baseline centred in bar

    # "Categories" box — horizontal bar matching sidebar style
    ET.SubElement(svg, "rect",
                  x="0", y=str(bar_y),
                  width=str(cat_bar_w), height=str(bar_h),
                  fill=bar_color)
    cat_label_el = ET.SubElement(svg, "text",
                                 x=str(cat_bar_w / 2),
                                 y=str(label_y),
                                 fill="white")
    cat_label_el.set("font-family", font)
    cat_label_el.set("font-size", "14")
    cat_label_el.set("font-weight", "bold")
    cat_label_el.set("text-anchor", "middle")
    cat_label_el.text = meta["x_label"]

    # category names centered under each column (bold)
    for cl in cat_labels:
        cx = cl["x"] + cl["width"] / 2
        lbl = ET.SubElement(svg, "text",
                            x=str(cx), y=str(label_y),
                            fill="#333333")
        lbl.set("font-family", font)
        lbl.set("font-size", "11")
        lbl.set("font-weight", "bold")
        lbl.set("text-anchor", "middle")
        lbl.text = cl["name"]

    tree = ET.ElementTree(svg)
    ET.indent(tree, space="  ")
    tree.write(output_path, xml_declaration=True, encoding="utf-8")
    print(f"SVG written to {output_path}")


# ---------------------------------------------------------------------------
# Draw.io renderer
# ---------------------------------------------------------------------------
def get_diagram_root():
    """Create the mxGraphModel + root + cell 0 + cell 1 skeleton."""
    mxGraphModel = etree.Element("mxGraphModel")
    for attr, val in [("dx", "1200"), ("dy", "800"), ("grid", "1"),
                      ("gridSize", "10"), ("guides", "1"), ("tooltips", "1"),
                      ("connect", "1"), ("arrows", "1"), ("fold", "1"),
                      ("page", "1"), ("pageScale", "1"),
                      ("pageWidth", "1600"), ("pageHeight", "900"),
                      ("math", "0"), ("shadow", "0")]:
        mxGraphModel.set(attr, val)
    root = etree.SubElement(mxGraphModel, "root")
    cell0 = etree.SubElement(root, "mxCell")
    cell0.set("id", "0")
    cell1 = etree.SubElement(root, "mxCell")
    cell1.set("id", "1")
    cell1.set("parent", "0")
    return mxGraphModel


def render_drawio(tiles, cat_labels, config, output_path):
    layout = config["layout"]
    tw = layout["tile_width"]
    th = layout["tile_height"]
    font = layout["font_family"]
    meta = config["meta"]
    mt = layout["margin_top"]
    gap = layout["gap"]
    sidebar_w = layout.get("sidebar_width", 40)
    bar_color = layout.get("bar_color", "#53565A")

    mxGraphModel = get_diagram_root()
    root = mxGraphModel.find("root")

    # compute grid extents
    max_x = max(t["x"] + tw for t in tiles)
    grid_bottom = max(t["y"] + th for t in tiles)

    # gray sidebar (left) — spans grid area only
    style_sidebar = (f"rounded=0;whiteSpace=wrap;html=1;"
                     f"fillColor={bar_color};strokeColor=none;")
    cell = etree.SubElement(root, "mxCell")
    cell.set("id", drawio_shared_functions.id_generator())
    cell.set("value", "")
    cell.set("style", style_sidebar)
    cell.set("parent", "1")
    cell.set("vertex", "1")
    geo = etree.SubElement(cell, "mxGeometry")
    geo.set("x", "0")
    geo.set("y", "0")
    geo.set("width", str(sidebar_w))
    geo.set("height", str(grid_bottom))
    geo.set("as", "geometry")

    # "Data Domains" text on sidebar (vertical)
    style_sidebar_text = (f"text;html=1;strokeColor=none;fillColor=none;"
                          f"align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;"
                          f"fontColor=#FFFFFF;fontSize=14;fontStyle=1;fontFamily={font};"
                          f"horizontal=0;")
    cell = etree.SubElement(root, "mxCell")
    cell.set("id", drawio_shared_functions.id_generator())
    cell.set("value", meta["y_label"])
    cell.set("style", style_sidebar_text)
    cell.set("parent", "1")
    cell.set("vertex", "1")
    geo = etree.SubElement(cell, "mxGeometry")
    geo.set("x", "0")
    geo.set("y", str(mt))
    geo.set("width", str(sidebar_w))
    geo.set("height", str(grid_bottom - mt))
    geo.set("as", "geometry")

    # tiles
    for t in tiles:
        style = (f"rounded=0;whiteSpace=wrap;html=1;"
                 f"fillColor={t['color']};fontColor=#FFFFFF;"
                 f"strokeColor=#000000;strokeWidth=2;"
                 f"fontSize=14;fontStyle=1;align=left;verticalAlign=top;spacing=8;"
                 f"fontFamily={font};")
        desc_html = t["desc"].replace("&", "&amp;")
        code_html = t["code"]
        value = (f'<b>{code_html}</b><br>'
                 f'<font style="font-size:9px">{desc_html}</font>')

        cell = etree.SubElement(root, "mxCell")
        cell.set("id", drawio_shared_functions.id_generator())
        cell.set("value", value)
        cell.set("style", style)
        cell.set("parent", "1")
        cell.set("vertex", "1")
        geo = etree.SubElement(cell, "mxGeometry")
        geo.set("x", str(t["x"]))
        geo.set("y", str(t["y"]))
        geo.set("width", str(tw))
        geo.set("height", str(th))
        geo.set("as", "geometry")

    # thin separator line below grid
    ml = layout["margin_left"]
    total_w = max(t["x"] + tw for t in tiles)
    style_line = (f"shape=line;strokeColor=#999999;strokeWidth=1;"
                  f"fillColor=none;align=left;verticalAlign=middle;"
                  f"spacingTop=-1;spacingBottom=-1;spacingLeft=-1;spacingRight=-1;"
                  f"points=[];portConstraint=eastwest;")
    cell = etree.SubElement(root, "mxCell")
    cell.set("id", drawio_shared_functions.id_generator())
    cell.set("value", "")
    cell.set("style", style_line)
    cell.set("parent", "1")
    cell.set("vertex", "1")
    geo = etree.SubElement(cell, "mxGeometry")
    geo.set("x", str(sidebar_w))
    geo.set("y", str(grid_bottom + 4))
    geo.set("width", str(total_w - sidebar_w + 20))
    geo.set("height", str(1))
    geo.set("as", "geometry")

    # bottom labels
    cat_bar_w = layout.get("cat_bar_width", 120)
    bar_h = 25
    bar_y = grid_bottom + 8
    label_y = bar_y

    # "Categories" box — horizontal bar matching sidebar style
    style_cat_box = (f"rounded=0;whiteSpace=wrap;html=1;"
                     f"fillColor={bar_color};strokeColor=none;")
    cell = etree.SubElement(root, "mxCell")
    cell.set("id", drawio_shared_functions.id_generator())
    cell.set("value", "")
    cell.set("style", style_cat_box)
    cell.set("parent", "1")
    cell.set("vertex", "1")
    geo = etree.SubElement(cell, "mxGeometry")
    geo.set("x", "0")
    geo.set("y", str(bar_y))
    geo.set("width", str(cat_bar_w))
    geo.set("height", str(bar_h))
    geo.set("as", "geometry")

    # "Categories" text centred in the box
    style_cat_label = (f"text;html=1;strokeColor=none;fillColor=none;"
                       f"align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;"
                       f"fontColor=#FFFFFF;fontSize=14;fontStyle=1;fontFamily={font};")
    cell = etree.SubElement(root, "mxCell")
    cell.set("id", drawio_shared_functions.id_generator())
    cell.set("value", meta["x_label"])
    cell.set("style", style_cat_label)
    cell.set("parent", "1")
    cell.set("vertex", "1")
    geo = etree.SubElement(cell, "mxGeometry")
    geo.set("x", "0")
    geo.set("y", str(bar_y))
    geo.set("width", str(cat_bar_w))
    geo.set("height", str(bar_h))
    geo.set("as", "geometry")

    # category names centered under each column (bold)
    for cl in cat_labels:
        style_cat = (f"text;html=1;strokeColor=none;fillColor=none;"
                     f"align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;"
                     f"fontColor=#333333;fontSize=11;fontStyle=1;"
                     f"fontFamily={font};")
        cell = etree.SubElement(root, "mxCell")
        cell.set("id", drawio_shared_functions.id_generator())
        cell.set("value", cl["name"])
        cell.set("style", style_cat)
        cell.set("parent", "1")
        cell.set("vertex", "1")
        geo = etree.SubElement(cell, "mxGeometry")
        geo.set("x", str(cl["x"]))
        geo.set("y", str(label_y))
        geo.set("width", str(cl["width"]))
        geo.set("height", str(25))
        geo.set("as", "geometry")

    # encode and write
    drawio_shared_functions.finish(mxGraphModel, output_path)
    print(f"Draw.io written to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Periodic Table of Data Domains")
    parser.add_argument("--yaml", default="data.yaml", help="Path to YAML config")
    parser.add_argument("--output-dir", default=".", help="Output directory")
    parser.add_argument("--scheme", default=None, help="Colour scheme name")
    args = parser.parse_args()

    yaml_path = args.yaml
    if not os.path.isabs(yaml_path):
        yaml_path = os.path.join(os.path.dirname(__file__), yaml_path)

    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.path.dirname(__file__), output_dir)
    os.makedirs(output_dir, exist_ok=True)

    config = load_yaml(yaml_path)
    if args.scheme:
        config["default_scheme"] = args.scheme
    tiles, cat_labels = compute_positions(config)

    stem = config["meta"].get("stem", "data-domains-overview")
    render_svg(tiles, cat_labels, config,
               os.path.join(output_dir, f"{stem}.svg"))
    render_drawio(tiles, cat_labels, config,
                  os.path.join(output_dir, f"{stem}.drawio"))


if __name__ == "__main__":
    main()
