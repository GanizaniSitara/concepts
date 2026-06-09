"""Reusable Safari/O'Reilly-style draw.io figure generator.

Single source of truth for a clean, monochrome technical-book diagram style
(heavy gray boxes, dashed bounded-domain containers with white data-product
sub-boxes, chunky block arrows, condensed Arial Narrow labels, left-hand layer
labels, a full-width foundation band, and a side band).

Two ways to use it:

1. Declarative spec (the repeatable path) -- describe a LAYERED architecture
   figure as a dict / JSON and call ``layered_figure(spec)``. This is the
   "landscape diagram" idiom and covers most data/solution architecture asks.

       python safari_figure.py spec.json -o out.drawio --png out.png

2. Builder API -- for free-form flows, construct a ``SafariFigure`` and place
   boxes / arrows by coordinate.

The style values were tuned against real technical-book figures with an
automated "odd-one-out" style discriminator (render -> anonymise -> ask a fresh
model which tile isn't from a book, on style only -> apply the cited fixes ->
repeat until indistinguishable). Treat the ``*_STYLE`` constants below as the
canonical look. The target is a clean born-digital figure that *reads* like a
book diagram -- do not add grain / noise / paper texture.

Requires the draw.io / diagrams.net desktop CLI for PNG export. Point at it with
the ``DRAWIO_CLI`` environment variable, or rely on the default install path.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


# --------------------------------------------------------------------------- #
# Canonical style constants
# --------------------------------------------------------------------------- #
BOX_STYLE = (
    "rounded=1;arcSize=6;whiteSpace=wrap;html=1;strokeColor=#111111;"
    "strokeWidth=3.2;fillColor=#D2D2D2;fontColor=#111111;fontFamily=Arial Narrow;"
    "fontSize=20;align=center;verticalAlign=middle;spacing=5;"
)
DOMAIN_FRAME_DASHED = (
    "rounded=1;arcSize=4;whiteSpace=wrap;html=1;dashed=1;dashPattern=8 6;"
    "strokeColor=#111111;strokeWidth=2;fillColor=none;fontColor=#111111;"
    "fontFamily=Arial Narrow;fontStyle=1;fontSize=17;align=center;verticalAlign=top;spacingTop=8;"
)
DOMAIN_FRAME_SOLID = (
    "rounded=1;arcSize=4;whiteSpace=wrap;html=1;strokeColor=#111111;strokeWidth=3.2;"
    "fillColor=#D2D2D2;fontColor=#111111;fontFamily=Arial Narrow;fontStyle=1;"
    "fontSize=17;align=center;verticalAlign=top;spacingTop=8;"
)
PRODUCT_STYLE = (
    "rounded=1;arcSize=8;whiteSpace=wrap;html=1;strokeColor=#111111;"
    "strokeWidth=1.6;fillColor=#FFFFFF;fontColor=#111111;fontFamily=Arial Narrow;"
    "fontSize=13;align=center;verticalAlign=middle;spacing=2;"
)
BAND_STYLE = (
    "rounded=1;arcSize=5;whiteSpace=wrap;html=1;strokeColor=#111111;"
    "strokeWidth=2.6;fillColor=#ECECEC;fontColor=#111111;fontFamily=Arial Narrow;"
    "fontSize=19;align=center;verticalAlign=middle;spacing=6;"
)
SIDE_STYLE = (
    "rounded=1;arcSize=5;whiteSpace=wrap;html=1;strokeColor=#111111;"
    "strokeWidth=2.6;fillColor=#ECECEC;fontColor=#111111;fontFamily=Arial Narrow;"
    "fontSize=18;align=center;verticalAlign=top;spacingTop=12;spacingLeft=8;spacingRight=8;"
)
EDGE_STYLE = (
    "edgeStyle=orthogonalEdgeStyle;rounded=0;strokeColor=#111111;"
    "strokeWidth=3.8;endArrow=block;endFill=1;endSize=18;html=1;"
)
EDGE_DASHED_STYLE = (
    "edgeStyle=orthogonalEdgeStyle;rounded=0;strokeColor=#111111;dashed=1;dashPattern=6 5;"
    "strokeWidth=2.4;endArrow=open;endFill=0;endSize=14;html=1;"
)
LAYER_LABEL_STYLE = (
    "text;html=0;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;"
    "whiteSpace=wrap;fontColor=#111111;fontFamily=Arial;fontSize=16;fontStyle=1;"
)
EDGE_LABEL_STYLE = (
    "text;html=0;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;"
    "whiteSpace=wrap;fontColor=#111111;fontFamily=Arial Narrow;fontSize=18;fontStyle=2;"
)
TITLE_STYLE = (
    "text;html=0;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;"
    "whiteSpace=wrap;fontColor=#111111;fontFamily=Times New Roman;fontSize=15;fontStyle=1;"
)

# draw.io / diagrams.net desktop CLI. Override with the DRAWIO_CLI env var.
DRAWIO_EXE = os.environ.get("DRAWIO_CLI", r"C:\Program Files\draw.io\draw.io.exe")


def _fmt(text):
    """Spec authors use '\\n' for line breaks; draw.io html=1 wants <br/>."""
    return str(text).replace("\n", "<br/>")


# --------------------------------------------------------------------------- #
# Low-level builder
# --------------------------------------------------------------------------- #
class SafariFigure:
    def __init__(self, title=None, page_w=1160, page_h=820):
        self.mxfile = ET.Element(
            "mxfile",
            {"host": "app.diagrams.net", "version": "24.7.17"},
        )
        diagram = ET.SubElement(self.mxfile, "diagram", {"id": "safari-figure", "name": "Safari figure"})
        self.model = ET.SubElement(
            diagram,
            "mxGraphModel",
            {
                "dx": "1200", "dy": "820", "grid": "1", "gridSize": "10", "guides": "1",
                "tooltips": "1", "connect": "1", "arrows": "1", "fold": "1", "page": "1",
                "pageScale": "1", "pageWidth": str(page_w), "pageHeight": str(page_h),
                "math": "0", "shadow": "0",
            },
        )
        self.root = ET.SubElement(self.model, "root")
        self.root.append(ET.Element("mxCell", {"id": "0"}))
        self.root.append(ET.Element("mxCell", {"id": "1", "parent": "0"}))
        self._n = 0
        if title:
            self.text("title", title, 0, 30, page_w, 30, TITLE_STYLE)

    def _id(self, prefix):
        self._n += 1
        return f"{prefix}{self._n}"

    def vertex(self, value, x, y, w, h, style, id_=None):
        id_ = id_ or self._id("v")
        cell = ET.SubElement(
            self.root, "mxCell",
            {"id": id_, "value": _fmt(value), "style": style, "parent": "1", "vertex": "1"},
        )
        ET.SubElement(cell, "mxGeometry",
                      {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"})
        return id_

    # semantic aliases
    def box(self, value, x, y, w, h, id_=None):
        return self.vertex(value, x, y, w, h, BOX_STYLE, id_)

    def band(self, value, x, y, w, h, id_=None):
        return self.vertex(value, x, y, w, h, BAND_STYLE, id_)

    def side(self, value, x, y, w, h, id_=None):
        return self.vertex(value, x, y, w, h, SIDE_STYLE, id_)

    def layer_label(self, value, x, y, w, h, id_=None):
        return self.vertex(value, x, y, w, h, LAYER_LABEL_STYLE, id_)

    def edge_label(self, value, x, y, w, h, id_=None):
        return self.vertex(value, x, y, w, h, EDGE_LABEL_STYLE, id_)

    def text(self, id_, value, x, y, w, h, style):
        return self.vertex(value, x, y, w, h, style, id_)

    def domain(self, title, products, x, y, w, h, dashed=True, id_=None):
        """A bounded-context container with white data-product sub-boxes."""
        frame_style = DOMAIN_FRAME_DASHED if dashed else DOMAIN_FRAME_SOLID
        self.vertex(title, x, y, w, h, frame_style, id_)
        k = max(1, len(products))
        margin, gap, ph = 10, 11, 56
        inner = w - 2 * margin
        pw = (inner - (k - 1) * gap) // k
        py = y + 48
        for i, p in enumerate(products):
            px = x + margin + i * (pw + gap)
            self.vertex(p, px, py, pw, ph, PRODUCT_STYLE)

    def arrow(self, src, dst, points=None, style=EDGE_STYLE):
        cell = ET.SubElement(self.root, "mxCell",
                             {"id": self._id("e"), "value": "", "style": style, "parent": "1", "edge": "1"})
        geom = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
        ET.SubElement(geom, "mxPoint", {"x": str(src[0]), "y": str(src[1]), "as": "sourcePoint"})
        if points:
            arr = ET.SubElement(geom, "Array", {"as": "points"})
            for px, py in points:
                ET.SubElement(arr, "mxPoint", {"x": str(px), "y": str(py)})
        ET.SubElement(geom, "mxPoint", {"x": str(dst[0]), "y": str(dst[1]), "as": "targetPoint"})

    def write(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ET.ElementTree(self.mxfile).write(path, encoding="utf-8", xml_declaration=True)
        return path


# --------------------------------------------------------------------------- #
# Layout constants for the declarative layered builder
# --------------------------------------------------------------------------- #
LEFT_MARGIN, LEFT_LABEL_W, GAP_LL_GRID = 20, 150, 15
X0 = LEFT_MARGIN + LEFT_LABEL_W + GAP_LL_GRID  # grid left edge = 185
BOX_W, COL_PITCH = 175, 200
ROW_H, DOMAIN_ROW_H, ROW_GAP = 78, 130, 147
Y_TITLE, Y_TOP = 30, 95
SIDE_GAP, SIDE_W = 25, 150
FOUND_GAP, FOUND_H = 65, 70
RIGHT_MARGIN, BOTTOM_MARGIN = 25, 30


def layered_figure(spec):
    """Build a layered architecture figure from a declarative spec.

    spec keys:
      title (str)
      layers (list, top->bottom): each {label, cells, kind?, dashed?}
        - kind "boxes" (default): cells are strings (use '\\n' to wrap)
        - kind "domains": cells are {title, products:[...], dashed?}
      flow ("up" | "down" | "none"): column-aligned arrows between adjacent
        layers that have equal cell counts. Default "up".
      foundation (str): full-width band beneath the grid.
      side ({title, items:[...]}): side band spanning the flow rows.
      stage_labels ({"From->To": text}): italic label in the gap between the
        two named layers (matched on their label).
    """
    layers = spec["layers"]
    flow = spec.get("flow", "up")
    n_cols = max(len(l["cells"]) for l in layers)
    grid_w = (n_cols - 1) * COL_PITCH + BOX_W

    has_side = bool(spec.get("side"))
    has_found = bool(spec.get("foundation"))

    # stack rows top -> bottom
    rows = []
    y = Y_TOP
    for layer in layers:
        h = DOMAIN_ROW_H if layer.get("kind") == "domains" else ROW_H
        rows.append({"layer": layer, "y": y, "h": h})
        y += h + ROW_GAP
    last_bottom = rows[-1]["y"] + rows[-1]["h"]

    found_y = last_bottom + FOUND_GAP
    page_w = (X0 + grid_w + (SIDE_GAP + SIDE_W if has_side else 0)) + RIGHT_MARGIN
    page_h = (found_y + FOUND_H if has_found else last_bottom) + BOTTOM_MARGIN

    fig = SafariFigure(page_w=page_w, page_h=page_h)
    if spec.get("title"):
        fig.text("title", spec["title"], X0, Y_TITLE, grid_w, 30, TITLE_STYLE)

    def col_center(layer, i):
        n = len(layer["cells"])
        total = (n - 1) * COL_PITCH + BOX_W
        offset = (grid_w - total) // 2
        return X0 + offset + i * COL_PITCH + BOX_W // 2

    def col_left(layer, i):
        return col_center(layer, i) - BOX_W // 2

    # place rows
    for row in rows:
        layer, ry, rh = row["layer"], row["y"], row["h"]
        fig.layer_label(layer["label"], LEFT_MARGIN, ry, LEFT_LABEL_W, rh)
        for i, cell in enumerate(layer["cells"]):
            x = col_left(layer, i)
            if layer.get("kind") == "domains":
                fig.domain(cell["title"], cell.get("products", []), x, ry, BOX_W, rh,
                           dashed=cell.get("dashed", layer.get("dashed", True)))
            else:
                fig.box(cell, x, ry, BOX_W, rh)

    # arrows between adjacent flow layers with equal cell counts
    if flow in ("up", "down"):
        for upper, lower in zip(rows, rows[1:]):
            la, lb = upper["layer"], lower["layer"]
            if len(la["cells"]) != len(lb["cells"]):
                continue
            for i in range(len(la["cells"])):
                cx = col_center(la, i)
                if flow == "up":
                    fig.arrow((cx, lower["y"]), (cx, upper["y"] + upper["h"]))
                else:
                    fig.arrow((cx, upper["y"] + upper["h"]), (cx, lower["y"]))

    # foundation band
    if has_found:
        fig.band(spec["foundation"], X0, found_y, grid_w, FOUND_H)

    # side band spanning the flow rows
    if has_side:
        side = spec["side"]
        body = side["title"]
        if side.get("items"):
            body += "\n\n" + "\n".join(side["items"])
        sx = X0 + grid_w + SIDE_GAP
        fig.side(body, sx, rows[0]["y"], SIDE_W, last_bottom - rows[0]["y"])

    # stage labels in the gaps (left of column 0's arrow)
    label_by = {r["layer"]["label"]: r for r in rows}
    for key, text in spec.get("stage_labels", {}).items():
        if "->" not in key:
            continue
        a, b = (s.strip() for s in key.split("->", 1))
        ra, rb = label_by.get(a), label_by.get(b)
        if not ra or not rb:
            continue
        top, bot = (ra, rb) if ra["y"] < rb["y"] else (rb, ra)
        gap_mid = (top["y"] + top["h"] + bot["y"]) // 2
        cx0 = col_center(top["layer"], 0)
        fig.edge_label(text, cx0 - 132, gap_mid - 12, 120, 24)

    return fig


def render(drawio_path, png_path, scale=2):
    """Export a .drawio to PNG via the draw.io / diagrams.net desktop CLI."""
    cmd = [DRAWIO_EXE, "-x", "-f", "png", "-s", str(scale), "--no-sandbox",
           "-o", str(png_path), str(drawio_path)]
    subprocess.run(cmd, check=True)
    return Path(png_path)


def main():
    ap = argparse.ArgumentParser(description="Generate a clean book-style draw.io figure from a JSON spec.")
    ap.add_argument("spec", type=Path, help="Path to a layered-figure JSON spec")
    ap.add_argument("-o", "--output", type=Path, help="Output .drawio (default: spec name)")
    ap.add_argument("--png", type=Path, nargs="?", const=True,
                    help="Also export a PNG (optionally to a given path) via the draw.io CLI")
    args = ap.parse_args()

    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    out = args.output or args.spec.with_suffix(".drawio")
    fig = layered_figure(spec)
    fig.write(out)
    print(out)
    if args.png:
        png = args.png if isinstance(args.png, Path) else out.with_suffix(".png")
        render(out, png)
        print(png)


if __name__ == "__main__":
    main()
