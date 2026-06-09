---
name: drawio-safari-figures
description: >-
  Generate clean, monochrome technical-book-style (O'Reilly/Manning) architecture
  diagrams as draw.io / diagrams.net files from a small JSON spec. Use when a user
  asks for a "book style" / textbook / grayscale architecture figure, a data-mesh /
  data-domain / landscape diagram, or wants consistent styling across diagrams.
---

# drawio-safari-figures

This skill turns a natural-language architecture description into a clean,
consistent book-style draw.io figure. You translate the request into a small
JSON **spec**, run the generator, and verify the rendered PNG.

The spec is the "language of the tool" — your job is to map the user's
architecture into it. Do not hand-author draw.io XML; do not invent new styles.

## When to use

- The user wants an architecture / data / solution diagram in a clean
  monochrome textbook style (grayscale boxes, dashed domain boundaries, chunky
  arrows, condensed labels).
- Layered or "landscape" shapes: sources → domains → consumers, ingestion →
  marts → consumption, producer/consumer planes, platform + governance layers.
- The user wants several diagrams that all look the same.

## Workflow

1. **Decompose** the request into stacked **layers** (top to bottom). Typical
   stacks: `Consumers / Data domains / Sources`, or
   `Consumption / Curated / Ingestion`. Identify any **bounded domains** (which
   own **data products**), a **foundation** layer (e.g. a platform), and a
   **side** concern (e.g. governance).
2. **Write a spec** as JSON (format below). Keep flow layers the same width
   (equal cell counts) so the arrows line up.
3. **Generate + render:**
   ```bash
   python safari_figure.py spec.json -o out.drawio --png out.png
   ```
   (Set the `DRAWIO_CLI` env var to your draw.io desktop CLI for PNG export.)
4. **Verify the PNG.** Open it and check for clipped/overlapping labels, arrows
   crossing text, or lopsided spacing. Fix the spec and re-run. Iterate a few
   times — issues only show up in the render. The target is a clean,
   born-digital figure that reads like a book diagram; do not add grain/noise.

## Spec cheat-sheet

```json
{
  "title": "Data domain architecture",
  "flow": "up",
  "layers": [
    {"label": "Consumers", "cells": ["Analytics\nand BI", "ML / data\nscience", "Regulatory\nreporting", "Data\nAPIs"]},
    {"label": "Data domains", "kind": "domains", "cells": [
        {"title": "Customer domain", "products": ["Profiles", "Segments"]}
    ]},
    {"label": "Sources", "cells": ["Core\nbanking", "CRM", "Payments\ngateway", "Market\ndata"]}
  ],
  "foundation": "Self-serve data platform  -  pipelines  ·  storage  ·  catalog  ·  compute",
  "side": {"title": "Federated\ngovernance", "items": ["Policies", "Lineage", "Access control"]},
  "stage_labels": {"Sources->Data domains": "ingest", "Data domains->Consumers": "publish data products"}
}
```

- `\n` wraps a label onto two lines.
- `"kind": "domains"` → dashed bounded-context boxes holding white data-product
  sub-boxes. Set `"dashed": false` (layer- or cell-level) for solid gray
  containers.
- `flow` (`"up"`/`"down"`/`"none"`) draws column-aligned arrows between adjacent
  layers **of equal cell count**.
- `foundation` = full-width band beneath the grid. `side` = band spanning the
  flow rows.

Full reference and worked examples: see `README.md` and `specs/`.

## Free-form figures

If the diagram isn't a clean layered stack (e.g. a state/flow graph), import the
builder API instead and place boxes/arrows by coordinate, reusing the canonical
style constants:

```python
from safari_figure import SafariFigure, BOX_STYLE, DOMAIN_FRAME_DASHED, EDGE_STYLE
```
