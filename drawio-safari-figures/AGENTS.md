# Agent guide

For any coding agent (GitHub Copilot, Codex, Cursor, Claude Code, etc.) asked to
produce a book-style architecture diagram with this tool.

**Goal:** translate the user's architecture into a JSON **spec** (the tool's
input language), run the generator, and verify the rendered PNG. Do not
hand-author draw.io XML and do not invent styles — the look is fixed.

## Steps

1. **Decompose** the request into stacked **layers** (top → bottom), e.g.
   `Consumers / Data domains / Sources`. Mark any **bounded domains** (which own
   **data products**), a **foundation** layer (a platform), and a **side**
   concern (e.g. governance). Keep flow layers the same width (equal cell count)
   so arrows align.
2. **Write `spec.json`** — see the format in `README.md` and the worked examples
   in `specs/`. Key fields: `title`, `flow` (`up`/`down`/`none`), `layers`
   (`kind: "domains"` for dashed bounded boxes with `products`), `foundation`,
   `side`, `stage_labels`. Use `\n` to wrap a label.
3. **Generate + render:**
   ```bash
   python safari_figure.py spec.json -o out.drawio --png out.png
   ```
   Set `DRAWIO_CLI` to the draw.io / diagrams.net desktop CLI for PNG export.
4. **Verify the PNG.** Open it; check for clipped/overlapping labels, arrows
   through text, uneven spacing. Fix the spec and re-run. Iterate a few times —
   layout problems only appear in the render.

## Conventions

- Reuse the canonical `*_STYLE` constants in `safari_figure.py`; never re-derive
  stroke weights, fonts, or arrow sizes.
- The target is a clean, born-digital figure that *reads* like a technical-book
  diagram. Do not add grain, noise, or paper texture.
- For non-layered figures (state/flow graphs), use the `SafariFigure` builder
  API and place boxes/arrows by coordinate.

Claude Code users: the same guidance is packaged as a skill in `SKILL.md`.
