# color-scheme-1-sapphire-dusk (light scheme)

A tiny, framework-agnostic theme you can **link from any HTML** (plain pages, Flask/FastAPI templates, React, etc.).
It’s intentionally *simple*:

- `sapphire-dusk-tokens.css` → design tokens (CSS variables)
- `sapphire-dusk-base.css` → minimal base styles (typography, buttons, inputs, cards)
- `tokens.json` → same tokens in JSON (optional)
- `example.html` → quick preview

## Use (local)
```html
<link rel="stylesheet" href="sapphire-dusk-tokens.css">
<link rel="stylesheet" href="sapphire-dusk-base.css">
```

## Use (hosted)
Host these files on a static host (e.g., GitHub Pages) and link them via URL:
```html
<link rel="stylesheet" href="https://YOUR_HOST/path/sapphire-dusk-tokens.css">
<link rel="stylesheet" href="https://YOUR_HOST/path/sapphire-dusk-base.css">
```

## Semantic variables you should use in apps
Prefer these (so you can remap palette later without touching app code):

- `--color-primary`
- `--color-accent`
- `--color-muted`
- `--color-success`
- `--color-warning`
- `--color-danger`

Example:
```css
.my-banner { background: var(--color-primary); color: white; }
```

## Notes
- Fonts: set to Arial-family to match the PPT typography shown in the screenshots.
- Colors: RGB values were transcribed from the PPT “Color reference” slide.

## Rendered semantic colours

| Token | Hex | Preview |
|---|---|---|
| `--color-primary` | `#022D5E` | ![](https://img.shields.io/badge/-%23022D5E-022D5E?style=flat-square) |
| `--color-accent` | `#005587` | ![](https://img.shields.io/badge/-%23005587-005587?style=flat-square) |
| `--color-muted` | `#53565A` | ![](https://img.shields.io/badge/-%2353565A-53565A?style=flat-square) |
| `--color-success` | `#009639` | ![](https://img.shields.io/badge/-%23009639-009639?style=flat-square) |
| `--color-warning` | `#FFD100` | ![](https://img.shields.io/badge/-%23FFD100-FFD100?style=flat-square) |
| `--color-danger` | `#D0002B` | ![](https://img.shields.io/badge/-%23D0002B-D0002B?style=flat-square) |

## Live demo (GitHub Pages)
- https://ganizanisitara.github.io/concepts/color-scheme-1-sapphire-dusk/example.html
