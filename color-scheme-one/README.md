# Corporate Lite Theme (neutral, CSS-only)

A tiny, framework-agnostic theme you can **link from any HTML** (plain pages, Flask/FastAPI templates, React, etc.).
It’s intentionally *simple*:

- `corporate-lite-tokens.css` → design tokens (CSS variables)
- `corporate-lite-base.css` → minimal base styles (typography, buttons, inputs, cards)
- `tokens.json` → same tokens in JSON (optional)
- `example.html` → quick preview

## Use (local)
```html
<link rel="stylesheet" href="corporate-lite-tokens.css">
<link rel="stylesheet" href="corporate-lite-base.css">
```

## Use (hosted)
Host these files on a static host (e.g., GitHub Pages) and link them via URL:
```html
<link rel="stylesheet" href="https://YOUR_HOST/path/corporate-lite-tokens.css">
<link rel="stylesheet" href="https://YOUR_HOST/path/corporate-lite-base.css">
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
