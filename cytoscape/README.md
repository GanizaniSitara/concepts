# Application Landscape Visualiser

Interactive application landscape visualisation using [Cytoscape.js](https://js.cytoscape.org).

Displays a network graph of applications and their dependencies within a generic universal bank, colour-coded by business domain (transaction cycle). Click an app to see its neighbourhood in a concentric layout, or use the search panel to filter.

## Data

The dataset is **fully synthetic** — 300 fictional applications across 25 business domains with ~530 edges. No real organisation data is included.

## Tech stack

- **Cytoscape.js** — graph/network visualisation
- **Preact** — lightweight UI components
- **Webpack + Babel** — bundling and transpilation
- **PostCSS + CSSNext** — CSS processing

## Building

```bash
npm install
npm run watch    # dev build with live reload
npm run prod     # production build
npm run clean    # delete dist/
npm run lint     # run linters
```
