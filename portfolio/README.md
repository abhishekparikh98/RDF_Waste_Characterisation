# Portfolio — Multi-Modal Waste Characterisation for RDF Production

Academic portfolio for the MSc dissertation project. It is a static React + Vite + TypeScript + Tailwind CSS site that reuses the actual project files:

- `src/content/reports/*.md` — the verbatim markdown reports from `reports/`
- `public/figures/*.png` — every 300 DPI figure from `reports/figures/` and `results/`
- `src/data/*.ts` — typed mirrors of the project facts (datasets, models, metrics, timeline, architecture)

The machine learning project at the repository root is not modified.

## Quick start

```bash
cd portfolio
npm install
npm run dev      # http://127.0.0.1:5173
npm run build    # production build in dist/
npm run preview  # preview the production build
```

## Tech stack

- React 18 + Vite 5 + TypeScript 5
- Tailwind CSS 3
- Framer Motion 11
- React Router 6
- `react-markdown` with `remark-gfm` and `rehype-raw` for rendering the imported reports

## Structure

```
portfolio/
├── public/figures/                 # real PNGs from the project
├── src/
│   ├── components/
│   │   ├── layout/                 # Navbar, Sidebar, Footer, AppShell, PageHeader
│   │   └── ui/                     # Card, Badge, Metric, Figure, Section, Markdown
│   ├── content/reports/            # raw .md imported via ?raw
│   ├── data/                       # typed facts (project, datasets, results, ...)
│   ├── lib/                        # navigation
│   ├── pages/                      # 15 routes
│   ├── App.tsx
│   ├── main.tsx
│   └── index.css
├── index.html
├── tailwind.config.js
├── vite.config.ts
└── package.json
```

## Routes

| Path | Page |
|---|---|
| `/` | Home |
| `/about` | About Project |
| `/research-problem` | Research Problem |
| `/objectives` | Objectives |
| `/literature-review` | Literature Review |
| `/datasets` | Datasets |
| `/architecture` | Project Architecture |
| `/methodology` | Methodology |
| `/model-comparison` | Model Comparison |
| `/results` | Results |
| `/timeline` | Development Timeline |
| `/reports` | Reports (the actual .md files) |
| `/documentation` | Documentation index |
| `/github` | GitHub Repository |
| `/contact` | Contact |
