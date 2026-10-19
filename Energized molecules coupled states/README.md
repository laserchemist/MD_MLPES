# Bridging the middle zone

**Physics-framed matrices for coupled states of energized molecules, from H₃ and water toward the Criegee intermediate CH₂OO**

Jonathan M. Smith, Temple University

A React slide viewer for the talk. The 29 slides are 1920×1080 HTML, scaled to the browser window.

## Viewing

| Key | Action |
|---|---|
| → / Space / PageDown | next slide |
| ← / PageUp | previous slide |
| Home / End | first / last slide |
| N | speaker notes |
| O | overview of all slides |
| F | full screen |

You can also click the left or right edge of a slide, or swipe on a touch screen. Each slide has its own address, such as `#/4`. The **PDF** button opens the browser's print dialog with one slide per page; choose "Save as PDF".

## Run locally

```bash
npm install
npm run dev        # http://localhost:5173
npm run build      # static site in dist/
```

## Publish on GitHub Pages

1. Create a repository (for example `laserchemist/middle-zone-talk`) and push this folder to `main`.
2. In the repository's **Settings → Pages**, set **Source** to **GitHub Actions**.
3. Each push to `main` runs `.github/workflows/deploy.yml`, which builds the site and publishes it to `https://laserchemist.github.io/middle-zone-talk/`.

`vite.config.js` uses `base: './'`, so the site works under any repository name. You can also serve it from a subfolder of an existing Pages site by copying `dist/` there, for example `MD_MLPES/middle-zone/`.

## Layout

```
src/slides.json        slide markup, titles, sections and speaker notes (generated)
src/App.jsx            viewer: scaling, navigation, notes, overview, printing
src/styles.css         viewer chrome and the slide HTML defaults
public/figures/        figures used on the slides
tools/sync_slides.py   regenerates src/slides.json from a Slides deck export
```

To change a slide, edit its `html` in `src/slides.json`, or edit the deck and resync:

```bash
python3 tools/sync_slides.py path/to/deck/project   # folder with deck.json and slides/*.html
```

## Related

- Earlier work: Belinta Simiyu & Jonathan M. Smith, ACS Fall 2025 poster, *Machine-learned potential energy surfaces for hot Criegee intermediates*. [Talk](https://laserchemist.github.io/MD_MLPES/talk/) · [poster image](https://laserchemist.github.io/MD_MLPES/talk/assets/acs2025_poster.jpg) · [poster PDF](https://laserchemist.github.io/MD_MLPES/talk/assets/Simiyu_Smith_ACS_Fall2025_poster.pdf)
- CDNN: Y. Shu, Z. Varga, D. Zhang, Q. Meng, A. M. Parameswaran, J.-G. Zhou, D. G. Truhlar, *J. Chem. Theory Comput.* **21**, 3342 (2025).
