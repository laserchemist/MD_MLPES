# Criegee intermediates: ML surfaces benchmarked on vibrational spectroscopy

**Criegee intermediates investigated with new machine-learned multireference and Δ-ML surfaces, benchmarked on vibrational spectroscopy**

Jonathan M. Smith, Temple University · project status, 4 November 2026

A React slide viewer (same viewer as `../Energized_molecules`). The 28 slides are 1920×1080 HTML, scaled to the browser window.
The built site is in `slides/`, served at `https://laserchemist.github.io/MD_MLPES/Criegee_ML/slides/`.

## Viewing

| Key | Action |
|---|---|
| → / Space / PageDown | next slide |
| ← / PageUp | previous slide |
| Home / End | first / last slide |
| N | speaker notes (references are in the notes) |
| O | overview of all slides |
| F | full screen |

The **PDF** button prints one slide per page; choose "Save as PDF".

## Update the slides

```bash
python3 tools/sync_slides.py path/to/deck/project   # folder with deck.json and slides/*.html
npm install        # once
npm run build      # writes dist/
rm -rf slides && cp -r dist slides
git add . && git commit -m "Update Criegee_ML slides" && git push
```

New images in the deck need an entry in `BLOBS` in `tools/sync_slides.py` and the file in `public/figures/`.

## Layout

```
src/slides.json        slide markup, titles, sections and speaker notes (generated)
src/App.jsx            viewer: scaling, navigation, notes, overview, printing
src/styles.css         viewer chrome and the slide HTML defaults
public/figures/        figures used on the slides
slides/                the built site (committed; served by GitHub Pages)
tools/sync_slides.py   regenerates src/slides.json from a Slides deck export
```

## Key references

- B. N. Simiyu and J. M. Smith, poster, ACS Fall 2025 National Meeting.
- K. T. Kuwata, *J. Phys. Chem. A* **128**, 7331 (2024).
- Y.-T. Su et al., *Science* **340**, 174 (2013).
- T. L. Nguyen et al., *J. Phys. Chem. A* **119**, 5524 (2015).
