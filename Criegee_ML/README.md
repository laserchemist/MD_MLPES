# Criegee intermediates: ML surfaces benchmarked on vibrational spectroscopy

**Criegee intermediates investigated with new machine-learned multireference and Δ-ML surfaces, benchmarked on vibrational spectroscopy**

Jonathan M. Smith, Temple University · research update, November 2026

A React slide viewer (same viewer as `../Energized_molecules`). The 44 slides are 1920×1080 HTML, scaled to the browser window.
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
python3 tools/sync_slides.py deck/project           # deck.json and slides/*.html (a copy of the Slides deck is kept in deck/)
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

- R. Criegee, *Angew. Chem. Int. Ed. Engl.* **14**, 745 (1975) (mechanism of ozonolysis).
- D. Stone et al., *Phys. Chem. Chem. Phys.* **20**, 24940 (2018) (CH₂OO and CD₂OO decomposition kinetics).
- B. N. Simiyu and J. M. Smith, poster, ACS Fall 2025 National Meeting.
- K. T. Kuwata, *J. Phys. Chem. A* **128**, 7331 (2024).
- Y.-T. Su et al., *Science* **340**, 174 (2013).
- T. L. Nguyen et al., *J. Phys. Chem. A* **119**, 5524 (2015).
- J. Li et al., *J. Phys. Chem. Lett.* **5**, 2364 (2014) (full-dimensional VCI for CH₂OO).
- Y.-H. Huang et al., *J. Chem. Phys.* **142**, 214301 (2015); **145**, 044305 (2016).
- Y.-P. Chang et al., *J. Chem. Phys.* **146**, 244302 (2017); J. Li, *Chin. J. Chem. Phys.* **33**, 65 (2020) (CD₂OO VCI).
- M. Nakajima and Y. Endo, *J. Chem. Phys.* **139**, 101103 (2013); M. C. McCarthy et al., *J. Phys. Chem. Lett.* **4**, 4133 (2013).
- C. A. Chung and Y.-P. Lee, *Commun. Chem.* **4**, 8 (2021) (MVK-oxide IR).
