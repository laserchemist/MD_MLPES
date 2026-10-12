# MD_MLPES project talk

A 30-slide reveal.js deck on the MD_MLPES project, served by GitHub Pages at
`https://laserchemist.github.io/MD_MLPES/talk/`. It opens with significance and
headline results for a general scientific audience, then moves into methods,
validation, spectra and software for research colleagues.

## Layout

```
talk/
  index.html          GENERATED, self-contained deck (figures inlined); this is what Pages serves
  deck_source.html    slide text and layout; edit this, then rebuild
  figures/*.svg       one SVG per data figure, drawn from committed outputs/
  scripts/
    talk_data.py        read-only loaders for outputs/ (CSV spectra, JSON summaries,
                        validation reports, digitized experiment in RESEARCH_LOG.html)
    make_figures.py     one fig_* function per figure
    build_deck.py       inlines figures/*.svg into deck_source.html -> index.html
    test_talk_data.py   unit tests for the loaders (+ checks against real outputs)
    test_build_deck.py  unit tests for SVG cleaning and inlining
```

## Rebuild

```bash
python3 talk/scripts/make_figures.py         # all figures (or name some: hot_emission expt_compare)
python3 talk/scripts/build_deck.py           # writes talk/index.html
python3 -m pytest talk/scripts -v            # 19 tests
```

Requirements: numpy, matplotlib, pytest; `pdftocairo` (poppler) for the conformer
figure. `talk_data.py` imports `HARTREE_TO_KCAL` from `modules/direct_md.py` so unit
constants stay defined in one place.

## Where each figure's data comes from

| Figure | Source in `outputs/` |
|---|---|
| descriptor_benchmark | RESEARCH_LOG.md, descriptor benchmark table (values in `make_figures.py`) |
| mvko_training_data | `mvko_20260319_081314/{nm_training_data,md_T300,md_T600,md_T1000}.npz` |
| ch2oo_ir | `ir_spectrum_20260318_232404/ir_spectrum.csv`, `ir_summary.json` |
| frequency_validation | `anti_cis_nm_pes_20260513/normal_modes.npz`, `validate_anti_cis_nm_pes/`, `validate_pes_20260418_101918/` |
| casscf_conformers | `casscf_wB97X_nm_grid_*/summary.json` (four conformers) |
| acf_vs_harmonic | `ir_mace_anti_cis_300K/ir_spectrum.csv`, `ir_nm_dipole_harmonic_anti_cis/ir_harmonic_spectrum.csv` |
| harmonic_ir | `ir_nm_dipole_harmonic_anti_cis/ir_harmonic_spectrum.csv` + RESEARCH_LOG Phase 13 table |
| hot_emission | `hot_emission_{anti_cis,syn_trans}_{300K,2000K}/ir_hot_emission.csv` |
| expt_compare | `hot_emission_anti_cis_*` + `CHUNG_EXPT` digitized trace in `RESEARCH_LOG.html` |
| mvko_conformers | `mvko_conformers_figure.pdf` (converted to SVG) |

## Adding a slide or figure

1. Add a `fig_<name>()` to `make_figures.py` that reads through `talk_data.py`, and
   register it in `FIGURES`.
2. Put `<!--FIG:<name>-->` inside a `<figure class="fig">` in `deck_source.html`.
3. Rebuild and run the tests.

## Navigating

Arrow keys or swipe; `F` fullscreen; `Esc` overview. Slide numbers are in the
lower right; links to the research log and guide are relative to the repo root.
