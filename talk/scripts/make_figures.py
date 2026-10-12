"""
make_figures.py — build every data figure in the MD_MLPES talk.

Each ``fig_*`` function reads committed result files (through talk_data.py),
draws one labelled matplotlib figure and writes it to ``talk/figures/`` as
SVG. Text is kept as real SVG text (not paths) so it stays sharp and uses
the slide font.

Usage
-----
    python3 talk/scripts/make_figures.py            # build all figures
    python3 talk/scripts/make_figures.py hot_emission expt_compare   # some

Style
-----
Colors are the validated three-slot categorical set (blue, orange, aqua);
every series is also direct-labelled or dashed so identity never rests on
color alone. Experiment is drawn in ink (near-black).
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

# The slide font is supplied by the web page; quiet matplotlib's fallback notes.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import talk_data as td  # noqa: E402

FIG_DIR = td.REPO_ROOT / "talk" / "figures"
O = td.OUTPUTS

# ------------------------------------------------------------------ style
INK = "#26313a"
MUTED = "#6b6b63"
GRID = "#d8d3c4"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SHADE = "#c9a24b"  # gold, used only for highlighted regions

plt.rcParams.update({
    "svg.fonttype": "none",
    "font.family": ["Source Sans 3", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 17,
    "axes.titlesize": 19,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 17,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "text.color": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "legend.fontsize": 15,
    "lines.linewidth": 2.0,
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.transparent": True,
})


def _save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.svg"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------------------------------------------------------ figures
def fig_descriptor_benchmark() -> Path:
    """Energy RMSE of MVKO KRR models by descriptor (April 2026 benchmark).

    Source: RESEARCH_LOG.md, "Descriptor Benchmark Summary" (no separate
    data file was committed for this benchmark).
    """
    rows = [  # (descriptor, dimension, RMSE kcal/mol)
        ("ACSF", 1188, 2.49),
        ("Pairwise distances", 66, 0.235),
        ("Coulomb matrix", 78, 0.152),
    ]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    y = np.arange(len(rows))
    vals = [r[2] for r in rows]
    ax.barh(y, vals, color=BLUE, height=0.55)
    for yi, (name, dim, v) in zip(y, rows):
        ax.text(v + 0.04, yi, f"{v:.3g} kcal/mol", va="center", fontsize=16)
    ax.set_yticks(y, [f"{r[0]}  (d = {r[1]})" for r in rows])
    ax.set_xlabel("Energy RMSE, MVKO (kcal/mol) — lower is better")
    ax.set_xlim(0, 2.95)
    ax.grid(axis="y", visible=False)
    ax.set_title("Global KRR works best with a compact descriptor")
    return _save(fig, "descriptor_benchmark")


def fig_mvko_training_data() -> Path:
    """Energy coverage of the 904-frame MVKO training set, by data source."""
    d = O / "mvko_20260319_081314"
    sources = [
        ("Normal-mode displacements", "nm_training_data.npz"),
        ("PSI4 MD, 300 K", "md_T300.npz"),
        ("PSI4 MD, 600 K", "md_T600.npz"),
        ("PSI4 MD, 1000 K", "md_T1000.npz"),
    ]
    ref = np.load(d / "combined_training_data.npz")["energies"].min()
    fig, ax = plt.subplots(figsize=(12, 4.6))
    rng = np.random.default_rng(0)
    for k, (label, fname) in enumerate(sources):
        e = td.relative_kcal(np.load(d / fname)["energies"], ref)
        e = np.clip(e, 0.05, None)
        yk = len(sources) - 1 - k
        ax.scatter(e, yk + rng.uniform(-0.28, 0.28, e.size), s=14,
                   color=BLUE, alpha=0.45, linewidths=0)
        med = np.median(e)
        ax.plot([med, med], [yk - 0.38, yk + 0.38], color=INK, lw=2.5)
        ax.text(560, yk, f"n = {e.size}\nmedian {med:.1f}", va="center",
                fontsize=14, color=MUTED)
    ax.set_yticks(range(len(sources))[::-1], [s[0] for s in sources])
    ax.set_xscale("log")
    ax.set_xlim(0.05, 1500)
    ax.set_xlabel("Energy above lowest training frame (kcal/mol, log scale)")
    ax.grid(axis="y", visible=False)
    ax.set_title("MVKO training set: 904 B3LYP/6-31G* frames "
                 "(black tick = median)")
    return _save(fig, "mvko_training_data")


def fig_ch2oo_ir() -> Path:
    """CH2OO IR spectrum from 20,000-step ML-MD dipole autocorrelation."""
    d = O / "ir_spectrum_20260318_232404"
    f, y = td.load_spectrum_csv(d / "ir_spectrum.csv")
    summ = td.load_json(d / "ir_summary.json")
    f, y = td.normalize_in_window(f, y, 150, 2400)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(f, y, color=BLUE, lw=1.6)
    labels = {803: "O–O str.", 1110: "C–O str."}
    for pk in summ["top_peaks_cm-1"]:
        fp = pk["frequency_cm-1"]
        yi = np.interp(fp, f, y)
        if 296 < fp < 300 or 457 < fp < 459:
            continue  # partner of the 307 / 461 doublets, labelled once
        txt = f"{fp:.0f}"
        for k, v in labels.items():
            if abs(fp - k) < 5:
                txt += f"\n{v}"
        ax.annotate(txt, (fp, yi), xytext=(0, 10), textcoords="offset points",
                    ha="center", va="bottom", fontsize=14)
    ax.set_xlim(150, 2400)
    ax.set_ylim(0, 1.25)
    ax.set_xlabel("Frequency (cm$^{-1}$)")
    ax.set_ylabel("Intensity (normalized)")
    ax.set_title("CH$_2$OO: ML-MD at 300 K → dipole ACF → IR "
                 "(10 ps, ZPE-floor initialization)")
    return _save(fig, "ch2oo_ir")


def fig_frequency_validation() -> Path:
    """Normal-mode frequencies: PSI4 reference vs. ML-PES backends (MVKO)."""
    ref = np.load(O / "anti_cis_nm_pes_20260513" / "normal_modes.npz")["freqs"]
    krr = td.parse_validation_report(
        O / "validate_anti_cis_nm_pes" / "validation_report.txt")
    mace = td.parse_validation_report(
        O / "validate_pes_20260418_101918" / "validation_report.txt")
    rows = [
        ("PSI4 reference\nωB97X-D/6-31G* Hessian", np.asarray(ref), INK),
        ("KRR in normal-mode\ncoordinates (anti-cis)", np.asarray(krr["freqs_cm1"]), BLUE),
        ("MACE\n(equivariant NN)", np.asarray(mace["freqs_cm1"]), ORANGE),
    ]
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.axvspan(5000, 12800, color=SHADE, alpha=0.16, lw=0)
    ax.text(8900, 2.62, "unphysical  (> 5000 cm$^{-1}$)", ha="center",
            fontsize=15, color=MUTED)
    ax.axvspan(-2400, 0, color=GRID, alpha=0.45, lw=0)
    ax.text(-1200, 2.62, "imaginary", ha="center", fontsize=15, color=MUTED)
    for k, (label, fr, col) in enumerate(rows):
        yk = len(rows) - 1 - k
        ax.vlines(fr, yk - 0.3, yk + 0.3, color=col, lw=2.2)
        n_imag = int((fr < 0).sum())
        n_bad = int((fr > 5000).sum())
        ax.text(12950, yk, f"max {fr.max():,.0f}\n{n_imag} imag · {n_bad} > 5000",
                va="center", fontsize=13.5, color=INK)
    ax.set_yticks(range(len(rows))[::-1], [r[0] for r in rows])
    ax.set_xlim(-2400, 12800)
    ax.set_ylim(-0.6, 2.9)
    ax.set_xlabel("Harmonic frequency of each of the 30 vibrational modes "
                  "(cm$^{-1}$; imaginary shown negative)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Kernel surfaces are too stiff; "
                 "MACE removes the > 5000 cm$^{-1}$ modes")
    return _save(fig, "frequency_validation")


CONFORMERS = [  # (label, CASSCF grid folder, active space)
    ("syn-trans", "casscf_wB97X_nm_grid_20260514_224806", "(2,2)"),
    ("syn-cis", "casscf_wB97X_nm_grid_20260519_132602", "(2,2)"),
    ("anti-trans", "casscf_wB97X_nm_grid_20260519_132839", "(4,4)"),
    ("anti-cis", "casscf_wB97X_nm_grid_20260407_184904", "(4,4)"),
]


def conformer_summaries() -> list[dict]:
    """Read the four CASSCF grid summary.json files (shared with the deck)."""
    out = []
    for label, folder, cas in CONFORMERS:
        s = td.load_json(O / folder / "summary.json")
        eq = s.get("eq_reference") or {}
        out.append({
            "label": label, "cas": cas, "n_clean": s["n_clean"],
            "s0": s["s0_loo_cv_kcal"], "s1": s["s1_loo_cv_kcal"],
            "t1": s["t1_loo_cv_kcal"],
            "gap_s1": eq.get("gap_s1_eq_kcal"),
            "gap_t1": eq.get("gap_t1_eq_kcal"),
        })
    return out


def fig_casscf_conformers() -> Path:
    """Multireference corrections: Δ-ML accuracy and S1/T1 gaps per conformer."""
    rows = conformer_summaries()
    # anti-cis summary.json lacks eq gaps for some versions; log values:
    log_gaps = {"syn-trans": (23.4, 20.7), "syn-cis": (21.6, 19.0),
                "anti-trans": (27.3, 24.8), "anti-cis": (29.0, 26.5)}
    x = np.arange(len(rows))
    w = 0.26
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14.5, 5.2),
                                 gridspec_kw={"wspace": 0.28})
    for j, (key, name, col) in enumerate(
            [("s0", "δ S$_0$", BLUE), ("s1", "S$_1$ gap", ORANGE),
             ("t1", "T$_1$ gap", AQUA)]):
        vals = [r[key] for r in rows]
        a1.bar(x + (j - 1) * w, vals, w * 0.92, color=col, label=name)
    a1.set_yscale("log")
    a1.set_ylim(0.02, 4)
    a1.set_xticks(x, [f"{r['label']}\nCAS{r['cas']}" for r in rows])
    a1.set_ylabel("Leave-one-out CV error (kcal/mol)")
    a1.set_title("NM-coordinate KRR Δ-models")
    a1.legend(loc="upper left", ncol=3, fontsize=14)
    a1.grid(axis="x", visible=False)
    at = next(i for i, r in enumerate(rows) if r["label"] == "anti-trans")
    a1.annotate("2.23 before\noutlier filter", (at - w, rows[at]["s0"]),
                xytext=(at - 0.9, 1.25), fontsize=13, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    for j, (idx, name, col) in enumerate(
            [(0, "S$_1$", ORANGE), (1, "T$_1$", AQUA)]):
        vals = [r["gap_s1" if idx == 0 else "gap_t1"] or log_gaps[r["label"]][idx]
                for r in rows]
        bars = a2.bar(x + (j - 0.5) * 0.36, vals, 0.33, color=col, label=name)
        for b, v in zip(bars, vals):
            a2.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}",
                    ha="center", fontsize=13)
    a2.set_xticks(x, [r["label"] for r in rows])
    a2.set_ylabel("Vertical gap at equilibrium (kcal/mol)")
    a2.set_ylim(0, 34)
    a2.set_title("CASSCF excitation gaps")
    a2.legend(loc="upper left", ncol=2, fontsize=14)
    a2.grid(axis="x", visible=False)
    return _save(fig, "casscf_conformers")


def fig_acf_vs_harmonic() -> Path:
    """Why ACF fails for MVKO: torsion-dominated ACF vs harmonic intensities."""
    fa, ya = td.load_spectrum_csv(O / "ir_mace_anti_cis_300K" / "ir_spectrum.csv")
    fh, yh = td.load_spectrum_csv(
        O / "ir_nm_dipole_harmonic_anti_cis" / "ir_harmonic_spectrum.csv")
    fa, ya = td.normalize_in_window(fa, ya, 50, 3500)
    fh, yh = td.normalize_in_window(fh, yh, 50, 3500)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12.5, 6.2), sharex=True,
                                 gridspec_kw={"hspace": 0.32})
    for ax in (a1, a2):
        ax.axvspan(100, 500, color=SHADE, alpha=0.16, lw=0)
        ax.set_ylim(0, 1.15)
    a1.plot(fa, ya, color=BLUE, lw=1.4)
    a1.set_title("Dipole autocorrelation, 5 × 30,000-step ML-MD at 300 K",
                 fontsize=17)
    a1.text(520, 0.9, "torsions reorient the 4.5 D\npermanent dipole",
            fontsize=14, color=MUTED, va="top")
    a2.plot(fh, yh, color=ORANGE, lw=1.6)
    a2.set_title("Harmonic dipole derivatives |∂μ/∂q$_k$|$^2$ "
                 "(local harmonic approximation)", fontsize=17)
    a2.text(1900, 0.9, "fingerprint bands recovered;\n"
            "torsional region ≈ 0.14", fontsize=14, color=MUTED, va="top")
    a2.set_xlim(50, 3500)
    a2.set_xlabel("Frequency (cm$^{-1}$)")
    for ax in (a1, a2):
        ax.set_ylabel("Intensity")
    fig.suptitle("anti-cis MVKO — same molecule, two routes to the IR spectrum",
                 x=0.125, ha="left", y=1.0, fontsize=19, fontweight="bold")
    return _save(fig, "acf_vs_harmonic")


HARMONIC_TABLE = [  # anti-cis MVKO, RESEARCH_LOG.md Phase 13
    (835.2, 0.041, "O–O"),
    (1011.2, 0.827, "C–O"),
    (1088.8, 1.000, "C–O"),
    (1130.8, 0.513, "C–O"),
    (1291.7, 0.399, "C=C"),
    (1480.4, 0.985, "CH$_2$ wag / C=C"),
    (1692.6, 0.550, "C=C"),
    (3201.3, 0.283, "C–H"),
]


def fig_harmonic_ir() -> Path:
    """anti-cis MVKO harmonic IR spectrum with band assignments."""
    f, y = td.load_spectrum_csv(
        O / "ir_nm_dipole_harmonic_anti_cis" / "ir_harmonic_spectrum.csv")
    fig, (a1, a2) = plt.subplots(
        1, 2, figsize=(13.5, 5.2), sharey=True,
        gridspec_kw={"width_ratios": [4.2, 1], "wspace": 0.05})
    for ax, (lo, hi) in ((a1, (700, 1800)), (a2, (3000, 3400))):
        m = (f >= lo) & (f <= hi)
        ax.plot(f[m], y[m], color=ORANGE, lw=1.8)
        ax.vlines([r[0] for r in HARMONIC_TABLE if lo <= r[0] <= hi], 0,
                  [r[1] for r in HARMONIC_TABLE if lo <= r[0] <= hi],
                  color=INK, lw=1.2, alpha=0.6)
        ax.set_xlim(lo, hi)
    a1.set_ylim(0, 1.32)
    offsets = {1011.2: (-22, 8), 1088.8: (0, 26), 1130.8: (24, 8)}
    for fr, inten, lab in HARMONIC_TABLE:
        ax = a1 if fr < 2000 else a2
        dx, dy = offsets.get(fr, (0, 8))
        top = max(inten, float(np.interp(fr, f, y)))
        ax.annotate(f"{fr:.0f}\n{lab}", (fr, top), xytext=(dx, dy),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=13.5)
    a1.set_xticks(range(800, 1801, 200)[:-1])
    a2.set_xticks([3100, 3300])
    a2.spines["left"].set_visible(False)
    a2.tick_params(axis="y", left=False)
    a1.set_ylabel("Intensity (normalized)")
    fig.supxlabel("Frequency (cm$^{-1}$)", y=0.0, fontsize=17)
    a1.set_title("anti-cis MVKO harmonic IR — curve: 10 cm$^{-1}$ Lorentzian; "
                 "sticks: mode intensities")
    return _save(fig, "harmonic_ir")


def fig_hot_emission() -> Path:
    """Thermally sampled local-harmonic IR at 300 K vs 2000 K, two conformers."""
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    for ax, conf in zip(axes, ("anti_cis", "syn_trans")):
        for T, col, ls in (("300K", BLUE, "--"), ("2000K", ORANGE, "-")):
            path = O / f"hot_emission_{conf}_{T}" / "ir_hot_emission.csv"
            meta = td.read_hot_emission_header(path)
            f, y = td.load_spectrum_csv(path)
            f, y = td.normalize_in_window(f, y, 700, 1900)
            ax.plot(f, y, color=col, ls=ls, lw=2,
                    label=(f"{T.replace('K', ' K')}  ({meta['n_samples']} "
                           f"snapshots, {meta['fwhm_cm1']:.0f} cm$^{{-1}}$)"))
        ax.set_xlim(700, 1900)
        ax.set_ylim(0, 1.12)
        ax.set_title(conf.replace("_", "-") + " MVKO")
        ax.set_xlabel("Frequency (cm$^{-1}$)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1,
                  fontsize=14)
    axes[0].set_ylabel("Intensity (normalized, 700–1900 cm$^{-1}$)")
    fig.suptitle("Thermally sampled spectra: MACE-MD snapshots, local "
                 "harmonic intensities at each snapshot",
                 x=0.125, ha="left", y=1.03, fontsize=18, fontweight="bold")
    return _save(fig, "hot_emission")


CHUNG_BANDS = {"B$_7$": 932, "B$_6$": 970, "B$_5$": 1013, "B$_4$": 1133,
               "B$_3$": 1405, "B$_2$": 1432, "B$_1$": 1457}


def fig_expt_compare() -> Path:
    """Theory (anti-cis, 300 K & 2000 K) vs digitized Chung & Lee 2021."""
    log_html = td.REPO_ROOT / "RESEARCH_LOG.html"
    ex = np.array(td.extract_js_const(log_html, "CHUNG_EXPT"))
    gray = td.extract_js_const(log_html, "CHUNG_GRAY")
    fe, ye = ex[:, 0], ex[:, 1] / ex[:, 1].max()
    fig, ax = plt.subplots(figsize=(13.5, 5.6))
    for lo, hi in gray:
        ax.axvspan(lo, hi, color=GRID, alpha=0.55, lw=0)
        ax.text((lo + hi) / 2, 1.2, "precursor\nmasked", ha="center",
                fontsize=12.5, color=MUTED)
    ax.plot(fe, ye, color=INK, lw=1.8, label="Experiment: Chung & Lee 2021")
    for T, col, ls in (("300K", BLUE, "--"), ("2000K", ORANGE, "-")):
        f, y = td.load_spectrum_csv(
            O / f"hot_emission_anti_cis_{T}" / "ir_hot_emission.csv")
        f, y = td.normalize_in_window(f, y, 850, 1500)
        ax.plot(f, y, color=col, ls=ls, lw=2.2,
                label=f"This work, anti-cis {T.replace('K', ' K')}")
    for lab, fr in CHUNG_BANDS.items():
        ax.text(fr, 1.08, lab, ha="center", fontsize=14, color=INK)
    ax.set_xlim(850, 1500)
    ax.set_ylim(-0.08, 1.32)
    ax.set_xlabel("Frequency (cm$^{-1}$)")
    ax.set_ylabel("Intensity (normalized, 850–1500 cm$^{-1}$)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3,
              fontsize=14)
    ax.set_title("MVKO fingerprint region: theory vs. step-scan FTIR "
                 "(Chung & Lee, Commun. Chem. 2021)")
    return _save(fig, "expt_compare")


def fig_conformers() -> Path:
    """Convert the committed conformer figure (PDF) to SVG for the deck."""
    src = O / "mvko_conformers_figure.pdf"
    dst = FIG_DIR / "mvko_conformers.svg"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(["pdftocairo", "-svg", str(src), str(dst)], check=True)
    return dst


FIGURES = {
    "descriptor_benchmark": fig_descriptor_benchmark,
    "mvko_training_data": fig_mvko_training_data,
    "ch2oo_ir": fig_ch2oo_ir,
    "frequency_validation": fig_frequency_validation,
    "casscf_conformers": fig_casscf_conformers,
    "acf_vs_harmonic": fig_acf_vs_harmonic,
    "harmonic_ir": fig_harmonic_ir,
    "hot_emission": fig_hot_emission,
    "expt_compare": fig_expt_compare,
    "conformers": fig_conformers,
}


def main(argv: list[str]) -> None:
    names = argv or list(FIGURES)
    unknown = [n for n in names if n not in FIGURES]
    if unknown:
        raise SystemExit(f"Unknown figure(s): {unknown}. "
                         f"Choose from {list(FIGURES)}")
    for n in names:
        print(f"  {n:22s} -> {FIGURES[n]().relative_to(td.REPO_ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1:])
