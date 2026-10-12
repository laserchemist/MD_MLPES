"""
talk_data.py — read MD_MLPES result files for the project talk.

Every number on a slide or in a figure comes through one of these loaders,
so each value can be traced back to a committed file under ``outputs/``.
The loaders only *read*; they never recompute physics.

Units follow the project conventions (see CLAUDE.md):
    energies   Hartree in files, reported here in kcal/mol where noted
    frequency  cm^-1
    dipole     Debye

Functions
---------
load_spectrum_csv      two-column spectrum CSV with '#' comment lines
parse_validation_report  validate_pes_frequencies.py text report -> dict
load_json              small JSON summary files
extract_js_const       pull a ``const NAME = <json>;`` literal out of an HTML page
find_peaks             simple local-maximum peak picker for labelling spectra
normalize_in_window    scale a spectrum to max = 1 inside a frequency window
relative_kcal          Hartree energies -> kcal/mol relative to a reference
coulomb_descriptor_dim length of the upper-triangle Coulomb-matrix descriptor
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

#: Repository root (talk/scripts/ -> repo)
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = REPO_ROOT / "outputs"

# Unit constants are defined once, in modules/direct_md.py (project rule).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from modules.direct_md import HARTREE_TO_KCAL  # noqa: E402


def load_spectrum_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a spectrum CSV written by the IR workflows.

    The files start with ``#`` comment lines and may contain one plain-text
    header row (e.g. ``frequency_cm-1,intensity_normalized``) before the data.

    Parameters
    ----------
    path : str or Path
        CSV file with two comma-separated numeric columns.

    Returns
    -------
    freq_cm1, intensity : np.ndarray
        Frequency in cm^-1 and intensity (as stored, usually normalized).
    """
    freqs, ints = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            try:
                f, i = float(parts[0]), float(parts[1])
            except (ValueError, IndexError):
                continue  # header row
            freqs.append(f)
            ints.append(i)
    if not freqs:
        raise ValueError(f"No numeric rows in {path}")
    return np.asarray(freqs), np.asarray(ints)


def read_hot_emission_header(path: str | Path) -> dict:
    """Read sampling metadata from an ``ir_hot_emission.csv`` comment header.

    Returns ``{"stride": int, "n_samples": int, "fwhm_cm1": float,
    "traj_dir": str}``; fields not present are ``None``.
    """
    head = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            head.append(line)
    text = "".join(head)

    def grab(pat, cast):
        m = re.search(pat, text)
        return cast(m.group(1)) if m else None

    return {
        "stride": grab(r"Stride:\s*(\d+)", int),
        "n_samples": grab(r"N_samples:\s*(\d+)", int),
        "fwhm_cm1": grab(r"FWHM:\s*([\d.]+)", float),
        "traj_dir": grab(r"Traj dir:\s*(\S+)", str),
    }


_REPORT_PATTERNS = {
    "driver": r"Driver type\s*:\s*(.+)",
    "model": r"Model\s*:\s*(\S+)",
    "energy_rmse_kcal": r"Energy RMSE\s*:\s*([-\d.eE+]+)\s*kcal/mol",
    "force_rmse_ha_per_A": r"Force\s+RMSE\s*:\s*([-\d.eE+]+)\s*Ha/",
    "n_imaginary": r"Imaginary(?: modes)?\s*:\s*(\d+)",
    "n_unphysical": r"Unphysical(?: modes)?\s*:\s*(\d+)",
    "max_freq_cm1": r"Max freq(?:uency)?\s*:\s*([-\d.]+)\s*cm",
}


def parse_validation_report(path: str | Path) -> dict:
    """Parse a ``validation_report.txt`` from validate_pes_frequencies.py.

    Returns a dict with keys ``driver``, ``model``, ``energy_rmse_kcal``,
    ``force_rmse_ha_per_A``, ``n_imaginary``, ``n_unphysical``,
    ``max_freq_cm1`` (missing fields are ``None``) and ``freqs_cm1`` — the
    list of vibrational frequencies (imaginary modes negative).
    """
    text = Path(path).read_text(encoding="utf-8")
    out: dict = {}
    for key, pat in _REPORT_PATTERNS.items():
        m = re.search(pat, text)
        if m is None:
            out[key] = None
        elif key in ("driver", "model"):
            out[key] = m.group(1).strip()
        elif key.startswith("n_"):
            out[key] = int(m.group(1))
        else:
            out[key] = float(m.group(1))
    out["freqs_cm1"] = [
        float(x) for x in re.findall(r"mode\s+\d+:\s+([-\d.]+)", text)
    ]
    return out


def load_json(path: str | Path) -> dict:
    """Read a JSON file into a dict."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def extract_js_const(html_path: str | Path, name: str):
    """Return the JSON value assigned by ``const NAME = ...;`` in an HTML file.

    Used to reuse the digitized Chung & Lee (2021) trace that lives in
    RESEARCH_LOG.html, so the talk and the log share one copy of the data.
    The literal must be valid JSON (arrays / objects of numbers and strings).
    """
    text = Path(html_path).read_text(encoding="utf-8")
    m = re.search(rf"const\s+{re.escape(name)}\s*=\s*", text)
    if m is None:
        raise KeyError(f"const {name} not found in {html_path}")
    start = m.end()
    opener = text[start]
    closer = {"[": "]", "{": "}"}.get(opener)
    if closer is None:
        raise ValueError(f"const {name} is not an array or object literal")
    depth = 0
    for j in range(start, len(text)):
        c = text[j]
        if c == opener:
            depth += 1
        elif c == closer:
            depth -= 1
            if depth == 0:
                return json.loads(text[start : j + 1])
    raise ValueError(f"Unbalanced literal for const {name}")


def find_peaks(
    freq: np.ndarray,
    inten: np.ndarray,
    n: int = 6,
    min_sep_cm1: float = 25.0,
    fmin: float | None = None,
    fmax: float | None = None,
) -> list[tuple[float, float]]:
    """Return the ``n`` tallest local maxima as (frequency, intensity) pairs.

    Peaks closer than ``min_sep_cm1`` to an already-chosen taller peak are
    skipped. Results are sorted by frequency.
    """
    freq = np.asarray(freq, float)
    inten = np.asarray(inten, float)
    mask = np.ones_like(freq, bool)
    if fmin is not None:
        mask &= freq >= fmin
    if fmax is not None:
        mask &= freq <= fmax
    idx = np.where(mask)[0]
    cand = [
        i for i in idx[1:-1]
        if inten[i] >= inten[i - 1] and inten[i] > inten[i + 1]
    ]
    cand.sort(key=lambda i: inten[i], reverse=True)
    chosen: list[int] = []
    for i in cand:
        if all(abs(freq[i] - freq[j]) >= min_sep_cm1 for j in chosen):
            chosen.append(i)
        if len(chosen) == n:
            break
    return sorted((float(freq[i]), float(inten[i])) for i in chosen)


def normalize_in_window(
    freq: np.ndarray, inten: np.ndarray, fmin: float, fmax: float
) -> tuple[np.ndarray, np.ndarray]:
    """Crop a spectrum to [fmin, fmax] and scale its maximum there to 1."""
    freq = np.asarray(freq, float)
    inten = np.asarray(inten, float)
    m = (freq >= fmin) & (freq <= fmax)
    f, y = freq[m], inten[m]
    peak = y.max()
    if peak <= 0:
        raise ValueError("Spectrum has no positive intensity in window")
    return f, y / peak


def relative_kcal(energies_ha, reference_ha: float) -> np.ndarray:
    """Convert Hartree energies to kcal/mol relative to ``reference_ha``."""
    return (np.asarray(energies_ha, float) - reference_ha) * HARTREE_TO_KCAL


def coulomb_descriptor_dim(n_atoms: int) -> int:
    """Length of the upper-triangle (incl. diagonal) Coulomb descriptor.

    5 atoms (CH2OO) -> 15, 12 atoms (MVKO) -> 78.
    """
    if n_atoms < 1:
        raise ValueError("n_atoms must be >= 1")
    return n_atoms * (n_atoms + 1) // 2
