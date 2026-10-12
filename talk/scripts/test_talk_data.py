"""Unit tests for talk_data.py.

Run:  python3 -m pytest talk/scripts -v
Synthetic-input tests always run; tests that read committed result files
are skipped automatically if those files are absent.
"""

from pathlib import Path

import numpy as np
import pytest

import talk_data as td


# ---------------------------------------------------------------- synthetic
def test_load_spectrum_csv_skips_comments_and_header(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("# title\n# T = 300 K\nfrequency_cm-1,intensity\n"
                 "100.0,0.5\n200.0,1.0\n\n")
    f, y = td.load_spectrum_csv(p)
    assert f.tolist() == [100.0, 200.0]
    assert y.tolist() == [0.5, 1.0]


def test_load_spectrum_csv_empty_raises(tmp_path):
    p = tmp_path / "e.csv"
    p.write_text("# nothing here\n")
    with pytest.raises(ValueError):
        td.load_spectrum_csv(p)


REPORT = """Driver type  : MACE
Model        : outputs/x/mace_model.pt
  Energy RMSE   : 11.0363 kcal/mol
  Force  RMSE   : 0.00234 Ha/Å  (1.470 kcal/mol/Å)
  Imaginary modes  : 1
  Unphysical modes : 0  (>5000 cm⁻¹)
  Max frequency    : 4598 cm⁻¹
  mode  1:  -1998.6 IMAGINARY
  mode  2:    495.8
  mode  3:   5186.5 UNPHYSICAL
"""


def test_parse_validation_report(tmp_path):
    p = tmp_path / "validation_report.txt"
    p.write_text(REPORT, encoding="utf-8")
    r = td.parse_validation_report(p)
    assert r["driver"] == "MACE"
    assert r["model"] == "outputs/x/mace_model.pt"
    assert r["energy_rmse_kcal"] == pytest.approx(11.0363)
    assert r["force_rmse_ha_per_A"] == pytest.approx(0.00234)
    assert r["n_imaginary"] == 1 and r["n_unphysical"] == 0
    assert r["max_freq_cm1"] == pytest.approx(4598)
    assert r["freqs_cm1"] == [-1998.6, 495.8, 5186.5]


def test_read_hot_emission_header(tmp_path):
    p = tmp_path / "ir_hot_emission.csv"
    p.write_text("# Hot-molecule IR emission spectrum\n"
                 "# # Traj dir: outputs/run_A  MACE: m.pt\n"
                 "# # Stride: 300  N_samples: 136  FWHM: 30.0 cm-1\n"
                 "0.0,1.0\n")
    h = td.read_hot_emission_header(p)
    assert h == {"stride": 300, "n_samples": 136, "fwhm_cm1": 30.0,
                 "traj_dir": "outputs/run_A"}


def test_extract_js_const(tmp_path):
    p = tmp_path / "page.html"
    p.write_text('<script>\nconst A = [[1, 2.5], [3, 4]];\n'
                 'const B = {"x": [1, [2]]};\n</script>')
    assert td.extract_js_const(p, "A") == [[1, 2.5], [3, 4]]
    assert td.extract_js_const(p, "B") == {"x": [1, [2]]}
    with pytest.raises(KeyError):
        td.extract_js_const(p, "MISSING")


def test_find_peaks_picks_tallest_separated():
    f = np.arange(0, 1000, 1.0)
    y = (np.exp(-((f - 200) / 5) ** 2) + 0.5 * np.exp(-((f - 210) / 5) ** 2)
         + 0.8 * np.exp(-((f - 600) / 5) ** 2))
    peaks = td.find_peaks(f, y, n=2, min_sep_cm1=50)
    assert [round(p[0]) for p in peaks] == [200, 600]


def test_find_peaks_window():
    f = np.arange(0, 1000, 1.0)
    y = np.exp(-((f - 200) / 5) ** 2) + 0.3 * np.exp(-((f - 700) / 5) ** 2)
    assert round(td.find_peaks(f, y, n=1, fmin=500)[0][0]) == 700


def test_normalize_in_window():
    f = np.array([0, 100, 200, 300.0])
    y = np.array([9, 1, 2, 9.0])
    fw, yw = td.normalize_in_window(f, y, 50, 250)
    assert fw.tolist() == [100, 200] and yw.tolist() == [0.5, 1.0]


def test_relative_kcal_uses_project_constant():
    assert td.relative_kcal([-1.0], -2.0)[0] == pytest.approx(627.509474)


@pytest.mark.parametrize("n,dim", [(1, 1), (5, 15), (12, 78)])
def test_coulomb_descriptor_dim(n, dim):
    assert td.coulomb_descriptor_dim(n) == dim


# --------------------------------------------------------- real repo files
HARM = td.OUTPUTS / "ir_nm_dipole_harmonic_anti_cis" / "ir_harmonic_spectrum.csv"
LOG_HTML = td.REPO_ROOT / "RESEARCH_LOG.html"


@pytest.mark.skipif(not HARM.exists(), reason="result file not committed")
def test_anti_cis_harmonic_peaks_match_research_log():
    """Broadened spectrum shows the logged C-O, CH2-wag and C=C bands.

    (The ~1012 cm^-1 band is tallest after 10 cm^-1 broadening because the
    1011.2 and 1016.0 cm^-1 modes overlap; the strongest single mode in the
    log table is 1088.8 cm^-1.)
    """
    f, y = td.load_spectrum_csv(HARM)
    found = [p[0] for p in td.find_peaks(f, y, n=10, min_sep_cm1=15)]
    for logged in (1088.8, 1130.8, 1291.7, 1480.4, 1692.6, 3201.3):
        assert min(abs(x - logged) for x in found) <= 2


@pytest.mark.skipif(not LOG_HTML.exists(), reason="RESEARCH_LOG.html absent")
def test_digitized_experiment_is_fingerprint_region():
    data = np.array(td.extract_js_const(LOG_HTML, "CHUNG_EXPT"))
    assert data.shape[1] == 2
    assert 800 <= data[:, 0].min() and data[:, 0].max() <= 1600
