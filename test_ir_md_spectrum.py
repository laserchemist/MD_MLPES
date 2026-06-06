#!/usr/bin/env python3
"""
Tests for ir_md_spectrum.py

Covers pure-Python utility functions (no PSI4 or real model files needed) and
lightweight component integration tests with synthetic data.

Run:
    python3 -m pytest test_ir_md_spectrum.py -v
    python3 test_ir_md_spectrum.py
"""

import csv
import io
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

REPO_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'modules'))

from ir_md_spectrum import (
    _hill_formula,
    _unicode_subscripts,
    identify_molecule,
    save_trajectory_xyz,
    save_spectrum_csv,
    compute_ir_spectrum,
    _parse_monitor_bonds,
    PESFamilyDriver,
    DeltaMLPESDriver,
    EnergyDeltaDriver,
    train_dipole_surface,
    predict_trajectory_dipoles,
    HARTREE_TO_KCAL,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared test fixtures
# ─────────────────────────────────────────────────────────────────────────────

class _MockBaseDriver:
    """Constant-energy, zero-force driver stub — no model files needed."""
    symbols = ['C', 'H', 'H', 'O', 'O']
    n_atoms = 5
    masses = np.array([12.011, 1.008, 1.008, 15.999, 15.999])
    _has_analytic = False
    _base_energy = -189.5  # Ha

    def energy(self, coords):
        return self._base_energy

    def forces(self, coords, delta=None):
        return np.zeros((self.n_atoms, 3))

    def predict(self, symbols, coords):
        return self.energy(coords)


# ─────────────────────────────────────────────────────────────────────────────
# _hill_formula
# ─────────────────────────────────────────────────────────────────────────────

class TestHillFormula(unittest.TestCase):
    def test_water(self):
        self.assertEqual(_hill_formula(['H', 'O', 'H']), 'H2O')

    def test_ch2oo(self):
        self.assertEqual(_hill_formula(['C', 'H', 'H', 'O', 'O']), 'CH2O2')

    def test_methane(self):
        self.assertEqual(_hill_formula(['C', 'H', 'H', 'H', 'H']), 'CH4')

    def test_co2(self):
        self.assertEqual(_hill_formula(['C', 'O', 'O']), 'CO2')

    def test_single_carbon(self):
        self.assertEqual(_hill_formula(['C']), 'C')

    def test_no_carbon_alphabetical(self):
        self.assertEqual(_hill_formula(['O', 'N', 'O']), 'NO2')

    def test_ozone(self):
        self.assertEqual(_hill_formula(['O', 'O', 'O']), 'O3')

    def test_mvko_c4h6o2(self):
        symbols = ['C'] * 4 + ['H'] * 6 + ['O'] * 2
        self.assertEqual(_hill_formula(symbols), 'C4H6O2')

    def test_c_before_h_before_rest(self):
        # Verify Hill ordering: C first, H second, then alphabetical
        formula = _hill_formula(['N', 'H', 'H', 'H', 'C'])
        self.assertTrue(formula.startswith('C'))
        self.assertIn('H', formula)


# ─────────────────────────────────────────────────────────────────────────────
# _unicode_subscripts
# ─────────────────────────────────────────────────────────────────────────────

class TestUnicodeSubscripts(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(_unicode_subscripts('CH2O2'), 'CH₂O₂')

    def test_no_numbers(self):
        self.assertEqual(_unicode_subscripts('CO'), 'CO')

    def test_double_digit(self):
        self.assertEqual(_unicode_subscripts('C10H16'), 'C₁₀H₁₆')

    def test_single_digit_every_element(self):
        self.assertEqual(_unicode_subscripts('H2O'), 'H₂O')

    def test_all_digits_converted(self):
        result = _unicode_subscripts('C0H1O2N3')
        for digit, subscript in zip('0123', '₀₁₂₃'):
            self.assertNotIn(digit, result)
            self.assertIn(subscript, result)


# ─────────────────────────────────────────────────────────────────────────────
# identify_molecule
# ─────────────────────────────────────────────────────────────────────────────

class TestIdentifyMolecule(unittest.TestCase):
    def test_ch2oo_known(self):
        info = identify_molecule(['C', 'H', 'H', 'O', 'O'])
        self.assertEqual(info['hill'], 'CH2O2')
        self.assertIn('Criegee', info['name'])
        self.assertEqual(info['n_atoms'], 5)
        self.assertIn('₂', info['unicode'])

    def test_water(self):
        info = identify_molecule(['H', 'O', 'H'])
        self.assertEqual(info['hill'], 'H2O')
        self.assertIn('water', info['name'].lower())
        self.assertIn('–', info['label'])  # known molecule gets name appended

    def test_mvko(self):
        symbols = ['C'] * 4 + ['H'] * 6 + ['O'] * 2
        info = identify_molecule(symbols)
        self.assertEqual(info['hill'], 'C4H6O2')
        self.assertIn('MVKO', info['name'])

    def test_unknown_molecule_label_is_unicode(self):
        # CF4 is not in _MOLECULE_NAMES
        info = identify_molecule(['C', 'F', 'F', 'F', 'F'])
        self.assertEqual(info['name'], info['hill'])
        self.assertEqual(info['label'], info['unicode'])

    def test_n_atoms_correct(self):
        symbols = ['C', 'H', 'H', 'O', 'O']
        info = identify_molecule(symbols)
        self.assertEqual(info['n_atoms'], len(symbols))

    def test_optional_coords_ignored(self):
        # coords kwarg should not change the output
        coords = np.zeros((5, 3))
        info_with = identify_molecule(['C', 'H', 'H', 'O', 'O'], coords)
        info_without = identify_molecule(['C', 'H', 'H', 'O', 'O'])
        self.assertEqual(info_with['hill'], info_without['hill'])


# ─────────────────────────────────────────────────────────────────────────────
# save_trajectory_xyz
# ─────────────────────────────────────────────────────────────────────────────

class TestSaveTrajectoryXyz(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        rng = np.random.default_rng(0)
        self.symbols = ['C', 'H', 'H', 'O', 'O']
        n_frames, n_atoms = 3, 5
        self.coords = rng.standard_normal((n_frames, n_atoms, 3))
        self.times = np.array([0.0, 0.5, 1.0])
        self.energies = np.array([-189.5, -189.4, -189.3])

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _path(self, name='traj.xyz'):
        return str(Path(self.tmpdir) / name)

    def test_file_created(self):
        with redirect_stdout(io.StringIO()):
            save_trajectory_xyz(self.coords, self.symbols, self.times,
                                self.energies, self._path())
        self.assertTrue(Path(self._path()).exists())

    def test_line_count(self):
        # Each frame: 1 atom-count line + 1 comment line + n_atoms coord lines
        with redirect_stdout(io.StringIO()):
            save_trajectory_xyz(self.coords, self.symbols, self.times,
                                self.energies, self._path())
        with open(self._path()) as f:
            lines = f.readlines()
        n_atoms = len(self.symbols)
        n_frames = len(self.coords)
        self.assertEqual(len(lines), n_frames * (n_atoms + 2))

    def test_first_line_is_atom_count(self):
        with redirect_stdout(io.StringIO()):
            save_trajectory_xyz(self.coords, self.symbols, self.times,
                                self.energies, self._path())
        with open(self._path()) as f:
            first = f.readline().strip()
        self.assertEqual(int(first), len(self.symbols))

    def test_comment_line_contains_energy(self):
        with redirect_stdout(io.StringIO()):
            save_trajectory_xyz(self.coords, self.symbols, self.times,
                                self.energies, self._path())
        with open(self._path()) as f:
            lines = f.readlines()
        comment = lines[1]  # second line of first frame
        self.assertIn('energy=', comment)
        self.assertIn('Ha', comment)

    def test_mol_info_label_in_comment(self):
        from ir_md_spectrum import identify_molecule
        mol = identify_molecule(self.symbols)
        with redirect_stdout(io.StringIO()):
            save_trajectory_xyz(self.coords, self.symbols, self.times,
                                self.energies, self._path(), mol_info=mol)
        with open(self._path()) as f:
            lines = f.readlines()
        self.assertIn(mol['hill'], lines[1])

    def test_coord_values_written(self):
        with redirect_stdout(io.StringIO()):
            save_trajectory_xyz(self.coords, self.symbols, self.times,
                                self.energies, self._path())
        with open(self._path()) as f:
            lines = f.readlines()
        # First coord line of first frame (line index 2)
        parts = lines[2].split()
        self.assertEqual(parts[0], self.symbols[0])
        np.testing.assert_allclose(
            [float(parts[1]), float(parts[2]), float(parts[3])],
            self.coords[0, 0], rtol=1e-6,
        )


# ─────────────────────────────────────────────────────────────────────────────
# save_spectrum_csv
# ─────────────────────────────────────────────────────────────────────────────

class TestSaveSpectrumCsv(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        rng = np.random.default_rng(1)
        self.freqs = np.linspace(100.0, 4000.0, 200)
        self.intens = rng.random(200)
        self.intens /= self.intens.max()
        self.meta = {
            'molecule': 'CH2O2', 'temperature': 300,
            'n_steps': 10000, 'dt_eff_fs': 0.5, 'date': '2026-05-11',
        }

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _csv_path(self):
        return str(Path(self.tmpdir) / 'spectrum.csv')

    def test_file_created(self):
        with redirect_stdout(io.StringIO()):
            save_spectrum_csv(self.freqs, self.intens, self._csv_path(), self.meta)
        self.assertTrue(Path(self._csv_path()).exists())

    def test_data_row_count(self):
        with redirect_stdout(io.StringIO()):
            save_spectrum_csv(self.freqs, self.intens, self._csv_path(), self.meta)
        with open(self._csv_path()) as f:
            rows = list(csv.reader(f))
        # Two-element rows that don't start with '#': column header + data rows
        data_rows = [r for r in rows if len(r) == 2 and not r[0].startswith('#')]
        self.assertEqual(len(data_rows) - 1, len(self.freqs))  # minus column header

    def test_column_header(self):
        with redirect_stdout(io.StringIO()):
            save_spectrum_csv(self.freqs, self.intens, self._csv_path(), self.meta)
        with open(self._csv_path()) as f:
            rows = list(csv.reader(f))
        headers = [r for r in rows if len(r) == 2 and r[0] == 'frequency_cm-1']
        self.assertEqual(len(headers), 1)

    def test_frequency_values_round_trip(self):
        with redirect_stdout(io.StringIO()):
            save_spectrum_csv(self.freqs, self.intens, self._csv_path(), self.meta)
        with open(self._csv_path()) as f:
            rows = list(csv.reader(f))
        data_rows = [r for r in rows if len(r) == 2 and not r[0].startswith('#')
                     and r[0] != 'frequency_cm-1']
        csv_freqs = np.array([float(r[0]) for r in data_rows])
        np.testing.assert_allclose(csv_freqs, self.freqs, rtol=1e-4)

    def test_molecule_name_in_header(self):
        with redirect_stdout(io.StringIO()):
            save_spectrum_csv(self.freqs, self.intens, self._csv_path(), self.meta)
        with open(self._csv_path()) as f:
            content = f.read()
        self.assertIn('CH2O2', content)


# ─────────────────────────────────────────────────────────────────────────────
# _parse_monitor_bonds
# ─────────────────────────────────────────────────────────────────────────────

class TestParseMonitrBonds(unittest.TestCase):
    def test_none_returns_none(self):
        self.assertIsNone(_parse_monitor_bonds(None))

    def test_empty_string_returns_none(self):
        self.assertIsNone(_parse_monitor_bonds(''))

    def test_single_unlabeled_pair(self):
        self.assertEqual(_parse_monitor_bonds('0-1'), [(0, 1)])

    def test_multiple_unlabeled_pairs(self):
        self.assertEqual(_parse_monitor_bonds('0-1,2-9'), [(0, 1), (2, 9)])

    def test_labeled_pair(self):
        self.assertEqual(_parse_monitor_bonds('0-1:O1-O2'), [(0, 1, 'O1-O2')])

    def test_mixed_labeled_and_unlabeled(self):
        result = _parse_monitor_bonds('0-1,2-9:C=O')
        self.assertEqual(result, [(0, 1), (2, 9, 'C=O')])

    def test_indices_are_ints(self):
        result = _parse_monitor_bonds('3-11')
        self.assertIsInstance(result[0][0], int)
        self.assertIsInstance(result[0][1], int)
        self.assertEqual(result[0], (3, 11))


# ─────────────────────────────────────────────────────────────────────────────
# PESFamilyDriver
# ─────────────────────────────────────────────────────────────────────────────

class TestPESFamilyDriver(unittest.TestCase):
    def setUp(self):
        self.family = MagicMock()
        self.family.symbols = ['C', 'H', 'H', 'O', 'O']
        self.family.blend_energy.return_value = -189.5
        self.family.blend_width = 3.0
        with redirect_stdout(io.StringIO()):
            self.driver = PESFamilyDriver(self.family)

    def test_energy_delegates_to_family(self):
        coords = np.random.randn(5, 3)
        self.assertAlmostEqual(self.driver.energy(coords), -189.5)
        self.family.blend_energy.assert_called()

    def test_predict_matches_energy(self):
        coords = np.random.randn(5, 3)
        self.assertAlmostEqual(
            self.driver.predict(self.family.symbols, coords),
            self.driver.energy(coords),
        )

    def test_forces_shape(self):
        coords = np.random.randn(5, 3)
        f = self.driver.forces(coords)
        self.assertEqual(f.shape, (5, 3))

    def test_forces_zero_for_constant_energy(self):
        # Constant energy ⟹ FD forces = 0 everywhere
        coords = np.random.randn(5, 3)
        f = self.driver.forces(coords)
        np.testing.assert_allclose(f, 0.0, atol=1e-10)

    def test_has_analytic_false(self):
        self.assertFalse(self.driver._has_analytic)

    def test_symbols_propagated(self):
        self.assertEqual(self.driver.symbols, ['C', 'H', 'H', 'O', 'O'])


# ─────────────────────────────────────────────────────────────────────────────
# DeltaMLPESDriver
# ─────────────────────────────────────────────────────────────────────────────

class TestDeltaMLPESDriver(unittest.TestCase):
    DELTA_E = 0.01  # Ha

    def _make_driver(self, mock_trainer_cls):
        mock_delta = MagicMock()
        mock_delta.predict.return_value = self.DELTA_E
        mock_trainer_cls.load.return_value = mock_delta
        with redirect_stdout(io.StringIO()):
            return DeltaMLPESDriver(_MockBaseDriver(), 'fake_delta.pkl')

    @patch('ir_md_spectrum.MLPESTrainer')
    def test_energy_is_base_plus_delta(self, MockTrainer):
        driver = self._make_driver(MockTrainer)
        coords = np.random.randn(5, 3)
        expected = _MockBaseDriver._base_energy + self.DELTA_E
        self.assertAlmostEqual(driver.energy(coords), expected)

    @patch('ir_md_spectrum.MLPESTrainer')
    def test_forces_shape(self, MockTrainer):
        driver = self._make_driver(MockTrainer)
        coords = np.random.randn(5, 3)
        f = driver.forces(coords)
        self.assertEqual(f.shape, (5, 3))

    @patch('ir_md_spectrum.MLPESTrainer')
    def test_forces_zero_for_constant_delta(self, MockTrainer):
        # Constant base + constant delta ⟹ forces ≈ 0
        driver = self._make_driver(MockTrainer)
        coords = np.random.randn(5, 3)
        f = driver.forces(coords)
        np.testing.assert_allclose(f, 0.0, atol=1e-8)

    @patch('ir_md_spectrum.MLPESTrainer')
    def test_symbols_and_n_atoms_propagated(self, MockTrainer):
        driver = self._make_driver(MockTrainer)
        self.assertEqual(driver.symbols, _MockBaseDriver.symbols)
        self.assertEqual(driver.n_atoms, _MockBaseDriver.n_atoms)

    @patch('ir_md_spectrum.MLPESTrainer')
    def test_predict_equals_energy(self, MockTrainer):
        driver = self._make_driver(MockTrainer)
        coords = np.random.randn(5, 3)
        self.assertAlmostEqual(
            driver.predict(driver.symbols, coords),
            driver.energy(coords),
        )


# ─────────────────────────────────────────────────────────────────────────────
# EnergyDeltaDriver
# ─────────────────────────────────────────────────────────────────────────────

class TestEnergyDeltaDriver(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        delta_data = {
            'dE_b3lyp_kcal': [1.0, 5.0, 10.0, 20.0, 30.0],
            'delta_kcal':    [0.1, 0.3,  0.5,  0.8,  1.0],
            'E_base_min_Ha': _MockBaseDriver._base_energy,
        }
        self.delta_json = str(Path(self.tmpdir) / 'delta.json')
        with open(self.delta_json, 'w') as fh:
            json.dump(delta_data, fh)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _driver(self, base=None):
        with redirect_stdout(io.StringIO()):
            return EnergyDeltaDriver(base or _MockBaseDriver(), self.delta_json)

    def test_zero_correction_at_minimum(self):
        # base energy == E_base_min ⟹ dE = 0 ⟹ delta = spline(0) = 0
        driver = self._driver()
        coords = np.zeros((5, 3))
        self.assertAlmostEqual(driver.energy(coords), _MockBaseDriver._base_energy, places=8)

    def test_positive_correction_above_min(self):
        # Move base energy up by 5 kcal/mol → spline(5) ≈ 0.3 kcal/mol
        shift_ha = 5.0 / HARTREE_TO_KCAL

        class ShiftedBase(_MockBaseDriver):
            def energy(self, coords):
                return _MockBaseDriver._base_energy + shift_ha

        driver = self._driver(ShiftedBase())
        delta_ha = driver._delta_energy(np.zeros((5, 3)))
        expected_ha = 0.3 / HARTREE_TO_KCAL
        self.assertAlmostEqual(delta_ha, expected_ha, places=5)

    def test_forces_shape(self):
        coords = np.random.randn(5, 3)
        self.assertEqual(self._driver().forces(coords).shape, (5, 3))

    def test_clamping_above_max_dE(self):
        # dE well above max_dE_kcal should still return a finite correction
        large_shift = 100.0 / HARTREE_TO_KCAL

        class HighBase(_MockBaseDriver):
            def energy(self, coords):
                return _MockBaseDriver._base_energy + large_shift

        driver = self._driver(HighBase())
        delta = driver._delta_energy(np.zeros((5, 3)))
        self.assertTrue(np.isfinite(delta))

    def test_symbols_propagated(self):
        driver = self._driver()
        self.assertEqual(driver.symbols, _MockBaseDriver.symbols)


# ─────────────────────────────────────────────────────────────────────────────
# compute_ir_spectrum
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeIrSpectrum(unittest.TestCase):
    """Use a simple sinusoidal dipole trajectory; test shapes and sanity."""

    @staticmethod
    def _make_dipoles(n_frames=2000, freq_cm1=500.0, dt_fs=0.5, amplitude=1.5):
        # ω [rad/fs] = 2π × freq_cm⁻¹ × c [cm/fs]
        omega = 2 * np.pi * freq_cm1 * 3e-5
        t = np.arange(n_frames) * dt_fs
        mu_x = amplitude * np.sin(omega * t)
        return np.column_stack([mu_x, np.zeros(n_frames), np.zeros(n_frames)])

    def _spectrum(self, **kw):
        dipoles = self._make_dipoles()
        with redirect_stdout(io.StringIO()):
            return compute_ir_spectrum(
                dipoles, timestep_fs=0.5, save_every=1,
                temperature=300.0, **kw,
            )

    def test_output_is_five_tuple(self):
        result = self._spectrum()
        self.assertEqual(len(result), 5)

    def test_freq_and_intensity_same_shape(self):
        freqs, intens, _, _, _ = self._spectrum()
        self.assertEqual(freqs.shape, intens.shape)
        self.assertGreater(len(freqs), 0)

    def test_acf_arrays_same_length(self):
        _, _, acf_t, acf_v, _ = self._spectrum()
        self.assertEqual(len(acf_t), len(acf_v))
        self.assertGreater(len(acf_t), 0)

    def test_intensities_non_negative(self):
        _, intens, _, _, _ = self._spectrum()
        self.assertTrue(np.all(intens >= -1e-12))

    def test_intensities_finite(self):
        _, intens, _, _, _ = self._spectrum()
        self.assertTrue(np.all(np.isfinite(intens)))

    def test_frequency_range_bounded_by_max_freq(self):
        max_freq = 2000.0
        freqs, _, _, _, _ = self._spectrum(max_freq=max_freq)
        self.assertTrue(np.all(freqs <= max_freq + 1.0))

    def test_peaks_are_pairs(self):
        _, _, _, _, peaks = self._spectrum()
        for peak in peaks:
            self.assertEqual(len(peak), 2)
            self.assertIsInstance(float(peak[0]), float)
            self.assertIsInstance(float(peak[1]), float)

    def test_save_every_scales_effective_dt(self):
        # save_every=2 halves the Nyquist; max frequency in output should be lower
        dipoles = self._make_dipoles(n_frames=2000, dt_fs=0.5)
        with redirect_stdout(io.StringIO()):
            freqs_1, _, _, _, _ = compute_ir_spectrum(
                dipoles, timestep_fs=0.5, save_every=1, temperature=300.0)
            freqs_2, _, _, _, _ = compute_ir_spectrum(
                dipoles, timestep_fs=0.5, save_every=2, temperature=300.0)
        self.assertLess(freqs_2.max(), freqs_1.max() + 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# predict_trajectory_dipoles
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictTrajectoryDipoles(unittest.TestCase):
    def test_output_shape(self):
        mock_surface = MagicMock()
        mock_surface.predict.side_effect = lambda c: np.array([1.0, 2.0, 3.0])
        coords_traj = np.random.randn(10, 5, 3)
        with redirect_stdout(io.StringIO()):
            dipoles = predict_trajectory_dipoles(mock_surface, coords_traj)
        self.assertEqual(dipoles.shape, (10, 3))

    def test_predict_called_once_per_frame(self):
        mock_surface = MagicMock()
        mock_surface.predict.return_value = np.zeros(3)
        n_frames = 7
        coords_traj = np.random.randn(n_frames, 5, 3)
        with redirect_stdout(io.StringIO()):
            predict_trajectory_dipoles(mock_surface, coords_traj)
        self.assertEqual(mock_surface.predict.call_count, n_frames)

    def test_values_match_mock(self):
        fixed_dipole = np.array([1.5, -0.3, 2.1])
        mock_surface = MagicMock()
        mock_surface.predict.return_value = fixed_dipole.copy()
        coords_traj = np.random.randn(5, 5, 3)
        with redirect_stdout(io.StringIO()):
            dipoles = predict_trajectory_dipoles(mock_surface, coords_traj)
        np.testing.assert_allclose(dipoles[0], fixed_dipole)
        np.testing.assert_allclose(dipoles[-1], fixed_dipole)


# ─────────────────────────────────────────────────────────────────────────────
# train_dipole_surface (integration)
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainDipoleSurface(unittest.TestCase):
    """Integration: train DipoleSurface on synthetic CH₂OO-like data."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        rng = np.random.default_rng(42)
        n, n_atoms = 30, 5
        base = np.array([
            [0.0,  0.0, 0.0],
            [0.9,  0.8, 0.0],
            [-0.9, 0.8, 0.0],
            [0.0, -1.2, 0.0],
            [0.0, -2.4, 0.0],
        ])
        coords = base[None] + rng.standard_normal((n, n_atoms, 3)) * 0.05
        dipoles = rng.standard_normal((n, 3)) + np.array([0.0, 0.0, 3.5])

        self.npz = str(Path(self.tmpdir) / 'train.npz')
        np.savez(self.npz,
                 symbols=np.array(['C', 'H', 'H', 'O', 'O']),
                 coordinates=coords,
                 dipoles=dipoles)
        self.out_pkl = str(Path(self.tmpdir) / 'dipole_surface.pkl')

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_trains_and_pkl_saved(self):
        with redirect_stdout(io.StringIO()):
            train_dipole_surface(self.npz, self.out_pkl)
        self.assertTrue(Path(self.out_pkl).exists())

    def test_loaded_model_predicts_shape(self):
        from ir_spectroscopy import DipoleSurface
        with redirect_stdout(io.StringIO()):
            train_dipole_surface(self.npz, self.out_pkl)
        loaded = DipoleSurface.load(self.out_pkl)
        coords_test = np.random.randn(5, 3)
        pred = loaded.predict(coords_test)
        self.assertEqual(pred.shape, (3,))

    def test_raises_on_all_zero_dipoles(self):
        bad_npz = str(Path(self.tmpdir) / 'bad.npz')
        np.savez(bad_npz,
                 symbols=np.array(['C', 'H', 'H', 'O', 'O']),
                 coordinates=np.random.randn(5, 5, 3),
                 dipoles=np.zeros((5, 3)))
        with self.assertRaises(RuntimeError):
            with redirect_stdout(io.StringIO()):
                train_dipole_surface(bad_npz, str(Path(self.tmpdir) / 'bad.pkl'))

    def test_returns_dipole_surface_instance(self):
        from ir_spectroscopy import DipoleSurface
        with redirect_stdout(io.StringIO()):
            surface = train_dipole_surface(self.npz, self.out_pkl)
        self.assertIsInstance(surface, DipoleSurface)

    def test_train_rmse_in_metadata(self):
        with redirect_stdout(io.StringIO()):
            surface = train_dipole_surface(self.npz, self.out_pkl)
        self.assertIn('train_rmse', surface.metadata)
        self.assertGreater(surface.metadata['train_rmse'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
