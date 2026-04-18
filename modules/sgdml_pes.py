#!/usr/bin/env python3
"""
sgdml_pes — sGDML-backed ML-PES module
=======================================
Provides:
  SGDMLModel   — wraps a trained sGDML model dict; save/load compatible
                 with the existing .pkl pipeline.
  SGDMLDriver  — bakken.py-compatible driver (same interface as MLPESDriver).
                 Forces are analytic (sGDML native); Hessian via FD on analytic forces.
  train_sgdml  — trains sGDML from our standard npz arrays; optional sig sweep.

Units (consistent with CLAUDE.md):
  coordinates  Angstrom
  energies     Hartree
  forces       Hartree/Angstrom
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

# ── Physical constants (mirrors bakken.py) ──────────────────────────────────
ATOMIC_MASSES = {
    'H':  1.00794,  'He': 4.002602, 'C':  12.011,   'N':  14.007,
    'O':  15.999,   'F':  18.9984,  'S':  32.06,    'Cl': 35.453,
    'Br': 79.904,   'I':  126.904,
}
ATOMIC_NUMBERS = {
    'H': 1, 'He': 2, 'C': 6, 'N': 7, 'O': 8,
    'F': 9, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53,
}


# =============================================================================
# Dataset conversion
# =============================================================================

def to_sgdml_dataset(symbols: List[str],
                     coordinates: np.ndarray,
                     energies: np.ndarray,
                     forces: np.ndarray,
                     name: str = 'molecule',
                     theory: str = 'DFT') -> tuple:
    """
    Convert our standard npz arrays to the sGDML dataset dict format.

    Energies are normalized to relative (E − mean) so that sGDML's internal
    energy-consistency validator works correctly with absolute Hartree values
    (which are ~−306 Ha for wB97X/MVKO, swamping the force-integrated scale).
    The offset is returned separately and must be stored in SGDMLModel.

    Parameters
    ----------
    symbols      : (n_atoms,) element symbols
    coordinates  : (n_frames, n_atoms, 3)  Angstrom
    energies     : (n_frames,)             Hartree
    forces       : (n_frames, n_atoms, 3)  Hartree/Angstrom

    Returns
    -------
    dataset      : dict in sGDML format
    energy_offset: float — mean energy subtracted (add back in predict())
    """
    z = np.array([ATOMIC_NUMBERS[s] for s in symbols], dtype=np.int32)
    energy_offset = float(np.mean(energies))
    E_rel = (energies - energy_offset).reshape(-1, 1).astype(np.float64)
    dataset = {
        'z':      z,
        'R':      coordinates.astype(np.float64),
        'E':      E_rel,
        'F':      forces.astype(np.float64),
        'name':   np.array(name),
        'theory': np.array(theory),
        'r_unit': np.array('Ang'),
        'e_unit': np.array('Hartree'),
        # No 'lattice' key — correct for isolated molecules
    }
    return dataset, energy_offset


# =============================================================================
# SGDMLModel — wraps sGDML model dict
# =============================================================================

class SGDMLModel:
    """
    Thin wrapper around a trained sGDML model dict.

    Provides .predict() / .predict_forces() / .predict_batch() with the same
    signatures as MLPESTrainer so it can be used anywhere MLPESTrainer is used.
    """

    def __init__(self, model_dict: dict, symbols: List[str], energy_offset: float = 0.0):
        self.model_dict    = model_dict
        self.symbols       = list(symbols)
        self.energy_offset = energy_offset  # mean energy subtracted during training
        self._predictor    = None
        self._init_predictor()

    def _init_predictor(self):
        from sgdml.predict import GDMLPredict
        self._predictor = GDMLPredict(self.model_dict)

    # ── Prediction ───────────────────────────────────────────────────────────

    def predict(self, symbols: List[str], coords: np.ndarray) -> float:
        """Single-point energy in Hartree.  coords: (n_atoms, 3) Angstrom."""
        r = coords.flatten().reshape(1, -1).astype(np.float64)
        E, _ = self._predictor.predict(r)
        return float(E[0]) + self.energy_offset

    def predict_forces(self, symbols: List[str], coords: np.ndarray) -> np.ndarray:
        """Analytic forces in Hartree/Angstrom. Returns (n_atoms, 3)."""
        r = coords.flatten().reshape(1, -1).astype(np.float64)
        _, F = self._predictor.predict(r)
        return F[0].reshape(-1, 3)

    def predict_batch(self, symbols: List[str], coords_batch: np.ndarray) -> np.ndarray:
        """
        Batch energy predictions.  coords_batch: (n_frames, n_atoms, 3).
        Returns (n_frames,) Hartree.
        """
        n = coords_batch.shape[0]
        R = coords_batch.reshape(n, -1).astype(np.float64)
        E, _ = self._predictor.predict(R)
        return E.flatten() + self.energy_offset

    # ── Serialisation ────────────────────────────────────────────────────────

    def save(self, filepath: str) -> None:
        with open(filepath, 'wb') as fh:
            pickle.dump({
                'type':          'sgdml',
                'model_dict':    self.model_dict,
                'symbols':       self.symbols,
                'energy_offset': self.energy_offset,
            }, fh, protocol=4)

    @classmethod
    def load(cls, filepath: str) -> 'SGDMLModel':
        with open(filepath, 'rb') as fh:
            state = pickle.load(fh)
        if not (isinstance(state, dict) and state.get('type') == 'sgdml'):
            raise ValueError(f"Not an SGDMLModel file: {filepath}")
        obj = cls.__new__(cls)
        obj.model_dict    = state['model_dict']
        obj.symbols       = state['symbols']
        obj.energy_offset = float(state.get('energy_offset', 0.0))
        obj._predictor    = None
        obj._init_predictor()
        return obj

    @staticmethod
    def is_sgdml_file(filepath: str) -> bool:
        try:
            with open(filepath, 'rb') as fh:
                state = pickle.load(fh)
            return isinstance(state, dict) and state.get('type') == 'sgdml'
        except Exception:
            return False

    # ── Metrics ──────────────────────────────────────────────────────────────

    @property
    def train_force_rmse(self) -> Optional[float]:
        """Training force RMSE (Hartree/Å) stored by sGDML during training."""
        ferr = self.model_dict.get('f_err')
        if ferr is None:
            return None
        if isinstance(ferr, dict):
            return float(ferr.get('rmse', float('nan')))
        return float(np.asarray(ferr).flat[0])

    @property
    def train_energy_rmse(self) -> Optional[float]:
        eerr = self.model_dict.get('e_err')
        if eerr is None:
            return None
        if isinstance(eerr, dict):
            return float(eerr.get('rmse', float('nan')))
        return float(np.asarray(eerr).flat[0])


# =============================================================================
# Training
# =============================================================================

def train_sgdml(symbols: List[str],
                coordinates: np.ndarray,
                energies: np.ndarray,
                forces: np.ndarray,
                sig: float = 0.5,
                lam: float = 1e-10,
                use_sym: bool = True,
                use_E_cstr: bool = False,
                n_train: Optional[int] = None,
                n_valid: int = 50,
                name: str = 'molecule',
                theory: str = 'wB97X-D/6-31G*',
                max_processes: int = 4,
                verbose: bool = True) -> SGDMLModel:
    """
    Train a single sGDML model.

    Parameters
    ----------
    sig           : kernel length scale (sGDML hyper-parameter, integer).
                    Larger → smoother PES.  Typical range: 10–500.
    lam           : Tikhonov regularisation (default 1e-10, sGDML default).
    use_sym       : discover and enforce molecular symmetries (recommended True).
    n_train       : training set size; defaults to len−n_valid.
    n_valid       : held-out validation set size.
    max_processes : parallel jobs for symmetry discovery.

    Returns
    -------
    SGDMLModel ready for prediction.
    """
    from sgdml.train import GDMLTrain

    dataset, energy_offset = to_sgdml_dataset(symbols, coordinates, energies, forces,
                                               name=name, theory=theory)
    n_total = len(energies)
    if n_train is None:
        n_train = n_total - n_valid
    n_valid = min(n_valid, n_total - n_train)

    if verbose:
        print(f"  sGDML training: n_train={n_train}, n_valid={n_valid}, "
              f"sig={sig}, lam={lam:.0e}, use_sym={use_sym}")

    trainer = GDMLTrain(max_processes=max_processes)
    task    = trainer.create_task(dataset, n_train, dataset, n_valid,
                                  sig=sig, lam=lam,
                                  use_sym=use_sym, use_E=True,
                                  use_E_cstr=use_E_cstr)
    model_dict = trainer.train(task)

    if verbose:
        def _rmse(d): return d['rmse'] if isinstance(d, dict) else float(np.asarray(d).flat[0])
        f_rmse = _rmse(model_dict.get('f_err', float('nan')))
        e_rmse = _rmse(model_dict.get('e_err', float('nan')))
        print(f"  Train force RMSE: {f_rmse:.4f} Ha/Å  |  energy RMSE: {e_rmse:.4f} Ha")

    return SGDMLModel(model_dict, symbols, energy_offset=energy_offset)


def train_sgdml_sweep(symbols: List[str],
                      coordinates: np.ndarray,
                      energies: np.ndarray,
                      forces: np.ndarray,
                      sig_values: List[float] = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0),
                      lam: float = 1e-10,
                      use_sym: bool = True,
                      use_E_cstr: bool = False,
                      n_train: Optional[int] = None,
                      n_valid: int = 80,
                      name: str = 'molecule',
                      theory: str = 'wB97X-D/6-31G*',
                      max_processes: int = 4) -> Tuple[SGDMLModel, dict]:
    """
    Sweep over sig values, return best model (lowest validation force RMSE).

    Returns
    -------
    (best_model, results_dict)
    results_dict has keys 'sig_values', 'force_rmse', 'energy_rmse', 'best_sig'.
    """
    from sgdml.train import GDMLTrain

    dataset, energy_offset = to_sgdml_dataset(symbols, coordinates, energies, forces,
                                               name=name, theory=theory)
    n_total = len(energies)
    if n_train is None:
        n_train = n_total - n_valid
    n_valid = min(n_valid, n_total - n_train)

    print(f"\n  sGDML sig sweep: {list(sig_values)}")
    print(f"  n_train={n_train}, n_valid={n_valid}, n_total={n_total}")
    print(f"  energy_offset={energy_offset:.6f} Ha (subtracted for training, restored in predict)")

    trainer     = GDMLTrain(max_processes=max_processes)
    best_model  = None
    best_f_rmse = np.inf
    best_sig    = None
    all_models  = []
    f_rmses     = []
    e_rmses     = []

    def _rmse(d):
        if isinstance(d, dict):
            v = d.get('rmse', float('nan'))
            return float('nan') if v is None else float(v)
        try:
            return float(np.asarray(d).flat[0])
        except Exception:
            return float('nan')

    for sig in sig_values:
        print(f"\n  --- sig={sig} ---")
        task = trainer.create_task(dataset, n_train, dataset, n_valid,
                                   sig=sig, lam=lam,
                                   use_sym=use_sym, use_E=True,
                                   use_E_cstr=use_E_cstr)
        model_dict = trainer.train(task)
        f_rmse = _rmse(model_dict.get('f_err', float('nan')))
        e_rmse = _rmse(model_dict.get('e_err', float('nan')))
        f_rmses.append(f_rmse)
        e_rmses.append(e_rmse)
        m = SGDMLModel(model_dict, symbols, energy_offset=energy_offset)
        all_models.append(m)
        print(f"  sig={sig!s:>6}: force RMSE={f_rmse:.4f} Ha/Å, energy RMSE={e_rmse:.4f} Ha")
        if np.isfinite(f_rmse) and f_rmse < best_f_rmse:
            best_f_rmse = f_rmse
            best_sig    = sig
            best_model  = m

    # Fallback: if sGDML internal RMSE is all NaN, select by manual validation
    if best_model is None:
        print("\n  [WARNING] sGDML internal RMSE all NaN — running manual validation to select sig")
        idxs_val = np.random.choice(n_total, min(30, n_total), replace=False)
        best_manual = np.inf
        for sig, m in zip(sig_values, all_models):
            preds = m.predict_batch(symbols, coordinates[idxs_val])
            manual_rmse = float(np.sqrt(np.mean((preds - energies[idxs_val])**2)))
            print(f"  sig={sig!s:>6}: manual energy RMSE={manual_rmse*627.51:.4f} kcal/mol")
            if manual_rmse < best_manual:
                best_manual = manual_rmse
                best_sig    = sig
                best_model  = m
        best_f_rmse = best_manual

    print(f"\n  Best sig={best_sig} (force RMSE={best_f_rmse:.4f} Ha/Å)")

    results = {
        'sig_values':  list(sig_values),
        'force_rmse':  f_rmses,
        'energy_rmse': e_rmses,
        'best_sig':    best_sig,
        'n_train':     n_train,
        'n_valid':     n_valid,
    }
    return best_model, results


# =============================================================================
# SGDMLDriver — bakken.py-compatible driver
# =============================================================================

class SGDMLDriver:
    """
    bakken-compatible PES driver backed by sGDML.

    Implements the same interface as MLPESDriver:
        .energy(coords)          → float [Hartree]
        .forces(coords)          → (n_atoms, 3) [Hartree/Å]  (analytic, sGDML native)
        .analytic_forces(coords) → (n_atoms, 3) [Hartree/Å]  (same as forces)
        .analytic_hessian(coords)→ (3N, 3N) [Hartree/Å²]     (FD on analytic forces)

    The driver accepts a path to either:
      - an SGDMLModel .pkl (saved via SGDMLModel.save())
      - OR an SGDMLModel object directly (for in-memory use)
    """

    def __init__(self, model_path_or_model):
        if isinstance(model_path_or_model, str):
            self.model = SGDMLModel.load(model_path_or_model)
        else:
            self.model = model_path_or_model

        self.symbols      = self.model.symbols
        self.n_atoms      = len(self.symbols)
        self.masses       = np.array([ATOMIC_MASSES[s] for s in self.symbols])
        self._has_analytic = True   # sGDML forces are always analytic

    # ── Energy / forces ──────────────────────────────────────────────────────

    def energy(self, coords: np.ndarray) -> float:
        """Predict energy in Hartree. coords: (n_atoms, 3) Angstrom."""
        return self.model.predict(self.symbols, coords)

    def forces(self, coords: np.ndarray, delta: float = 0.005) -> np.ndarray:
        """Analytic forces (Hartree/Å). delta is ignored; provided for API compat."""
        return self.model.predict_forces(self.symbols, coords)

    def analytic_forces(self, coords: np.ndarray) -> np.ndarray:
        """Analytic forces (Hartree/Å)."""
        return self.model.predict_forces(self.symbols, coords)

    def analytic_hessian(self, coords: np.ndarray,
                         delta: float = 0.005) -> np.ndarray:
        """
        Hessian via central FD on analytic forces.

        Cost: 6N analytic force evaluations (each a single sGDML forward pass),
        vs 6N energy evaluations for the KRR FD Hessian — similar cost but
        the sGDML forces are exact gradients, giving a symmetric, accurate Hessian.

        Returns (3N, 3N) Hartree/Å².
        """
        n3 = self.n_atoms * 3
        H  = np.zeros((n3, n3))
        r  = coords.flatten()
        for i in range(n3):
            rp = r.copy(); rp[i] += delta
            rm = r.copy(); rm[i] -= delta
            fp = self.analytic_forces(rp.reshape(-1, 3)).flatten()
            fm = self.analytic_forces(rm.reshape(-1, 3)).flatten()
            H[i] = -(fp - fm) / (2.0 * delta)
        H = 0.5 * (H + H.T)   # enforce symmetry
        return H
