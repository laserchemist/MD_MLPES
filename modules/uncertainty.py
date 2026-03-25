#!/usr/bin/env python3
"""
uncertainty.py — Ensemble uncertainty estimation for ML-PES models
===================================================================
Provides a `CommitteeModel` that trains K bootstrap KRR models and
uses their prediction variance as a proxy for epistemic uncertainty.

The bootstrap committee approach is preferred over exact KRR posterior
variance for two reasons:
  1. Exact GP posterior variance requires solving an (n_train × n_train)
     linear system per query point — expensive for n_train ≈ 900.
  2. Bootstrap variance naturally propagates through the StandardScaler
     and reuses existing MLPESTrainer without any changes.

Typical use:
  committee = CommitteeModel(symbols, training_data, k_models=5,
                             gamma=0.001, alpha=1e-5)
  committee.train()
  energy_mean, sigma_kcal = committee.predict_with_uncertainty(symbols, coords)

  # Batch scoring for adaptive sampling
  energies, sigmas = committee.batch_uncertainty(symbols, coords_batch)

Calibration:
  Bootstrapped variance underestimates true uncertainty when training
  frames are highly correlated.  Call .calibrate(val_coords, val_energies)
  to fit a scalar factor so that sigma predicts PSI4 errors on held-out data.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

HARTREE_TO_KCAL = 627.509474


class CommitteeModel:
    """
    Ensemble of K bootstrapped KRR models for uncertainty estimation.

    Each member is trained on a random 80% subset of the training frames,
    using the same hyperparameters (gamma, alpha).  The prediction variance
    across committee members gives the epistemic uncertainty.

    Parameters
    ----------
    symbols : list of str
        Atomic symbols (must match training data ordering).
    training_coords : np.ndarray, shape (N, n_atoms, 3)
        Training geometries in Angstrom.
    training_energies : np.ndarray, shape (N,)
        Training energies in Hartree.
    k_models : int
        Number of committee members (default 5).
    gamma : float
        RBF kernel width — use the same as the best single model.
    alpha : float
        KRR regularisation.
    bootstrap_fraction : float
        Fraction of training data for each member (default 0.80).
    seed : int
        RNG seed for reproducibility.
    """

    def __init__(self,
                 symbols: List[str],
                 training_coords: np.ndarray,
                 training_energies: np.ndarray,
                 k_models: int = 5,
                 gamma: float = 0.001,
                 alpha: float = 1e-5,
                 bootstrap_fraction: float = 0.80,
                 seed: int = 42):
        self.symbols           = symbols
        self.training_coords   = training_coords
        self.training_energies = training_energies
        self.k_models          = k_models
        self.gamma             = gamma
        self.alpha             = alpha
        self.bootstrap_fraction = bootstrap_fraction
        self.seed              = seed
        self.models: List      = []   # list of trained MLPESTrainer
        self._calibration_scale = 1.0  # scalar recalibration factor

    # ── Training ─────────────────────────────────────────────────────────────

    def train(self, verbose: bool = True) -> None:
        """Train K bootstrap committee members."""
        # Lazy import to avoid circular deps
        try:
            from .ml_pes import MLPESTrainer, MLPESConfig
            from .data_formats import TrajectoryData
        except ImportError:
            from ml_pes import MLPESTrainer, MLPESConfig
            from data_formats import TrajectoryData

        N = len(self.training_coords)
        n_boot = max(10, int(N * self.bootstrap_fraction))
        rng = np.random.default_rng(self.seed)

        if verbose:
            print(f"  Training committee of {self.k_models} models "
                  f"(boot_size={n_boot}/{N}, γ={self.gamma}, α={self.alpha})")

        self.models = []
        for k in range(self.k_models):
            idx = rng.choice(N, size=n_boot, replace=True)
            cfg = MLPESConfig(
                gamma=self.gamma,
                alpha=self.alpha,
                tune_hyperparameters=False,
                validation_split=0.0,
            )
            traj = TrajectoryData(
                symbols=self.symbols,
                coordinates=self.training_coords[idx],
                energies=self.training_energies[idx],
                forces=np.zeros((len(idx), len(self.symbols), 3)),
            )
            trainer = MLPESTrainer(cfg)
            trainer._train_committee_member(traj)
            self.models.append(trainer)
            if verbose:
                print(f"    member {k+1}/{self.k_models} done", flush=True)

    # ── Prediction ───────────────────────────────────────────────────────────

    def predict_with_uncertainty(self,
                                  symbols: List[str],
                                  coords: np.ndarray
                                  ) -> Tuple[float, float]:
        """
        Predict energy and uncertainty for a single geometry.

        Returns
        -------
        energy_mean : float
            Mean prediction of the committee (Hartree).
        sigma_kcal : float
            Standard deviation of committee predictions (kcal/mol).
            Scaled by the calibration factor if .calibrate() was called.
        """
        preds = np.array([m.predict(symbols, coords) for m in self.models])
        return float(preds.mean()), float(preds.std() * HARTREE_TO_KCAL * self._calibration_scale)

    def batch_uncertainty(self,
                          symbols: List[str],
                          coords_batch: np.ndarray,
                          ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorised prediction for a batch of geometries.

        Parameters
        ----------
        coords_batch : (N, n_atoms, 3)

        Returns
        -------
        energies : (N,) Hartree — mean committee prediction
        sigmas   : (N,) kcal/mol — committee std × calibration scale
        """
        N = len(coords_batch)
        preds_all = np.zeros((self.k_models, N))
        for k, m in enumerate(self.models):
            preds_all[k] = m.predict_batch(symbols, coords_batch)
        energies = preds_all.mean(axis=0)
        sigmas   = preds_all.std(axis=0) * HARTREE_TO_KCAL * self._calibration_scale
        return energies, sigmas

    # ── Calibration ──────────────────────────────────────────────────────────

    def calibrate(self,
                  symbols: List[str],
                  val_coords: np.ndarray,
                  val_energies: np.ndarray) -> float:
        """
        Calibrate the committee variance against actual PSI4 errors on
        held-out validation frames.

        Fits a scalar factor `s` so that  s × sigma ≈ |E_pred - E_psi4|
        on average (mean-absolute calibration).

        Returns the calibration scale factor.
        """
        energies, sigmas = self.batch_uncertainty(symbols, val_coords)
        abs_errors = np.abs((energies - val_energies) * HARTREE_TO_KCAL)

        # Avoid division by zero for near-zero sigmas
        valid = sigmas > 1e-6
        if valid.sum() < 3:
            print("  WARNING: too few valid committee members for calibration")
            return 1.0

        # Least-squares scale: s = mean(abs_errors) / mean(sigma)
        scale = abs_errors[valid].mean() / sigmas[valid].mean()
        self._calibration_scale = scale
        print(f"  Committee calibration: scale = {scale:.3f}  "
              f"(mean|error| = {abs_errors.mean():.4f} kcal/mol, "
              f"mean sigma = {sigmas.mean():.4f} kcal/mol before calibration)")
        return float(scale)

    # ── Serialisation ────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save all committee members and metadata to a single pickle file."""
        data = {
            'symbols':            self.symbols,
            'k_models':           self.k_models,
            'gamma':              self.gamma,
            'alpha':              self.alpha,
            'bootstrap_fraction': self.bootstrap_fraction,
            'seed':               self.seed,
            'calibration_scale':  self._calibration_scale,
            'models':             self.models,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"  CommitteeModel saved → {path}")

    @classmethod
    def load(cls, path: str) -> 'CommitteeModel':
        """Load a previously saved CommitteeModel."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        obj = cls.__new__(cls)
        obj.symbols            = data['symbols']
        obj.k_models           = data['k_models']
        obj.gamma              = data['gamma']
        obj.alpha              = data['alpha']
        obj.bootstrap_fraction = data['bootstrap_fraction']
        obj.seed               = data['seed']
        obj._calibration_scale = data.get('calibration_scale', 1.0)
        obj.models             = data['models']
        obj.training_coords    = None
        obj.training_energies  = None
        return obj
