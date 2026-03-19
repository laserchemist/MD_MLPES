#!/usr/bin/env python3
"""
IR Spectroscopy Module for ML-PES Framework

Computes infrared spectra from molecular dynamics simulations using:
1. Dipole moment autocorrelation function
2. ML-trained dipole moment surfaces
3. Fourier transform to frequency domain

All calculations use 100% real data from PSI4/ML predictions.

Theory:
    I(ω) ∝ ω² ∫ <μ(0)·μ(t)> exp(-iωt) dt
    
Where:
    μ(t) = dipole moment vector at time t
    <...> = ensemble average
    ω = frequency

Classes:
    DipoleSurface: ML model for dipole moment prediction
    IRSpectrumCalculator: Compute IR spectra from trajectories
    
Functions:
    train_dipole_surface: Train ML model for dipole moments
    compute_autocorrelation: Dipole autocorrelation function
    compute_ir_spectrum: FFT to get spectrum
    
Author: PSI4-MD ML-PES Framework
Date: January 2026
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from scipy import signal
from scipy.fft import fft, fftfreq

try:
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, GridSearchCV
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Physical constants
HARTREE_TO_KCAL = 627.509474
BOHR_TO_ANGSTROM = 0.529177
DEBYE_TO_AU = 0.393456  # 1 Debye = 0.393456 e*Bohr
AU_TO_DEBYE = 2.541746  # 1 e*Bohr = 2.541746 Debye
FS_TO_AU = 41.341374  # 1 fs = 41.341 atomic time units
SPEED_OF_LIGHT = 299792458  # m/s
HBAR = 1.054571817e-34  # J*s
KB = 1.380649e-23  # J/K


class DipoleSurface:
    """
    Machine learning model for predicting dipole moments.
    
    Similar to ML-PES but predicts 3-component dipole vector instead of scalar energy.
    """
    
    def __init__(self, symbols: List[str]):
        """
        Initialize dipole surface.
        
        Args:
            symbols: Atomic symbols (e.g., ['C', 'O', 'O', 'H', 'H'])
        """
        self.symbols = symbols
        self.n_atoms = len(symbols)
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.metadata = {}
    
    def compute_descriptor(self, coords: np.ndarray) -> np.ndarray:
        """
        Compute Coulomb matrix descriptor.
        
        Args:
            coords: Atomic coordinates (n_atoms, 3) in Angstroms
            
        Returns:
            Flattened Coulomb matrix
        """
        # Nuclear charges
        charge_map = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
        charges = np.array([charge_map[s] for s in self.symbols])
        
        # Coulomb matrix
        n = len(coords)
        coulomb = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    coulomb[i, j] = 0.5 * charges[i] ** 2.4
                else:
                    r = np.linalg.norm(coords[i] - coords[j])
                    coulomb[i, j] = charges[i] * charges[j] / r
        
        # Flatten upper triangle
        descriptor = coulomb[np.triu_indices(n)]
        
        return descriptor
    
    def train(self, coords: np.ndarray, dipoles: np.ndarray,
              test_size: float = 0.2, verbose: bool = True,
              n_jobs: int = 1) -> Dict:
        """
        Train ML model for dipole prediction.
        
        Args:
            coords: Training coordinates (n_samples, n_atoms, 3)
            dipoles: Training dipole moments (n_samples, 3) in Debye
            test_size: Fraction for test set
            verbose: Print progress
            
        Returns:
            Training statistics
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for training")
        
        if verbose:
            print("\n" + "=" * 80)
            print("  TRAINING DIPOLE MOMENT SURFACE")
            print("=" * 80)
        
        # Compute descriptors
        if verbose:
            print(f"\n📊 Computing descriptors for {len(coords)} configurations...")
        
        X = np.array([self.compute_descriptor(c) for c in coords])
        y = dipoles.copy()  # (n_samples, 3)
        
        if verbose:
            print(f"   Descriptor shape: {X.shape}")
            print(f"   Dipole shape: {y.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        if verbose:
            print(f"\n📂 Data split:")
            print(f"   Training: {len(X_train)} samples")
            print(f"   Testing: {len(X_test)} samples")
        
        # Scale data
        if verbose:
            print(f"\n⚙️  Scaling data...")
        
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)
        
        # Scale each dipole component
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        y_test_scaled = self.scaler_y.transform(y_test)
        
        # Hyperparameter tuning
        if verbose:
            print(f"\n🔍 Hyperparameter optimization...")

        param_grid = {
            'kernel': ['rbf'],
            'gamma': [0.001, 0.01, 0.1, 1.0],
            'alpha': [1e-5, 1e-4, 1e-3, 1e-2]
        }

        # Hard cap: never use more than 2 parallel jobs.
        # On Apple Silicon (unified memory), n_jobs=-1 with BLAS-threaded KRR
        # spawns one process per core; combined CPU/memory saturation caused a
        # kernel watchdog timeout panic (94 s with no checkins).
        # The dataset is small (<<1000 samples) so parallelism provides no
        # meaningful speedup over the forking overhead anyway.
        safe_jobs = min(n_jobs, 2) if n_jobs > 0 else 1
        grid_search = GridSearchCV(
            KernelRidge(),
            param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=safe_jobs,
            verbose=0
        )
        
        grid_search.fit(X_train_scaled, y_train_scaled)
        self.model = grid_search.best_estimator_
        
        if verbose:
            print(f"   Best parameters: {grid_search.best_params_}")
        
        # Evaluate
        y_train_pred_scaled = self.model.predict(X_train_scaled)
        y_test_pred_scaled = self.model.predict(X_test_scaled)
        
        y_train_pred = self.scaler_y.inverse_transform(y_train_pred_scaled)
        y_test_pred = self.scaler_y.inverse_transform(y_test_pred_scaled)
        
        # Calculate errors for each component
        train_rmse = np.sqrt(((y_train - y_train_pred) ** 2).mean(axis=0))
        test_rmse = np.sqrt(((y_test - y_test_pred) ** 2).mean(axis=0))
        
        train_rmse_total = np.sqrt(((y_train - y_train_pred) ** 2).mean())
        test_rmse_total = np.sqrt(((y_test - y_test_pred) ** 2).mean())
        
        # R² score
        from sklearn.metrics import r2_score
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)
        
        if verbose:
            print(f"\n📊 Training Results:")
            print(f"   Training RMSE (μx, μy, μz): {train_rmse[0]:.4f}, {train_rmse[1]:.4f}, {train_rmse[2]:.4f} Debye")
            print(f"   Testing RMSE (μx, μy, μz):  {test_rmse[0]:.4f}, {test_rmse[1]:.4f}, {test_rmse[2]:.4f} Debye")
            print(f"   Training RMSE (total): {train_rmse_total:.4f} Debye")
            print(f"   Testing RMSE (total):  {test_rmse_total:.4f} Debye")
            print(f"   R² (train): {r2_train:.6f}")
            print(f"   R² (test):  {r2_test:.6f}")
        
        # Store metadata
        self.metadata = {
            'training_date': datetime.now().isoformat(),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'train_rmse': train_rmse_total,
            'test_rmse': test_rmse_total,
            'train_rmse_components': train_rmse.tolist(),
            'test_rmse_components': test_rmse.tolist(),
            'r2_train': r2_train,
            'r2_test': r2_test,
            'hyperparameters': grid_search.best_params_
        }
        
        return self.metadata
    
    def predict(self, coords: np.ndarray) -> np.ndarray:
        """
        Predict dipole moment for given coordinates.
        
        Args:
            coords: Coordinates (n_atoms, 3) or (n_samples, n_atoms, 3)
            
        Returns:
            Dipole moment (3,) or (n_samples, 3) in Debye
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Handle single or multiple configurations
        if coords.ndim == 2:
            coords = coords[np.newaxis, ...]
            single = True
        else:
            single = False
        
        # Compute descriptors
        X = np.array([self.compute_descriptor(c) for c in coords])
        
        # Scale and predict
        X_scaled = self.scaler_X.transform(X)
        y_scaled = self.model.predict(X_scaled)
        y = self.scaler_y.inverse_transform(y_scaled)
        
        if single:
            return y[0]
        else:
            return y
    
    def save(self, filename: str):
        """Save model to file."""
        data = {
            'symbols': self.symbols,
            'model': self.model,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'metadata': self.metadata,
            'version': '1.0'
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✅ Dipole surface saved: {filename}")
    
    @classmethod
    def load(cls, filename: str):
        """Load model from file."""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        surface = cls(data['symbols'])
        surface.model = data['model']
        surface.scaler_X = data['scaler_X']
        surface.scaler_y = data['scaler_y']
        surface.metadata = data.get('metadata', {})
        
        return surface


class IRSpectrumCalculator:
    """
    Calculate IR spectrum from MD trajectory with dipole moments.
    
    Uses dipole autocorrelation function and Fourier transform.
    """
    
    def __init__(self, temperature: float = 300.0):
        """
        Initialize calculator.
        
        Args:
            temperature: Temperature in Kelvin
        """
        self.temperature = temperature
        self.spectrum = None
        self.autocorrelation = None
    
    def compute_autocorrelation(self, dipoles: np.ndarray, 
                                max_lag: Optional[int] = None,
                                verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute dipole autocorrelation function.
        
        C(t) = <μ(0) · μ(t)>
        
        Args:
            dipoles: Dipole moments (n_frames, 3) in Debye
            max_lag: Maximum lag time in frames (default: n_frames // 2)
            verbose: Print progress
            
        Returns:
            (lags, autocorrelation): Time lags and C(t) values
        """
        if verbose:
            print("\n📊 Computing dipole autocorrelation function...")
        
        n_frames = len(dipoles)
        
        if max_lag is None:
            max_lag = n_frames // 2
        
        # Compute autocorrelation for each component
        autocorr = np.zeros((max_lag, 3))
        
        for i in range(3):  # x, y, z components
            # Center the data
            dipole_centered = dipoles[:, i] - dipoles[:, i].mean()
            
            # Compute autocorrelation using FFT (fast method)
            # This is equivalent to: C(τ) = Σ μ(t) μ(t+τ) / N
            fft_dipole = fft(dipole_centered, n=2*n_frames)
            power_spectrum = np.abs(fft_dipole) ** 2
            autocorr_full = np.real(fft(power_spectrum, n=2*n_frames))[:n_frames]
            autocorr_full = autocorr_full / (n_frames - np.arange(n_frames))
            
            autocorr[:, i] = autocorr_full[:max_lag]
        
        # Total autocorrelation (sum over components)
        total_autocorr = autocorr.sum(axis=1)
        
        # Normalize
        total_autocorr = total_autocorr / total_autocorr[0]
        
        lags = np.arange(max_lag)
        
        if verbose:
            print(f"   Computed for {max_lag} lag times")
            print(f"   C(0) = {total_autocorr[0]:.6f} (normalized to 1)")
        
        self.autocorrelation = total_autocorr
        
        return lags, total_autocorr
    
    def compute_ir_spectrum(self, dipoles: np.ndarray, timestep: float,
                           max_freq: float = 4000.0,
                           window: str = 'hann',
                           zero_padding: int = 4,
                           verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute IR spectrum from dipole moments.
        
        I(ω) ∝ ω² * |FT[C(t)]|²
        
        Args:
            dipoles: Dipole moments (n_frames, 3) in Debye
            timestep: Time between frames in fs
            max_freq: Maximum frequency in cm⁻¹
            window: Window function ('hann', 'hamming', 'blackman', None)
            zero_padding: Factor for zero padding (increases resolution)
            verbose: Print progress
            
        Returns:
            (frequencies, intensities): IR spectrum in cm⁻¹ and arbitrary units
        """
        if verbose:
            print("\n" + "=" * 80)
            print("  IR SPECTRUM CALCULATION")
            print("=" * 80)
        
        # Compute autocorrelation
        lags, autocorr = self.compute_autocorrelation(
            dipoles, 
            max_lag=len(dipoles) // 2,
            verbose=verbose
        )
        
        # Apply window function to reduce spectral leakage
        if window:
            if verbose:
                print(f"\n🪟 Applying {window} window function...")
            
            if window == 'hann':
                w = np.hanning(len(autocorr))
            elif window == 'hamming':
                w = np.hamming(len(autocorr))
            elif window == 'blackman':
                w = np.blackman(len(autocorr))
            else:
                raise ValueError(f"Unknown window: {window}")
            
            autocorr = autocorr * w
        
        # Zero padding for better frequency resolution
        n_pad = len(autocorr) * zero_padding
        autocorr_padded = np.zeros(n_pad)
        autocorr_padded[:len(autocorr)] = autocorr
        
        if verbose:
            print(f"\n📏 Zero padding:")
            print(f"   Original points: {len(autocorr)}")
            print(f"   Padded points: {n_pad}")
        
        # Fourier transform
        if verbose:
            print(f"\n🔄 Computing Fourier transform...")
        
        # Time step in atomic units
        dt_au = timestep * FS_TO_AU
        
        # FFT
        spectrum_complex = fft(autocorr_padded)
        frequencies_au = fftfreq(n_pad, d=dt_au)
        
        # Convert to cm⁻¹
        # ω (cm⁻¹) = ω (au) * 219474.63
        CM_INV_PER_AU = 219474.63
        frequencies = frequencies_au * CM_INV_PER_AU
        
        # Take positive frequencies only
        positive_freq_idx = frequencies > 0
        frequencies = frequencies[positive_freq_idx]
        spectrum_complex = spectrum_complex[positive_freq_idx]
        
        # Intensity: I(ω) ∝ ω² * |FT[C(t)]|²
        # The ω² factor accounts for the quantum correction
        intensities = (frequencies ** 2) * np.abs(spectrum_complex) ** 2
        
        # Limit to max_freq
        freq_mask = frequencies <= max_freq
        frequencies = frequencies[freq_mask]
        intensities = intensities[freq_mask]
        
        # Normalize
        intensities = intensities / intensities.max()
        
        if verbose:
            print(f"   Frequency range: 0 - {frequencies.max():.1f} cm⁻¹")
            print(f"   Frequency resolution: {frequencies[1] - frequencies[0]:.2f} cm⁻¹")
            print(f"   Number of points: {len(frequencies)}")
        
        self.spectrum = (frequencies, intensities)
        
        return frequencies, intensities
    
    def find_peaks(self, threshold: float = 0.1, 
                   verbose: bool = True) -> List[Tuple[float, float]]:
        """
        Find peaks in IR spectrum.
        
        Args:
            threshold: Minimum relative intensity (0-1)
            verbose: Print results
            
        Returns:
            List of (frequency, intensity) tuples
        """
        if self.spectrum is None:
            raise ValueError("Spectrum not computed yet")
        
        frequencies, intensities = self.spectrum
        
        # Find peaks using scipy
        from scipy.signal import find_peaks
        
        peak_indices, properties = find_peaks(
            intensities,
            height=threshold,
            distance=10  # Minimum distance between peaks
        )
        
        peaks = [(frequencies[i], intensities[i]) for i in peak_indices]
        
        # Sort by intensity
        peaks.sort(key=lambda x: x[1], reverse=True)
        
        if verbose:
            print(f"\n🎯 Found {len(peaks)} peaks above threshold {threshold:.2f}:")
            print(f"\n   {'Frequency (cm⁻¹)':<20} {'Intensity':<15} {'Assignment':<20}")
            print(f"   {'-' * 60}")
            
            for freq, intensity in peaks[:10]:  # Top 10
                assignment = self._assign_peak(freq)
                print(f"   {freq:<20.1f} {intensity:<15.3f} {assignment:<20}")
        
        return peaks
    
    def _assign_peak(self, frequency: float) -> str:
        """Simple peak assignment based on typical ranges."""
        if frequency < 500:
            return "Torsion/bending"
        elif frequency < 1000:
            return "C-O, C-N stretch"
        elif frequency < 1500:
            return "C-H bend"
        elif frequency < 1800:
            return "C=C, C=O stretch"
        elif frequency < 2500:
            return "C≡C, C≡N stretch"
        elif frequency < 3000:
            return "C-H stretch"
        elif frequency < 3500:
            return "O-H, N-H stretch"
        else:
            return "O-H, N-H stretch (H-bonded)"


def train_dipole_surface(coords: np.ndarray, 
                         dipoles: np.ndarray,
                         symbols: List[str],
                         output_file: str,
                         test_size: float = 0.2,
                         verbose: bool = True) -> DipoleSurface:
    """
    Train ML model for dipole moment prediction.
    
    Args:
        coords: Training coordinates (n_samples, n_atoms, 3) in Angstroms
        dipoles: Training dipole moments (n_samples, 3) in Debye
        symbols: Atomic symbols
        output_file: Where to save trained model
        test_size: Fraction for test set
        verbose: Print progress
        
    Returns:
        Trained DipoleSurface object
    """
    surface = DipoleSurface(symbols)
    surface.train(coords, dipoles, test_size=test_size, verbose=verbose)
    surface.save(output_file)
    
    return surface


def compute_ir_spectrum_from_trajectory(dipoles: np.ndarray,
                                       timestep: float,
                                       temperature: float = 300.0,
                                       max_freq: float = 4000.0,
                                       window: str = 'hann',
                                       verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute IR spectrum from trajectory dipoles.
    
    Args:
        dipoles: Dipole moments (n_frames, 3) in Debye
        timestep: Time between frames in fs
        temperature: Temperature in K
        max_freq: Maximum frequency in cm⁻¹
        window: Window function
        verbose: Print progress
        
    Returns:
        (frequencies, intensities): IR spectrum
    """
    calculator = IRSpectrumCalculator(temperature=temperature)
    frequencies, intensities = calculator.compute_ir_spectrum(
        dipoles, timestep, max_freq=max_freq, window=window, verbose=verbose
    )
    calculator.find_peaks(verbose=verbose)
    
    return frequencies, intensities


# For backward compatibility
compute_autocorrelation = IRSpectrumCalculator(300.0).compute_autocorrelation
compute_ir_spectrum = compute_ir_spectrum_from_trajectory
