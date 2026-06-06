"""Quick test: 5000-step with tau=50000 vs tau=200 — show ACF spectrum."""
import numpy as np, sys, pickle
sys.path.insert(0, '/Users/jmsmith1/Documents/Research/code/MD_MLPES')
from modules.mace_pes import MACEDriver
from ir_md_spectrum import run_ml_md_dense
from modules.normal_modes import compute_normal_modes
from ir_spectroscopy import IRSpectrumCalculator
from modules.ml_pes import CoulombMatrixDescriptor

print('Loading MACE model...')
driver = MACEDriver('outputs/mace_wB97X_20260417/mace_model.pt')
syms = driver.symbols
eq_coords = np.load('outputs/anti_cis_nm_pes_20260513/eq_coords.npy')

# Compute Hessian + normal modes for ZPE init
print('Computing Hessian...')
H = driver.analytic_hessian(eq_coords)
freqs, U_vib, eigenvalues, mass_vec = compute_normal_modes(syms, H)
nm_data = (freqs, U_vib, eigenvalues, mass_vec)

# Load dipole surface
with open('outputs/ir_mace_anti_cis_300K/dipole_surface.pkl', 'rb') as f:
    ds = pickle.load(f)
model_dip, scX, scY = ds['model'], ds['scaler_X'], ds['scaler_y']
desc = CoulombMatrixDescriptor()

def predict_dipoles(coords_traj):
    feats = np.array([desc.compute(syms, c) for c in coords_traj])
    feats_sc = scX.transform(feats)
    return scY.inverse_transform(model_dip.predict(feats_sc))

for tau_label, tau in [('tau=200', 200.0), ('tau=50000', 50000.0)]:
    print(f'\n=== {tau_label} ===')
    md = run_ml_md_dense(
        driver, eq_coords, n_steps=5000, temperature=300,
        timestep=0.5, save_every=1,
        nm_data=nm_data,
        min_freq_zpe=50, max_freq_zpe=1700,
        preminimize=True, seed=42,
        thermostat_tau=tau,
    )
    dips = predict_dipoles(md['coords_traj'])
    calc = IRSpectrumCalculator(temperature=300)
    freqs_sp, ints = calc.compute_ir_spectrum(
        dips, timestep=0.5, max_freq=1800, window='hann', zero_padding=4, verbose=False)
    ints_n = ints / ints.max()
    print('Spectrum by region:')
    for lo, hi in [(50,300),(300,500),(500,700),(700,1000),(1000,1400),(1400,1800)]:
        mask = (freqs_sp >= lo) & (freqs_sp < hi)
        if mask.sum() > 0:
            mx = ints_n[mask].max()
            pk = freqs_sp[mask][np.argmax(ints_n[mask])]
            print(f'  {lo:4d}-{hi:4d}: max={mx:.4f}  @ {pk:.1f} cm⁻¹')
