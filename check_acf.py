"""Recompute ACF from traj_01 dipoles and show spectrum to 2000 cm-1."""
import numpy as np, pickle, sys
sys.path.insert(0, '/Users/jmsmith1/Documents/Research/code/MD_MLPES')
from modules.ml_pes import CoulombMatrixDescriptor

with open('outputs/ir_mace_anti_cis_300K/dipole_surface.pkl', 'rb') as f:
    ds = pickle.load(f)
model, scX, scY = ds['model'], ds['scaler_X'], ds['scaler_y']

with open('outputs/ir_mace_anti_cis_300K/traj_01.xyz') as f:
    lines = f.readlines()

n_atoms = 12
syms = ['C','O','O','C','C','C','H','H','H','H','H','H']

def get_coords(lines, idx):
    start = idx * (n_atoms + 2)
    return np.array([[float(x) for x in lines[start+2+j].split()[1:4]]
                     for j in range(n_atoms)])

# Predict dipoles for ALL 30000 frames
print('Predicting dipoles for 30000 frames...')
desc = CoulombMatrixDescriptor()
feats = []
for i in range(30000):
    coords = get_coords(lines, i)
    feats.append(desc.compute(syms, coords))
feats = np.array(feats)
feats_sc = scX.transform(feats)
dips_sc = model.predict(feats_sc)
dips = scY.inverse_transform(dips_sc)   # (30000, 3) Debye

# Mean-centre
dips -= dips.mean(axis=0)
print(f'Dipole std (D): x={dips[:,0].std():.4f}  y={dips[:,1].std():.4f}  z={dips[:,2].std():.4f}')

# ACF + FFT
dt_fs = 0.5
n = len(dips)
# Use autocorrelation
from numpy.fft import rfft, rfftfreq
# Welch-style: Hann window, split into 4 segments
seg_len = n // 4
window = np.hanning(seg_len)
psd_total = np.zeros(seg_len // 2 + 1)
for start in range(0, n - seg_len + 1, seg_len // 2):
    seg = dips[start:start+seg_len]
    for comp in range(3):
        sig = seg[:, comp] * window
        fft_seg = rfft(sig, n=seg_len)
        psd_total += np.abs(fft_seg)**2

freq_thz = rfftfreq(seg_len, d=dt_fs * 1e-15) * 1e-12   # THz
freq_cm = freq_thz / 0.02998                              # cm⁻¹

# Print spectrum in bands
print('\nManual ACF spectrum (Welch, normalized):')
psd_norm = psd_total / psd_total.max()
for lo, hi in [(0,200),(200,400),(400,600),(600,800),(800,1000),(1000,1300),(1300,1800)]:
    mask = (freq_cm >= lo) & (freq_cm < hi)
    if mask.sum() > 0:
        local_max = psd_norm[mask].max()
        peak_idx = np.where(mask)[0][np.argmax(psd_norm[mask])]
        print(f'  {lo:4d}-{hi:4d} cm⁻¹:  max_rel={local_max:.6f}  @ {freq_cm[peak_idx]:.1f} cm⁻¹')
