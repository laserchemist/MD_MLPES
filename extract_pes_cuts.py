#!/usr/bin/env python3
"""
extract_pes_cuts.py
-------------------
Extract 1D PES cuts along NM mode 10 (O-O stretch, ~835 cm-1) and mode 1
(torsion, ~119 cm-1) from the trained ML models.

Outputs: outputs/pes_cut_data.npz
"""

import sys, json, pickle, warnings
import numpy as np

sys.path.insert(0, '.')

HARTREE_TO_KCAL = 627.509474
ANGSTROM_TO_BOHR = 1.88972612463

# ── load NMKRRDeltaModel from casscf_nm_delta.py ─────────────────────────────
print("Loading NMKRRDeltaModel class...")
import importlib.util, types
spec = importlib.util.spec_from_file_location("casscf_nm_delta",
    "casscf_nm_delta.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
NMKRRDeltaModel = mod.NMKRRDeltaModel
print("  NMKRRDeltaModel loaded from casscf_nm_delta.py")

# ── load CASSCF delta/gap models ─────────────────────────────────────────────
base_dir = "outputs/casscf_wB97X_nm_grid_20260407_184904"
print("\nLoading CASSCF delta/gap models...")
delta_s0_model  = NMKRRDeltaModel.load(f"{base_dir}/nm_delta_s0_model.pkl")
gap_s1_model    = NMKRRDeltaModel.load(f"{base_dir}/nm_gap_s1_model.pkl")
gap_t1_model    = NMKRRDeltaModel.load(f"{base_dir}/nm_gap_t1_model.pkl")
print(f"  delta_s0: γ={delta_s0_model.gamma}, α={delta_s0_model.alpha_reg}")
print(f"  gap_s1:   γ={gap_s1_model.gamma}, α={gap_s1_model.alpha_reg}")
print(f"  gap_t1:   γ={gap_t1_model.gamma}, α={gap_t1_model.alpha_reg}")

# ── load NMKRRPESModel (anti-cis) for U_vib and eq_coords ────────────────────
print("\nLoading NMKRRPESModel for NM eigenvectors...")
from modules.nm_pes import NMKRRPESModel
nm_model = NMKRRPESModel.load("outputs/anti_cis_nm_pes_20260513/mlpes_nm.pkl")
U_vib      = nm_model.U_vib           # (3N, n_vib)
eq_coords  = nm_model.eq_coords_ang   # (n_atoms, 3) Angstrom
sqrt_mass  = nm_model.sqrt_mass       # (3N,)
freqs_vib  = nm_model.freqs_vib       # (n_vib,)
symbols    = nm_model.symbols
n_atoms    = nm_model.n_atoms
n_vib      = nm_model.n_vib
print(f"  n_atoms={n_atoms}, n_vib={n_vib}")
print(f"  freqs_vib[0:5] = {freqs_vib[:5]}")
print(f"  freqs_vib[9]   = {freqs_vib[9]:.1f} cm-1  (mode 10 = O-O stretch?)")
print(f"  eq O1-O2 dist  = {np.linalg.norm(eq_coords[1]-eq_coords[2]):.4f} Ang")

# ── load wB97X ML-PES ─────────────────────────────────────────────────────────
print("\nLoading wB97X ML-PES...")
from modules.ml_pes import MLPESTrainer
wb97x_model = MLPESTrainer.load(
    "outputs/wB97X_surface_20260406_223155/mlpes_wB97X_aug.pkl")
print(f"  symbols: {wb97x_model.symbols[:4]}...")

# ── load raw CASSCF grid data ─────────────────────────────────────────────────
print("\nLoading CASSCF grid data...")
grid_data  = np.load(f"{base_dir}/grid_only.npz")
X_q_train  = np.load(f"{base_dir}/X_q_train.npy")       # (232, 30)
delta_s0   = np.load(f"{base_dir}/delta_s0_ha.npy")      # (232,)
gap_s1_raw = np.load(f"{base_dir}/gap_s1_ha.npy")        # (232,)
gap_t1_raw = np.load(f"{base_dir}/gap_t1_ha.npy")        # (232,)
coords_all = grid_data['coordinates']                     # (232, 12, 3)
energies_all = grid_data['energies']                     # (232,) wB97X Ha

print(f"  grid shape: {coords_all.shape}, energies: {energies_all.shape}")
print(f"  X_q_train:  {X_q_train.shape}")
print(f"  delta_s0 range: {delta_s0.min()*HARTREE_TO_KCAL:.3f} "
      f"to {delta_s0.max()*HARTREE_TO_KCAL:.3f} kcal/mol")
print(f"  gap_s1 eq:  {gap_s1_raw[0]*HARTREE_TO_KCAL:.3f} kcal/mol (frame 0)")
print(f"  gap_t1 eq:  {gap_t1_raw[0]*HARTREE_TO_KCAL:.3f} kcal/mol (frame 0)")

# ── helper: q-vector → Cartesian coords ──────────────────────────────────────
def q_to_cart(q_vec):
    """
    Convert NM coordinates to Cartesian (Angstrom).
    R = R_eq + M^{-1/2} U_vib q   (q in sqrt(amu)·Bohr)
    """
    # dR_mw = U_vib q  (in sqrt(amu)·Bohr)
    dR_mw   = U_vib @ q_vec               # (3N,)
    # convert to Angstrom: divide by sqrt_mass, then Bohr->Ang
    dR_ang  = (dR_mw / sqrt_mass) / ANGSTROM_TO_BOHR   # (3N,)
    return eq_coords + dR_ang.reshape(n_atoms, 3)       # (n_atoms, 3)

# ── helper: evaluate all models for one q-vector ─────────────────────────────
def eval_at_q(q_vec):
    """Return (E_wb97x, d_s0, g_s1, g_t1) all in Hartree for a NM coord vector."""
    coords = q_to_cart(q_vec)
    E_wb97x = wb97x_model.predict(symbols, coords)
    d_s0    = delta_s0_model.predict_delta_ha(q_vec)
    g_s1    = gap_s1_model.predict_delta_ha(q_vec)
    g_t1    = gap_t1_model.predict_delta_ha(q_vec)
    return E_wb97x, d_s0, g_s1, g_t1

# ── MODE 10: O-O stretch ──────────────────────────────────────────────────────
print("\n--- Mode 10 (O-O stretch) ---")
MODE_IDX = 9  # 0-based
q10_range = (-0.60, 0.60)
n_grid    = 200

q10_grid = np.linspace(*q10_range, n_grid)
E_wb97x_ml   = np.zeros(n_grid)
E_s0_corr    = np.zeros(n_grid)
E_s1         = np.zeros(n_grid)
E_t1         = np.zeros(n_grid)
r_oo_grid    = np.zeros(n_grid)

print(f"  Computing 1D scan along mode 10 ({freqs_vib[MODE_IDX]:.1f} cm-1)...")
for i, q10 in enumerate(q10_grid):
    q_vec = np.zeros(n_vib)
    q_vec[MODE_IDX] = q10
    coords = q_to_cart(q_vec)
    r_oo_grid[i] = np.linalg.norm(coords[1] - coords[2])  # O1-O2
    E_wb97x, d_s0, g_s1, g_t1 = eval_at_q(q_vec)
    E_wb97x_ml[i] = E_wb97x
    E_s0_corr[i]  = E_wb97x + d_s0
    E_s1[i]       = E_wb97x + d_s0 + g_s1
    E_t1[i]       = E_wb97x + d_s0 + g_t1
    if i % 40 == 0:
        print(f"  [{i:3d}/{n_grid}] q10={q10:+.3f}, rOO={r_oo_grid[i]:.4f}Å, "
              f"E_wb97x={E_wb97x*HARTREE_TO_KCAL:.2f} kcal/mol")

# Reference: minimum of wB97X ML-PES along this scan
E_ref_wb97x = E_wb97x_ml.min()
print(f"\n  E_ref (wB97X ML min): {E_ref_wb97x:.6f} Ha")

# Convert to kcal/mol relative to wB97X ML min
E_wb97x_ml_kcal = (E_wb97x_ml - E_ref_wb97x) * HARTREE_TO_KCAL
E_s0_corr_kcal  = (E_s0_corr  - E_ref_wb97x) * HARTREE_TO_KCAL
E_s1_kcal       = (E_s1       - E_ref_wb97x) * HARTREE_TO_KCAL
E_t1_kcal       = (E_t1       - E_ref_wb97x) * HARTREE_TO_KCAL

print(f"  r_OO at q10=0:    {r_oo_grid[n_grid//2]:.4f} Ang")
print(f"  r_OO at q10=+0.6: {r_oo_grid[-1]:.4f} Ang")
print(f"  E_S0 max:   {E_s0_corr_kcal.max():.2f} kcal/mol")
print(f"  E_S1 at eq: {E_s1_kcal[n_grid//2]:.2f} kcal/mol")
print(f"  E_T1 at eq: {E_t1_kcal[n_grid//2]:.2f} kcal/mol")

# ── Mode 10 scatter points (pure mode 10 training frames) ────────────────────
print("\n  Selecting pure mode-10 scatter points...")
q_other_norm = np.sqrt(np.sum(X_q_train[:, [k for k in range(n_vib) if k != MODE_IDX]]**2, axis=1))
mask10 = q_other_norm < 0.4
print(f"  Frames with |q_other| < 0.4: {mask10.sum()} of {len(X_q_train)}")

q10_pts      = X_q_train[mask10, MODE_IDX]
r_oo_pts     = np.array([np.linalg.norm(coords_all[i,1] - coords_all[i,2])
                          for i in np.where(mask10)[0]])
E_wb97x_raw  = energies_all[mask10]
d_s0_pts     = delta_s0[mask10]
g_s1_pts     = gap_s1_raw[mask10]
g_t1_pts     = gap_t1_raw[mask10]

E_wb97x_pts  = (E_wb97x_raw   - E_ref_wb97x) * HARTREE_TO_KCAL
E_s0_pts     = (E_wb97x_raw + d_s0_pts - E_ref_wb97x) * HARTREE_TO_KCAL
E_s1_pts     = (E_wb97x_raw + d_s0_pts + g_s1_pts - E_ref_wb97x) * HARTREE_TO_KCAL
E_t1_pts     = (E_wb97x_raw + d_s0_pts + g_t1_pts - E_ref_wb97x) * HARTREE_TO_KCAL
print(f"  q10 scatter range: {q10_pts.min():.3f} to {q10_pts.max():.3f}")

# ── MODE 1: Torsion ───────────────────────────────────────────────────────────
print("\n--- Mode 1 (torsion) ---")
MODE_IDX1 = 0  # 0-based
q1_range  = (-4.0, 4.0)

q1_grid   = np.linspace(*q1_range, n_grid)
E_wb97x_ml_mode1 = np.zeros(n_grid)
E_s0_corr_mode1  = np.zeros(n_grid)
E_s1_mode1       = np.zeros(n_grid)
E_t1_mode1       = np.zeros(n_grid)

print(f"  Computing 1D scan along mode 1 ({freqs_vib[MODE_IDX1]:.1f} cm-1)...")
for i, q1 in enumerate(q1_grid):
    q_vec = np.zeros(n_vib)
    q_vec[MODE_IDX1] = q1
    E_wb97x, d_s0, g_s1, g_t1 = eval_at_q(q_vec)
    E_wb97x_ml_mode1[i] = E_wb97x
    E_s0_corr_mode1[i]  = E_wb97x + d_s0
    E_s1_mode1[i]       = E_wb97x + d_s0 + g_s1
    E_t1_mode1[i]       = E_wb97x + d_s0 + g_t1
    if i % 40 == 0:
        print(f"  [{i:3d}/{n_grid}] q1={q1:+.3f}, E_wb97x={E_wb97x*HARTREE_TO_KCAL:.2f}")

E_ref_mode1 = E_wb97x_ml_mode1.min()
E_wb97x_ml_mode1_kcal = (E_wb97x_ml_mode1 - E_ref_mode1) * HARTREE_TO_KCAL
E_s0_corr_mode1_kcal  = (E_s0_corr_mode1  - E_ref_mode1) * HARTREE_TO_KCAL
E_s1_mode1_kcal       = (E_s1_mode1       - E_ref_mode1) * HARTREE_TO_KCAL
E_t1_mode1_kcal       = (E_t1_mode1       - E_ref_mode1) * HARTREE_TO_KCAL
print(f"  E_S0 max: {E_s0_corr_mode1_kcal.max():.2f} kcal/mol")
print(f"  E_S1 at q1=0: {E_s1_mode1_kcal[n_grid//2]:.2f} kcal/mol")

# ── Mode 1 scatter points ─────────────────────────────────────────────────────
print("\n  Selecting pure mode-1 scatter points...")
q_other_norm1 = np.sqrt(np.sum(X_q_train[:, [k for k in range(n_vib) if k != MODE_IDX1]]**2, axis=1))
mask1 = q_other_norm1 < 0.5
print(f"  Frames with |q_other excl mode1| < 0.5: {mask1.sum()}")

q1_pts              = X_q_train[mask1, MODE_IDX1]
E_wb97x_pts_mode1   = (energies_all[mask1]  - E_ref_mode1) * HARTREE_TO_KCAL
E_s0_pts_mode1      = ((energies_all[mask1] + delta_s0[mask1] - E_ref_mode1) * HARTREE_TO_KCAL)
print(f"  q1 scatter range: {q1_pts.min():.3f} to {q1_pts.max():.3f}")

# ── save all data ─────────────────────────────────────────────────────────────
out_path = "outputs/pes_cut_data.npz"
print(f"\nSaving to {out_path}...")
np.savez(out_path,
    # Mode 10 (O-O stretch) grid
    q10_grid          = q10_grid,
    r_oo_grid         = r_oo_grid,
    E_wb97x_ml        = E_wb97x_ml_kcal,
    E_s0_corr         = E_s0_corr_kcal,
    E_s1              = E_s1_kcal,
    E_t1              = E_t1_kcal,
    # Mode 10 scatter
    q10_pts           = q10_pts,
    r_oo_pts          = r_oo_pts,
    E_wb97x_pts       = E_wb97x_pts,
    E_s0_pts          = E_s0_pts,
    E_s1_pts          = E_s1_pts,
    E_t1_pts          = E_t1_pts,
    # Mode 1 (torsion) grid
    q1_grid           = q1_grid,
    E_wb97x_ml_mode1  = E_wb97x_ml_mode1_kcal,
    E_s0_corr_mode1   = E_s0_corr_mode1_kcal,
    E_s1_mode1        = E_s1_mode1_kcal,
    E_t1_mode1        = E_t1_mode1_kcal,
    # Mode 1 scatter
    q1_pts            = q1_pts,
    E_wb97x_pts_mode1 = E_wb97x_pts_mode1,
    E_s0_pts_mode1    = E_s0_pts_mode1,
    # Metadata
    freqs_vib         = freqs_vib,
    eq_roo            = np.array([np.linalg.norm(eq_coords[1]-eq_coords[2])]),
)
print(f"  Saved {out_path}")
print("\nDone.")
