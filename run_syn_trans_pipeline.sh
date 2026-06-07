#!/bin/bash
# Syn-trans MVKO NM-PES local-harmonic IR pipeline
# Steps: NMDipoleSurface (NM-PES 300K) → MACE MD 300K → MACE MD 2000K
#        → hot_emission 300K → hot_emission 2000K
set -e
cd /Users/jmsmith1/Documents/Research/code/MD_MLPES

echo "============================================================"
echo "Syn-trans MVKO IR pipeline — started $(date)"
echo "============================================================"

# ── Step 1: NM-PES 300 K — trains NMDipoleSurface ────────────────────────
echo ""
echo "=== Step 1: NM-PES 300 K — NMDipoleSurface training ==="
python3 ir_md_spectrum.py \
    --nm-pes-model  outputs/syn_trans_nm_pes_20260513/mlpes_nm.pkl \
    --training-data outputs/syn_trans_nm_pes_20260513/combined_training_data.npz \
    --steps 30000 --temp 300 --preminimize \
    --zpe-min-freq 50 --zpe-max-freq 4000 \
    --n-trajectories 5 --max-bond-extension 2.0 \
    --output-dir outputs/ir_nm_pes_syn_trans_300K
echo "Step 1 done at $(date)"

# ── Step 2: MACE MD 300 K ─────────────────────────────────────────────────
echo ""
echo "=== Step 2: MACE MD 300 K ==="
python3 ir_md_spectrum.py \
    --mace-model    outputs/mace_wB97X_20260417/mace_model.pt \
    --dipole-model  outputs/ir_nm_pes_syn_trans_300K/dipole_surface.pkl \
    --training-data outputs/syn_trans_nm_pes_20260513/combined_training_data.npz \
    --steps 30000 --temp 300 --preminimize \
    --zpe-min-freq 50 --zpe-max-freq 4000 \
    --n-trajectories 5 --max-bond-extension 2.0 \
    --output-dir outputs/ir_mace_syn_trans_300K
echo "Step 2 done at $(date)"

# ── Step 3: MACE MD 2000 K ───────────────────────────────────────────────
echo ""
echo "=== Step 3: MACE MD 2000 K ==="
python3 ir_md_spectrum.py \
    --mace-model    outputs/mace_wB97X_20260417/mace_model.pt \
    --dipole-model  outputs/ir_nm_pes_syn_trans_300K/dipole_surface.pkl \
    --training-data outputs/syn_trans_nm_pes_20260513/combined_training_data.npz \
    --steps 30000 --temp 2000 --preminimize \
    --zpe-min-freq 50 --zpe-max-freq 4000 \
    --n-trajectories 5 --max-bond-extension 2.0 \
    --output-dir outputs/ir_mace_syn_trans_2000K
echo "Step 3 done at $(date)"

# ── Step 4: hot_emission 300 K ───────────────────────────────────────────
echo ""
echo "=== Step 4: hot_emission_spectrum.py 300 K ==="
python3 hot_emission_spectrum.py \
    --traj-dir      outputs/ir_mace_syn_trans_300K \
    --mace-model    outputs/mace_wB97X_20260417/mace_model.pt \
    --dipole-model  outputs/ir_nm_pes_syn_trans_300K/dipole_surface.pkl \
    --nm-delta-model outputs/casscf_wB97X_nm_grid_20260514_224806/nm_delta_s0_model.pkl \
    --stride 300 --fwhm 15.0 \
    --ref-harmonic  outputs/syn_trans_nm_pes_20260513/eq_coords.npy \
    --output-dir    outputs/hot_emission_syn_trans_300K
echo "Step 4 done at $(date)"

# ── Step 5: hot_emission 2000 K ──────────────────────────────────────────
echo ""
echo "=== Step 5: hot_emission_spectrum.py 2000 K ==="
python3 hot_emission_spectrum.py \
    --traj-dir      outputs/ir_mace_syn_trans_2000K \
    --mace-model    outputs/mace_wB97X_20260417/mace_model.pt \
    --dipole-model  outputs/ir_nm_pes_syn_trans_300K/dipole_surface.pkl \
    --nm-delta-model outputs/casscf_wB97X_nm_grid_20260514_224806/nm_delta_s0_model.pkl \
    --stride 300 --fwhm 30.0 \
    --ref-harmonic  outputs/syn_trans_nm_pes_20260513/eq_coords.npy \
    --output-dir    outputs/hot_emission_syn_trans_2000K
echo "Step 5 done at $(date)"

echo ""
echo "============================================================"
echo "Pipeline complete — $(date)"
echo "Results:"
echo "  NMDipoleSurface : outputs/ir_nm_pes_syn_trans_300K/dipole_surface.pkl"
echo "  Hot emission 300K : outputs/hot_emission_syn_trans_300K/ir_hot_emission.csv"
echo "  Hot emission 2000K: outputs/hot_emission_syn_trans_2000K/ir_hot_emission.csv"
echo "============================================================"
