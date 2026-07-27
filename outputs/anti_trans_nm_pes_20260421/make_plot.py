import sys
sys.path.insert(0, '.')
import numpy as np, pickle, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from modules.nm_pes import NMKRRPESModel

# ── Load ──────────────────────────────────────────────────────────────────────
pes = NMKRRPESModel.load('outputs/anti_trans_nm_pes_20260421/mlpes_nm.pkl')

nm_data   = np.load('outputs/anti_trans_nm_pes_20260421/nm_displacements_with_eq.npz', allow_pickle=True)
md_data   = np.load('outputs/anti_trans_nm_pes_20260421/md_training.npz',              allow_pickle=True)
comb_data = np.load('outputs/anti_trans_nm_pes_20260421/combined_training_data.npz',  allow_pickle=True)

e_nm   = nm_data['energies']
e_md   = md_data['energies']
e_comb = comb_data['energies']
e_min  = e_comb.min()
de_nm  = (e_nm - e_min) * 627.509
de_md  = (e_md - e_min) * 627.509

freqs = pes.freqs_vib
cs    = pes.coord_scale
rmse  = pes.cv_rmse_kcal

print(f"NM frames={len(e_nm)}, MD frames={len(e_md)}, combined={len(e_comb)}, kernel rows={len(pes.X_train_q)}")
print(f"LOO-CV RMSE (stored): {rmse:.3f} kcal/mol")
print(f"\nFrequencies (cm-1):")
for i,f in enumerate(freqs):
    mark = " *** B4-B7" if 980<f<1170 else (" C-H" if f>2800 else "")
    print(f"  Mode {i+1:2d}: {f:8.1f}  cs={cs[i]:.4f}{mark}")

# KRR-only prediction on training NM coords (no wall penalty)
# Use the internal kernel directly to bypass wall
X_qs = pes._X_train_qs   # scaled coords
y_t  = pes.y_train_ha
y_p_raw = np.dot(pes._alpha_vec[None, :],
                 np.exp(-pes.gamma *
                        np.sum((X_qs[:, None, :] - X_qs[None, :, :]) ** 2, axis=2))
                 ).squeeze() + pes._y_mean
# Simpler: just use kernel matrix diagonal = 1
K_train = np.exp(-pes.gamma * np.sum((X_qs[:, None, :] - X_qs[None, :, :])**2, axis=-1))
y_pred_ha = K_train @ pes._alpha_vec + pes._y_mean
dy_t = (y_t    - y_t.mean()) * 627.509
dy_p = (y_pred_ha - y_t.mean()) * 627.509
resid = dy_p - dy_t
train_rmse = np.sqrt(np.mean(resid**2))
print(f"Train-set RMSE (KRR only, no wall): {train_rmse:.4f} kcal/mol")

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14,10))
gs  = gridspec.GridSpec(2,3, figure=fig, hspace=0.44, wspace=0.36)

# Panel 1: Frequencies
ax1 = fig.add_subplot(gs[0,0])
colors_f = []
for f in freqs:
    if 980<f<1170: colors_f.append('#d62728')
    elif f>2800:   colors_f.append('#2ca02c')
    else:          colors_f.append('steelblue')
ax1.barh(range(1,31), freqs, color=colors_f, height=0.75)
ax1.axvline(0, color='k', lw=0.5)
ax1.legend(handles=[Patch(color='#d62728',label='B4-B7 (991-1108 cm$^{-1}$)'),
                    Patch(color='#2ca02c',label='C-H stretch (~3000 cm$^{-1}$)'),
                    Patch(color='steelblue',label='other')], fontsize=7)
ax1.set_xlabel('Frequency (cm$^{-1}$)', fontsize=10)
ax1.set_ylabel('Mode', fontsize=10)
ax1.set_title('PSI4 wB97X-D/6-31G*\nNormal mode frequencies', fontsize=10)
ax1.invert_yaxis()

# Panel 2: coord_scale
ax2 = fig.add_subplot(gs[0,1])
bar_colors = ['#d62728' if 980<f<1170 else 'steelblue' for f in freqs]
ax2.bar(range(1,31), cs, color=bar_colors, width=0.75)
ax2.set_xlabel('Mode index', fontsize=10)
ax2.set_ylabel('Thermal amplitude ($\sqrt{amu}\cdot$Bohr)', fontsize=10)
ax2.set_title('coord_scale at 300 K\n(RBF kernel normalization per mode)', fontsize=10)

# Panel 3: Energy distribution
ax3 = fig.add_subplot(gs[0,2])
bins = np.linspace(0, 65, 28)
ax3.hist(de_nm, bins=bins, alpha=0.65, label=f'NM-displaced ({len(e_nm)})', color='steelblue')
ax3.hist(de_md, bins=bins, alpha=0.65, label=f'MD 300/600/1000K ({len(e_md)})', color='tomato')
ax3.set_xlabel('$\Delta$E above min (kcal/mol)', fontsize=10)
ax3.set_ylabel('Count', fontsize=10)
ax3.set_title(f'Training data  N={len(e_comb)} total, 387 used\n(ΔE<50 kcal/mol filter)', fontsize=10)
ax3.legend(fontsize=8)

# Panel 4: Train-set parity (KRR interpolation, no wall)
ax4 = fig.add_subplot(gs[1,0])
lim = max(np.abs(dy_t).max(), np.abs(dy_p).max()) * 1.05
ax4.scatter(dy_t, dy_p, s=4, alpha=0.35, color='steelblue')
ax4.plot([-lim,lim],[-lim,lim],'r--',lw=1)
ax4.set_xlabel('PSI4 $\Delta$E (kcal/mol)', fontsize=10)
ax4.set_ylabel('KRR predicted $\Delta$E (kcal/mol)', fontsize=10)
ax4.set_title(f'Train-set parity (KRR interpolation)\nTrain RMSE={train_rmse:.3f} kcal/mol', fontsize=10)
ax4.set_xlim(-lim,lim); ax4.set_ylim(-lim,lim)
ax4.text(0.05, 0.88, f'LOO-CV RMSE\n{rmse:.3f} kcal/mol',
         transform=ax4.transAxes, fontsize=9, color='darkred',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='orange'))

# Panel 5: Train residuals
ax5 = fig.add_subplot(gs[1,1])
ax5.scatter(dy_t, resid, s=4, alpha=0.35, color='steelblue')
ax5.axhline(0, color='r', lw=1, ls='--')
ax5.axhline( rmse, color='orange', lw=0.8, ls=':', label=f'LOO-CV $\pm${rmse:.2f} kcal/mol')
ax5.axhline(-rmse, color='orange', lw=0.8, ls=':')
ax5.set_xlabel('PSI4 $\Delta$E (kcal/mol)', fontsize=10)
ax5.set_ylabel('Train residual (kcal/mol)', fontsize=10)
ax5.set_title('Train residuals vs energy\n(LOO-CV error band shown)', fontsize=10)
ax5.legend(fontsize=8)

# Panel 6: 1D PES cuts (B4-B7 modes)
ax6 = fig.add_subplot(gs[1,2])
q_grid = np.linspace(-4, 4, 200)
b_modes = [i for i,f in enumerate(freqs) if 980<f<1170]
cols6 = plt.cm.Reds(np.linspace(0.4, 0.95, len(b_modes)))
for ci, mi in enumerate(b_modes):
    q_mat = np.zeros((200, len(freqs)))
    q_mat[:,mi] = q_grid * cs[mi]
    q_sc = q_mat / cs
    k_cut = np.exp(-pes.gamma * np.sum((q_sc[:, None, :] - X_qs[None, :, :])**2, axis=-1))
    y_cut = k_cut @ pes._alpha_vec + pes._y_mean
    ax6.plot(q_grid, (y_cut - y_t.min())*627.509, color=cols6[ci], lw=1.8,
             label=f'M{mi+1} {freqs[mi]:.0f} cm$^{{-1}}$')
# torsion for reference
q_mat0 = np.zeros((200, len(freqs)))
q_mat0[:,0] = q_grid * cs[0]
q_sc0 = q_mat0 / cs
k_c0 = np.exp(-pes.gamma * np.sum((q_sc0[:, None, :] - X_qs[None, :, :])**2, axis=-1))
y_c0 = k_c0 @ pes._alpha_vec + pes._y_mean
ax6.plot(q_grid, (y_c0 - y_t.min())*627.509, 'k--', lw=1, alpha=0.5,
         label=f'M1 torsion {freqs[0]:.0f} cm$^{{-1}}$')
ax6.set_xlabel('Amplitude ($\\times$ coord_scale)', fontsize=10)
ax6.set_ylabel('$\Delta$E (kcal/mol)', fontsize=10)
ax6.set_title('1D NM-PES cuts\nB4-B7 modes (red) + torsion (dashed)', fontsize=10)
ax6.legend(fontsize=7, ncol=2)
ax6.set_ylim(-1, 30)

fig.suptitle('Anti-trans MVKO NM-KRR Surface  |  wB97X-D/6-31G*\n'
             '$\gamma$=0.1, $\\alpha$=1e-7  |  387/403 frames'
             '  (NM-displaced + 300K/600K/1000K PSI4 MD)  |  LOO-CV RMSE=1.079 kcal/mol',
             fontsize=11, fontweight='bold')

plt.savefig('outputs/anti_trans_nm_pes_20260421/surface_summary.png', dpi=150, bbox_inches='tight')
print("Saved: outputs/anti_trans_nm_pes_20260421/surface_summary.png")
