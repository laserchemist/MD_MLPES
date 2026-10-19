"""Standard figures. Every function returns the matplotlib Figure so notebooks can save or restyle it."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# fixed categorical order (colour follows the model, never its rank)
COLORS = {'reference': '#3d3d3a', 'physics prior': '#2a78d6', 'CDNN': '#eb6834', 'CDNN + permutations': '#eda100',
          'independent NN': '#e87ba4', 'framed CDNN': '#1baf7a', 'KRR': '#7a5a00', 'framed + KRR': '#008300',
          'framed (mixing)': '#9a6fd0'}
DIABAT_COLORS = {'covalent': '#2a78d6', 'biradical': '#1baf7a', 'ionic': '#eb6834'}
DIVERGE = LinearSegmentedColormap.from_list('div', ['#2a78d6', '#e9e8e4', '#eb6834'])


def style():
    plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': 0.6,
                         'lines.linewidth': 2, 'font.size': 10, 'legend.frameon': False})


def color(name):
    return COLORS.get(name, '#555555')


def cut_plot(x, ref, models: dict, xlabel, ylabel='energy (eV)', title=None, ax=None, ref_x=None, states=None):
    """Adiabatic energies along a 1D cut. ref: (n, n_states) at ref_x (default x); models: {name: (N, n_states)}."""
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(6.4, 4))
    ref_x = x if ref_x is None else ref_x
    states = range(ref.shape[1]) if states is None else states
    for s in states:
        ax.plot(ref_x, ref[:, s], 'o', ms=4, color=color('reference'), mfc='white' if s else color('reference'),
                label='reference' if s == list(states)[0] else None)
    for name, E in models.items():
        for k, s in enumerate(states):
            ax.plot(x, E[:, s], color=color(name), lw=1.6, ls=':' if name == 'physics prior' else '-',
                    label=name if k == 0 else None)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend(fontsize=8)
    return fig


def learning_curves(ns, results: dict, baseline=None, ylabel='test RMSE (kcal/mol)', title=None, ax=None,
                    baseline_label='physics prior, no target data'):
    """results: {model name: list over ns of lists over splits}. Shows mean with min-max bars on log-log axes."""
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(6.4, 4))
    for name, v in results.items():
        v = [np.asarray(x) for x in v]
        m = np.array([x.mean() for x in v])
        lo, hi = m - np.array([x.min() for x in v]), np.array([x.max() for x in v]) - m
        ax.errorbar(ns, m, yerr=[lo, hi], color=color(name), marker='o', ms=6, capsize=3, label=name)
    if baseline is not None:
        ax.axhline(baseline, color=color('physics prior'), ls=':', lw=1.8, label=baseline_label)
    ax.set(xscale='log', yscale='log', xlabel='training points', ylabel=ylabel, title=title)
    ax.set_xticks(ns); ax.set_xticklabels(ns); ax.xaxis.set_minor_formatter(plt.NullFormatter())
    lo, hi = ax.get_ylim()
    yt = [t for t in [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100] if lo <= t <= hi]
    ax.set_yticks(yt); ax.set_yticklabels([f'{t:g}' for t in yt]); ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.legend(fontsize=8)
    return fig


def diabat_plot(x, diabats, adiabats, labels, kinds, ref_x=None, ref=None, xlabel='r (Å)', ylabel='energy (eV)',
                title='Learned diabats (dashed) and adiabats (solid)', ax=None, ylim=None):
    """Diagonal matrix elements (diabats) coloured by structure type, with the adiabatic eigenvalues on top."""
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(7, 4.4))
    for i, (lab, kind) in enumerate(zip(labels, kinds)):
        ax.plot(x, diabats[:, i], ls='--', lw=1.4, color=DIABAT_COLORS.get(kind, '#888888'), label=lab)
    for s in range(adiabats.shape[1]):
        ax.plot(x, adiabats[:, s], color='#3d3d3a', lw=1.8, label='adiabats (model)' if s == 0 else None)
    if ref is not None:
        for s in range(ref.shape[1]):
            ax.plot(ref_x, ref[:, s], 'o', ms=3.5, color='#3d3d3a', mfc='white', label='reference' if s == 0 else None)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=7, ncol=2)
    return fig


def matrix_maps(X, Y, M, xlabel, ylabel, marker=None, titles=('Ω₁₁ − Ω₂₂', 'Ω₁₂')):
    """Two-panel map of a 2x2 matrix over a 2D grid: diagonal difference and off-diagonal, with zero contours.
    A conical intersection sits where both zero lines cross."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for ax, Z, t in [(axes[0], M[..., 0, 0] - M[..., 1, 1], titles[0]), (axes[1], M[..., 0, 1], titles[1])]:
        lim = max(np.abs(Z).max(), 1e-9)
        im = ax.pcolormesh(X, Y, Z, cmap=DIVERGE, vmin=-lim, vmax=lim, shading='auto')
        ax.contour(X, Y, Z, levels=[0], colors='#3d3d3a', linewidths=1)
        plt.colorbar(im, ax=ax, label=f'{t} (eV)')
        if marker is not None:
            ax.plot(*marker, 'k*', ms=10)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=t)
    return fig


def character_plot(x, charge, n_unpaired, xlabel='r(O–O) (Å)', states=None):
    """Per-state charge on a fragment (zwitterion character) and effective unpaired electrons (biradical character)."""
    states = range(charge.shape[1]) if states is None else states
    cols = ['#3d3d3a', '#2a78d6', '#eb6834', '#1baf7a', '#e87ba4']
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for s in states:
        axes[0].plot(x, charge[:, s], color=cols[s % 5], marker='o', ms=3, label=f'S{s}')
        axes[1].plot(x, n_unpaired[:, s], color=cols[s % 5], marker='o', ms=3, label=f'S{s}')
    axes[0].set(xlabel=xlabel, ylabel='Mulliken charge on terminal O (e)', title='Charge separation (zwitterion character)')
    axes[1].set(xlabel=xlabel, ylabel='effective unpaired electrons', title='Open-shell (biradical) character')
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return fig
