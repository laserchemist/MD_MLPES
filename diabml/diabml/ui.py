"""Optional interactive explorer (needs ipywidgets). Falls back to a static plot without it."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .plotting import cut_plot


def cut_explorer(make_coords, models: dict, x_range, fixed: dict, xlabel, n=150, reference_fn=None):
    """Slide the fixed coordinates and watch every model along a 1D cut.

    make_coords(x, **fixed) -> (n, d) coordinates along the cut; models: {name: object with .predict}.
    reference_fn(coords) -> reference energies (optional, e.g. a cached lookup)."""
    x = np.linspace(*x_range, n)

    def draw(**kw):
        C = make_coords(x, **kw)
        preds = {k: m.predict(C) if hasattr(m, 'predict') else m(C) for k, m in models.items()}
        ref = reference_fn(C) if reference_fn else np.full((n, next(iter(preds.values())).shape[1]), np.nan)
        cut_plot(x, ref, preds, xlabel, title=', '.join(f'{k} = {v:.2f}' for k, v in kw.items()))
        plt.show()

    try:
        import ipywidgets as w
        sliders = {k: w.FloatSlider(value=v[0], min=v[1], max=v[2], step=(v[2] - v[1]) / 50, description=k)
                   for k, v in fixed.items()}
        return w.interact(draw, **sliders)
    except ImportError:
        draw(**{k: v[0] for k, v in fixed.items()})
