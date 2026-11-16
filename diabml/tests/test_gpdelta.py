"""GP Delta model: fits a smooth function, its std is small at data and grows away from it, the mean falls back to
zero far away, batch selection spreads out."""
import numpy as np

from diabml.gpdelta import GPDelta, select_uncertain


def test_gp_fit_predict_and_select():
    rng = np.random.default_rng(1)
    names = ['r_OO', 'a_COO']
    X = np.c_[rng.uniform(1.25, 1.45, 60), rng.uniform(110, 125, 60)]
    f = lambda Z: 0.05 * np.sin((Z[:, 0] - 1.3) * 10) + 0.002 * (Z[:, 1] - 117)
    gp = GPDelta.fit(X, f(X), names, noise_eV=1e-4)
    T = np.c_[rng.uniform(1.27, 1.43, 30), rng.uniform(111, 124, 30)]
    mu, sd = gp.predict(T)
    assert np.sqrt(np.mean((mu - f(T)) ** 2)) < 2e-3
    far = np.array([[1.35 + 12 * gp.ls[0], 117.0]])                   # many length scales from the data
    mf, sf = gp.predict(far)
    assert abs(mf[0]) < 1e-3 and sf[0] > 100 * sd.mean()             # no data: back to zero, uncertain
    C = np.c_[np.linspace(1.0, 2.0, 41), np.full(41, 117.0)]
    pick = select_uncertain(gp, C, 4)
    assert len(set(pick.tolist())) == 4 and np.ptp(C[pick, 0]) > 0.4  # spread out, outside the data
    s = gp.sample(T[:5], n_samples=50)
    assert s.shape == (50, 5) and np.allclose(s.mean(0), mu[:5], atol=4 * sd[:5].max() + 1e-4)
