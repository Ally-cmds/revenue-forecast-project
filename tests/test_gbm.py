import numpy as np
import pandas as pd

from fin_forecast.backtest import rolling_origin
from fin_forecast.models.gbm import GBMGrowth


def _data(seed=1, n=300):
    rng = np.random.default_rng(seed)
    cols = ["g1", "g2", "is_hb", "m_l1", "m_l2"]
    X = rng.normal(size=(n, len(cols)))
    y = np.where(X[:, 3] > 0, 0.10, -0.10) + rng.normal(size=n) * 0.01  # step signal
    return X, y, cols


def test_gbm_learns_a_nonlinear_signal():
    X, y, cols = _data()
    m = GBMGrowth(("m",), max_depth=2)
    hi = m._fit_predict(X, y, np.array([0, 0, 0, 2.0, 0]), cols)
    lo = m._fit_predict(X, y, np.array([0, 0, 0, -2.0, 0]), cols)
    assert hi > 0.05 and lo < -0.05


def test_gbm_is_deterministic_and_returns_float():
    X, y, cols = _data()
    x_now = np.zeros(len(cols))
    a = GBMGrowth(("m",))._fit_predict(X, y, x_now, cols)
    b = GBMGrowth(("m",))._fit_predict(X, y, x_now, cols)
    assert isinstance(a, float) and a == b


def test_no_future_information_reaches_gbm(panel_data):
    clean, macro = panel_data
    make = lambda: {"gbm_d2": GBMGrowth(("mortgage30", "starts"), max_depth=2)}
    base = rolling_origin(clean, make(), macro)
    target = clean.period_end.unique()[35]
    origin = pd.Timestamp(clean.period_end.unique()[34]) + pd.Timedelta(days=1)
    c2, m2 = clean.copy(), macro.copy()
    c2.loc[c2.period_end >= origin, "revenue"] *= 5
    m2.loc[m2.index >= origin] *= 5
    alt = rolling_origin(c2, make(), m2)
    pick = lambda d: d[(d.ticker == "A") & (d.period_end == target)].forecast.values
    assert np.allclose(pick(base), pick(alt))
