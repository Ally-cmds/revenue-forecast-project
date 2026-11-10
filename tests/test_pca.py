import numpy as np
import pandas as pd

from fin_forecast.backtest import rolling_origin
from fin_forecast.models.pca import GrowthPCA, make_pca_grid
from fin_forecast.models.pooled import standardise


def test_pca_with_all_components_equals_plain_regression():
    rng = np.random.default_rng(1)
    cols = ["g1", "g2", "is_hb"] + [f"m_l{k}" for k in range(1, 5)] + [f"s_l{k}" for k in range(1, 5)]
    X = rng.normal(size=(200, len(cols)))
    X[:, 4:7] += 0.8 * X[:, [3]]                      # correlated driver lags
    y = X @ rng.normal(size=len(cols)) + rng.normal(size=200) * 0.1
    x_now = rng.normal(size=len(cols))
    got = GrowthPCA(("m", "s"), n_components=8)._fit_predict(X, y, x_now, cols)
    Z, mu, sd = standardise(X)
    beta = np.linalg.lstsq(np.column_stack([np.ones(200), Z]), y, rcond=None)[0]
    want = float(np.concatenate([[1.0], (x_now - mu) / sd]) @ beta)
    assert np.isclose(got, want)


def test_pca_keeps_requested_number_of_components():
    rng = np.random.default_rng(2)
    cols = ["g1", "g2", "is_hb", "m_l1", "m_l2", "m_l3"]
    m = GrowthPCA(("m",), n_components=2)
    m._fit_predict(rng.normal(size=(60, 6)), rng.normal(size=60), rng.normal(size=6), cols)
    assert len(m._explained) == 2 and 0 < sum(m._explained) <= 1


def test_no_future_information_reaches_pca(panel_data):
    clean, macro = panel_data
    make = lambda: make_pca_grid([2], drivers=("mortgage30", "starts"))
    base = rolling_origin(clean, make(), macro)
    target = clean.period_end.unique()[35]
    origin = pd.Timestamp(clean.period_end.unique()[34]) + pd.Timedelta(days=1)
    c2, m2 = clean.copy(), macro.copy()
    c2.loc[c2.period_end >= origin, "revenue"] *= 5
    m2.loc[m2.index >= origin] *= 5
    alt = rolling_origin(c2, make(), m2)
    pick = lambda d: d[(d.ticker == "A") & (d.period_end == target)].forecast.values
    assert np.allclose(pick(base), pick(alt))
