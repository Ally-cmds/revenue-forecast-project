import numpy as np
import pandas as pd

from fin_forecast.backtest import rolling_origin
from fin_forecast.features import driver_features, forecast_from_growth
from fin_forecast.models.baselines import BASELINES
from fin_forecast.models.ridge_growth import make_ridge_models


def _data():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2010-03-31", periods=50, freq="QE")
    rows = []
    for t, sec in [("A", "homebuilders"), ("B", "homebuilders"),
                   ("C", "home_improvement"), ("D", "home_improvement")]:
        v = 100 * 1.02 ** np.arange(50) * np.tile([0.9, 1.0, 1.1, 1.0], 13)[:50]
        v = v * (1 + rng.normal(0, 0.02, 50))
        rows += [dict(ticker=t, sector=sec, period_end=d, revenue=x)
                 for d, x in zip(idx, v)]
    qs = pd.date_range("2005-03-31", "2026-12-31", freq="QE")
    macro = pd.DataFrame({"mortgage30": 4 + rng.normal(0, 0.5, len(qs)).cumsum() * 0.1,
                          "starts": 1000 + rng.normal(0, 50, len(qs)).cumsum()},
                         index=qs)
    return pd.DataFrame(rows), macro


def test_no_future_information_reaches_any_model():
    clean, macro = _data()
    models = {**BASELINES, **make_ridge_models()}
    base = rolling_origin(clean, models, macro)
    row = base[(base.ticker == "A") & (base.period_end == clean.period_end.unique()[35])]
    origin = pd.Timestamp(clean.period_end.unique()[34]) + pd.Timedelta(days=1)
    c2, m2 = clean.copy(), macro.copy()
    c2.loc[c2.period_end >= origin, "revenue"] *= 5      # scramble the future
    m2.loc[m2.index >= origin] *= 5
    alt = rolling_origin(c2, make_models := {**BASELINES, **make_ridge_models()}, m2)
    alt = alt[(alt.ticker == "A") & (alt.period_end == row.period_end.iloc[0])]
    a = row.set_index("model").forecast.sort_index()
    b = alt.set_index("model").forecast.sort_index()
    assert np.allclose(a.values, b.values)


def test_zero_growth_returns_revenue_from_four_quarters_ago():
    h = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    assert forecast_from_growth(h, 0.0) == 20.0


def test_driver_lag_one_is_most_recent_available_quarter():
    qs = pd.date_range("2005-03-31", "2026-12-31", freq="QE")
    macro = pd.DataFrame({"mortgage30": np.arange(len(qs)) ** 2.0}, index=qs)
    f = driver_features(macro, pd.Timestamp("2020-04-01"), ["mortgage30"])
    k = list(qs).index(pd.Timestamp("2020-03-31"))       # latest quarter ended
    assert f["mortgage30_l1"] == k ** 2 - (k - 4) ** 2
    assert f["mortgage30_l2"] == (k - 1) ** 2 - (k - 5) ** 2
