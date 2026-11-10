import numpy as np
import pandas as pd

from fin_forecast.backtest import rolling_origin
from fin_forecast.models.baselines import BASELINES
from fin_forecast.models.elastic_net import make_elastic_net_grid


def test_no_future_information_reaches_elastic_net(panel_data):
    clean, macro = panel_data
    make = lambda: make_elastic_net_grid([0.01], [0.5])
    base = rolling_origin(clean, make(), macro)
    target = clean.period_end.unique()[35]
    origin = pd.Timestamp(clean.period_end.unique()[34]) + pd.Timedelta(days=1)
    c2, m2 = clean.copy(), macro.copy()
    c2.loc[c2.period_end >= origin, "revenue"] *= 5
    m2.loc[m2.index >= origin] *= 5
    alt = rolling_origin(c2, make(), m2)
    pick = lambda d: d[(d.ticker == "A") & (d.period_end == target)].forecast.values
    assert np.allclose(pick(base), pick(alt))


def test_stop_before_never_computes_later_forecasts(panel_data):
    clean, macro = panel_data
    res = rolling_origin(clean, BASELINES, macro, stop_before="2018-01-01")
    assert res["period_end"].max() < pd.Timestamp("2018-01-01")
    assert len(res) > 0


def test_elastic_net_is_sane_on_easy_data(panel_data):
    clean, macro = panel_data
    models = {"naive_last": BASELINES["naive_last"], **make_elastic_net_grid([0.01], [0.5])}
    res = rolling_origin(clean, models, macro)
    mape = res.groupby("model")["abs_pct_error"].mean()
    assert mape["enet_a0.01_l0.5"] < mape["naive_last"]
