import numpy as np
import pandas as pd

from fin_forecast.backtest import rolling_origin
from fin_forecast.models.baselines import BASELINES


def _clean():
    rng = np.random.default_rng(0)
    qe = pd.date_range("2015-03-31", periods=24, freq="QE")
    return pd.DataFrame({"ticker": "AAA", "sector": "homebuilders", "period_end": qe,
                         "revenue": (1000 + rng.normal(0, 50, 24).cumsum()).round()})


def _macro(index):
    return pd.DataFrame({"mortgage30": 5.0, "starts": 1300.0}, index=index)


def test_baselines_run_with_and_without_macro():
    clean = _clean()
    a = rolling_origin(clean, BASELINES)
    b = rolling_origin(clean, BASELINES, macro=_macro(clean["period_end"]))
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == (24 - 12) * len(BASELINES)


def test_ticker_must_be_a_string():
    clean = _clean()
    clean["ticker"] = b"AAA"

    try:
        rolling_origin(clean, BASELINES)
    except TypeError as exc:
        assert str(exc) == "ticker values must be strings, got bytes"
    else:
        raise AssertionError("non-string ticker should be rejected")


def test_forecaster_never_sees_future_drivers():
    clean = _clean()
    seen = []

    def spy(history, ctx):
        seen.append((ctx.origin, ctx.macro))
        return float(history.iloc[-1])

    rolling_origin(clean, {"spy": spy}, macro=_macro(clean["period_end"]))
    for origin, m in seen:
        # no value from a quarter that ends on/after the forecast origin
        assert m["mortgage30"].dropna().index.max() < origin
        # starts carry an extra quarter of delay
        assert m["starts"].dropna().index.max() <= m["mortgage30"].dropna().index.max()
