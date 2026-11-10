import numpy as np
import pandas as pd

from fin_forecast import splits
from fin_forecast.calibration import add_interval


def _res():
    """Two companies in one sector with different fiscal calendars (B is 40 days later)."""
    rng = np.random.default_rng(3)
    rows = []
    for t, shift in (("A", 0), ("B", 40)):
        ends = pd.date_range("2010-03-31", periods=40, freq="QE") + pd.Timedelta(days=shift)
        for end in ends:
            actual = 100.0
            rows.append({"ticker": t, "sector": "s", "period_end": end,
                         "origin": end - pd.Timedelta(days=90), "model": "m",
                         "actual": actual,
                         "forecast": actual * np.exp(rng.normal(0.02, 0.05))})
    res = pd.DataFrame(rows)
    res["pct_error"] = (res.forecast - res.actual) / res.actual
    res["abs_pct_error"] = res.pct_error.abs()
    res["split"] = splits.assign_split(res["period_end"])
    return res


def _scramble_from(res, cutoff):
    r = res.copy()
    late = r["period_end"] >= cutoff
    r.loc[late, "forecast"] = r.loc[late, "actual"] * 50    # absurd errors
    return r


def _bounds(d, ticker, period_end):
    """lo, hi, point relative to the row own forecast (the test scrambles that
    forecast too, so only the ratios say anything about the pool used)."""
    row = d[(d.ticker == ticker) & (d.period_end == period_end)].iloc[0]
    f = row["forecast"]
    return row["lo"] / f, row["hi"] / f, row["point"] / f


def test_expanding_interval_ignores_errors_not_yet_known_at_origin():
    res = _res()
    row = res[(res.ticker == "A") & (res.period_end == pd.Timestamp("2016-06-30"))].iloc[0]
    # A quarter of B ends after A origin but before A target date. The old rule
    # (cutoff = target date) let its error in; the origin rule must not.
    scrambled = _scramble_from(res, row["origin"])
    a = add_interval(res, "m", mode="expanding")
    b = add_interval(scrambled, "m", mode="expanding")
    assert np.allclose(_bounds(a, "A", row["period_end"]), _bounds(b, "A", row["period_end"]))


def test_frozen_interval_ignores_everything_after_the_calibration_period():
    res = _res()
    row = res[(res.ticker == "A") & (res.period_end == pd.Timestamp("2019-06-30"))].iloc[0]
    scrambled = _scramble_from(res, pd.Timestamp("2018-01-01"))
    a = add_interval(res, "m", mode="frozen")
    b = add_interval(scrambled, "m", mode="frozen")
    assert np.allclose(_bounds(a, "A", row["period_end"]), _bounds(b, "A", row["period_end"]))


def test_bias_correction_removes_a_constant_overforecast():
    res = _res()
    res["forecast"] = res["actual"] * 1.10                   # always 10% too high
    d = add_interval(res, "m", mode="expanding").dropna(subset=["lo"])
    assert np.allclose(d["point"], d["actual"], rtol=1e-6)