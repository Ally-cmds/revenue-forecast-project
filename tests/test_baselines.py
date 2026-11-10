import pandas as pd
import pytest

from fin_forecast.models.baselines import naive_last, seasonal_drift, seasonal_naive


def _series(vals, dtype=float):
    idx = pd.date_range("2020-03-31", periods=len(vals), freq="QE")
    return pd.Series(vals, index=idx, dtype=dtype)


def test_known_values():
    h = _series([100, 110, 120, 130, 120, 132])
    assert naive_last(h) == 132
    assert seasonal_naive(h) == 120
    assert seasonal_drift(h) == pytest.approx(120 * 132 / 110)


def test_seasonal_drift_no_int64_overflow():
    big = 9_000_000_000_000  # a*b would overflow int64
    h = _series([big] * 6, dtype="int64")
    assert seasonal_drift(h) == pytest.approx(big)


def test_ctx_is_optional():
    h = _series([1, 2, 3, 4, 5, 6])
    assert naive_last(h, None) == naive_last(h)
