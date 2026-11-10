"""Baseline forecasters. Contract (shared with ML models):
    forecaster(history: pd.Series indexed by period_end, ctx: Ctx | None) -> float
Baselines ignore ctx; it is optional so they still work standalone."""
import pandas as pd


def naive_last(history: pd.Series, ctx=None) -> float:
    """Next quarter = the latest quarter."""
    return float(history.iloc[-1])


def seasonal_naive(history: pd.Series, ctx=None) -> float:
    """Next quarter = the same quarter one year ago."""
    return float(history.iloc[-4])


def seasonal_drift(history: pd.Series, ctx=None) -> float:
    """Same quarter last year, scaled by the latest year-on-year growth."""
    a, b, c = (float(history.iloc[i]) for i in (-4, -1, -5))  # float: int64 overflows
    return a * b / c


BASELINES = {
    "naive_last": naive_last,
    "seasonal_naive": seasonal_naive,
    "seasonal_drift": seasonal_drift,
}
