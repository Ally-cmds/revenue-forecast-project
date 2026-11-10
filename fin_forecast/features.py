"""Shared feature builders. One function builds BOTH the training rows and the
forecast row, so the two can never drift apart.
Target: log YoY revenue growth. Forecast = revenue 4 quarters ago * exp(growth)."""
import numpy as np
import pandas as pd

from fin_forecast.data.macro_loader import as_of

HOMEBUILDERS = "homebuilders"
RATE_DRIVERS = {"mortgage30", "dgs10"}  # YoY change in pct points; others: log change
N_LAGS = 4
MIN_OWN_HISTORY = 6                     # g2 needs the quarter 6 back


def driver_features(macro, origin, drivers, cache=None):
    """Lags 1..N_LAGS of each driver's YoY change, using only data published
    before `origin` (lag 1 = most recent available quarter). None if too short.
    cache: dict owned by the caller, keyed by origin (valid within one backtest)."""
    key = (origin, tuple(drivers))
    if cache is not None and key in cache:
        return cache[key]
    masked, feats = as_of(macro, origin), {}
    for d in drivers:
        x = masked[d]
        yoy = x.diff(4) if d in RATE_DRIVERS else np.log(x).diff(4)
        v = yoy.dropna().iloc[-N_LAGS:]
        if len(v) < N_LAGS:
            feats = None
            break
        feats.update({f"{d}_l{k}": float(val) for k, val in enumerate(v.iloc[::-1], 1)})
    if cache is not None:
        cache[key] = feats
    return feats


def build_features(history, sector, origin, macro=None, drivers=(),
                   interactions=False, cache=None):
    """Feature dict for forecasting the quarter after `history`, or None."""
    h = history.astype(float)
    if len(h) < MIN_OWN_HISTORY:
        return None
    f = {"g1": float(np.log(h.iloc[-1] / h.iloc[-5])),
         "g2": float(np.log(h.iloc[-2] / h.iloc[-6])),
         "is_hb": float(sector == HOMEBUILDERS)}
    if drivers:
        if macro is None:
            raise ValueError("drivers requested but no macro data in context")
        d = driver_features(macro, origin, drivers, cache)
        if d is None:
            return None
        f.update(d)
        if interactions:
            f.update({f"hb_x_{k}": f["is_hb"] * v for k, v in d.items()})
    return f


def training_set(panel, macro=None, drivers=(), interactions=False, cache=None):
    """Pooled rows from every company's past quarters (panel is pre-filtered to
    quarters that ended before the forecast origin). Returns (X, y)."""
    rows, ys = [], []
    for _, g in panel.groupby("ticker"):
        g = g.sort_values("period_end")
        s = g.set_index("period_end")["revenue"].astype(float)
        sector = g["sector"].iloc[0]
        for i in range(MIN_OWN_HISTORY, len(s)):
            origin = s.index[i - 1] + pd.Timedelta(days=1)   # same rule as backtest
            f = build_features(s.iloc[:i], sector, origin, macro, drivers,
                               interactions, cache)
            if f is not None:
                rows.append(f)
                ys.append(float(np.log(s.iloc[i] / s.iloc[i - 4])))
    return pd.DataFrame(rows), np.array(ys)


def forecast_from_growth(history, y_hat):
    """Revenue 4 quarters ago * exp(predicted log YoY growth)."""
    return float(history.iloc[-4]) * float(np.exp(y_hat))
