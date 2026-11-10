"""Rolling-origin backtest. Each forecast uses only data available at the time.
Run: python -m scripts.run_backtest"""
import pandas as pd

from fin_forecast.context import Ctx
from fin_forecast.data.macro_loader import as_of

MAX_GAP_DAYS = 120
PANEL_COLS = ["ticker", "sector", "period_end", "revenue"]


def rolling_origin(clean: pd.DataFrame, forecasters: dict,
                   macro: pd.DataFrame | None = None,
                   min_history: int = 12,
                   stop_before: str | None = None) -> pd.DataFrame:
    """forecasters: name -> fn(history, ctx). macro: quarterly drivers (optional);
    each forecast sees only as_of(macro, origin).
    stop_before: do not compute forecasts for quarters ending on/after this date
    (tuning passes the test start, so test forecasts are never even computed)."""
    rows = []
    for ticker, g in clean.groupby("ticker"):
        if not isinstance(ticker, str):
            raise TypeError(f"ticker values must be strings, got {type(ticker).__name__}")
        g = g.sort_values("period_end")
        sector = g["sector"].iloc[0]
        s = g.set_index("period_end")["revenue"].astype(float)
        if (s.index.to_series().diff().dt.days > MAX_GAP_DAYS).any():
            raise ValueError(f"{ticker}: missing quarter; fix the data first "
                             "(quarter-offset baselines assume no gaps).")
        for i in range(min_history, len(s)):
            if stop_before is not None and s.index[i] >= pd.Timestamp(stop_before):
                break
            hist, actual = s.iloc[:i], float(s.iloc[i])
            # Origin = day after the last known quarter ends. Drivers are on
            # calendar quarters; fiscal alignment is handled in features.py.
            origin = s.index[i - 1] + pd.Timedelta(days=1)
            ctx = Ctx(ticker, sector, origin,
                      None if macro is None else as_of(macro, origin),
                      clean.loc[clean["period_end"] < origin, PANEL_COLS])
            for name, fc in forecasters.items():
                rows.append({"ticker": ticker, "sector": sector,
                             "period_end": s.index[i], "origin": origin, "model": name,
                             "actual": actual, "forecast": fc(hist, ctx)})
    df = pd.DataFrame(rows)
    df["pct_error"] = (df["forecast"] - df["actual"]) / df["actual"]
    df["abs_pct_error"] = df["pct_error"].abs()
    return df