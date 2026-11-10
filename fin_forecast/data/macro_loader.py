"""FRED macro drivers -> one quarterly frame, cached like the SEC data.
Check with: python -m scripts.check_macro

Availability rule lives HERE (see as_of) so no model can leak future data."""
import pandas as pd

from fin_forecast import config

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"

# name -> (FRED id, extra publication lag in quarters).
# Rates are never revised -> lag 0. Starts/permits are revised -> 1-quarter guard.
# Quarterly value = mean of the period (monthly/daily SAAR series stay in SAAR units).
SERIES = {
    "mortgage30": ("MORTGAGE30US", 0),
    "starts": ("HOUST", 1),
    "dgs10": ("DGS10", 0),       # optional: makes PCA meaningful
    "permits": ("PERMIT", 1),    # optional
}


def _cache_path(series_id: str):
    return config.CACHE_DIR / f"fred_{series_id}.parquet"


def download_series(series_id: str, refresh: bool = False) -> pd.Series:
    """Raw FRED series (date-indexed floats). Uses cache; falls back to it offline."""
    path = _cache_path(series_id)
    if path.exists() and not refresh:
        return pd.read_parquet(path)["value"]
    try:
        df = pd.read_csv(FRED_URL.format(id=series_id), na_values=".")
    except Exception:
        if path.exists():
            return pd.read_parquet(path)["value"]
        raise
    s = pd.Series(df.iloc[:, 1].astype(float).values,
                  index=pd.to_datetime(df.iloc[:, 0]), name="value").dropna()
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    s.to_frame().to_parquet(path)
    return s


def to_quarterly(s: pd.Series) -> pd.Series:
    """Calendar-quarter mean, indexed by quarter-end date. Drops a trailing
    quarter that is still incomplete (last observation more than ~a month
    before quarter end), so a partial-quarter mean never looks like a full one."""
    idx = pd.DatetimeIndex(s.index)
    q = s.groupby(idx.to_period("Q")).mean()
    q.index = pd.PeriodIndex(q.index).end_time.normalize()
    return q[q.index - pd.Timedelta(days=31) <= idx.max()]


def load_macro(names=None, refresh: bool = False) -> pd.DataFrame:
    """Wide quarterly frame (index = calendar quarter end), one column per driver."""
    names = list(names or SERIES)
    cols = {n: to_quarterly(download_series(SERIES[n][0], refresh)) for n in names}
    return pd.DataFrame(cols).sort_index()


def as_of(macro: pd.DataFrame, origin) -> pd.DataFrame:
    """Only values published before `origin`: quarter must have ended, plus the
    series' extra lag. Unavailable cells become NaN (rows kept, no look-ahead).
    `origin` = first day of the quarter being forecast (prior quarter end + 1 day)."""
    origin, out = pd.Timestamp(origin), macro.copy()
    for col in out.columns:
        lag = SERIES[col][1]
        published = out.index + pd.DateOffset(months=3 * lag)
        out.loc[published >= origin, col] = float("nan")
    return out


