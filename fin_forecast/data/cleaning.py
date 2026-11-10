"""Raw EDGAR rows -> one clean quarterly revenue series per ticker."""
import pandas as pd

from fin_forecast import config

QUARTER_DAYS = (80, 100)    # 13-14 week quarters
ANNUAL_DAYS = (350, 380)    # 52-53 week years
START_TOLERANCE = pd.Timedelta(days=7)


def _prepare(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[raw["metric"] == "revenue"].copy()
    for col in ("start", "end", "filed"):
        df[col] = pd.to_datetime(df[col])
    df["days"] = (df["end"] - df["start"]).dt.days
    return df


def _dedupe_filings(df: pd.DataFrame) -> pd.DataFrame:
    """Per ticker+tag+period keep the latest filing (handles restatements)."""
    df = df.sort_values(
        ["ticker", "tag", "start", "end", "filed"],
        ascending=[True, True, True, True, False],
    )
    return df.drop_duplicates(["ticker", "tag", "start", "end"], keep="first")


def _derive_q4(df: pd.DataFrame) -> pd.DataFrame:
    """Q4 = full-year minus the three quarters, using the SAME tag for all four
    numbers so the pieces measure the same thing."""
    rows = []
    for (ticker, tag), g in df.groupby(["ticker", "tag"]):
        quarters = g[g["days"].between(*QUARTER_DAYS)]
        for _, fy in g[g["days"].between(*ANNUAL_DAYS)].iterrows():
            inside = quarters[
                (quarters["start"] >= fy["start"] - START_TOLERANCE)
                & (quarters["end"] < fy["end"])
            ]
            if len(inside) == 3:
                rows.append({
                    "ticker": ticker, "sector": fy["sector"], "tag": tag,
                    "end": fy["end"], "val": fy["val"] - inside["val"].sum(),
                    "derived": True,
                })
    return pd.DataFrame(rows)


def _choose_by_priority(df: pd.DataFrame) -> pd.DataFrame:
    """One row per ticker+quarter: best tag by config order, direct over derived."""
    prio = {t: i for i, t in enumerate(config.METRIC_TAGS["revenue"])}
    df = df.assign(prio=df["tag"].map(prio))
    df = df.sort_values(["ticker", "end", "prio", "derived"])
    return df.drop_duplicates(["ticker", "end"], keep="first")


def clean_revenue(raw: pd.DataFrame) -> pd.DataFrame:
    df = _dedupe_filings(_prepare(raw))
    q = df[df["days"].between(*QUARTER_DAYS)].assign(derived=False)
    out = _choose_by_priority(pd.concat([q, _derive_q4(df)], ignore_index=True))
    out = out.rename(columns={"end": "period_end", "val": "revenue"})
    cols = ["ticker", "sector", "period_end", "revenue", "tag", "derived"]
    return out[cols].sort_values(["ticker", "period_end"]).reset_index(drop=True)


def trim_early_gaps(clean: pd.DataFrame, before: str = "2012-01-01",
                    max_days: int = 120) -> pd.DataFrame:
    """Drop the stub before a gap in early XBRL years. Later gaps stay visible."""
    cutoff, parts = pd.Timestamp(before), []
    for _, g in clean.groupby("ticker"):
        g = g.sort_values("period_end").reset_index(drop=True)
        gap = g["period_end"].diff().dt.days
        early = g.index[(gap > max_days) & (g["period_end"] < cutoff)]
        parts.append(g.iloc[int(early.max()):] if len(early) else g)
    return pd.concat(parts, ignore_index=True)


def validate(clean: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker health check: depth, gaps, tag switches, suspicious values."""
    rows = []
    for t, g in clean.groupby("ticker"):
        gaps = g["period_end"].diff().dt.days
        rows.append({
            "ticker": t, "quarters": len(g),
            "first": g["period_end"].min().date(),
            "last": g["period_end"].max().date(),
            "gaps_over_120d": int((gaps > 120).sum()),
            "tags_used": g["tag"].nunique(),
            "derived_q4": int(g["derived"].sum()),
            "min_revenue": g["revenue"].min(),
        })
    return pd.DataFrame(rows).set_index("ticker")


def find_gaps(clean: pd.DataFrame, max_days: int = 120) -> pd.DataFrame:
    """Where a quarter is missing: the two period ends either side of each gap."""
    rows = []
    for t, g in clean.groupby("ticker"):
        g = g.reset_index(drop=True)
        gap = g["period_end"].diff().dt.days
        for i in gap[gap > max_days].index:
            rows.append({"ticker": t, "before": g.loc[i - 1, "period_end"].date(),
                         "after": g.loc[i, "period_end"].date(), "days": int(gap[i])})
    return pd.DataFrame(rows)


def tag_switches(clean: pd.DataFrame) -> pd.DataFrame:
    """Rows where the revenue tag changes, with the quarter-on-quarter ratio.
    Compare against normal seasonality before calling a ratio a definition jump."""
    rows = []
    for t, g in clean.groupby("ticker"):
        g = g.reset_index(drop=True)
        for i in range(1, len(g)):
            if g.loc[i, "tag"] != g.loc[i - 1, "tag"]:
                rows.append({
                    "ticker": t, "period_end": g.loc[i, "period_end"].date(),
                    "from_tag": g.loc[i - 1, "tag"], "to_tag": g.loc[i, "tag"],
                    "qoq_ratio": round(g.loc[i, "revenue"] / g.loc[i - 1, "revenue"], 2),
                })
    return pd.DataFrame(rows)


