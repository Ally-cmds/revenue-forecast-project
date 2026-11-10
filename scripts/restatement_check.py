"""Measures look-ahead bias: how much quarterly revenue changed between its
first and latest filing. Run: python -m scripts.restatement_check"""
import pandas as pd

from fin_forecast import config
from fin_forecast.data.cleaning import QUARTER_DAYS


def restatement_summary(raw: pd.DataFrame) -> pd.DataFrame:
    r = raw[raw["metric"] == "revenue"].copy()
    for col in ("start", "end", "filed"):
        r[col] = pd.to_datetime(r[col])
    r = r[(r["end"] - r["start"]).dt.days.between(*QUARTER_DAYS)].sort_values("filed")
    g = r.groupby(["ticker", "tag", "start", "end"])["val"]
    chk = pd.DataFrame({"first": g.first(), "last": g.last()})
    chk["pct_diff"] = (chk["last"] / chk["first"] - 1).abs() * 100
    return chk.groupby("ticker").agg(
        mean=("pct_diff", "mean"),
        median=("pct_diff", "median"),
        max=("pct_diff", "max"),
    ).round(2)


if __name__ == "__main__":
    raw = pd.read_parquet(config.CACHE_DIR / "raw_all.parquet")
    keep = [t for tks in config.SECTORS.values() for t in tks]
    summary = restatement_summary(raw[raw["ticker"].isin(keep)])
    print(summary)
    out_dir = config.PROJECT_ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "restatement_check.csv")
