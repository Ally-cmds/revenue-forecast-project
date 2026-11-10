"""Error summaries for backtest results. Percent units throughout."""
import pandas as pd


def summarize(results: pd.DataFrame, by=("sector", "model"),
              exclude_years=()) -> pd.DataFrame:
    df = results[~results["period_end"].dt.year.isin(exclude_years)]
    grp = df.groupby(list(by))
    out = pd.DataFrame({
        "mape": grp["abs_pct_error"].mean() * 100,
        "median_ape": grp["abs_pct_error"].median() * 100,
        "bias": grp["pct_error"].mean() * 100,   # >0 means over-forecasting
        "n": grp.size(),
    })
    return out.round(2)
