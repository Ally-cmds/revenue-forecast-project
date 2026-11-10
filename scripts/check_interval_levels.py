"""Step 1 of the report plan: how well do 80% / 90% / 95% intervals for seasonal_drift
cover actuals on the VALIDATION years? Test years are never computed (stop_before).
Decides how the red flag tier is defined. Run from repo root:
    python -m scripts.check_interval_levels"""
import pandas as pd

from fin_forecast import config
from fin_forecast.backtest import rolling_origin
from fin_forecast.calibration import add_interval
from fin_forecast.models.baselines import BASELINES
from fin_forecast.run_log import log_run
from fin_forecast.splits import assign_split, only

MODEL, LEVELS, MODE = "seasonal_drift", (0.8, 0.9, 0.95), "expanding"


def level_table(d: pd.DataFrame, level: float, by: str = None) -> pd.DataFrame:
    d = d.dropna(subset=["lo"]).copy()
    d["below"], d["above"] = d["actual"] < d["lo"], d["actual"] > d["hi"]
    d["covered"] = ~(d["below"] | d["above"])
    d["width_pct"] = (d["hi"] - d["lo"]) / d["point"] * 100
    g = d.groupby(by) if by else d.groupby(lambda _: "pooled")
    return pd.DataFrame({"nominal": level * 100, "observed": g["covered"].mean() * 100,
                         "below_lo": g["below"].mean() * 100, "above_hi": g["above"].mean() * 100,
                         "width_pct": g["width_pct"].mean(), "n": g.size()}).round(1)


if __name__ == "__main__":
    proc = config.PROJECT_ROOT / "data" / "processed"
    clean = pd.read_parquet(proc / "revenue_quarterly.parquet")
    models = {MODEL: BASELINES[MODEL]}
    res = rolling_origin(clean, models, macro=None, stop_before=config.TEST_START)
    assert res["period_end"].max() < pd.Timestamp(config.TEST_START), "test rows computed"
    res["split"] = assign_split(res["period_end"])

    pooled, by_sector = [], []
    for level in LEVELS:
        val = only(add_interval(res, MODEL, level, MODE), "validation")
        pooled.append(level_table(val, level))
        by_sector.append(level_table(val, level, by="sector"))
    print(f"VALIDATION ONLY: {MODEL}, {MODE} intervals (observed vs nominal coverage, %)")
    print("\nPOOLED\n", pd.concat(pooled).to_string())
    print("\nBY SECTOR\n", pd.concat(by_sector).to_string())
    note = (f"interval level check for {MODEL}: levels {LEVELS}, {MODE}; validation only, "
            "no test forecasts computed")
    print("\nRun record saved:", log_run("check_interval_levels", res, models, clean, notes=note))
