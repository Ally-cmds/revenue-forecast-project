"""Calibrate ridge_drivers on VALIDATION only: per-sector bias shift + 80% intervals.
frozen    = calibrated on pre-2018 errors only (clean holdout check)
expanding = adds errors as they are realised (the version to use on the test set)
Run from repo root: python -m scripts.calibrate"""
from fin_forecast.calibration import add_interval, coverage_report
from fin_forecast.run_log import log_run
from fin_forecast.splits import only
from scripts.tune_common import backtest

MODEL, LEVEL = "ridge_drivers", 0.8

if __name__ == "__main__":
    res, models, clean = backtest({})
    for mode in ("frozen", "expanding"):
        d = only(add_interval(res, MODEL, LEVEL, mode), "validation")
        d["year"] = d["period_end"].dt.year
        print(f"\n{mode.upper()}: {LEVEL:.0%} interval + bias shift, validation")
        print(coverage_report(d.assign(all="pooled"), by=("all",)).to_string())
        print(coverage_report(d).to_string())
        print(coverage_report(d, by=("sector", "year")).to_string())
    note = (f"calibration of {MODEL}: per-sector median log-error shift + {LEVEL:.0%} "
            "empirical-quantile intervals, modes frozen and expanding "
            "(1 extra configuration); no test forecasts computed")
    print("\nRun record saved:", log_run("calibrate", res, models, clean, notes=note))
