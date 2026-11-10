"""Shared validation-only tuning runner. Test years are never computed (stop_before).
A tune script supplies only its candidate models, drivers and note."""
import pandas as pd

from fin_forecast import config
from fin_forecast.backtest import rolling_origin
from fin_forecast.data.macro_loader import load_macro
from fin_forecast.evaluation import summarize
from fin_forecast.models.baselines import BASELINES
from fin_forecast.models.ridge_growth import make_ridge_models
from fin_forecast.run_log import log_run
from fin_forecast.splits import assign_split, only


def reference_models():
    """Models every tuning run is compared against."""
    return {"seasonal_drift": BASELINES["seasonal_drift"],
            "ridge_drivers": make_ridge_models()["ridge_drivers"]}


REFERENCE_DRIVERS = ("mortgage30", "starts")  # assumed to match ridge_drivers


def backtest(candidates, drivers=REFERENCE_DRIVERS):
    """Rolling-origin backtest of references + candidates; test years never computed.
    Returns (results with a split column, models, clean data)."""
    proc = config.PROJECT_ROOT / "data" / "processed"
    clean = pd.read_parquet(proc / "revenue_quarterly.parquet")
    macro = load_macro(list(drivers))
    models = {**reference_models(), **candidates}
    res = rolling_origin(clean, models, macro=macro, stop_before=config.TEST_START)
    res["split"] = assign_split(res["period_end"])
    return res, models, clean


def run_validation(name, candidates, drivers, note, report=None):
    """Backtest references + candidates on validation, print results, log the run.
    report: optional zero-argument function for model-specific printing."""
    res, models, clean = backtest(candidates, drivers)
    val = only(res, "validation")

    overall = val.groupby("model")["abs_pct_error"].mean().mul(100).round(2).sort_values()
    print("VALIDATION MAPE, all companies pooled (lower is better)\n", overall.to_string())
    print("\nBY SECTOR\n", summarize(val).to_string())
    if report:
        report()
    print("\nFALLBACKS:", {k: m.n_fallback for k, m in models.items()
                           if hasattr(m, "n_fallback")})
    print("\nRun record saved:", log_run(name, res, models, clean, notes=note))
    return res