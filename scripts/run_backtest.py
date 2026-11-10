"""Run baselines + ridge ladder through the backtest. Test period stays locked.
Run from repo root: python -m scripts.run_backtest"""
import pandas as pd

from fin_forecast import config
from fin_forecast.backtest import rolling_origin
from fin_forecast.data.macro_loader import load_macro
from fin_forecast.evaluation import summarize
from fin_forecast.run_log import log_run
from fin_forecast.models.baselines import BASELINES
from fin_forecast.models.ridge_growth import make_ridge_models
from fin_forecast.splits import assign_split, hide_test, only

if __name__ == "__main__":
    proc = config.PROJECT_ROOT / "data" / "processed"
    clean = pd.read_parquet(proc / "revenue_quarterly.parquet")
    macro = load_macro(["mortgage30", "starts"])
    ridge = make_ridge_models()   # new objects for every run
    models = {**BASELINES, **ridge}
    res = rolling_origin(clean, models, macro=macro)
    res["split"] = assign_split(res["period_end"])
    res.to_parquet(proc / "backtest_results.parquet")

    visible = hide_test(res)
    print("BY SPLIT AND SECTOR (test period hidden)\n",
          summarize(visible, by=("split", "sector", "model")).to_string())
    val = only(res, "validation")
    print("\nVALIDATION: MAPE BY COMPANY (%)\n",
          summarize(val, by=("ticker", "model"))["mape"].unstack().to_string())
    print("\nRIDGE FALLBACKS (used seasonal_drift instead):",
          {k: m.n_fallback for k, m in ridge.items()})
    print("\nRun record saved:", log_run("baselines_ridge", res, models, clean))