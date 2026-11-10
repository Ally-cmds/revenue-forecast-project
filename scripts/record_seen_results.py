"""One-off: record the all-period results seen before the split existed.
Run after scripts.run_backtest:  python -m scripts.record_seen_results"""
import pandas as pd

from fin_forecast import config
from fin_forecast.run_log import log_seen_results

if __name__ == "__main__":
    proc = config.PROJECT_ROOT / "data" / "processed"
    res = pd.read_parquet(proc / "backtest_results.parquet")
    clean = pd.read_parquet(proc / "revenue_quarterly.parquet")
    print("Saved:", log_seen_results(res, clean))
