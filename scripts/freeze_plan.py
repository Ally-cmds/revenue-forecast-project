"""Write the pre-registered test plan to the run log. Run ONCE, before the test.
Run from repo root: python -m scripts.freeze_plan"""
import pandas as pd

from fin_forecast import config
from fin_forecast.freeze import write_plan

if __name__ == "__main__":
    clean = pd.read_parquet(config.PROJECT_ROOT / "data" / "processed" / "revenue_quarterly.parquet")
    print("Frozen plan written to:", write_plan(clean))
