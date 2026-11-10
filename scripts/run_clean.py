"""Raw SEC cache -> clean quarterly revenue + health checks.
Run from repo root: python -m scripts.run_clean"""
import pandas as pd

from fin_forecast import config
from fin_forecast.data.cleaning import (clean_revenue, find_gaps, tag_switches,
                                        trim_early_gaps, validate)

if __name__ == "__main__":
    raw = pd.read_parquet(config.CACHE_DIR / "raw_all.parquet")
    clean = trim_early_gaps(clean_revenue(raw))
    print(validate(clean).to_string())
    print("\nGAPS\n", find_gaps(clean).to_string(index=False))
    print("\nTAG SWITCHES\n", tag_switches(clean).to_string(index=False))
    out_dir = config.PROJECT_ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(out_dir / "revenue_quarterly.parquet")
