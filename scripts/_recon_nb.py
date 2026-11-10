"""One-off recon: parquet schemas for the SQL notebook (delete after)."""
import pandas as pd

for p in ("data/processed/backtest_results.parquet",
          "data/processed/revenue_quarterly.parquet",
          "data/raw/raw_all.parquet",
          "data/raw/fred_MORTGAGE30US.parquet"):
    df = pd.read_parquet(p)
    print("=" * 20, p, f"{len(df):,} rows")
    print(df.dtypes.to_string())
    print(df.head(3).to_string())