"""Download/cache FRED drivers and print a health check.
Run from repo root: python -m scripts.check_macro"""
from fin_forecast.data.macro_loader import as_of, load_macro

if __name__ == "__main__":
    m = load_macro()
    print(m.describe().T[["count", "min", "max"]].to_string())
    print("\nLAST ROWS\n", m.tail(4).to_string())
    # Forecasting Q3 2024: Q2 rates usable, Q2 starts/permits not (1-quarter guard)
    print("\nAS OF 2024-07-01\n",
          as_of(m, "2024-07-01").loc[:"2024-09-30"].tail(3).to_string())
