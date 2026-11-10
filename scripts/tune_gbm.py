"""Tune GBM on the VALIDATION years only (no test forecasts computed).
Run from repo root: python -m scripts.tune_gbm"""
from fin_forecast.models.gbm import DEFAULT_DRIVERS, make_gbm_grid
from scripts.tune_common import run_validation

DEPTHS = (2, 3)

if __name__ == "__main__":
    grid = make_gbm_grid(DEPTHS)
    note = (f"GBM grid: max_depth={DEPTHS} on drivers {DEFAULT_DRIVERS} "
            f"({len(grid)} configurations tried on validation); no test forecasts computed")
    run_validation("tune_gbm", grid, DEFAULT_DRIVERS, note)
