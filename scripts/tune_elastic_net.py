"""Tune elastic net on the VALIDATION years only. Test-year forecasts are never
computed (stop_before). Run from repo root: python -m scripts.tune_elastic_net"""
from fin_forecast.models.elastic_net import make_elastic_net_grid
from scripts.tune_common import run_validation

ALPHAS = (0.0003, 0.001, 0.003, 0.01)
L1_RATIOS = (0.1, 0.3, 0.5)
DRIVERS = ("mortgage30", "starts")

if __name__ == "__main__":
    grid = make_elastic_net_grid(ALPHAS, L1_RATIOS)
    note = (f"elastic net grid: alphas={ALPHAS}, l1_ratios={L1_RATIOS} "
            f"({len(grid)} configurations tried on validation); no test forecasts computed")
    run_validation("tune_elastic_net", grid, DRIVERS, note)