"""Pooled ridge regression on log YoY growth.
    forecaster(history, ctx) -> float   (same contract as the baselines)"""
import numpy as np

from fin_forecast.models.pooled import PooledGrowthModel, standardise


class GrowthRidge(PooledGrowthModel):
    def __init__(self, drivers=(), interactions=False, alpha=10.0, min_rows=40):
        super().__init__(drivers, interactions, min_rows)
        self.alpha = alpha

    def _fit_predict(self, X, y, x_now, columns):
        Z, mu, sd = standardise(X)
        ym = y.mean()
        beta = np.linalg.solve(Z.T @ Z + self.alpha * np.eye(Z.shape[1]), Z.T @ (y - ym))
        return float(ym + ((x_now - mu) / sd) @ beta)


def make_ridge_models(alpha=10.0, drivers=("mortgage30", "starts")):
    """The comparison ladder: own growth -> + drivers -> + sector x driver."""
    return {
        "ridge_own": GrowthRidge((), False, alpha),
        "ridge_drivers": GrowthRidge(drivers, False, alpha),
        "ridge_drivers_x_sector": GrowthRidge(drivers, True, alpha),
    }