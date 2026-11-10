"""Shared base for models trained on pooled log-YoY growth. Subclasses only say how
to fit and predict (_fit_predict); data prep, fallback and the forecast formula
live here once. Create a NEW model object per backtest run (it caches features)."""
import numpy as np

from fin_forecast.features import build_features, forecast_from_growth, training_set
from fin_forecast.models.baselines import seasonal_drift


def standardise(X):
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd == 0, 1.0, sd)
    return (X - mu) / sd, mu, sd


class PooledGrowthModel:
    def __init__(self, drivers=(), interactions=False, min_rows=40):
        self.drivers, self.interactions, self.min_rows = tuple(drivers), interactions, min_rows
        self._cache, self.n_fallback = {}, 0

    def _fit_predict(self, X, y, x_now, columns) -> float:
        """Fit on (X, y) and return predicted log YoY growth for features x_now.
        columns: feature names, in the column order of X and x_now."""
        raise NotImplementedError

    def __call__(self, history, ctx):
        f = build_features(
            history, ctx.sector, ctx.origin, macro=ctx.macro,
            drivers=self.drivers, interactions=self.interactions,
            cache=self._cache)
        X, y = training_set(
            ctx.panel, macro=ctx.macro, drivers=self.drivers,
            interactions=self.interactions, cache=self._cache)
        if f is None or len(y) < self.min_rows:      # too little data: visible fallback
            self.n_fallback += 1
            return seasonal_drift(history)
        x_now = np.array([f[c] for c in X.columns])
        return forecast_from_growth(history, self._fit_predict(X.to_numpy(float), y, x_now, list(X.columns)))