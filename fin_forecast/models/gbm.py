"""Gradient-boosted trees on pooled log-YoY growth. Only defines how to fit;
data prep, fallback and the forecast formula live in PooledGrowthModel."""
from sklearn.ensemble import HistGradientBoostingRegressor

from fin_forecast.models.pooled import PooledGrowthModel


class GBMGrowth(PooledGrowthModel):
    def __init__(self, drivers=(), max_depth=2, learning_rate=0.05, max_iter=200,
                 min_samples_leaf=20, l2_regularization=1.0, min_rows=40):
        super().__init__(drivers=drivers, interactions=False, min_rows=min_rows)
        self.params = dict(
            max_depth=max_depth, learning_rate=learning_rate, max_iter=max_iter,
            min_samples_leaf=min_samples_leaf, l2_regularization=l2_regularization,
            early_stopping=False, random_state=0)  # fixed seed: reproducible runs

    def _fit_predict(self, X, y, x_now, columns) -> float:
        # trees need no standardising
        model = HistGradientBoostingRegressor(**self.params).fit(X, y)
        return float(model.predict(x_now.reshape(1, -1))[0])


# Must match the drivers used by ridge_drivers, so the comparison is fair.
DEFAULT_DRIVERS = ("mortgage30", "starts")


def make_gbm_grid(depths=(2, 3), drivers=DEFAULT_DRIVERS):
    """One fresh model per depth (pooled.py needs a new object per run)."""
    return {f"gbm_d{d}": GBMGrowth(drivers, max_depth=d) for d in depths}