"""Pooled elastic net on log YoY growth. Same contract as the other models.
alpha = overall penalty strength; l1_ratio = share of the L1 penalty (1 = lasso,
which can zero out useless driver lags; 0 = ridge). Settings are tuned on the
VALIDATION years only (scripts/tune_elastic_net.py)."""
import numpy as np
from sklearn.linear_model import ElasticNet

from fin_forecast.models.pooled import PooledGrowthModel, standardise


class GrowthElasticNet(PooledGrowthModel):
    def __init__(self, drivers=(), interactions=False, alpha=0.01, l1_ratio=0.5,
                 min_rows=40):
        super().__init__(drivers, interactions, min_rows)
        self.alpha, self.l1_ratio = alpha, l1_ratio

    def _fit_predict(self, X, y, x_now, columns):
        Z, mu, sd = standardise(X)
        model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=10000)
        model.fit(Z, y)
        return float(model.predict(((x_now - mu) / sd).reshape(1, -1))[0])


def make_elastic_net_grid(alphas, l1_ratios, drivers=("mortgage30", "starts"),
                          interactions=False):
    return {f"enet_a{a}_l{l}": GrowthElasticNet(drivers, interactions, a, l)
            for a in alphas for l in l1_ratios}