"""Principal-components regression on log YoY growth. The correlated driver lags
are compressed into n_components PCA scores, fitted INSIDE each training window
(so nothing about the forecast quarter reaches the components); own-growth and
sector terms stay as ordinary regressors. Same contract as the other models."""
import numpy as np

from fin_forecast.models.pooled import PooledGrowthModel, standardise

BASE_COLS = ("g1", "g2", "is_hb")
ALL_DRIVERS = ("mortgage30", "starts", "dgs10", "permits")


class GrowthPCA(PooledGrowthModel):
    def __init__(self, drivers=ALL_DRIVERS, n_components=3, min_rows=40):
        super().__init__(drivers, False, min_rows)
        self.n_components = n_components
        self._explained = []        # variance share of the kept components, last fit

    def _fit_predict(self, X, y, x_now, columns):
        Z, mu, sd = standardise(X)
        z = (x_now - mu) / sd
        base = [i for i, c in enumerate(columns) if c in BASE_COLS]
        drv = [i for i, c in enumerate(columns) if c not in BASE_COLS]
        _, s, vt = np.linalg.svd(Z[:, drv], full_matrices=False)  # Z is centred
        k = min(self.n_components, len(drv))
        V = vt[:k].T
        self._explained = (s ** 2 / (s ** 2).sum())[:k].tolist()
        A = np.column_stack([np.ones(len(y)), Z[:, base], Z[:, drv] @ V])
        beta = np.linalg.lstsq(A, y, rcond=None)[0]
        a_now = np.concatenate([[1.0], z[base], z[drv] @ V])
        return float(a_now @ beta)


def make_pca_grid(ks, drivers=ALL_DRIVERS):
    return {f"pca_k{k}": GrowthPCA(drivers, k) for k in ks}
