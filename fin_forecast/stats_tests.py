"""Paired forecast-comparison tests. Pure functions; no I/O.
Loss differential d = loss(model) - loss(baseline), so d < 0 means the model is better."""
import numpy as np
import pandas as pd
from scipy import stats


def quarterly_loss(res: pd.DataFrame, model: str) -> pd.Series:
    """Mean absolute percentage error per calendar quarter of period_end.
    Firms in the same quarter share shocks, so the quarter (not the forecast) is
    the unit of inference."""
    d = res[res["model"] == model]
    return d.groupby(d["period_end"].dt.to_period("Q"))["abs_pct_error"].mean()


def aligned_losses(res, model, baseline):
    a, b = quarterly_loss(res, model), quarterly_loss(res, baseline)
    idx = a.index.intersection(b.index)
    return a.loc[idx].to_numpy(), b.loc[idx].to_numpy()


def dm_test(d) -> dict:
    """Diebold-Mariano for one-step-ahead forecasts with the Harvey-Leybourne-Newbold
    small-sample correction; two-sided p-value from a t distribution with T-1 df."""
    d = np.asarray(d, float)
    T = len(d)
    stat = d.mean() / np.sqrt(d.var(ddof=0) / T) * np.sqrt((T - 1) / T)
    return {"T": T, "mean_diff": float(d.mean()), "stat": float(stat),
            "p_value": float(2 * stats.t.sf(abs(stat), df=T - 1))}


def _circular_block_indices(T, block, rng):
    n_blocks = int(np.ceil(T / block))
    starts = rng.integers(0, T, size=n_blocks)
    return ((starts[:, None] + np.arange(block)[None, :]) % T).ravel()[:T]


def block_bootstrap(series, stat_fn, block=4, n_boot=10000, seed=0, ci=0.95) -> dict:
    """Circular block bootstrap over time. The SAME resampled quarters are applied to
    every series in `series`, so paired comparisons stay paired."""
    series = [np.asarray(s, float) for s in series]
    T, rng = len(series[0]), np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = _circular_block_indices(T, block, rng)
        vals[i] = stat_fn(*[s[idx] for s in series])
    lo, hi = np.quantile(vals, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {"estimate": float(stat_fn(*series)), "ci_low": float(lo),
            "ci_high": float(hi), "share_below_zero": float((vals < 0).mean())}


def bootstrap_mean_diff(d, **kw) -> dict:
    return block_bootstrap([d], lambda x: x.mean(), **kw)


def bootstrap_relative_improvement(loss_model, loss_base, **kw) -> dict:
    """1 - mean(model loss) / mean(baseline loss): the 'cut error by X%' number."""
    return block_bootstrap([loss_model, loss_base],
                           lambda m, b: 1 - m.mean() / b.mean(), **kw)


def holm_adjust(p) -> np.ndarray:
    p = np.asarray(p, float)
    order, adj, running = np.argsort(p), np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def coverage_test(covered, level) -> dict:
    """Exact binomial test of observed coverage against the nominal level. Treats
    forecasts as independent, which they are not: read the p-value as optimistic."""
    covered = np.asarray(covered, bool)
    k, n = int(covered.sum()), len(covered)
    return {"n": n, "covered": k, "rate": k / n,
            "p_value": float(stats.binomtest(k, n, level).pvalue)}
