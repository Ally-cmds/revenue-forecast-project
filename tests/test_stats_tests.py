import numpy as np
import pandas as pd
from scipy import stats

from fin_forecast.stats_tests import (block_bootstrap, bootstrap_mean_diff,
                                      bootstrap_relative_improvement, coverage_test,
                                      dm_test, holm_adjust, quarterly_loss)


def test_dm_matches_a_manual_calculation():
    d = np.array([-0.02, 0.01, -0.03, -0.01, 0.00, -0.02, -0.04, 0.01])
    T = len(d)
    stat = d.mean() / np.sqrt(d.var() / T) * np.sqrt((T - 1) / T)
    out = dm_test(d)
    assert np.isclose(out["stat"], stat)
    assert np.isclose(out["p_value"], 2 * stats.t.sf(abs(stat), T - 1))


def test_dm_flags_a_clear_improvement_and_not_pure_noise():
    rng = np.random.default_rng(0)
    better = dm_test(-0.02 + rng.normal(0, 0.005, 18))
    noise = dm_test(rng.normal(0, 0.02, 18))
    assert better["stat"] < 0 and better["p_value"] < 0.01
    assert noise["p_value"] > 0.05


def test_bootstrap_is_reproducible_and_brackets_the_estimate():
    d = np.random.default_rng(1).normal(-0.02, 0.01, 18)
    a, b = bootstrap_mean_diff(d, n_boot=2000, seed=5), bootstrap_mean_diff(d, n_boot=2000, seed=5)
    assert a == b and a["ci_low"] < a["estimate"] < a["ci_high"]


def test_constant_series_gives_a_degenerate_interval():
    r = bootstrap_mean_diff(np.full(18, -0.01), n_boot=500)
    assert np.isclose(r["ci_low"], -0.01) and np.isclose(r["ci_high"], -0.01)


def test_relative_improvement_recovers_a_known_halving():
    base = np.random.default_rng(2).uniform(0.05, 0.12, 18)
    r = bootstrap_relative_improvement(0.5 * base, base, n_boot=500)
    assert np.isclose(r["estimate"], 0.5) and np.isclose(r["ci_low"], 0.5)


def test_paired_bootstrap_resamples_the_same_quarters_for_both_series():
    x = np.arange(18, dtype=float)
    r = block_bootstrap([x, x], lambda a, b: float((a - b).max()), n_boot=200)
    assert r["estimate"] == 0.0 and r["ci_high"] == 0.0


def test_holm_adjustment():
    assert np.allclose(holm_adjust([0.01, 0.04]), [0.02, 0.04])
    assert np.allclose(holm_adjust([0.03, 0.03]), [0.06, 0.06])


def test_coverage_test_accepts_nominal_and_rejects_a_clear_miss():
    assert coverage_test([True] * 80 + [False] * 20, 0.8)["p_value"] > 0.9
    assert coverage_test([True] * 50 + [False] * 50, 0.8)["p_value"] < 0.001


def test_quarterly_loss_groups_by_calendar_quarter():
    res = pd.DataFrame({"model": "m", "abs_pct_error": [0.1, 0.3, 0.2],
                        "period_end": pd.to_datetime(["2022-01-31", "2022-03-31", "2022-06-30"])})
    q = quarterly_loss(res, "m")
    assert len(q) == 2 and np.isclose(q.iloc[0], 0.2)
