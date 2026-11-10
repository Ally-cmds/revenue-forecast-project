"""The one-time test evaluation named in the frozen plan. Opening the test set is
logged by splits.reveal_test. Only the finalist and the baseline are run."""
import pandas as pd

from fin_forecast import plan
from fin_forecast.backtest import rolling_origin
from fin_forecast.calibration import add_interval, coverage_report
from fin_forecast.evaluation import summarize
from fin_forecast.models.baselines import BASELINES
from fin_forecast.models.ridge_growth import GrowthRidge
from fin_forecast.splits import assign_split, reveal_test
from fin_forecast.stats_tests import (aligned_losses, block_bootstrap, bootstrap_mean_diff,
                                      bootstrap_relative_improvement, coverage_test,
                                      dm_test, holm_adjust)

BOOT = dict(block=plan.BOOT_BLOCK, n_boot=plan.BOOT_N, seed=plan.BOOT_SEED, ci=plan.CI_LEVEL)


def _compare(sub: pd.DataFrame) -> dict:
    lm, lb = aligned_losses(sub, plan.FINALIST, plan.BASELINE)
    d = lm - lb
    pct = lambda r: {k: (v * 100 if k != "share_below_zero" else v) for k, v in r.items()}
    return {"quarters": len(d), "mape_model_qweighted": lm.mean() * 100,
            "mape_baseline_qweighted": lb.mean() * 100,
            "dm": dm_test(d), "mean_diff_pp": pct(bootstrap_mean_diff(d, **BOOT)),
            "relative_improvement": bootstrap_relative_improvement(lm, lb, **BOOT)}


def run_frozen_test(clean, macro, reason: str) -> dict:
    models = {plan.BASELINE: BASELINES[plan.BASELINE],
              plan.FINALIST: GrowthRidge(plan.DRIVERS, False, plan.RIDGE_ALPHA, plan.MIN_ROWS)}
    res = rolling_origin(clean, models, macro=macro)       # all periods (intervals need history)
    res["split"] = assign_split(res["period_end"])
    test = reveal_test(res, reason)                         # logged opening
    out = {"n_test_forecasts_per_model": int(len(test) / 2)}

    t = test.assign(scope="pooled")
    out["mape_rowlevel"] = pd.concat([
        summarize(t, by=("scope", "model")).rename_axis(index=["scope", "model"]),
        summarize(test, by=("sector", "model")).rename_axis(index=["scope", "model"]),
    ]).reset_index().to_dict("records")

    out["primary"] = _compare(test)
    sectors = sorted(test["sector"].unique())
    by_sector = {s: _compare(test[test["sector"] == s]) for s in sectors}
    for s, p in zip(sectors, holm_adjust([by_sector[s]["dm"]["p_value"] for s in sectors])):
        by_sector[s]["dm_p_holm"] = float(p)
    out["by_sector"] = by_sector
    out["decision"] = ("finalist better" if out["primary"]["mean_diff_pp"]["ci_high"] < 0
                       else "finalist NOT shown to be better")

    iv = add_interval(res, plan.FINALIST, plan.INTERVAL_LEVEL, plan.INTERVAL_MODE)
    iv = iv[iv["split"] == "test"].dropna(subset=["lo"])
    covered = (iv["actual"] >= iv["lo"]) & (iv["actual"] <= iv["hi"])
    out["interval"] = {"level": plan.INTERVAL_LEVEL, "mode": plan.INTERVAL_MODE,
                       "coverage_test": coverage_test(covered, plan.INTERVAL_LEVEL),
                       "report": coverage_report(iv.assign(all="pooled"), by=("all",))
                       .reset_index().to_dict("records"),
                       "report_by_sector": coverage_report(iv).reset_index().to_dict("records")}
    return out


def format_report(out: dict) -> str:
    p = out["primary"]
    lines = [f"TEST RESULT (frozen plan) - {out['n_test_forecasts_per_model']} forecasts per model, "
             f"{p['quarters']} calendar quarters",
             f"  MAPE (quarter-weighted): {plan.FINALIST} {p['mape_model_qweighted']:.2f}%  vs  "
             f"{plan.BASELINE} {p['mape_baseline_qweighted']:.2f}%",
             f"  mean loss difference: {p['mean_diff_pp']['estimate']:+.2f} pp, "
             f"{plan.CI_LEVEL:.0%} bootstrap CI [{p['mean_diff_pp']['ci_low']:+.2f}, "
             f"{p['mean_diff_pp']['ci_high']:+.2f}]",
             f"  DM test: stat {p['dm']['stat']:.2f}, p = {p['dm']['p_value']:.3f}",
             f"  relative improvement: {p['relative_improvement']['estimate']:.1%} "
             f"[{p['relative_improvement']['ci_low']:.1%}, {p['relative_improvement']['ci_high']:.1%}]",
             f"  DECISION (pre-registered rule): {out['decision']}", "", "BY SECTOR (secondary)"]
    for s, r in out["by_sector"].items():
        lines.append(f"  {s}: {r['mape_model_qweighted']:.2f}% vs {r['mape_baseline_qweighted']:.2f}%, "
                     f"diff CI [{r['mean_diff_pp']['ci_low']:+.2f}, {r['mean_diff_pp']['ci_high']:+.2f}] pp, "
                     f"DM p = {r['dm']['p_value']:.3f} (Holm {r['dm_p_holm']:.3f})")
    c = out["interval"]["coverage_test"]
    lines += ["", f"INTERVAL ({out['interval']['level']:.0%}, {out['interval']['mode']}): "
              f"coverage {c['rate']:.1%} ({c['covered']}/{c['n']}), binomial p = {c['p_value']:.3f} (optimistic)"]
    return "\n".join(lines)
