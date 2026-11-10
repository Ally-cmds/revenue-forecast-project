"""Pre-registration. write_plan() records the test plan plus a fingerprint of the
code, data and driver files in the run log BEFORE the test set is opened.
verify() lets the one-time test script refuse to run if any of them changed."""
import hashlib
from datetime import datetime
from pathlib import Path

import pandas as pd

from fin_forecast import config, plan, splits
from fin_forecast.run_log import _git_commit, append_record, data_fingerprint, load_runs

PKG = Path(__file__).resolve().parent
FROZEN_CODE = ["features.py", "backtest.py", "calibration.py", "evaluation.py",
               "stats_tests.py", "splits.py", "context.py", "plan.py", "freeze.py",
               "frozen_test.py", "data/macro_loader.py", "models/pooled.py",
               "models/ridge_growth.py", "models/baselines.py"]
FRED_FILES = ["fred_MORTGAGE30US.parquet", "fred_HOUST.parquet"]


def _sha(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:12]


def code_fingerprint(pkg=PKG, files=FROZEN_CODE) -> dict:
    return {f: _sha(pkg / f) for f in files}


def fred_fingerprint(cache_dir=None, files=FRED_FILES) -> dict:
    cache_dir = cache_dir or config.CACHE_DIR
    return {f: _sha(cache_dir / f) for f in files}


def latest_plan(runs_dir=None):
    plans = [r for r in load_runs(runs_dir) if r.get("kind") == "frozen_plan"]
    return plans[-1] if plans else None


def build_plan(clean: pd.DataFrame) -> dict:
    return {
        "kind": "frozen_plan",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "statement": ("Pre-registered BEFORE the test set was opened. Nothing below may "
                      "change after the test result is seen."),
        "split": {"validation_start": config.VALIDATION_START, "test_start": config.TEST_START},
        "finalist": {"model": plan.FINALIST, "class": "GrowthRidge", "alpha": plan.RIDGE_ALPHA,
                     "drivers": list(plan.DRIVERS), "interactions": False,
                     "min_rows": plan.MIN_ROWS},
        "baseline": plan.BASELINE,
        "primary_question": (f"Is the {plan.FINALIST} MAPE lower than the {plan.BASELINE} MAPE "
                             "on the test forecasts, pooled over both sectors?"),
        "inference": {"unit": "calendar quarter of period_end (mean abs pct error per quarter)",
                      "dm_test": "Diebold-Mariano, h=1, Harvey-Leybourne-Newbold correction, two-sided",
                      "bootstrap": {"type": "circular block", "block_quarters": plan.BOOT_BLOCK,
                                    "resamples": plan.BOOT_N, "seed": plan.BOOT_SEED,
                                    "ci_level": plan.CI_LEVEL}},
        "decision_rule": ("The finalist is declared better ONLY if the bootstrap CI for the "
                          "mean loss difference (model - baseline) lies entirely below zero. "
                          "The DM p-value is reported alongside; if they disagree, both are reported."),
        "secondary": ["by-sector comparison with Holm correction across the two sectors",
                      f"{plan.INTERVAL_MODE} {plan.INTERVAL_LEVEL:.0%} interval coverage on test "
                      "(binomial test, optimistic: forecasts are correlated)",
                      "bias-corrected point forecast MAPE",
                      "relative improvement 1 - MAPE_model / MAPE_baseline with bootstrap CI"],
        "not_tested": ["naive_last", "seasonal_naive", "ridge_own", "ridge_drivers_x_sector",
                       "elastic net", "PCA", "GBM (validation-only candidates)"],
        "validation_candidates": plan.VALIDATION_CANDIDATES,
        "known_contamination": ("Test-period errors of the baselines and the ridge models can be "
                                "backed out from all-period tables viewed before the split existed "
                                "(see the pre_split_results_SEEN record). Validation selected the "
                                "finalist; the test set was not computed during tuning."),
        "data": data_fingerprint(clean),
        "fred_cache": fred_fingerprint(),
        "code": code_fingerprint(),
    }


def write_plan(clean: pd.DataFrame, runs_dir=None):
    if latest_plan(runs_dir) is not None:
        raise RuntimeError("A frozen plan already exists in the run log. Not overwriting.")
    return append_record(build_plan(clean), runs_dir)


def verify(frozen: dict, clean: pd.DataFrame) -> list:
    """Differences between the frozen plan and the current state (empty = unchanged)."""
    problems = []
    for name, now, then in (("code", code_fingerprint(), frozen["code"]),
                            ("fred_cache", fred_fingerprint(), frozen["fred_cache"])):
        problems += [f"{name}: {k} changed" for k in then if now.get(k) != then[k]]
    if data_fingerprint(clean)["hash"] != frozen["data"]["hash"]:
        problems.append("data: cleaned revenue data changed")
    if frozen["split"] != {"validation_start": config.VALIDATION_START,
                           "test_start": config.TEST_START}:
        problems.append("split: dates in config.py changed")
    return problems
