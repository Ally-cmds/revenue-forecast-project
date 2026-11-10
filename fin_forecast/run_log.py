"""Run records: ONE append-only log, runs/run_log.jsonl (one JSON record per line,
each with its own timestamp), plus the latest forecasts CSV per run name.
Test-period numbers are never written here."""
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

from fin_forecast import config, splits
from fin_forecast.evaluation import summarize
from fin_forecast.splits import hide_test

RUNS_DIR = config.PROJECT_ROOT / "runs"
LOG_NAME = "run_log.jsonl"


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=config.PROJECT_ROOT,
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def data_fingerprint(clean: pd.DataFrame) -> dict:
    ordered = clean.sort_values(["ticker", "period_end"]).reset_index(drop=True)
    digest = hashlib.sha256(pd.util.hash_pandas_object(ordered, index=False)
                            .to_numpy().tobytes()).hexdigest()[:16]
    return {"rows": len(clean), "tickers": sorted(clean["ticker"].unique()),
            "first_period": str(clean["period_end"].min().date()),
            "last_period": str(clean["period_end"].max().date()), "hash": digest}


def _params(fc) -> dict:
    """Settings of a model object (plain functions have none)."""
    if hasattr(fc, "__code__"):
        return {}
    return {k: v for k, v in vars(fc).items()
            if not k.startswith("_") and k != "n_fallback"}


def _test_openings() -> int:
    """One definition lives in splits.count_test_openings (skips blank lines)."""
    return splits.count_test_openings()


def _native(o):
    return o.item() if hasattr(o, "item") else str(o)


def append_record(record: dict, runs_dir=None) -> Path:
    """Add one record as a single line (safe to append; never rewrites old lines)."""
    runs_dir = runs_dir or RUNS_DIR
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / LOG_NAME
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=_native) + "\n")
    return path


def load_runs(runs_dir=None) -> list:
    path = (runs_dir or RUNS_DIR) / LOG_NAME
    if not path.exists():
        return []
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def log_run(name: str, res: pd.DataFrame, models: dict, clean: pd.DataFrame,
            notes: str = "", runs_dir=None):
    runs_dir = runs_dir or RUNS_DIR
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    visible = hide_test(res)
    record = {
        "kind": "run",
        "run_id": f"{stamp}_{name}",
        "run_name": name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "split": {"validation_start": config.VALIDATION_START,
                  "test_start": config.TEST_START},
        "data": data_fingerprint(clean),
        "models": {n: _params(fc) for n, fc in models.items()},
        "fallbacks": {n: fc.n_fallback for n, fc in models.items()
                      if hasattr(fc, "n_fallback")},
        "metrics_excluding_test": summarize(
            visible, by=("split", "sector", "model")).reset_index().to_dict("records"),
        "test_set_openings": _test_openings(),
        "notes": notes,
    }
    visible.to_csv(runs_dir / f"forecasts_{name}.csv", index=False)  # latest only
    return append_record(record, runs_dir)


SEEN_MODELS = ["naive_last", "seasonal_naive", "seasonal_drift",
               "ridge_own", "ridge_drivers", "ridge_drivers_x_sector"]


def log_seen_results(res: pd.DataFrame, clean: pd.DataFrame, runs_dir=None) -> Path:
    """One-off disclosure record: results the analyst viewed BEFORE the split
    existed (all periods, test years included). Computed from the saved backtest
    results, only for the models listed in SEEN_MODELS, so it cannot be used to
    peek at models that have a pristine test set."""
    runs_dir = runs_dir or RUNS_DIR
    runs_dir.mkdir(parents=True, exist_ok=True)
    r = res[res["model"].isin(SEEN_MODELS)]
    record = {
        "kind": "seen_results",
        "status": ("SEEN BEFORE THE SPLIT EXISTED. All-period results, including "
                   "what is now the test period. These models have no pristine test set."),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "data": data_fingerprint(clean),
        "models_covered": SEEN_MODELS,
        "forecasts_per_model": r.groupby("model").size().to_dict(),
        "by_sector_all_periods": summarize(r).reset_index().to_dict("records"),
        "by_sector_excluding_2020": summarize(
            r, exclude_years=(2020,)).reset_index().to_dict("records"),
        "mape_by_company_all_periods": summarize(
            r, by=("ticker", "model"))["mape"].unstack().round(2).to_dict("index"),
    }
    return append_record(record, runs_dir)