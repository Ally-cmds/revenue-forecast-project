"""Time split of FORECAST dates (the period_end of the quarter being forecast).
  train      : before VALIDATION_START  (warm-up; fits models, not used to choose them)
  validation : VALIDATION_START .. before TEST_START  (tune settings, pick finalist)
  test       : TEST_START onward  (locked: opened once, every opening is logged)
Dates live in config.py (VALIDATION_START, TEST_START)."""
from datetime import datetime

import pandas as pd

from fin_forecast import config

ACCESS_LOG = config.PROJECT_ROOT / "data" / "processed" / "test_access.log"


def assign_split(period_end: pd.Series) -> pd.Series:
    labels = pd.Series("train", index=period_end.index)
    labels[period_end >= pd.Timestamp(config.VALIDATION_START)] = "validation"
    labels[period_end >= pd.Timestamp(config.TEST_START)] = "test"
    return labels


def hide_test(res: pd.DataFrame) -> pd.DataFrame:
    """Everything except the locked test rows (needs a 'split' column)."""
    return res[res["split"] != "test"]


def only(res: pd.DataFrame, split: str) -> pd.DataFrame:
    if split == "test":
        raise ValueError("Use reveal_test(res, reason) to open the test set.")
    return res[res["split"] == split]


def reveal_test(res: pd.DataFrame, reason: str) -> pd.DataFrame:
    """The only way to get test rows. Each call is appended to the access log,
    so the README can state how many times the test set was opened."""
    ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(ACCESS_LOG, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')}\t{reason}\n")
    return res[res["split"] == "test"]


def count_test_openings() -> int:
    """How many times the test set was opened through reveal_test."""
    if not ACCESS_LOG.exists():
        return 0
    return sum(1 for line in ACCESS_LOG.read_text(encoding="utf-8").splitlines()
               if line.strip())