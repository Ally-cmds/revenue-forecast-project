"""Open the test set ONCE, exactly as the frozen plan says.
Refuses to run if code, data or drivers changed since the freeze, or if the test
set was already opened. Run from repo root: python -m scripts.run_test_once"""
import sys

import pandas as pd

from fin_forecast import config, plan
from fin_forecast.data.macro_loader import load_macro
from fin_forecast.freeze import latest_plan, verify
from fin_forecast.frozen_test import format_report, run_frozen_test
from fin_forecast.run_log import append_record, data_fingerprint
from fin_forecast.splits import count_test_openings

if __name__ == "__main__":
    clean = pd.read_parquet(config.PROJECT_ROOT / "data" / "processed" / "revenue_quarterly.parquet")
    frozen = latest_plan()
    if frozen is None:
        sys.exit("No frozen plan in the run log. Run: python -m scripts.freeze_plan")
    problems = verify(frozen, clean)
    if problems:
        sys.exit("Refusing to open the test set; changed since the freeze:\n  " + "\n  ".join(problems))
    if count_test_openings() > 0:
        sys.exit("The test set was already opened. Not running again.")
    out = run_frozen_test(clean, load_macro(list(plan.DRIVERS)),
                          reason=f"frozen plan {frozen['timestamp']}: {plan.FINALIST} vs {plan.BASELINE}")
    print(format_report(out))
    append_record({"kind": "test_result", "timestamp": frozen["timestamp"],
                   "plan_timestamp": frozen["timestamp"],
                   "data": data_fingerprint(clean), "result": out})
    print("\nTest result saved to the run log. The test set is now OPEN: any further look must be disclosed.")
