"""Compact view of the run log. Run from repo root: python -m scripts.show_runs"""
import pandas as pd

from fin_forecast.run_log import load_runs

if __name__ == "__main__":
    for rec in load_runs():
        # every record has a kind; timestamp is absent on test_result records
        stamp = rec.get("timestamp", f"(started by {rec.get('plan_timestamp', '?')})")
        print(f"\n{stamp}  [{rec.get('kind', '?')}]  {rec.get('run_name', '')}"
              f"  data={rec.get('data', {}).get('hash', '?')}")
        rows = rec.get("metrics_excluding_test")
        if rows:
            val = pd.DataFrame(rows).query("split == 'validation'")
            print("  validation MAPE (%)")
            print(val.pivot(index="model", columns="sector", values="mape").to_string())
        else:
            summary = {"test_result": "one-time frozen test evaluation (see 'result' in the log)",
                       "frozen_plan": "pre-registered plan + fingerprints",
                       "seen_results": "disclosure: results seen before the split existed",
                       }
            print("  " + summary.get(rec.get("kind", ""), "record without metrics"))
