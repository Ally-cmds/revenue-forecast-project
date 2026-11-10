"""One-off: move the old per-run JSON files into runs/run_log.jsonl.
Run once from repo root: python -m scripts.merge_old_runs   (then delete the old files)"""
import json

from fin_forecast.run_log import RUNS_DIR, append_record

if __name__ == "__main__":
    files = sorted(RUNS_DIR.glob("*.json"))
    records = []
    for p in files:
        r = json.loads(p.read_text(encoding="utf-8"))
        r["kind"] = "seen_results" if "status" in r else "run"
        r["run_id"] = p.stem
        records.append(r)
    for r in sorted(records, key=lambda r: r["timestamp"]):
        append_record(r)
    print(f"Appended {len(records)} records. You can now delete:", [p.name for p in files])
