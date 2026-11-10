import json

import pandas as pd

from fin_forecast import run_log, splits


def _res():
    d = pd.to_datetime(["2016-03-31", "2019-06-30", "2023-03-31"])
    res = pd.DataFrame({"ticker": "A", "sector": "s", "period_end": d, "model": "m",
                        "actual": 100.0, "forecast": [101.0, 99.0, 150.0]})
    res["pct_error"] = (res.forecast - res.actual) / res.actual
    res["abs_pct_error"] = res.pct_error.abs()
    res["split"] = splits.assign_split(res["period_end"])
    return res


def _clean():
    return pd.DataFrame({"ticker": ["A", "A"], "sector": "s",
                         "period_end": pd.to_datetime(["2019-03-31", "2019-06-30"]),
                         "revenue": [100, 110]})


class _M:  # model-like object with settings
    def __init__(self):
        self.alpha, self.n_fallback, self._cache = 10.0, 3, {}


def test_record_is_one_json_line_with_no_test_numbers(tmp_path, monkeypatch):
    monkeypatch.setattr(splits, "ACCESS_LOG", tmp_path / "test_access.log")
    run_log.log_run("unit", _res(), {"m": _M()}, _clean(), runs_dir=tmp_path)
    lines = (tmp_path / "run_log.jsonl").read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["kind"] == "run" and rec["models"]["m"] == {"alpha": 10.0}
    assert rec["fallbacks"] == {"m": 3}
    assert {r["split"] for r in rec["metrics_excluding_test"]} == {"train", "validation"}
    forecasts = pd.read_csv(tmp_path / "forecasts_unit.csv")
    assert "test" not in set(forecasts["split"])


def test_runs_append_and_never_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(splits, "ACCESS_LOG", tmp_path / "test_access.log")
    for _ in range(2):
        run_log.log_run("unit", _res(), {"m": _M()}, _clean(), runs_dir=tmp_path)
    assert len(run_log.load_runs(tmp_path)) == 2


def test_fingerprint_changes_when_data_changes():
    a, b = _clean(), _clean()
    b.loc[0, "revenue"] = 101
    assert run_log.data_fingerprint(a)["hash"] != run_log.data_fingerprint(b)["hash"]


def test_seen_record_covers_only_listed_models(tmp_path):
    res = pd.concat([_res(), _res().assign(model="brand_new_model")])
    res["model"] = res["model"].where(res["model"] != "m", "seasonal_drift")
    run_log.log_seen_results(res, _clean(), runs_dir=tmp_path)
    rec = run_log.load_runs(tmp_path)[0]
    assert rec["kind"] == "seen_results" and "SEEN" in rec["status"]
    assert set(rec["forecasts_per_model"]) == {"seasonal_drift"}