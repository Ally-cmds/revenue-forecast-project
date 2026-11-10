import pandas as pd
import pytest

from fin_forecast import freeze, splits


def _clean(x=100):
    return pd.DataFrame({"ticker": ["A", "A"], "sector": "s", "tag": "t", "derived": False,
                         "period_end": pd.to_datetime(["2019-03-31", "2019-06-30"]),
                         "revenue": [x, 110]})


@pytest.fixture(autouse=True)
def _stub_fingerprints(monkeypatch):
    monkeypatch.setattr(freeze, "code_fingerprint", lambda *a, **k: {"f.py": "aaa"})
    monkeypatch.setattr(freeze, "fred_fingerprint", lambda *a, **k: {"x.parquet": "bbb"})


def test_code_fingerprint_detects_an_edit(tmp_path):
    (tmp_path / "f.py").write_text("x = 1\n")
    from fin_forecast.freeze import _sha
    before = _sha(tmp_path / "f.py")
    (tmp_path / "f.py").write_text("x = 2\n")
    assert _sha(tmp_path / "f.py") != before
    assert _sha(tmp_path / "missing.py") == "MISSING"


def test_line_endings_do_not_change_the_fingerprint(tmp_path):
    from fin_forecast.freeze import _sha
    (tmp_path / "a.py").write_bytes(b"x = 1\ny = 2\n")
    (tmp_path / "b.py").write_bytes(b"x = 1\r\ny = 2\r\n")
    assert _sha(tmp_path / "a.py") == _sha(tmp_path / "b.py")


def test_plan_is_written_once_only(tmp_path):
    freeze.write_plan(_clean(), runs_dir=tmp_path)
    with pytest.raises(RuntimeError):
        freeze.write_plan(_clean(), runs_dir=tmp_path)
    assert freeze.latest_plan(tmp_path)["kind"] == "frozen_plan"


def test_verify_passes_when_unchanged_and_flags_changed_data(tmp_path):
    freeze.write_plan(_clean(), runs_dir=tmp_path)
    frozen = freeze.latest_plan(tmp_path)
    assert freeze.verify(frozen, _clean()) == []
    assert any("data" in p for p in freeze.verify(frozen, _clean(x=101)))


def test_verify_flags_changed_code(tmp_path, monkeypatch):
    freeze.write_plan(_clean(), runs_dir=tmp_path)
    frozen = freeze.latest_plan(tmp_path)
    monkeypatch.setattr(freeze, "code_fingerprint", lambda *a, **k: {"f.py": "CHANGED"})
    assert any("f.py" in p for p in freeze.verify(frozen, _clean()))


def test_test_openings_are_counted(tmp_path, monkeypatch):
    monkeypatch.setattr(splits, "ACCESS_LOG", tmp_path / "test_access.log")
    assert splits.count_test_openings() == 0
    res = pd.DataFrame({"period_end": pd.to_datetime(["2023-03-31"]), "split": ["test"]})
    splits.reveal_test(res, "unit test")
    assert splits.count_test_openings() == 1
