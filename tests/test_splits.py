import pandas as pd
import pytest

from fin_forecast import splits


@pytest.fixture(autouse=True)
def _tmp_log(tmp_path, monkeypatch):
    monkeypatch.setattr(splits, "ACCESS_LOG", tmp_path / "test_access.log")


def _res():
    d = pd.to_datetime(["2017-12-31", "2018-01-31", "2021-12-31", "2022-01-31", "2026-06-30"])
    res = pd.DataFrame({"period_end": d, "x": range(5)})
    res["split"] = splits.assign_split(res["period_end"])
    return res


def test_boundaries():
    assert _res()["split"].tolist() == ["train", "validation", "validation", "test", "test"]


def test_hide_test_and_only_validation():
    res = _res()
    assert splits.hide_test(res)["split"].eq("test").sum() == 0
    assert splits.only(res, "validation")["x"].tolist() == [1, 2]


def test_only_refuses_test():
    with pytest.raises(ValueError):
        splits.only(_res(), "test")


def test_reveal_test_is_logged():
    out = splits.reveal_test(_res(), "unit test")
    assert len(out) == 2
    assert "unit test" in splits.ACCESS_LOG.read_text()
