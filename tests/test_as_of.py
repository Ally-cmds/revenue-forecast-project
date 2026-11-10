import pandas as pd

from fin_forecast.data.macro_loader import as_of, to_quarterly


def _macro():
    idx = pd.to_datetime(["2024-03-31", "2024-06-30", "2024-09-30"])
    return pd.DataFrame({"mortgage30": [1.0, 2.0, 3.0], "starts": [1.0, 2.0, 3.0]},
                        index=idx)


def test_unrevised_series_available_after_quarter_end():
    out = as_of(_macro(), "2024-07-01")
    assert out.loc["2024-06-30", "mortgage30"] == 2.0


def test_revised_series_has_one_quarter_guard():
    out = as_of(_macro(), "2024-07-01")
    assert pd.isna(out.loc["2024-06-30", "starts"])
    assert out.loc["2024-03-31", "starts"] == 1.0


def test_future_quarter_never_visible():
    out = as_of(_macro(), "2024-07-01")
    assert out.loc["2024-09-30"].isna().all()


def test_partial_trailing_quarter_dropped():
    idx = pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01",
                          "2024-05-01", "2024-06-01", "2024-07-01"])
    s = pd.Series(range(1, 8), index=idx, dtype=float)
    q = to_quarterly(s)
    assert list(q.index) == list(pd.to_datetime(["2024-03-31", "2024-06-30"]))
