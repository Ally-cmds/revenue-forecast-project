"""Tests for the raw-EDGAR -> clean quarterly series rules in cleaning.py:
restatement dedupe, derived Q4, tag priority, gap trimming and health checks."""
import pandas as pd
import pytest

from fin_forecast.data.cleaning import (clean_revenue, find_gaps, tag_switches,
                                        trim_early_gaps, validate)

A = "Revenues"                       # first in the tag priority list
B = "SalesRevenueNet"                # later in the list


def _row(ticker="AAA", sector="homebuilders", metric="revenue", tag=A,
         start="2018-01-01", end="2018-03-31", val=100.0, filed="2018-05-10",
         form="10-Q", fp="Q1", fy=2018):
    return dict(ticker=ticker, sector=sector, metric=metric, tag=tag,
                start=start, end=end, val=val, form=form, fp=fp, fy=fy,
                filed=filed)


def _q4_derivable(fy_start="2018-01-01", fy_end="2018-12-31",
                  vals=(100.0, 110.0, 120.0), fy_val=460.0, tag=A):
    """Three direct quarters plus the annual total (Q4 must come out derived)."""
    s = pd.Timestamp(fy_start)
    e1, e2, e3 = s + pd.Timedelta(days=89), s + pd.Timedelta(days=180), s + pd.Timedelta(days=272)
    return [
        _row(tag=tag, start=str(s.date()), end=str(e1.date()), val=vals[0]),
        _row(tag=tag, start=str((e1 + pd.Timedelta(days=1)).date()),
             end=str(e2.date()), val=vals[1]),
        _row(tag=tag, start=str((e2 + pd.Timedelta(days=1)).date()),
             end=str(e3.date()), val=vals[2]),
        _row(tag=tag, start=fy_start, end=fy_end, val=fy_val, form="10-K", fp="FY"),
    ]


def test_latest_filing_wins_on_restatement():
    raw = pd.DataFrame([_row(val=1000.0, filed="2018-11-05"),
                        _row(val=1200.0, filed="2018-11-08")])
    clean = clean_revenue(raw)
    assert len(clean) == 1 and clean["revenue"].iloc[0] == 1200.0


def test_non_revenue_metrics_are_dropped():
    raw = pd.DataFrame([_row(), _row(metric="other")])
    assert len(clean_revenue(raw)) == 1


def test_derived_q4_equals_fy_minus_three_quarters():
    clean = clean_revenue(pd.DataFrame(_q4_derivable(vals=(100.0, 110.0, 120.0), fy_val=460.0)))
    q4 = clean[clean["derived"]].iloc[0]
    assert q4["period_end"] == pd.Timestamp("2018-12-31")
    assert q4["revenue"] == pytest.approx(460.0 - 330.0)
    assert q4["tag"] == A


def test_q4_not_derived_when_a_quarter_is_missing():
    # Q1, Q2 and the FY row only: Q3 was never filed under this tag, so the
    # annual total cannot be split into exactly three pieces.
    rows = _q4_derivable()[:2] + _q4_derivable()[-1:]
    clean = clean_revenue(pd.DataFrame(rows))
    assert not clean["derived"].any()
    assert set(clean["period_end"]) == {pd.Timestamp("2018-03-31"), pd.Timestamp("2018-06-30")}


def test_tag_priority_prefers_first_tag_in_config():
    raw = pd.DataFrame([_row(tag=A, val=1000.0), _row(tag=B, val=990.0)])
    clean = clean_revenue(raw)
    assert len(clean) == 1
    assert clean["tag"].iloc[0] == A and clean["revenue"].iloc[0] == 1000.0


def test_direct_row_beats_derived_row_for_the_same_quarter():
    rows = _q4_derivable(fy_val=460.0) + [
        dict(_row(start="2018-10-01", end="2018-12-31", val=135.0, form="10-Q", fp="Q4"))]
    clean = clean_revenue(pd.DataFrame(rows))
    q4 = clean[clean["period_end"] == pd.Timestamp("2018-12-31")].iloc[0]
    assert q4["revenue"] == 135.0 and not q4["derived"]


def test_trim_early_gaps_drops_only_the_pre_cutoff_stub():
    spans = [("2014-01-01", "2014-03-31"), ("2014-04-01", "2014-06-30"),
             ("2015-07-01", "2015-09-30"), ("2015-10-01", "2015-12-30"),
             ("2016-04-01", "2016-06-30")]
    clean = clean_revenue(pd.DataFrame([_row(start=s, end=e) for s, e in spans]))
    assert len(find_gaps(clean)) == 2             # one long gap before the cutoff, one after

    trimmed = trim_early_gaps(clean, before="2016-01-01")
    # the stub before the last pre-cutoff gap is dropped; the later gap stays
    assert trimmed["period_end"].min() == pd.Timestamp("2015-09-30")
    assert find_gaps(trimmed).iloc[-1]["days"] == 183


def test_validate_and_find_gaps_describe_the_data():
    rows = [_row(end="2018-03-31"),
            _row(start="2018-04-01", end="2018-06-30", val=110.0),
            _row(start="2018-10-01", end="2018-12-30", val=130.0)]   # 183-day gap
    clean = clean_revenue(pd.DataFrame(rows))
    v = validate(clean)
    assert v.loc["AAA", "quarters"] == 3 and v.loc["AAA", "gaps_over_120d"] == 1
    gaps = find_gaps(clean)
    assert len(gaps) == 1
    assert (gaps.iloc[0]["before"], gaps.iloc[0]["after"], gaps.iloc[0]["days"]) == (
        pd.Timestamp("2018-06-30").date(), pd.Timestamp("2018-12-30").date(), 183)


def test_tag_switches_records_the_ratio():
    raw = pd.DataFrame([_row(tag=A, end="2018-03-31", val=100.0),
                        _row(tag=B, start="2018-04-01", end="2018-06-30", val=110.0)])
    sw = tag_switches(clean_revenue(raw))
    assert len(sw) == 1
    assert (sw.iloc[0]["from_tag"], sw.iloc[0]["to_tag"]) == (A, B)
    assert sw.iloc[0]["qoq_ratio"] == 1.1


def test_output_is_sorted_and_has_the_contract_columns():
    raw = pd.DataFrame([_row(start="2018-04-01", end="2018-06-30"),
                        _row(end="2018-03-31")])
    clean = clean_revenue(raw)
    assert list(clean.columns) == ["ticker", "sector", "period_end", "revenue", "tag", "derived"]
    assert clean["period_end"].is_monotonic_increasing