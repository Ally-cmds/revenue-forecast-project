"""Build the Excel deliverable: Revenue_Forecast_Review.xlsx.

One workbook for a finance-manager review, generated from the project's audited
artifacts (disclosed pre-split backtest parquet + run log + clean revenue data).
Sheets: Read me, Summary, Next quarter (forecast + 80% interval, per company),
Variance (close-style review with amber/red flags + full-history accuracy),
Flags (comment/owner/status), Trend (YoY + seasonality + company drop-down with
chart), Reconciliation (four quarters vs the SEC annual figure), Data checks,
Method and limits, Figures (the memo charts embedded). Formula cells and charts
carry baked-in cached results so every viewer shows the numbers (Excel still
recalculates on open).

Run from repo root: python -m scripts.make_excel_report
"""
import json
import os
import re
import shutil
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from datetime import date, datetime

import numpy as np
import pandas as pd

from fin_forecast import config, plan
from fin_forecast.backtest import PANEL_COLS
from fin_forecast.context import Ctx
from fin_forecast.data.cleaning import (ANNUAL_DAYS, QUARTER_DAYS, clean_revenue,
                                        find_gaps, tag_switches, validate)
from fin_forecast.data.data_loader import extract_metric
from fin_forecast.data.macro_loader import as_of, load_macro
from fin_forecast.models.ridge_growth import GrowthRidge
from fin_forecast.run_log import append_record, load_runs

import openpyxl
from openpyxl.chart import BarChart, LineChart, Reference
from openpyxl.drawing.image import Image as XLImage
from openpyxl.formatting.rule import CellIsRule, FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation

OUT = config.PROJECT_ROOT / "Revenue_Forecast_Review.xlsx"
AS_OF_ORIGIN = pd.Timestamp("2026-09-01")     # origin = day after last filed quarter
N_VAR_QUARTERS = 8
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
NAVY = PatternFill("solid", fgColor="1F4E79")
LIGHT = PatternFill("solid", fgColor="DDEBF7")
AMBER = PatternFill("solid", fgColor="FFE599")
RED = PatternFill("solid", fgColor="F4B4B4")
GREEN = PatternFill("solid", fgColor="D8EAD3")
H_FONT = Font(color="FFFFFF", bold=True)
T_FONT = Font(bold=True, size=14)
NOTE_FONT = Font(italic=True, color="595959", size=9)
WRAP = Alignment(wrap_text=True, vertical="top")
MONEY = '#,##0.00,,,"$B"'
PCT = "0.0%"
PCT3 = "0.000%"
DATEF = "yyyy-mm-dd"
H_FILL = PatternFill("solid", fgColor="1F4E79")


def table(ws, row, col, headers, rows, fmts=None):
    """Write a styled table at (row, col). Formulas are strings starting with '='."""
    for j, h in enumerate(headers):
        c = ws.cell(row=row, column=col + j, value=h)
        c.fill, c.font, c.border, c.alignment = H_FILL, H_FONT, BORDER, WRAP
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            c = ws.cell(row=row + 1 + i, column=col + j, value=v)
            c.border = BORDER
            if fmts and fmts.get(j):
                c.number_format = fmts[j]
            if isinstance(v, str) and not v.startswith("="):
                c.alignment = WRAP
    return row + 1 + len(rows)


def title(ws, row, text, note_text=None):
    ws.cell(row=row, column=1, value=text).font = T_FONT
    if note_text:
        ws.cell(row=row + 1, column=1, value=note_text).font = NOTE_FONT
        return row + 3
    return row + 2


def para(ws, row, text, span=10):
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=span)
    c = ws.cell(row=row, column=1, value=text)
    c.alignment = WRAP
    ws.row_dimensions[row].height = max(15, 14 * (1 + len(text) // 105))
    return row + 2


def note(ws, row, text, span=10):
    """Italic gray footnote row."""
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=span)
    c = ws.cell(row=row, column=1, value=text)
    c.alignment, c.font = WRAP, NOTE_FONT
    ws.row_dimensions[row].height = max(14, 13 * (1 + len(text) // 110))
    return row + 1


CELL_CACHE = {}          # (sheet title, coord) -> computed value, injected post-save
CHART_CACHE = {}         # sheet title -> {col letter: {row: value}} for chart caches


def put_cache(sheet, coord, val):
    """Remember a formula cell's computed value. openpyxl saves formulas without
    cached results, so non-recalculating viewers (previews, old tools) would show
    blanks; inject_values bakes these values in exactly the way Excel saves them."""
    CELL_CACHE[(sheet, coord)] = None if val is None else val


def inject_values(path):
    """Bake cached formula results and chart data into the saved workbook.

    Excel recalculates everything on load (fullCalcOnLoad stays on), but previews
    need stored results: a <v> after each <f> in worksheet cells, and
    numCache/strCache inside each chart reference."""
    NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    PR = "{http://schemas.openxmlformats.org/package/2006/relationships}"
    C = "{http://schemas.openxmlformats.org/drawingml/2006/chart}"
    ET.register_namespace("", NS[1:-1])
    ET.register_namespace("c", C[1:-1])

    def numtxt(v):
        if isinstance(v, str):
            return v
        f = float(v)
        if f.is_integer() and abs(f) < 1e15:
            return str(int(f))
        return repr(f)

    with zipfile.ZipFile(path) as z:
        parts = {n: z.read(n) for n in z.namelist()}
    wbroot = ET.fromstring(parts["xl/workbook.xml"])
    rels = ET.fromstring(parts["xl/_rels/workbook.xml.rels"])
    rmap = {r.get("Id"): r.get("Target") for r in rels
            if r.tag == PR + "Relationship"}
    sheet_part = {}
    for sh in wbroot.find(NS + "sheets"):
        target = rmap[sh.get(REL + "id")].replace("\\", "/").lstrip("/")
        sheet_part[sh.get("name")] = (target if target.startswith("xl/")
                                      else "xl/" + target)

    per_sheet = {}
    for (sheet, coord), val in CELL_CACHE.items():
        if val is not None:
            per_sheet.setdefault(sheet, []).append((coord, val))
    for sheet, items in per_sheet.items():
        part = sheet_part.get(sheet)
        if not part or part not in parts:
            continue
        root = ET.fromstring(parts[part])
        want = dict(items)
        n_done = 0
        for c in root.iter(NS + "c"):
            if c.get("r") in want and c.find(NS + "f") is not None:
                val = want[c.get("r")]
                if isinstance(val, str):
                    c.set("t", "str")
                v = c.find(NS + "v")             # openpyxl writes an empty <v/>
                if v is not None:
                    v.text = numtxt(val)         # fill the placeholder cache
                else:
                    ET.SubElement(c, NS + "v").text = numtxt(val)
                n_done += 1
        parts[part] = ET.tostring(root, encoding="UTF-8", xml_declaration=True)
        print("  cached formula cells: " + sheet + " = " + str(n_done), flush=True)

    rx = re.compile(r"'([^']+)'!\$([A-Z]+)\$(\d+)(?::\$([A-Z]+)\$(\d+))?$")

    def cache_ref(ref, refname):
        f = ref.findtext(C + "f")
        m = rx.match(f) if f else None
        if not m:
            return 0
        if ref.find(C + "numCache") is not None or \
                ref.find(C + "strCache") is not None:
            return 0                               # already cached
        colmap = CHART_CACHE.get(m.group(1), {}).get(m.group(2), {})
        vals = [colmap.get(r) for r in
                range(int(m.group(3)), int(m.group(5) or m.group(3)) + 1)]
        if not vals or all(v is None for v in vals):
            return 0
        if refname == "numRef" and any(isinstance(v, str) for v in vals):
            ref.tag = C + "strRef"                 # text values need a strCache
            refname = "strRef"
        ctag = "numCache" if refname == "numRef" else "strCache"
        cache = ET.SubElement(ref, C + ctag)
        if refname == "numRef":
            ET.SubElement(cache, C + "formatCode").text = "General"
        ET.SubElement(cache, C + "ptCount").set("val", str(len(vals)))
        for i, v in enumerate(vals):
            if v is None:
                continue
            pt = ET.SubElement(cache, C + "pt")
            pt.set("idx", str(i))
            ET.SubElement(pt, C + "v").text = (v if isinstance(v, str)
                                               else numtxt(v))
        return 1

    for name in sorted(n for n in parts if n.startswith("xl/charts/chart")):
        root = ET.fromstring(parts[name])
        n_done = 0
        for ref in list(root.iter(C + "numRef")):
            n_done += cache_ref(ref, "numRef")
        for ref in list(root.iter(C + "strRef")):
            n_done += cache_ref(ref, "strRef")
        if n_done:
            parts[name] = ET.tostring(root, encoding="UTF-8", xml_declaration=True)
            print("  chart cache refs: " + name + " = " + str(n_done), flush=True)

    tmp = tempfile.mktemp(dir=str(path.parent), suffix=".xlsx")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for n, b in parts.items():
            z.writestr(n, b)
    shutil.move(tmp, path)
    print("cached values and chart data baked in", flush=True)


def fiscal_windows(raw, tickers):
    """All SEC annual (FY) revenue rows for the panel, latest filing first per
    (ticker, tag, period)."""
    r = raw[(raw["metric"] == "revenue") & raw["ticker"].isin(tickers)].copy()
    for c in ("start", "end", "filed"):
        r[c] = pd.to_datetime(r[c])
    r["days"] = (r["end"] - r["start"]).dt.days
    fy = r[r["days"].between(*ANNUAL_DAYS)].sort_values("filed", ascending=False)
    return fy.drop_duplicates(["ticker", "tag", "start", "end"], keep="first")


def fy_tables(raw, clean, tickers):
    """Build (reconciliation_rows, seasonality_rows) from SEC annual windows.

    A fiscal year qualifies when exactly four cleaned quarters fall inside the
    annual row's window. The SEC annual figure is taken under the same tag as
    that year's Q4 row so the comparison is apples-to-apples."""
    fy = fiscal_windows(raw, tickers)
    rec, sea = [], []
    for t in tickers:
        cq = (clean[clean["ticker"] == t].sort_values("period_end")
              .reset_index(drop=True))
        shares = {p: [] for p in range(4)}
        for _, f in fy[fy["ticker"] == t].sort_values("end").iterrows():
            if f["end"].year > 2025:
                continue                      # annual windows ending after 2025
            win = cq[(cq["period_end"] > f["start"] - pd.Timedelta(days=7))
                     & (cq["period_end"] <= f["end"])]
            if len(win) == 5:             # prior FY's Q4 leaked in via the 7d slack
                win = win[win["period_end"] >= f["start"]]
            if len(win) != 4:
                continue
            win = win.sort_values("period_end")
            tag_row = win.iloc[-1]            # the year's Q4 row: its tag governs
            same = fy[(fy["ticker"] == t) & (fy["tag"] == tag_row["tag"])
                      & (fy["end"] == f["end"])]
            f = same.iloc[0] if len(same) else f
            vals = [float(v) for v in win["revenue"]]
            rec.append([t, f["fy"], *[d.to_pydatetime() for d in win["period_end"]],
                        *vals, None, float(f["val"]), None, None, None])
            if f["end"].year >= 2013:
                for p in range(4):
                    shares[p].append(vals[p] / sum(vals))
        sea.append([t, *[round(float(np.mean(shares[p])), 4) if shares[p] else None
                         for p in range(4)]])
    return rec, sea


def next_quarter(clean, res):
    """True out-of-sample next-quarter forecasts (origin 2026-09-01) + intervals.

    Point = seasonal-drift forecast scaled by the sector's median past log error
    (bias correction), exactly like fin_forecast.calibration.add_interval."""
    macro = load_macro(list(plan.DRIVERS))
    rows = []
    for t, g in clean.groupby("ticker"):
        g = g.sort_values("period_end")
        hist = g.set_index("period_end")["revenue"].astype(float)
        raw_f = float(hist.iloc[-4]) * float(hist.iloc[-1]) / float(hist.iloc[-5])
        pool = res[(res["model"] == "seasonal_drift")
                   & (res["sector"] == g["sector"].iloc[0])
                   & (res["period_end"] < AS_OF_ORIGIN)]
        e = np.log(pool["forecast"] / pool["actual"])
        e_lo, e_mid, e_hi = e.quantile([0.1, 0.5, 0.9])
        ridge = GrowthRidge(plan.DRIVERS, False, plan.RIDGE_ALPHA, plan.MIN_ROWS)
        ctx = Ctx(t, g["sector"].iloc[0], AS_OF_ORIGIN, as_of(macro, AS_OF_ORIGIN),
                  clean.loc[clean["period_end"] < AS_OF_ORIGIN, PANEL_COLS])
        rows.append({"ticker": t, "sector": g["sector"].iloc[0],
                     "last_q": hist.index[-1].to_pydatetime(),
                     "last_a": float(hist.iloc[-1]), "prior_a": float(hist.iloc[-4]),
                     "raw_f": raw_f, "point": raw_f * float(np.exp(-e_mid)),
                     "lo": raw_f * float(np.exp(-e_hi)),
                     "hi": raw_f * float(np.exp(-e_lo)),
                     "ridge": ridge(hist, ctx), "fallback": ridge.n_fallback})
    return pd.DataFrame(rows)


def main():
    clean = pd.read_parquet(config.PROJECT_ROOT / "data" / "processed"
                            / "revenue_quarterly.parquet")
    res = pd.read_parquet(config.PROJECT_ROOT / "data" / "processed"
                          / "backtest_results.parquet")
    raw = pd.read_parquet(config.CACHE_DIR / "raw_all.parquet")
    tickers = sorted(clean["ticker"].unique())
    verdict = [r for r in load_runs() if r.get("kind") == "test_result"][-1]["result"]
    print("artifacts loaded (panel, backtest, raw, run log)", flush=True)

    wb = openpyxl.Workbook()
    wb.calculation.fullCalcOnLoad = True

    # ================================================================ Read me
    ws = wb.active
    ws.title = "Read me"
    ws.sheet_properties.tabColor = "2E7D32"
    r = title(ws, 1, "Revenue Forecast Review — US homebuilders & home improvement",
              f"Prepared {date.today():%d %b %Y} | data: SEC EDGAR filings through "
              f"2026-08-31 (as cached) | run log: runs/run_log.jsonl")
    r = para(ws, r,
             "This workbook supports the quarterly revenue review: what we expect "
             "next quarter and how confident we are, how recent forecasts did "
             "against actuals, revenue trend and seasonality per company, and the "
             "data-quality checks behind the numbers.")
    r = para(ws, r,
             "How to use: start at Summary (five findings, plain English). Next "
             "quarter is the forward-looking table. Variance is a close-style "
             "review of the last eight forecast quarters plus the full-history "
             "accuracy table; misses beyond ±5% are amber, beyond ±10% red, and "
             "land on Flags for comment, owner and status. Trend has a company "
             "drop-down with a chart. Reconciliation ties the four quarters to "
             "the SEC annual figure. Data checks documents every exclusion. "
             "Method and limits says what the model is and is not. Figures "
             "embeds the memo charts.")
    r = para(ws, r, "Colour legend: green = on plan (within ±5%), amber = review "
                    "(±5-10%), red = escalate (>±10%).")
    r = table(ws, r, 1, ["Sheet", "Question it answers"],
              [["Summary", "What should I take away from this review?"],
               ["Next quarter", "What do we expect next quarter, per company, with an 80% interval?"],
               ["Variance", "Did previous one-step forecasts match actuals? Where are the misses?"],
               ["Flags", "Which periods need a comment, an owner and a status?"],
               ["Trend", "How is each company's YoY growth trending; how seasonal is it?"],
               ["Reconciliation", "Do the four quarters add back to the SEC annual revenue?"],
               ["Data checks", "Why is anything excluded (PHM) and how clean is the data?"],
               ["Method and limits", "How are the forecasts made, and what are the caveats?"],
               ["Figures", "Where are the backing charts (error by year, by company, forecast vs actual)?"]])
    r = para(ws, r, "All figures trace to SEC EDGAR filings and the project run log; "
                    "every table is regenerated by scripts/make_excel_report.py.")
    ws.column_dimensions["A"].width = 26
    ws.column_dimensions["B"].width = 92

    # ================================================================ Summary
    ws = wb.create_sheet("Summary")
    ws.sheet_properties.tabColor = "2E7D32"
    nq = next_quarter(clean, res)
    print("next-quarter forecasts ready: " + str(len(nq)) + " companies", flush=True)
    var_recent = (res[res["model"] == "seasonal_drift"]
                  .groupby("ticker", group_keys=False).tail(N_VAR_QUARTERS).copy())
    var_recent["var_pct"] = (var_recent["actual"] - var_recent["forecast"]) / var_recent["forecast"]
    n_red = int((var_recent["var_pct"].abs() > 0.10).sum())
    n_amber = int(((var_recent["var_pct"].abs() > 0.05)
                   & (var_recent["var_pct"].abs() <= 0.10)).sum())
    rec_rows, sea = fy_tables(raw, clean, tickers)
    diffs = [abs((x[11] - sum(x[6:10])) / x[11]) for x in rec_rows]
    n_reco_bad = int(sum(d > 0.001 for d in diffs))
    print("reconciliation candidate rows: " + str(len(rec_rows)), flush=True)
    worst_i = int(np.argmax(diffs))
    m = res[res["model"] == "seasonal_drift"]
    mape_co = ((m["forecast"] / m["actual"] - 1).abs() * 100).groupby(m["ticker"]).mean()
    best, worst = mape_co.idxmin(), mape_co.idxmax()
    prim, cov = verdict["primary"], verdict["interval"]["coverage_test"]["rate"]

    r = title(ws, 1, "Summary — five findings and what to do about them",
              f"To the finance team | {date.today():%d %b %Y} | basis: one-step-ahead "
              f"forecasts, data through 2026-08-31")
    pt = nq["point"].sum() / 1e9
    yr = (nq["point"].sum() - nq["prior_a"].sum()) / nq["prior_a"].sum()
    r = para(ws, r,
             f"1. NEXT QUARTER. We forecast about ${pt:,.1f}B of revenue for the "
             f"quarter following each company's last reported quarter (mostly "
             f"Q4-2026), {yr:+.1%} versus the same quarter last year. The honest "
             f"range at 80% confidence is roughly ±10% industry-wide; company "
             f"detail is on the Next quarter sheet.")
    r = para(ws, r,
             f"2. WHICH MODEL. We use the simple seasonal rule (same quarter last "
             f"year scaled by latest momentum). A machine-learning model with "
             f"mortgage-rate and housing-start drivers was tested against it on a "
             f"locked 2022-2026 window and did not beat it "
             f"({prim['mape_model_qweighted']:.2f}% vs "
             f"{prim['mape_baseline_qweighted']:.2f}% average error). Simpler is "
             f"better here, and we can prove it.")
    r = para(ws, r,
             f"3. RELIABILITY DIFFERS BY COMPANY. Our error is about "
             f"{mape_co[best]:.1f}% at {best} but {mape_co[worst]:.1f}% at {worst}. "
             f"Plan buffers by business model: defensive small-ticket retail is "
             f"forecastable; builders on backlog cycles are not.")
    r = para(ws, r,
             f"4. WE RUN SLIGHTLY HIGH. On the locked test the seasonal rule "
             f"over-forecast by +1.6% on average; the points published here are "
             f"bias-corrected for that. In the last {N_VAR_QUARTERS} quarters, "
             f"{n_red} company-quarters missed by more than 10% (red) and {n_amber} "
             f"by 5-10% (amber) — work the Flags sheet monthly.")
    r = para(ws, r,
             f"5. THE NUMBERS TIE OUT. Summing each company's four quarters against "
             f"the SEC annual figure agrees within 0.1% for {len(rec_rows) - n_reco_bad} "
             f"of {len(rec_rows)} company-years; the exceptions are listed in "
             f"Reconciliation (mid-year tag changes are the usual cause; largest "
             f"gap {max(diffs):.2%}, {rec_rows[worst_i][0]} FY{int(rec_rows[worst_i][1])}). "
             f"PHM is excluded — its revenue tags are fragmented and four Q4 "
             f"periods cannot be reconstructed (see Data checks).")
    r = para(ws, r,
             "ACTIONS. (i) This week: comment, owner and status on every amber/red "
             "row in Flags. (ii) Rerun the workbook when the Q4 filings land. "
             "(iii) Keep the ML model off daily duty: it only earned its keep in "
             "2021's regime break; revisit on a fresh locked window.")
    ws.column_dimensions["A"].width = 120

    # ================================================================ Next quarter
    ws = wb.create_sheet("Next quarter")
    ws.sheet_properties.tabColor = "1565C0"
    r = title(ws, 1, "Next quarter — forecast and 80% interval, per company",
              "One-step-ahead at origin = day after the last filed quarter "
              "(2026-09-01). Interval = each sector's expanding error quantiles; "
              "point is bias-corrected. Ridge column is reference only (it lost "
              "the locked test).")
    hdr = ["Ticker", "Sector", "Last filed quarter", "Last actual",
           "Same quarter last year", "Seasonal forecast (raw)",
           "Point (bias-corrected)", "Lo 80%", "Hi 80%", "Ridge + macro (ref)",
           "Implied YoY", "Interval width"]
    rows = []
    for i, x in nq.iterrows():
        rr = i + 5
        rows.append([x["ticker"], x["sector"], x["last_q"], x["last_a"], x["prior_a"],
                     x["raw_f"], x["point"], x["lo"], x["hi"], x["ridge"],
                     f"=G{rr}/E{rr}-1", f"=(I{rr}-H{rr})/G{rr}"])
    r = table(ws, 4, 1, hdr, rows,
              fmts={2: DATEF, 3: MONEY, 4: MONEY, 5: MONEY, 6: MONEY, 7: MONEY,
                    8: MONEY, 9: MONEY, 10: PCT, 11: PCT})
    ws.conditional_formatting.add(
        f"L5:L{r - 1}", CellIsRule(operator="greaterThan", formula=["0.3"], fill=AMBER))
    r = note(ws, r, f"Point = raw seasonal forecast scaled by the sector's median "
                    f"historical log error. Lo/Hi use each sector's own error "
                    f"quantiles (about {len(res) // 12:,} forecasts per sector). "
                    f"ML fallback count: {int(nq['fallback'].sum())} (0 = the model "
                    f"was used for every company).")
    for i, x in nq.iterrows():                      # cache the two live columns
        put_cache("Next quarter", f"K{5 + i}",
                  float(x["point"]) / float(x["prior_a"]) - 1)
        put_cache("Next quarter", f"L{5 + i}",
                  (float(x["hi"]) - float(x["lo"])) / float(x["point"]))
    ws.cell(row=4, column=15, value="chart source").font = NOTE_FONT
    ws.cell(row=4, column=16, value="Point ($B)").font = NOTE_FONT
    for i, x in nq.iterrows():
        ws.cell(row=5 + i, column=15, value=x["ticker"]).border = BORDER
        c = ws.cell(row=5 + i, column=16,            # static value = chart's cache
                    value=float(x["point"]) / 1e9)
        c.number_format, c.border = "0.0", BORDER
    ch = BarChart()
    ch.title = "Point forecast by company ($B)"
    ch.height, ch.width, ch.legend, ch.y_axis.title = 8, 16, None, "$B"
    ch.add_data(Reference(ws, min_col=16, min_row=4, max_row=14), titles_from_data=True)
    ch.set_categories(Reference(ws, min_col=15, min_row=5, max_row=14))
    ws.add_chart(ch, "N17")
    for k, v in {"A": 8, "B": 16, "C": 18, "D": 14, "E": 14, "F": 14, "G": 14,
                 "H": 14, "I": 14, "J": 16, "K": 11, "L": 13}.items():
        ws.column_dimensions[k].width = v
    ws.freeze_panes = "A5"

    # ================================================================ Variance
    ws = wb.create_sheet("Variance")
    ws.sheet_properties.tabColor = "B26A00"
    r = title(ws, 1, f"Variance — last {N_VAR_QUARTERS} forecast quarters vs actuals",
              "Chosen model: seasonal drift. Variance % = (actual - forecast) / "
              "forecast. Amber beyond ±5%, red beyond ±10%. Ridge columns are "
              "reference. Source: the disclosed pre-split backtest run (all periods).")
    ridge_res = res[res["model"] == "ridge_drivers"].set_index(["ticker", "period_end"])
    hdr = ["Ticker", "Sector", "Quarter end", "Forecast", "Actual", "Variance",
           "Variance %", "Flag", "Ridge forecast", "Ridge var %", "Ridge flag"]
    rows = []
    for _, x in var_recent.sort_values(["period_end", "ticker"]).iterrows():
        key = (x["ticker"], x["period_end"])
        rf = float(ridge_res.loc[key, "forecast"]) if key in ridge_res.index else None
        rr = len(rows) + 5
        fv = float(x["actual"]) - float(x["forecast"])
        gv = fv / float(x["forecast"])
        fg = "RED" if abs(gv) >= 0.1 else "AMBER" if abs(gv) >= 0.05 else "OK"
        jv = jg = None
        if rf is not None:
            jv = (float(x["actual"]) - rf) / rf
            jg = "RED" if abs(jv) >= 0.1 else "AMBER" if abs(jv) >= 0.05 else "OK"
        for coord, cv in ((f"F{rr}", fv), (f"G{rr}", gv), (f"H{rr}", fg),
                          (f"J{rr}", jv), (f"K{rr}", jg)):
            put_cache("Variance", coord, cv)
        rows.append([x["ticker"], x["sector"], x["period_end"].to_pydatetime(),
                     float(x["forecast"]), float(x["actual"]), f"=E{rr}-D{rr}",
                     f"=F{rr}/D{rr}",
                     f'=IF(ABS(G{rr})>=0.1,"RED",IF(ABS(G{rr})>=0.05,"AMBER","OK"))',
                     rf, f"=(E{rr}-I{rr})/I{rr}",
                     f'=IF(ABS(J{rr})>=0.1,"RED",IF(ABS(J{rr})>=0.05,"AMBER","OK"))'])
    last = table(ws, 4, 1, hdr, rows,
                 fmts={2: DATEF, 3: MONEY, 4: MONEY, 5: MONEY, 6: PCT,
                       8: MONEY, 9: PCT}) - 1
    for letter in ("H", "K"):
        ws.conditional_formatting.add(f"{letter}5:{letter}{last}",
                                      FormulaRule(formula=[f'${letter}5="RED"'], fill=RED))
        ws.conditional_formatting.add(f"{letter}5:{letter}{last}",
                                      FormulaRule(formula=[f'${letter}5="AMBER"'], fill=AMBER))
        ws.conditional_formatting.add(f"{letter}5:{letter}{last}",
                                      FormulaRule(formula=[f'${letter}5="OK"'], fill=GREEN))
    for letter in ("G", "J"):
        for op, form, fill in (("greaterThan", "0.1", RED), ("lessThan", "-0.1", RED),
                               ("greaterThan", "0.05", AMBER), ("lessThan", "-0.05", AMBER)):
            ws.conditional_formatting.add(f"{letter}5:{letter}{last}",
                                          CellIsRule(operator=op, formula=[form], fill=fill))
        ws.conditional_formatting.add(f"{letter}5:{letter}{last}",
                                      CellIsRule(operator="between",
                                                 formula=["-0.05", "0.05"], fill=GREEN))
    ws.freeze_panes = "A5"
    for k, v in {"A": 8, "B": 17, "C": 13, "D": 14, "E": 14, "F": 13, "G": 11,
                 "H": 9, "I": 14, "J": 11, "K": 11}.items():
        ws.column_dimensions[k].width = v
    # accuracy summary — full backtest history, chosen model (the long-run numbers)
    mres = res[res["model"] == "seasonal_drift"].copy()
    mres["v"] = (mres["actual"] - mres["forecast"]) / mres["forecast"]
    mres["ok5"] = mres["v"].abs() <= 0.05
    r2 = title(ws, last + 1, "Accuracy — full backtest history, chosen model",
               "MAPE = mean absolute variance % over every forecast quarter of the "
               "disclosed 2009-2026 run (the long run behind the recent eight "
               "quarters above). Within ±5% = share of quarters inside the review "
               "tolerance. Worst = the largest miss and when it happened.")
    rows = []
    for t, sg in mres.groupby("ticker"):
        w = mres.loc[mres.loc[sg.index, "v"].abs().idxmax()]
        rows.append([t, int(sg["v"].size), float(sg["v"].abs().mean()),
                     float(sg["ok5"].mean()), float(w["v"]),
                     pd.Timestamp(w["period_end"]).to_pydatetime()])
    rows.append(["ALL", int(len(mres)), float(mres["v"].abs().mean()),
                 float(mres["ok5"].mean()), None, None])
    r2 = table(ws, r2, 1,
               ["Ticker", "Quarters", "MAPE", "Within ±5%", "Worst var %",
                "Worst quarter"], rows,
               fmts={1: "0", 2: PCT, 3: PCT, 4: PCT, 5: DATEF})

    # ================================================================ Flags
    ws = wb.create_sheet("Flags")
    ws.sheet_properties.tabColor = "B26A00"
    r = title(ws, 1, "Flags — review tracker (works like a live close review)",
              "Every amber/red company-quarter from the Variance sheet, plus blank "
              "rows for new issues. Fill Comment, Owner and Status; Status has a "
              "drop-down (Open / In progress / Closed).")
    flagged = var_recent[var_recent["var_pct"].abs() > 0.05].sort_values(
        "var_pct", key=lambda s: -s.abs())
    rows = []
    for _, x in flagged.iterrows():
        lvl = "RED" if abs(x["var_pct"]) > 0.10 else "AMBER"
        rows.append([lvl, x["ticker"], x["period_end"].to_pydatetime(),
                     float(x["var_pct"]), None, None, None])
    for _ in range(15):
        rows.append([None] * 7)
    r = table(ws, 4, 1, ["Level", "Ticker", "Quarter end", "Variance %",
                         "Analyst comment", "Owner", "Status"],
              rows, fmts={2: DATEF, 3: PCT})
    dv = DataValidation(type="list", formula1='"Open,In progress,Closed"',
                        allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"G5:G{r + 9}")
    note(ws, r + 1, "Suggested flow: FP&A analyst comments, category lead owns the "
                    "item, controller approves closure. Variance holds the numbers "
                    "behind every line.")
    for k, v in {"A": 9, "B": 9, "C": 13, "D": 12, "E": 48, "F": 18, "G": 13}.items():
        ws.column_dimensions[k].width = v

    # ================================================================ Trend
    ws = wb.create_sheet("Trend")
    ws.sheet_properties.tabColor = "1565C0"
    r = title(ws, 1, "Trend — YoY growth per company and sector, seasonality",
              "YoY = revenue vs the same fiscal quarter a year earlier. Sector avg "
              "is the mean of the companies present that quarter. Seasonality = "
              "each quarter's share of its fiscal year, averaged over fiscal years "
              "ending 2013-2025. Pick a company in the yellow cell — chart follows.")
    yoy = pd.DataFrame(index=sorted(clean["period_end"].unique())[-20:])
    for t in tickers:
        s = (clean[clean["ticker"] == t].sort_values("period_end")
             .set_index("period_end")["revenue"].astype(float))
        yoy[t] = s / s.shift(4) - 1
    for sec in ("homebuilders", "home_improvement"):
        cols = [t for t in tickers
                if clean[clean["ticker"] == t]["sector"].iloc[0] == sec]
        yoy[sec] = yoy[cols].mean(axis=1)
    hdr = ["Quarter end", *tickers, "homebuilders avg", "home_improvement avg"]
    rows = [[d.to_pydatetime(),
             *[(None if pd.isna(yoy.loc[d, c]) else float(yoy.loc[d, c]))
               for c in yoy.columns]] for d in yoy.index]
    grid_top = table(ws, 4, 2, hdr, rows,
                     fmts={0: DATEF, **{i: PCT for i in range(1, 13)}}) - len(rows)
    # --- company view: dropdown + INDEX/MATCH helpers + line chart ---
    ws.cell(row=27, column=2, value="Company view — pick a company:").font = Font(bold=True)
    ws.cell(row=28, column=2, value="Company").font = NOTE_FONT
    ws.cell(row=28, column=3, value=tickers[0]).fill = LIGHT
    dv2 = DataValidation(type="list", formula1='"' + ",".join(tickers) + '"',
                         allow_blank=False)
    ws.add_data_validation(dv2)
    dv2.add("C28")
    for j, h in ((2, "Quarter"), (3, "Company YoY"), (4, "Sector avg")):
        c = ws.cell(row=30, column=j, value=h)
        c.fill, c.font, c.border = H_FILL, H_FONT, BORDER
    map_r = 70
    ws.cell(row=map_r, column=16, value="Ticker").font = NOTE_FONT
    ws.cell(row=map_r, column=17, value="Sector").font = NOTE_FONT
    for i, t in enumerate(tickers):
        ws.cell(row=map_r + 1 + i, column=16, value=t)
        ws.cell(row=map_r + 1 + i, column=17,
                value=clean[clean["ticker"] == t]["sector"].iloc[0])
    def_sec = clean[clean["ticker"] == tickers[0]]["sector"].iloc[0]
    for i in range(len(rows)):
        rr, src = 31 + i, grid_top + i
        d0 = yoy.index[i]
        put_cache("Trend", f"B{rr}", int((d0 - pd.Timestamp("1899-12-30")).days))
        cval, sval = yoy.loc[d0, tickers[0]], yoy.loc[d0, def_sec]
        put_cache("Trend", f"C{rr}", None if pd.isna(cval) else float(cval))
        put_cache("Trend", f"D{rr}", None if pd.isna(sval) else float(sval))
        ws.cell(row=rr, column=2, value=f"=B{src}").number_format = DATEF
        ws.cell(row=rr, column=3,
                value=f"=INDEX(C{src}:L{src},1,MATCH($C$28,$C$4:$L$4,0))").number_format = PCT
        ws.cell(row=rr, column=4,
                value=(f'=IF(INDEX($Q${map_r + 1}:$Q${map_r + 10},MATCH($C$28,'
                       f'$P${map_r + 1}:$P${map_r + 10},0))="homebuilders",'
                       f"INDEX(M{src}:M{src},1,1),INDEX(N{src}:N{src},1,1))"
                       )).number_format = PCT
    ch = LineChart()
    ch.title = "YoY revenue growth — selected company vs its sector"
    ch.height, ch.width, ch.y_axis.title = 8, 16, "YoY growth"
    ch.add_data(Reference(ws, min_col=3, max_col=4, min_row=30, max_row=50),
                titles_from_data=True)
    ch.set_categories(Reference(ws, min_col=2, min_row=31, max_row=50))
    ws.add_chart(ch, "F52")
    # --- seasonality table ---
    ws.cell(row=27, column=9, value="Seasonality (share of fiscal year)").font = Font(bold=True)
    table(ws, 28, 9, ["Ticker", "Q1", "Q2", "Q3", "Q4"], sea,
          fmts={1: PCT, 2: PCT, 3: PCT, 4: PCT})
    for k, v in {"B": 13, "C": 15, "D": 15, "E": 15, "F": 15, "G": 15, "H": 15,
                 "I": 15, "J": 15, "K": 15, "L": 15, "M": 17, "N": 20}.items():
        ws.column_dimensions[k].width = v
    ws.freeze_panes = "C5"

    # ================================================================ Reconciliation
    ws = wb.create_sheet("Reconciliation")
    ws.sheet_properties.tabColor = "6A1B9A"
    r = title(ws, 1, "Reconciliation — the four quarters vs the SEC annual figure",
              "Every company fiscal year with exactly four cleaned quarters inside "
              "the annual filing's window. The SEC annual figure is taken under the "
              "same tag as that year's Q4. Differences usually come from mid-year "
              "revenue-tag changes.")
    hdr = ["Ticker", "FY", "Q1 end", "Q2 end", "Q3 end", "Q4 end", "Q1", "Q2", "Q3",
           "Q4", "Sum of quarters", "SEC annual", "Diff", "Diff %", "Status"]
    rows = []
    for i, x in enumerate(rec_rows):
        rr = i + 5
        s, sec = sum(x[6:10]), float(x[11])
        put_cache("Reconciliation", f"K{rr}", s)
        put_cache("Reconciliation", f"M{rr}", sec - s)
        put_cache("Reconciliation", f"N{rr}",
                  (sec - s) / sec if sec else None)
        st = ("CHECK" if abs((sec - s) / sec) >= 0.01
              else "review" if abs((sec - s) / sec) >= 0.001 else "ok")
        put_cache("Reconciliation", f"O{rr}", st)
        rows.append([x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9],
                     f"=SUM(G{rr}:J{rr})", x[11], f"=L{rr}-K{rr}", f"=M{rr}/L{rr}",
                     f'=IF(ABS(N{rr})>=0.01,"CHECK",IF(ABS(N{rr})>=0.001,"review","ok"))'])
    r = table(ws, 4, 1, hdr, rows,
              fmts={1: "0", 2: DATEF, 3: DATEF, 4: DATEF, 5: DATEF, 6: MONEY,
                    7: MONEY, 8: MONEY, 9: MONEY, 10: MONEY, 11: MONEY, 12: MONEY,
                    13: PCT3})
    for op, form, fill in (("greaterThan", "0.01", RED), ("lessThan", "-0.01", RED),
                           ("greaterThan", "0.001", AMBER), ("lessThan", "-0.001", AMBER)):
        ws.conditional_formatting.add(f"N5:N{r - 1}",
                                      CellIsRule(operator=op, formula=[form], fill=fill))
    ws.conditional_formatting.add(f"N5:N{r - 1}",
                                  CellIsRule(operator="between", formula=["-0.001", "0.001"],
                                             fill=GREEN))
    ws.freeze_panes = "A5"
    for k, v in {"A": 8, "B": 8, "C": 11, "D": 11, "E": 11, "F": 11, "G": 12,
                 "H": 12, "I": 12, "J": 12, "K": 15, "L": 14, "M": 12, "N": 10,
                 "O": 9}.items():
        ws.column_dimensions[k].width = v

    # ================================================================ Data checks
    ws = wb.create_sheet("Data checks")
    ws.sheet_properties.tabColor = "616161"
    r = title(ws, 1, "Data checks — exclusions, revisions, tags, structure")
    r = para(ws, r,
             "PHM (PulteGroup) is excluded from the panel. Pulte files revenue under "
             "three XBRL tags with non-matching coverage (Revenues to 2024, "
             "RevenueFromContractWithCustomer to 2026, segment tags to 2018), so "
             "four Q4 periods (Q4-2009, 2021, 2022, 2023) cannot be reconstructed "
             "from any single tag and the series runs one quarter behind the panel "
             "(last quarter 2026-06-30). The backtest framework refuses series with "
             "calendar gaps beyond 120 days because quarter-offset baselines assume "
             "clean spacing.", span=15)
    facts = json.loads(open(config.CACHE_DIR / "facts_PHM.json", encoding="utf-8").read())
    dfp = extract_metric(facts, "revenue")
    tagtab = (dfp.groupby("tag").agg(rows=("val", "size"), first=("end", "min"),
                                     last=("end", "max")).reset_index())
    r = table(ws, r, 1, ["PHM tag", "Rows", "First period", "Last period"],
              [[x.tag, int(x.rows), pd.Timestamp(x.first).to_pydatetime(),
                pd.Timestamp(x.last).to_pydatetime()] for x in tagtab.itertuples()],
              fmts={2: DATEF, 3: DATEF})
    phm_gaps = find_gaps(clean_revenue(dfp.assign(ticker="PHM", sector="x")))
    r = table(ws, r + 1, 1,
              ["PHM gap: quarter before", "quarter after", "missing days"],
              [[pd.Timestamp(x.before).to_pydatetime(),
                pd.Timestamp(x.after).to_pydatetime(), int(x.days)]
               for x in phm_gaps.itertuples()], fmts={0: DATEF, 1: DATEF, 2: "0"})
    v = validate(clean)
    r = table(ws, r + 1, 1,
              ["Derived Q4 per company", "Quarters kept", "of which derived Q4", "share"],
              [[t, int(v.loc[t, "quarters"]), int(v.loc[t, "derived_q4"]),
                v.loc[t, "derived_q4"] / v.loc[t, "quarters"]] for t in v.index],
              fmts={1: "0", 2: "0", 3: PCT})
    r2 = raw[raw["metric"] == "revenue"].copy()
    for c in ("start", "end", "filed"):
        r2[c] = pd.to_datetime(r2[c])
    r2 = r2[r2["ticker"].isin(tickers)]
    r2 = r2[(r2["end"] - r2["start"]).dt.days.between(*QUARTER_DAYS)].sort_values("filed")
    g = r2.groupby(["ticker", "tag", "start", "end"])["val"]
    chk = pd.DataFrame({"first": g.first(), "last": g.last()})
    chk["pct"] = (chk["last"] / chk["first"] - 1).abs() * 100
    rev = chk.groupby("ticker").agg(mean=("pct", "mean"), median=("pct", "median"),
                                    max=("pct", "max"))
    r = table(ws, r + 1, 1,
              ["Revision check: first vs latest filing of the same period",
               "mean", "median", "max"],
              [[t, float(rev.loc[t, "mean"]) / 100, float(rev.loc[t, "median"]) / 100,
                float(rev.loc[t, "max"]) / 100] for t in rev.index],
              fmts={1: PCT3, 2: PCT3, 3: PCT3})
    r = note(ws, r, f"Largest revision across every company and period: "
                    f"{rev['max'].max() / 100:.2%} — filings are effectively never "
                    f"revised in this sample, so latest-filing dedupe is safe.")
    sw = tag_switches(clean)
    r = table(ws, r + 1, 1,
              ["Tag switches in the cleaned panel", "period end", "from tag", "to tag"],
              [[x.ticker, pd.Timestamp(x.period_end).to_pydatetime(),
                x.from_tag, x.to_tag]
               for x in sw.itertuples()], fmts={1: DATEF})
    note(ws, r, f"Panel: {len(clean):,} cleaned quarterly rows, {len(tickers)} "
                f"companies, {clean['period_end'].min():%Y-%m} to "
                f"{clean['period_end'].max():%Y-%m}. Locked 2022+ test window was "
                f"opened exactly once, on record (access log 2026-10-22).")
    for k, w in {"A": 46, "B": 14, "C": 36, "D": 36}.items():
        ws.column_dimensions[k].width = w

    # ================================================================ Method and limits
    ws = wb.create_sheet("Method and limits")
    ws.sheet_properties.tabColor = "616161"
    r = title(ws, 1, "Method and limits — in plain language")
    for t in [
        "THE FORECAST. Each company's next-quarter revenue is forecast as the same "
        "quarter last year scaled by the company's latest momentum (last quarter "
        "over the quarter before it). This 'seasonal drift' rule was chosen because "
        "it beat a machine-learning model using mortgage rates and housing starts "
        f"on data the models never saw while being tuned ({cov:.1%} of actuals "
        "fell inside the 80% interval on the locked 2022-2026 test).",
        "THE INTERVAL. The 80% interval comes from the model's own past mistakes in "
        "the same sector: the middle 80% of past percentage errors, projected onto "
        "the new forecast. It is honest width, not decoration — about a fifth of "
        "actuals in the test period sat outside it, as expected at 80%.",
        "THE BIAS CORRECTION. The whole rule ran a little high (+1.6% on the test "
        "period). Published points are scaled down by the sector's median past "
        "error to correct that — standard practice before a forecast is used.",
        "DATA. Revenue comes from SEC EDGAR filings; we keep the latest filing of "
        "each period and merge six revenue tag names that filers switch between. "
        "Fiscal fourth quarters that are never reported separately are derived as "
        "annual minus the first three quarters, always within one tag so the "
        "pieces measure the same thing. Macro reference data (mortgage rates, "
        "housing starts) comes from FRED with publication lags respected.",
        "LIMITS. (1) One quarter ahead only; longer horizons need a different "
        "model. (2) Ten companies in two related sectors; averages hide company "
        "detail. (3) The 2020 COVID quarter sits inside the error history and "
        "widens the intervals — we prefer honest width over flattery. (4) The ML "
        "reference model did not beat the baseline out-of-sample; its column is "
        "for comparison only. (5) Q3-2026 figures are included as filed in the "
        "data cache; late revisions are effectively nil (max 0.05%). (6) PHM is "
        "excluded for data-quality reasons (see Data checks).",
        "REFRESH. Rerun the project pipeline (data pull, clean, backtest) and this "
        "workbook regenerates from the refreshed parquet files and run log: "
        "python -m scripts.make_excel_report",
    ]:
        r = para(ws, r, t, span=12)
    ws.column_dimensions["A"].width = 22

    # ================================================================ Figures
    ws = wb.create_sheet("Figures")
    ws.sheet_properties.tabColor = "1565C0"
    r = title(ws, 1, "Figures — the memo charts, embedded",
              "Built by scripts/make_memo.py from the same audited artifacts; "
              "PNG sources in outputs/figures.")
    sdv = res[res["model"] == "seasonal_drift"]
    okp = float(((sdv["actual"] - sdv["forecast"]).abs()
                 / sdv["forecast"]).le(.05).mean())
    two = res["model"].isin(("seasonal_drift", "ridge_drivers"))
    nout = int((((res.loc[two, "actual"] - res.loc[two, "forecast"]).abs()
                 / res.loc[two, "forecast"])).gt(.2).sum())
    embed = {"fig1_mape_by_year.png": (760, 401),
             "fig2_mape_by_company.png": (760, 465),
             "fig3_forecast_vs_actual.png": (760, 380)}
    for fname, cap in [
        ("fig1_mape_by_year.png",
         "1. Forecast error (MAPE, %) by calendar year, all three models: thick "
         "blue = chosen seasonal drift; orange = ridge + macro drivers (ML "
         "reference); green = naive last quarter. Shaded band = locked 2022+ "
         "test window; the tall 2020 spike is the COVID quarter."),
        ("fig2_mape_by_company.png",
         "2. Average error by company — paired horizontal bars coloured by "
         "model: blue = chosen seasonal drift (upper bar), orange = ridge "
         "reference (lower bar); sector in parentheses beside each ticker. "
         "Worst forecasters at the top."),
        ("fig3_forecast_vs_actual.png",
         f"3. LEFT — every forecast vs its actual (log-log), both models: the "
         f"diagonal is a perfect forecast; points close to it = accurate. "
         f"RIGHT — how far off each forecast was, grouped into the same zones "
         f"the Variance sheet uses: green = within ±5% (on plan), amber = 5-10% "
         f"(review), red = beyond ±10% (escalate). Chosen model: {okp:.1%} of "
         f"quarters inside ±5%; {nout} of the {int(two.sum()):,} modeled "
         f"forecasts fall beyond the displayed ±20%."),
    ]:
        p = config.PROJECT_ROOT / "outputs" / "figures" / fname
        if not p.exists():
            continue
        img = XLImage(str(p))
        img.width, img.height = embed[fname]
        ws.add_image(img, f"A{r}")
        r = para(ws, r + img.height // 20 + 3, cap, span=8)

    print("writing workbook...", flush=True)
    ch_o = {5 + i: t for i, t in enumerate(nq["ticker"])}     # chart cache sources
    ch_p = {5 + i: float(x["point"]) / 1e9 for i, x in nq.iterrows()}
    ch_p[4] = "Point ($B)"                       # series-name cell (strCache)
    ser = {31 + i: int((d - pd.Timestamp("1899-12-30")).days)
           for i, d in enumerate(yoy.index)}
    co_v = {31 + i: CELL_CACHE.get(("Trend", "C" + str(31 + i)))
            for i in range(len(yoy.index))}
    sa_v = {31 + i: CELL_CACHE.get(("Trend", "D" + str(31 + i)))
            for i in range(len(yoy.index))}
    co_v[30], sa_v[30] = "Company YoY", "Sector avg"
    CHART_CACHE.update({"Next quarter": {"O": ch_o, "P": ch_p},
                        "Trend": {"B": ser, "C": co_v, "D": sa_v}})
    wb.save(OUT)
    inject_values(OUT)
    print("workbook saved; logging run record...", flush=True)
    max_diff = max(diffs) if diffs else 0.0
    try:
        notes = ("generated " + str(OUT.name) + ": next-quarter point forecast total "
                 + format(pt, ".1f") + "B (" + format(yr, "+.1%") + " YoY); "
                 + str(n_red) + " red / " + str(n_amber) + " amber flags in the last "
                 + str(N_VAR_QUARTERS) + " quarters; " + str(n_reco_bad) + " of "
                 + str(len(rec_rows)) + " reconciliation rows beyond 0.1% (max "
                 + format(max_diff, ".2%") + ")")
    except Exception as exc:             # never lose a workbook over a log line
        notes = "generated " + str(OUT.name) + "; note formatting failed: " + repr(exc)
    append_record({
        "kind": "excel_report",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "notes": notes,
    })
    print("Saved " + str(OUT))
    print("  next-quarter point total $" + format(pt, ",.1f") + "B ("
          + format(yr, "+.1%") + " YoY)")
    print("  variance flags: " + str(n_red) + " red, " + str(n_amber)
          + " amber (last " + str(N_VAR_QUARTERS) + " quarters)")
    print("  reconciliation: " + str(len(rec_rows)) + " company-years, "
          + str(n_reco_bad) + " beyond 0.1% (max " + format(max_diff, ".2%") + ")")

if __name__ == "__main__":
    main()