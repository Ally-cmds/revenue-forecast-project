# Revenue Forecast Project

Forecast next-quarter revenue for 10 homebuilders and home-improvement stocks
(SEC EDGAR data) and compare machine-learning models against simple baselines —
under strict no-leakage rules and pre-registered testing.

**Headline result:** on the locked test period (2022-01-01 onward, opened once on
2026-10-22), the pre-registered finalist `ridge_drivers` did **not** beat the
`seasonal_drift` baseline: quarter-weighted MAPE **7.27% vs 6.54%** (mean loss
difference +0.73 pp, 95% bootstrap CI [-0.63, +1.93], DM p = 0.185). 80% empirical
intervals covered **78.4%** of test actuals (nominal 80%). The baseline is a
tough benchmark for quarterly revenue.

## Layout

```
fin_forecast/            the library (everything a forecast needs)
  config.py              split dates, tickers, XBRL tags, cache dir, SEC user agent
  data/
    data_loader.py       raw SEC EDGAR fetch (on-disk JSON cache, data/raw/)
    cleaning.py          restatement dedupe, derived Q4, tag priority, health checks
    macro_loader.py      FRED quarterly drivers + the as_of() publication-lag rule
  features.py            leakage-safe feature builder (training rows AND forecast row)
  splits.py              time split + the one logged door to the locked test set
  backtest.py            rolling-origin backtest; Ctx built in one place
  context.py             Ctx: ticker, sector, origin, as_of-masked macro, past panel
  evaluation.py          MAPE / median APE / bias summaries (percent units)
  calibration.py         empirical per-sector intervals + bias correction
  stats_tests.py         Diebold-Mariano (HLN), circular block bootstrap, Holm
  plan.py                the pre-registered plan values (single source of truth)
  freeze.py              writes/verifies the frozen plan + code/data fingerprints
  frozen_test.py         the ONE-TIME test evaluation named in the plan
  models/
    baselines.py         naive_last, seasonal_naive, seasonal_drift
    pooled.py            shared base: pooled log-growth models + seasonal_drift fallback
    ridge_growth.py      ridge ladder (own -> +drivers -> +drivers x sector)
    elastic_net.py       elastic net (validation-only candidate)
    pca.py               PCR with PCA fitted inside each training window
    gbm.py               gradient-boosted trees (validation-only candidate)
  run_log.py             append-only runs/run_log.jsonl + latest forecasts CSV per run
scripts/                 CLI entry points (run from repo root: python -m scripts.<name>)
tests/                   pytest suite (pytest.ini sets pythonpath=.)
runs/                    run_log.jsonl, disclosure record, forecast CSVs (versioned)
data/raw/                SEC/FRED caches (gitignored)
data/processed/          cleaned series, backtest results, test_access.log (gitignored)
outputs/results_memo.md  the findings memo (with figures) built by scripts/make_memo
Revenue_Forecast_Review.xlsx  the Excel deliverable built by scripts/make_excel_report
outputs/tableau/           BI fact-table CSVs for Tableau / Power BI (make_tableau_export)
forecast.sql               banquet-style SQL mirror of the MIS fact tables (schema + data)
outputs/powerbi/README.md  the Power BI dashboard build guide (measures, layout, RAG rules)
```

## Pipeline

| Step | Command | What it does |
|------|---------|--------------|
| 1. fetch | `python -m fin_forecast.data.data_loader` | SEC companyfacts per ticker → `data/raw/facts_*.json` |
| 2. clean | `python -m scripts.run_clean` | → `data/processed/revenue_quarterly.parquet` + health checks |
| 3. macro | `python -m scripts.check_macro` | FRED drivers cached + `as_of` sanity check |
| 4. data audit | `python -m scripts.restatement_check` | how much quarterly revenue changed after first filing |
| 5. backtest | `python -m scripts.run_backtest` | baselines + ridge ladder; test rows stay locked |
| 6. validation tuning | `python -m scripts.tune_elastic_net` / `tune_pca` / `tune_gbm` / `calibrate` / `check_interval_levels` | validation years only; `stop_before` never computes test forecasts |
| 7. pre-register | `python -m scripts.freeze_plan` | frozen plan + fingerprints to the run log (once) |
| 8. the test | `python -m scripts.run_test_once` | refuses to run without a plan, with changed inputs, or twice |
| 9. history | `python -m scripts.show_runs` | compact view of the append-only run log |
| 10. excel report | `python -m scripts.make_excel_report` | builds `Revenue_Forecast_Review.xlsx` — review workbook with live formulas (with baked-in results), charts, drop-downs and the embedded memo figures |
| 11. bi extracts | `python -m scripts.make_tableau_export` | DuckDB fact-table CSVs for Tableau / Power BI; `forecast.sql` is the MySQL-style mirror |

Tests: `.venv\Scripts\python.exe -m pytest` (or `pytest` with the venv active).
The full suite takes ~6 minutes; the five "no future information reaches X"
tests dominate by design (they scramble the future and assert the forecast
cannot change).

## The leakage rules (what keeps this honest)

1. **Origin rule** — a forecast for quarter *t* is made at
   `origin = end of quarter t-1 + 1 day`. Every look at the world uses only
   things *known at its origin*: backtest panel rows (`period_end < origin`),
   macro values (`macro_loader.as_of`), calibration errors
   (`calibration._pool`). `features.py` builds training rows with the same
   rule, so train-time and forecast-time features cannot drift apart.
2. **Publication lag** — never-revised series (mortgage rates, DGS10) may be
   used once their quarter ends; revised series (housing starts, permits) get
   a guarded extra quarter.
3. **Locked test** — `splits.reveal_test()` is the only way to see test rows;
   each call appends to `data/processed/test_access.log`
   (`count_test_openings` reports the tally). `scripts.run_test_once` checks
   code/data/FRED fingerprints against the frozen plan and self-refuses if the
   test was already opened.
4. **Append-only run log** — `runs/run_log.jsonl`; test-period numbers are
   never written to it. `kind: "seen_results"` is the one-off disclosure of
   results the analyst saw before the split existed (those models have no
   pristine test set).

## Method detail

- **Target**: log YoY revenue growth; forecast = revenue 4 quarters ago ×
  exp(predicted growth). Features: own last two YoY growths, sector flag,
  4 lagged year-over-year macro drivers (rates in points, others in logs),
  optional sector × driver interactions.
- **Cleaning**: per (ticker, tag, period) keep the latest filing (restatement
  dedupe); Q4 = FY − Q1..Q3 computed per tag so the pieces measure the same
  thing; one row per quarter via config tag priority, direct over derived;
  trailing early stubs before a pre-2012–era gap are dropped (`trim_early_gaps`).
- **Fallbacks**: pooled models fall back to `seasonal_drift` and count it
  (`n_fallback`), never fail silently.
- **Validation tuning** (2018–2021 only): 34 configurations tried —
  3 baselines, 3 ridge, a 4-driver ridge control, 18 elastic net, 5 PCA,
  2 GBM, 2 calibration modes. The finalist (`ridge_drivers`, α=10,
  drivers (mortgage30, starts)) was registered *before* the test was opened.
- **Inference**: the calendar quarter is the unit (firms share shocks);
  DM test with Harvey– Leybourne–Newbold correction, circular block bootstrap
  (4-quarter blocks, 10 000 resamples, seed 0), Holm across the two sectors.

## State and disclosures

- `runs/run_log.jsonl` is the permanent history: 17 run records, the
  `seen_results` disclosure, the frozen plan (2026-10-22T12:06:22), and the
  test result (2026-10-22T12:07:50, one opening).
- The pre-registered decision rule (bootstrap CI of the mean loss difference
  entirely below zero) was **not** met; per the plan the baseline stands.
  Model errors partly stem from small-sample fits and the 2020 structural
  break visible in validation tables (`exclude_years=(2020,)`).
- **Post-test code changes (2026-10-27)** were limited to non-frozen
  infrastructure: repaired broken imports (`restatement_check`,
  `diagnose_tags`, `data_loader`), removed a duplicated tuning run
  (`tune_gbm`), unified the test-opening counter (`run_log`), housekeeping
  (`scripts/__init__.py`, stale comments, deleted empty `fin_forecast/reporting`),
  plus README, requirements and new cleaning tests. No frozen modeling file
  (features, backtest, models/pooled, ridge, baselines, calibration,
  stats_tests, splits, freeze, frozen_test, plan, context, evaluation,
  data/macro_loader) was modified.

## Known limitations / next steps

- `config.py` is not in `freeze.FROZEN_CODE`; a future plan should fingerprint
  it too (today it is guarded indirectly via the data fingerprint).
- The project is deliberately small (10 tickers, 2 sectors); more history and
  sectors would sharpen inference, and the test conclusion (ridge ≯ baseline)
  suggests model form, not tuning, is the binding constraint.
- `opencode.json` (gitignored) holds a plaintext API key — it stays local; never
  stage, commit, or share it.
- Initialize git if you want `_git_commit` fingerprints to resolve (they are
  currently `null` in the run log).