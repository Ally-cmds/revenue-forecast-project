"""Export Tableau-ready extracts from the audited parquet artifacts via SQL.

A MySQL-style standalone mirror of these fact tables lives in forecast.sql at
the repo root (banquet.sql format), and outputs/powerbi/README.md documents the
Power BI build on top of the same files.

Runs DuckDB directly over the project parquet files (no server, no copy) and
writes three tidy extracts to outputs/tableau/:

  fact_forecast_accuracy.csv  one row per company-quarter: chosen / ridge / naive
                              forecasts side by side, variance %, review zone
  fact_company_league.csv     one row per company: full-period accuracy of the
                              chosen model vs the ridge reference, ranked
  fact_error_by_year.csv      calendar-year MAPE per model + average 30-year
                              mortgage rate for context

Run: python -m scripts.make_tableau_export   (also the notebook's final step)
"""
from pathlib import Path

import duckdb

from fin_forecast import config

ROOT = config.PROJECT_ROOT
DEST = ROOT / "outputs" / "tableau"

VIEWS = f"""
CREATE OR REPLACE VIEW vw_forecasts AS SELECT * FROM
    read_parquet('{(ROOT / "data/processed/backtest_results.parquet").as_posix()}');
CREATE OR REPLACE VIEW vw_panel AS SELECT * FROM
    read_parquet('{(ROOT / "data/processed/revenue_quarterly.parquet").as_posix()}');
CREATE OR REPLACE VIEW vw_sec AS SELECT ticker, tag, val, fy, form,
    TRY_CAST(start AS DATE) AS fy_start, TRY_CAST("end" AS DATE) AS fy_end,
    TRY_CAST(filed AS DATE) AS filed
    FROM read_parquet('{(ROOT / "data/raw/raw_all.parquet").as_posix()}')
    WHERE metric = 'revenue';
CREATE OR REPLACE VIEW vw_rate AS
    SELECT TRY_CAST(observation_date AS DATE) AS d, value
    FROM read_parquet('{(ROOT / "data/raw/fred_MORTGAGE30US.parquet").as_posix()}');
"""

FACT_SQL = """
WITH w AS (
    SELECT ticker, ANY_VALUE(sector) AS sector,
           period_end AS quarter_end,
           year(period_end) AS year, quarter(period_end) AS quarter_num,
           MAX(actual) AS actual,
           MAX(CASE WHEN model = 'seasonal_drift' THEN forecast END) AS fc_chosen,
           MAX(CASE WHEN model = 'ridge_drivers'  THEN forecast END) AS fc_ridge,
           MAX(CASE WHEN model = 'naive_last'     THEN forecast END) AS fc_naive
    FROM vw_forecasts
    GROUP BY 1, 3, 4, 5
), a AS (
    SELECT *,
           ROUND(100.0 * (actual - fc_chosen) / fc_chosen, 2) AS var_chosen_pct,
           ROUND(100.0 * (actual - fc_ridge)  / fc_ridge, 2)  AS var_ridge_pct
    FROM w
)
SELECT *, CASE WHEN ABS(var_chosen_pct) <= 5 THEN 'on plan'
               WHEN ABS(var_chosen_pct) <= 10 THEN 'review'
               ELSE 'escalate' END AS zone_chosen,
       CASE WHEN ABS(var_chosen_pct) <= 5 THEN 1 ELSE 0 END AS within_5pct
FROM a
ORDER BY quarter_end, ticker
"""

LEAGUE_SQL = """
WITH chosen AS (
    SELECT ticker, ANY_VALUE(sector) AS sector, COUNT(*) AS n_quarters,
           ROUND(AVG(abs_pct_error) * 100, 2) AS mape_chosen_pct,
           ROUND(AVG(CASE WHEN abs_pct_error <= .05 THEN 1.0 ELSE 0 END) * 100, 1)
               AS within_5pct_share
    FROM vw_forecasts
    WHERE model = 'seasonal_drift'
    GROUP BY 1
), ridge AS (
    SELECT ticker, ROUND(AVG(abs_pct_error) * 100, 2) AS mape_ridge_pct
    FROM vw_forecasts
    WHERE model = 'ridge_drivers'
    GROUP BY 1
)
SELECT c.*, r.mape_ridge_pct,
       ROUND(c.mape_chosen_pct - r.mape_ridge_pct, 2) AS chosen_minus_ridge,
       RANK() OVER (ORDER BY c.mape_chosen_pct) AS ease_rank
FROM chosen c
JOIN ridge r USING (ticker)
ORDER BY ease_rank
"""

YEAR_SQL = """
WITH yr AS (
    SELECT year(period_end) AS year, model,
           ROUND(AVG(abs_pct_error) * 100, 2) AS mape_pct
    FROM vw_forecasts
    GROUP BY 1, 2
), p AS (
    SELECT year,
           MAX(CASE WHEN model = 'seasonal_drift' THEN mape_pct END) AS mape_chosen_pct,
           MAX(CASE WHEN model = 'ridge_drivers'  THEN mape_pct END) AS mape_ridge_pct,
           MAX(CASE WHEN model = 'naive_last'     THEN mape_pct END) AS mape_naive_pct
    FROM yr
    GROUP BY 1
), r AS (
    SELECT year(d) AS year, ROUND(AVG(value), 2) AS avg_mortgage_rate
    FROM vw_rate
    GROUP BY 1
)
SELECT p.*, r.avg_mortgage_rate
FROM p LEFT JOIN r USING (year)
ORDER BY year
"""


def connect():
    con = duckdb.connect()
    con.execute(VIEWS)
    return con


def export_all(dest=None):
    """Write the three extracts; returns list of (path, rows)."""
    dest = Path(dest) if dest else DEST
    dest.mkdir(parents=True, exist_ok=True)
    con = connect()
    out = []
    for name, sql in (("fact_forecast_accuracy", FACT_SQL),
                      ("fact_company_league", LEAGUE_SQL),
                      ("fact_error_by_year", YEAR_SQL)):
        df = con.execute(sql).df()
        p = dest / f"{name}.csv"
        df.to_csv(p, index=False)
        out.append((p, len(df)))
    con.close()
    return out


if __name__ == "__main__":
    for p, n in export_all():
        print(f"  {p.relative_to(ROOT)}  {n:,} rows")
    print("wrote Tableau extracts to outputs/tableau/")