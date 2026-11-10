"""Build the results memo assets: summary tables + publication figures.
Numbers come ONLY from two auditable sources:
  1. data/processed/backtest_results.parquet  (the pre-split run, disclosed in the
     runs/ run-log record kind == "seen_results": all 6 models, all periods)
  2. runs/run_log.jsonl record kind == "test_result" (the one-time frozen test)
Run: python -m scripts.make_memo    (writes outputs/figures/*.png and prints tables)
"""
import json
import os
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from fin_forecast import config
from fin_forecast.run_log import load_runs

DOCS = config.PROJECT_ROOT / "outputs"
FIGS = DOCS / "figures"
MODELS = ["naive_last", "seasonal_drift", "ridge_drivers"]     # the story's cast
LABELS = {"naive_last": "naive (last quarter)",
          "seasonal_drift": "seasonal drift (chosen)",
          "ridge_drivers": "ridge + macro drivers (ML reference)"}
MCOLOR = {"naive_last": "#55a868", "seasonal_drift": "#4c72b0",
          "ridge_drivers": "#dd8452"}


def mape(df):
    return df["abs_pct_error"].mean() * 100


def test_verdict() -> dict:
    rec = [r for r in load_runs() if r.get("kind") == "test_result"][-1]["result"]
    return rec


def save_png(fig, path, tries=6):
    """Save via temp file + os.replace with brief retries. Explorer previews,
    thumbnail extraction or antivirus can hold a share on a PNG that was just
    viewed, and a direct overwrite can race that handle (Windows Errno 22/13)."""
    tmp = path.with_name(path.stem + ".tmp.png")
    err = None
    for _ in range(tries):
        try:
            fig.savefig(tmp)
            os.replace(tmp, path)
            return
        except OSError as e:
            err = e
            time.sleep(.7)
    if Path.exists(tmp):
        try:
            os.remove(tmp)
        except OSError:
            pass
    raise err


def main():
    res = pd.read_parquet(config.PROJECT_ROOT / "data" / "processed"
                          / "backtest_results.parquet")
    res = res[res["model"].isin(MODELS)].copy()
    res["year"] = res["period_end"].dt.year

    # ---------- printed summary tables ----------
    scope = res[res["model"] == "seasonal_drift"]
    print(f"STUDY SCALE: {res['ticker'].nunique()} companies, "
          f"{res['sector'].nunique()} sectors, "
          f"{scope['period_end'].nunique()} forecast quarters "
          f"({scope['period_end'].min():%Y-%m} to {scope['period_end'].max():%Y-%m}), "
          f"{len(scope):,} forecast observations/model, "
          f"${scope['actual'].sum() / 1e9:,.0f}B of forecasted revenue")

    print("\nMAPE BY YEAR (%)")
    print(res.groupby(["year", "model"])["abs_pct_error"].mean().unstack()
          .mul(100).round(2).to_string())

    print("\nMAPE EXCLUDING 2020 (%)")
    no20 = res[res["year"] != 2020]
    print(no20.groupby("model")["abs_pct_error"].mean().mul(100).round(2).to_string())

    print("\nMAPE BY COMPANY, ALL PERIODS (%)")
    byco = res.groupby(["ticker", "model"])["abs_pct_error"].mean().unstack().mul(100).round(2)
    byco = byco.join(scope.groupby("ticker")["sector"].first())
    print(byco.sort_values("seasonal_drift").to_string())

    print("\nBIAS BY YEAR (mean pct error, + = over-forecast, %)")
    print(res.groupby(["year", "model"])["pct_error"].mean().unstack().mul(100)
          .round(2).to_string())

    # ---------- figures ----------
    FIGS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    yearly = res.groupby(["year", "model"])["abs_pct_error"].mean().unstack().mul(100)

    # 1. MAPE by year — all three models on one axis, chosen model emphasised
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for m in MODELS:
        ax.plot(yearly.index, yearly[m], marker="o", ms=3.5,
                lw=2.2 if m == "seasonal_drift" else 1.3,
                zorder=3 if m == "seasonal_drift" else 2,
                color=MCOLOR[m], label=LABELS[m])
    ax.axvspan(2021.5, res["year"].max() + .5, color="#f2c94c", alpha=.15,
               label="locked test window (2022+, opened once)")
    ax.set_ylabel("MAPE (%)")
    ax.set_xlabel("calendar year of quarter-end")
    ax.set_title("One-step-ahead forecast error by year", fontsize=11)
    if 2021 in yearly.index:
        rb, sb = (float(yearly.loc[2021, "ridge_drivers"]),
                  float(yearly.loc[2021, "seasonal_drift"]))
        ax.annotate(f"2021 boom — ridge {rb:.1f}% vs chosen {sb:.1f}%",
                    xy=(2021, rb), xytext=(2013.3, 15.4), fontsize=8, color="#555",
                    arrowprops=dict(arrowstyle="->", lw=.8, color="#555"))
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center",
              bbox_to_anchor=(.5, -.19))
    fig.subplots_adjust(left=.075, right=.99, top=.9, bottom=.27)
    save_png(fig, FIGS / "fig1_mape_by_year.png")

    # 2. MAPE by company — horizontal paired bars coloured by MODEL
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    order = byco.sort_values("seasonal_drift")          # best at the bottom
    short = {"homebuilders": "builder", "home_improvement": "improvement"}
    for i, (t, r) in enumerate(order.iterrows()):
        ax.barh(i + .2, r["seasonal_drift"], .38, color=MCOLOR["seasonal_drift"])
        ax.barh(i - .2, r["ridge_drivers"], .38, color=MCOLOR["ridge_drivers"])
    ax.set_yticks(list(range(len(order))))
    ax.set_yticklabels([f"{t}  ({short.get(sec, sec)})"
                        for t, sec in zip(order.index, order["sector"])], fontsize=9)
    ax.set_xlabel("MAPE, all periods (%)")
    ax.set_title("Average forecast error by company", fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=MCOLOR["seasonal_drift"]),
               plt.Rectangle((0, 0), 1, 1, fc=MCOLOR["ridge_drivers"])]
    ax.legend(handles, [LABELS["seasonal_drift"], LABELS["ridge_drivers"]],
              frameon=False, fontsize=8.5, loc="lower right")
    ax.grid(axis="x", alpha=.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_png(fig, FIGS / "fig2_mape_by_company.png")

    # 3. accuracy view: forecast vs actual (left) + the error distribution
    #    against the Variance-sheet tolerance zones (right)
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(8.2, 4.1),
                                   gridspec_kw={"width_ratios": [1, 1.05]})
    lims = [.05, 1e3]
    for m in ("seasonal_drift", "ridge_drivers"):
        r = res[res["model"] == m]
        axl.scatter(r["actual"] / 1e9, r["forecast"] / 1e9, s=7, alpha=.4,
                    color=MCOLOR[m], label=LABELS[m])
    axl.plot(lims, lims, color="#444", lw=.8)
    axl.set_xscale("log"), axl.set_yscale("log")
    axl.set_xlim(lims), axl.set_ylim(lims)
    axl.set_xlabel("actual revenue ($B, log)")
    axl.set_ylabel("forecast ($B, log)")
    axl.set_title("Forecast vs actual — every quarter", fontsize=10)
    axl.legend(frameon=False, fontsize=7.5, loc="upper left")

    zg, za, zr = "#c8e6c9", "#ffe082", "#ef9a9a"
    for a, b, c in ((-25, -10, zr), (-10, -5, za), (-5, 5, zg),
                    (5, 10, za), (10, 25, zr)):
        axr.axvspan(a, b, color=c, alpha=.55, lw=0)
    v = (res["actual"] - res["forecast"]) / res["forecast"] * 100
    axr.hist(v[res["model"] == "seasonal_drift"],
             bins=np.arange(-20, 20.1, 1.25), color=MCOLOR["seasonal_drift"],
             alpha=.55, label=LABELS["seasonal_drift"])
    axr.hist(v[res["model"] == "ridge_drivers"],
             bins=np.arange(-20, 20.1, 1.25), color=MCOLOR["ridge_drivers"],
             alpha=.55, label=LABELS["ridge_drivers"])
    axr.set_xlim(-25, 25)
    axr.set_xticks([-20, -10, -5, 0, 5, 10, 20])
    axr.set_xlabel("forecast variance %")
    axr.set_ylabel("number of forecasts")
    axr.set_title("Error vs the tolerance zones", fontsize=10)
    okp = float((v[res["model"] == "seasonal_drift"].abs() <= 5).mean())
    handles, labs = axr.get_legend_handles_labels()
    handles.append(plt.Rectangle((0, 0), 1, 1, fc=zg))
    labs.append(f"chosen model: {okp:.0%} inside ±5%")
    axr.legend(handles, labs, frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    save_png(fig, FIGS / "fig3_forecast_vs_actual.png")

    # ---------- frozen test verdict (printed for the memo) ----------
    v = test_verdict()
    print("\nFROZEN TEST VERDICT (from run log, opened once 2026-10-22)")
    print(f"  n forecasts/model: {v['n_test_forecasts_per_model']}, quarters: "
          f"{v['primary']['quarters']}")
    print(f"  quarter-weighted MAPE  model {v['primary']['mape_model_qweighted']:.2f}% "
          f"vs baseline {v['primary']['mape_baseline_qweighted']:.2f}%")
    print(f"  decision: {v['decision']}")
    print(f"  80% interval coverage on test: {v['interval']['coverage_test']['rate']:.1%} "
          f"(nominal 80%)")
    print("\nWrote:", [p.name for p in FIGS.glob("*.png")])


if __name__ == "__main__":
    main()