"""Bias correction and prediction intervals from past forecast errors.
Pure functions on a backtest results frame (no I/O). Error e = log(forecast/actual),
so e > 0 means over-forecasting and actual = forecast * exp(-e).
One-step-ahead forecasts: the forecast for quarter t is made once quarter t-1 is known.
Two look-ahead-free modes, calibrated per sector. The cutoff for every forecast is
ITS OWN ORIGIN (the day after the previous quarter ends, column `origin` from the
backtest): only errors of quarters that ended before that day were known then.
  expanding: a forecast uses all errors known at its origin
  frozen:    a forecast uses only split == frozen_split errors known at its origin"""
import numpy as np
import pandas as pd


def _pool(g, cur, mode, frozen_split):
    known = g["period_end"] < cur["origin"].iloc[0]   # actual known at this origin
    if mode == "expanding":
        return g.loc[known, "log_err"]
    if cur["split"].iloc[0] == frozen_split:      # calibration rows get no interval
        return g["log_err"].iloc[0:0]
    return g.loc[known & (g["split"] == frozen_split), "log_err"]


def _apply(cur, pool, lo_q, hi_q, min_n, correct_bias):
    cur = cur.copy()
    if len(pool) < min_n:
        cur[["point", "lo", "hi"]] = np.nan
        return cur
    e_lo, e_mid, e_hi = pool.quantile([lo_q, 0.5, hi_q])
    f = cur["forecast"]
    cur["point"] = f * np.exp(-e_mid) if correct_bias else f
    cur["lo"], cur["hi"] = f * np.exp(-e_hi), f * np.exp(-e_lo)
    return cur


def add_interval(res, model, level=0.8, mode="expanding", min_n=20,
                 correct_bias=True, frozen_split="train"):
    """Adds point (bias-corrected forecast), lo, hi columns for one model."""
    if "origin" not in res.columns:
        raise ValueError("results need an 'origin' column: rerun the backtest")
    df = res[res["model"] == model].sort_values("period_end").copy()
    df["log_err"] = np.log(df["forecast"] / df["actual"])
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    parts = []
    for _, g in df.groupby("sector"):
        for i in range(len(g)):
            cur = g.iloc[[i]]
            parts.append(_apply(cur, _pool(g, cur, mode, frozen_split),
                                lo_q, hi_q, min_n, correct_bias))
    return pd.concat(parts).sort_values(["period_end", "ticker"])


def coverage_report(df, by=("sector",)):
    """Coverage, interval width and raw vs bias-corrected error. Percent units."""
    d = df.dropna(subset=["lo"]).copy()
    d["covered"] = (d["actual"] >= d["lo"]) & (d["actual"] <= d["hi"])
    d["width"] = (d["hi"] - d["lo"]) / d["point"] * 100
    d["pe_corr"] = (d["point"] - d["actual"]) / d["actual"]
    g = d.groupby(list(by))
    return pd.DataFrame({
        "coverage": g["covered"].mean() * 100,
        "width_pct": g["width"].mean(),
        "mape_raw": g["abs_pct_error"].mean() * 100,
        "mape_corrected": g["pe_corr"].apply(lambda s: s.abs().mean()) * 100,
        "bias_raw": g["pct_error"].mean() * 100,
        "bias_corrected": g["pe_corr"].mean() * 100,
        "n": g.size(),
    }).round(2)