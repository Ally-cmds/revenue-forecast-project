"""Raw SEC EDGAR loader. No cleaning here: fetch, cache, tag, return."""
import json
import time

import pandas as pd
import requests

from fin_forecast import config

_HEADERS = {"User-Agent": config.USER_AGENT}


def _get_json(url: str, cache_name: str) -> dict:
    """GET with on-disk cache so each file is downloaded once."""
    path = config.CACHE_DIR / cache_name
    if path.exists():
        return json.loads(path.read_text())
    resp = requests.get(url, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    path.write_text(resp.text)
    time.sleep(config.REQUEST_PAUSE_SEC)
    return resp.json()


def get_cik(ticker: str) -> str:
    data = _get_json("https://www.sec.gov/files/company_tickers.json",
                     "company_tickers.json")
    for row in data.values():
        if row["ticker"].upper() == ticker.upper():
            return str(row["cik_str"]).zfill(10)
    raise ValueError(f"CIK not found for {ticker}")


def get_company_facts(ticker: str) -> dict:
    cik = get_cik(ticker)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    return _get_json(url, f"facts_{ticker}.json")


def extract_metric(facts: dict, metric: str) -> pd.DataFrame:
    """All reported USD entries for a metric, across every candidate tag."""
    frames = []
    gaap = facts.get("facts", {}).get("us-gaap", {})
    for tag in config.METRIC_TAGS[metric]:
        entries = gaap.get(tag, {}).get("units", {}).get("USD", [])
        if entries:
            df = pd.DataFrame(entries)
            df["tag"] = tag
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["metric"] = metric
    return out


def load_raw(tickers: list[str], sector: str) -> pd.DataFrame:
    """Long table: ticker, sector, metric, tag, start, end, val, form, fp, fy, filed."""
    rows = []
    for t in tickers:
        facts = get_company_facts(t)
        for metric in config.METRIC_TAGS:
            df = extract_metric(facts, metric)
            if not df.empty:
                df["ticker"], df["sector"] = t, sector
                rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_all_sectors() -> pd.DataFrame:
    return pd.concat(
        [load_raw(tks, name) for name, tks in config.SECTORS.items()],
        ignore_index=True,
    )


if __name__ == "__main__":
    raw = load_all_sectors()
    print(raw.groupby(["sector", "ticker", "metric"]).size().unstack(fill_value=0))
    raw.to_parquet(config.CACHE_DIR / "raw_all.parquet")
