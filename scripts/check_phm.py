"""One-off: document exactly why PulteGroup (PHM) is not in the panel."""
import json
import sys

import pandas as pd

sys.path.insert(0, ".")

from fin_forecast.data.cleaning import (clean_revenue, find_gaps, tag_switches,
                                        validate)
from fin_forecast.data.data_loader import extract_metric

facts = json.loads(open("data/raw/facts_PHM.json").read())
df = extract_metric(facts, "revenue")
print("PHM rows from configured tags:", len(df))
if len(df):
    print(df.groupby("tag").agg(n=("val", "size"), first=("end", "min"),
                                last=("end", "max")).to_string())
    clean = clean_revenue(df.assign(ticker="PHM", sector="x"))
    v = validate(clean)
    r = v.loc["PHM"]
    print("quarters kept after cleaning:", r["quarters"], "| span:", r["first"], "-",
          r["last"], "| gaps>120d:", r["gaps_over_120d"], "| tags used:", r["tags_used"])
    sw = tag_switches(clean.reindex(columns=["ticker", "sector", "period_end",
                                             "revenue", "tag", "derived"]))
    print(sw.to_string(index=False) if len(sw) else "no tag switches")
    gaps = find_gaps(clean)
    print("GAPS >120d:")
    print(gaps.to_string(index=False) if len(gaps) else "none")

gaap = facts["facts"]["us-gaap"]
cands = [t for t in gaap
         if any(k in t.lower() for k in ("revenue", "sales"))
         and gaap[t].get("units", {}).get("USD")]
print("\nrevenue-ish tags PHM files:")
for t in cands:
    e = gaap[t]["units"]["USD"]
    q = [x for x in e if x.get("form") == "10-Q"
         and 80 <= (pd.Timestamp(x["end"]) - pd.Timestamp(x["start"])).days <= 100]
    last = max((x["end"] for x in q), default="-")
    print(f"  {t}: {len(e)} rows, {len(q)} quarterly 10-Q rows, last quarterly end {last}")