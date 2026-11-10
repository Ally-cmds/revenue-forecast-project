"""List candidate XBRL tags per ticker so we can fix the metric mapping."""
import sys

from fin_forecast.data import data_loader

KEYWORDS = ("revenue", "sales", "costof", "grossprofit", "homebuilding", "costs")
DEFAULT = ["TOL", "NVR", "PHM", "LEN", "KBH", "DHI"]


def tag_summary(ticker: str):
    gaap = data_loader.get_company_facts(ticker)["facts"].get("us-gaap", {})
    rows = []
    for tag, body in gaap.items():
        if any(k in tag.lower() for k in KEYWORDS):
            entries = body.get("units", {}).get("USD", [])
            if entries:
                rows.append((tag, len(entries), max(e["end"] for e in entries)))
    return sorted(rows, key=lambda r: -r[1])


if __name__ == "__main__":
    for t in (sys.argv[1:] or DEFAULT):
        print(f"\n=== {t} ===")
        for tag, n, last in tag_summary(t)[:15]:
            print(f"{n:5d}  {last}  {tag}")
