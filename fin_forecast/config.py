"""Single place for all project settings."""
from pathlib import Path

VALIDATION_START = "2018-01-01"   # forecast dates before this = train
TEST_START = "2022-01-01"         # forecast dates from here = locked test

# SEC requires a real contact in the User-Agent. EDIT THIS.
USER_AGENT = "Ally Fan allyfan923@gmail.com"

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # the repo root above fin_forecast/
CACHE_DIR = PROJECT_ROOT / "data" / "raw"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Confirm membership later; kept small and pure-play on purpose.
SECTORS = {
    "homebuilders": ["DHI", "LEN", "NVR", "TOL", "KBH"],
    "home_improvement": ["HD", "LOW", "FND", "SHW", "BLDR"],
}

# Companies switch XBRL tags over time, so we collect all and merge in cleaning.
METRIC_TAGS = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "HomeBuildingRevenue",
        "RealEstateRevenueNet",
    ],
}

REQUEST_PAUSE_SEC = 0.2  # stay under SEC's 10 requests/sec limit
