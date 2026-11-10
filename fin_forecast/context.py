"""Per-forecast context handed to every forecaster (baselines ignore it).
Built in ONE place (backtest.py) so the leakage rule can't be bypassed."""
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Ctx:
    ticker: str
    sector: str
    origin: pd.Timestamp                    # first day of the quarter being forecast
    macro: pd.DataFrame | None = None       # drivers already masked by as_of(origin)
    panel: pd.DataFrame | None = None       # every company's quarters that ended before origin