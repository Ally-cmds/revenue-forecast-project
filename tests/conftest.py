import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def panel_data():
    """Small synthetic panel: 4 companies, 50 quarters, plus quarterly drivers."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2010-03-31", periods=50, freq="QE")
    rows = []
    for t, sec in [("A", "homebuilders"), ("B", "homebuilders"),
                   ("C", "home_improvement"), ("D", "home_improvement")]:
        v = 100 * 1.02 ** np.arange(50) * np.tile([0.9, 1.0, 1.1, 1.0], 13)[:50]
        v = v * (1 + rng.normal(0, 0.02, 50))
        rows += [dict(ticker=t, sector=sec, period_end=d, revenue=x) for d, x in zip(idx, v)]
    qs = pd.date_range("2005-03-31", "2026-12-31", freq="QE")
    macro = pd.DataFrame({"mortgage30": 4 + rng.normal(0, 0.5, len(qs)).cumsum() * 0.1,
                          "starts": 1000 + rng.normal(0, 50, len(qs)).cumsum()}, index=qs)
    return pd.DataFrame(rows), macro
