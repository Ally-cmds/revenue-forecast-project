"""The pre-registered test plan. ONE source of truth: freeze.py records these values,
frozen_test.py reads them, so the plan and the code that runs it cannot drift apart."""
FINALIST = "ridge_drivers"
BASELINE = "seasonal_drift"
RIDGE_ALPHA = 10.0                    # fixed before any validation result was seen
DRIVERS = ("mortgage30", "starts")
MIN_ROWS = 40

INTERVAL_LEVEL = 0.8
INTERVAL_MODE = "expanding"           # bias-corrected point forecast is SECONDARY

CI_LEVEL = 0.95
BOOT_BLOCK = 4                        # quarters per block (circular block bootstrap)
BOOT_N = 10000
BOOT_SEED = 0

# Candidates compared on the validation years only (distinct configurations).
VALIDATION_CANDIDATES = {"baselines": 3, "ridge": 3, "ridge_drivers4_control": 1,
                         "elastic_net": 18, "pca": 5, "gbm": 2, "calibration_modes": 2}
