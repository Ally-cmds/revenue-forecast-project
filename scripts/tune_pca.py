"""Tune PCA regression on the VALIDATION years only (no test forecasts computed).
Includes a control: ridge on the same 4 drivers WITHOUT PCA, so we can tell whether
any gain comes from the extra drivers or from the PCA step itself.
Run from repo root: python -m scripts.tune_pca"""
from fin_forecast.models.pca import ALL_DRIVERS, make_pca_grid
from fin_forecast.models.ridge_growth import GrowthRidge
from scripts.tune_common import run_validation

KS = (1, 2, 3, 4, 6)

if __name__ == "__main__":
    grid = make_pca_grid(KS)
    control = {"ridge_drivers4": GrowthRidge(ALL_DRIVERS, False, 10.0)}

    def report():
        print("\nVARIANCE OF THE 16 DRIVER FEATURES KEPT BY PCA (last fit, data through 2021):")
        for name, m in grid.items():
            print(f"  {name}: {sum(m._explained):.0%}")

    note = (f"PCA grid k={KS} on 4 drivers {ALL_DRIVERS} ({len(KS)} configurations on "
            "validation) plus ridge_drivers4 control; no test forecasts computed")
    run_validation("tune_pca", {**control, **grid}, ALL_DRIVERS, note, report)