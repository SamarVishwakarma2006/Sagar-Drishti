"""
Unit tests for Copernicus 10-Year Dataset Validation Module.
Verifies variable matching, coordinate monotonicity, depth level filtering,
and baseline comparison mechanisms.
"""

import os
import pytest
from app.services.validate_10yr_copernicus import Copernicus10YearValidator

BASELINE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "copernicus", "copernicus_phy_2yr_surface.nc")
)


def test_validator_file_not_found():
    with pytest.raises(FileNotFoundError):
        Copernicus10YearValidator.validate("non_existent_dataset.nc")


@pytest.mark.skipif(not os.path.exists(BASELINE_PATH), reason="Baseline 2yr dataset not available")
def test_baseline_dataset_structural_consistency():
    # Validates structural properties on the available baseline file
    rep = Copernicus10YearValidator.validate(
        filepath=BASELINE_PATH,
        sample_time_indices=[0, 100, 365, 729],
    )
    # The 2-year file has 730 timesteps (not 3652), so temporal is False, but variables, spatial, depth must pass!
    checks = rep["checks"]
    assert checks["variables"]["passed"] is True
    assert set(checks["variables"]["found"]) == {"mlotst", "so", "thetao", "uo", "vo", "zos"}
    assert checks["spatial"]["passed"] is True
    assert checks["spatial"]["lat_count"] == 301
    assert checks["spatial"]["lon_count"] == 601
    assert checks["depth"]["passed"] is True
    assert checks["depth"]["is_surface_only"] is True
    assert checks["physical_ranges"]["passed"] is True
