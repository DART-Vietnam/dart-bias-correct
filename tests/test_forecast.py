"Snapshot testing of bias correction for forecast data"

from pathlib import Path

import pytest
import xarray as xr
import numpy.testing as npt
from dart_bias_correct.forecast import bias_correct_forecast_from_paths

OBS = Path("tests/data/HCMC-era5.nc")
FORECAST = "tests/data/HCMC-2025-06-24-ecmwf.forecast.nc"
# This file is not open access and must be fetched from dart-pipeline-private
HISTORICAL_FORECAST = Path("tests/data/HCMC-historical.forecast.nc")


@pytest.fixture(scope="module")
def corrected_forecast():
    if not HISTORICAL_FORECAST.exists():
        raise FileNotFoundError(f"""Could not find historical forecast at: {HISTORICAL_FORECAST}
    This file is not open access and needs to be downloaded from the 'dart-pipeline-private' bucket,
    or alternatively, obtained from the authors for testing.""")
    corrected_forecast_file = bias_correct_forecast_from_paths(
        OBS, HISTORICAL_FORECAST, FORECAST
    )
    ds = xr.open_dataset(corrected_forecast_file, decode_timedelta=True)
    return ds


# TODO: Once forecast code is finalised, remove xfail marker
@pytest.mark.xfail(reason="Snapshot tests")
def test_forecast_snapshot(corrected_forecast):
    npt.assert_approx_equal(
        (corrected_forecast.t2m_bc - corrected_forecast.t2m).max().item(),
        2.6195068359375,
    )
    npt.assert_approx_equal(
        (corrected_forecast.r_bc - corrected_forecast.r).max().item(), 5.165809631347656
    )
    npt.assert_approx_equal(
        (corrected_forecast.tp_bc - corrected_forecast.tp).max().item(),
        0.11165580153465271,
    )


def test_forecast_value_bounds(corrected_forecast):
    assert "pevt" in corrected_forecast.data_vars
    assert (corrected_forecast.t2m_bc - corrected_forecast.t2m).max().item() > 1
    assert (corrected_forecast.r_bc - corrected_forecast.r).max().item() > 1
    assert (corrected_forecast.tp_bc - corrected_forecast.tp).max().item() > 0.1
    assert corrected_forecast.r_bc.min().item() >= 0
    assert corrected_forecast.r_bc.max().item() <= 100
