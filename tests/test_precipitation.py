"Snapshot testing of bias correction for precipitation data"

from pathlib import Path

import pytest
import xarray as xr
import numpy.testing as npt
from dart_bias_correct.precipitation import bias_correct_precipitation

# This file is not open access and must be fetched from dart-pipeline-private
TP_REF = Path("tests/data/HCMC-regrid_era_remoclic.nc")

ERA_TP = Path("tests/data/HCMC-era5.nc")
DATASET_TO_CORRECT = Path("tests/data/HCMC-2015-era5.accum.nc")


@pytest.fixture(scope="module")
def uncorrected_precipitation():
    return xr.open_dataset(DATASET_TO_CORRECT)


@pytest.fixture(scope="module")
def corrected_precipitation(uncorrected_precipitation):
    if not TP_REF.exists():
        raise FileNotFoundError(f"""Could not find precipitation reference at: {TP_REF}
    This file is not open access and needs to be downloaded from the 'dart-pipeline-private' bucket,
    or alternatively, obtained from the authors for testing.""")
    tp_ref = xr.open_dataset(TP_REF)
    era_tp = xr.open_dataset(ERA_TP)
    uncorrected_precipitation = uncorrected_precipitation.resample(valid_time="D").sum()
    return bias_correct_precipitation(tp_ref, era_tp, uncorrected_precipitation)


def test_precipitation_is_corrected(corrected_precipitation, uncorrected_precipitation):
    max_diff = (corrected_precipitation.tp - uncorrected_precipitation.tp).max().values
    npt.assert_approx_equal(max_diff, 0.11450434145720108)
