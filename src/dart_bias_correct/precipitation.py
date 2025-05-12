"""Bias correction module (precipitation)"""

from pathlib import Path

import xarray as xr
import cmethods


def adjust_wrapper_tp(
    obs: xr.DataArray, simh: xr.DataArray, simp: xr.DataArray
) -> xr.DataArray:
    """Perform quantile mapping and correct total precipitation

    Parameters
    ----------
    obs
        Observations or data of reference
    simh
        Dataset of uncorrected data
    simp
        Dataset to correct

    Returns
    -------
    xr.DataArray
    """
    return cmethods.adjust(
        method="quantile_delta_mapping",  # methodology to correct data
        obs=obs,
        simh=simh,
        simp=simp,
        n_quantiles=100,
        kind="*",
    )


def bias_correct_precipitation(
    reference_dataset: Path, uncorrected_dataset: Path, dataset_to_correct: Path
) -> Path:
    """Bias correct precipitation data

    Parameters
    ----------
    reference_dataset : Path
        Reference dataset to use. This is usually externally provided
        from independent studies performing rainfall gauge measurements.
    uncorrected_dataset : Path
        Dataset to correct. This is usually concatenated ERA5 total_precipitation
        data, which can be obtained from dart-pipeline:

        .. shell::
            dart-pipeline process era5.prep_bias_correct VNM 2000-2020 profile=precipitation

        The above command would produce a concatenated netCDF file that comprised
        the years 2000-2020 for Vietnam, and would be saved in the dart-pipeline
        output folder as

        .. shell::
            output/VNM/era5/VNM-2000-2020-era5.prep_bias_correct.precipitation.nc

    dataset_to_correct : Path
        Dataset to correct, this is usually a single year file such as
        VNM-2020-era5.accum.nc but can also span multiple files, such as the
        concatenated dataset produced above

    Returns
    -------
    Path
        Path where corrected dataset was written to
    """
    tp_ref = xr.open_dataset(reference_dataset)
    era_tp = xr.open_dataset(uncorrected_dataset).rename({"valid_time": "time"})
    accum_vars = xr.open_dataset(dataset_to_correct).rename({"valid_time": "time"})
    corrected_tp = adjust_wrapper_tp(tp_ref.tp, era_tp.tp, accum_vars.tp).rename(
        {"time": "valid_time"}
    )
    output_file = dataset_to_correct.parent / (
        dataset_to_correct.stem + "_tp_corrected.nc"
    )
    corrected_tp.to_netcdf(output_file)
    return output_file
