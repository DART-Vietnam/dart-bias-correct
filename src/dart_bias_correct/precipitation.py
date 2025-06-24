"""Bias correction module (precipitation)"""

import logging
from pathlib import Path

import xarray as xr
import cmethods
from geoglue.types import Bbox
from geoglue.region import gadm
from geoglue.cds import ReanalysisSingleLevels

from .util import is_hourly, get_dart_root

logger = logging.getLogger(__name__)


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


def crop_bbox(ds: xr.Dataset, bbox: Bbox) -> xr.Dataset:
    return ds.sel(
        latitude=slice(bbox.maxy, bbox.miny), longitude=slice(bbox.minx, bbox.maxx)
    )


def align_geo_extents(
    ds1: xr.Dataset, ds2: xr.Dataset
) -> tuple[xr.Dataset, xr.Dataset]:
    ds1_bbox = Bbox.from_xarray(ds1)
    ds2_bbox = Bbox.from_xarray(ds2)
    if ds1_bbox == ds2_bbox:
        return ds1, ds2
    if ds1_bbox < ds2_bbox:
        return ds1, crop_bbox(ds2, ds1_bbox)
    if ds2_bbox < ds1_bbox:
        return crop_bbox(ds1, ds2_bbox), ds2
    raise ValueError(f"""Could not align geospatial extents for:
    ds1: {ds1_bbox}
    ds2: {ds2_bbox}""")


def bias_correct_precipitation(
    reference_dataset: Path, uncorrected_dataset: Path, dataset_to_correct: str
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

    dataset_to_correct : str
        Dataset to correct, this is expressed in the form ISO3-YEAR, which loads
        the dataset (and preceding and succeeding datasets according to timeshift)
        for correction

    Returns
    -------
    Path
        Path where corrected dataset was written to
    """
    tp_ref = xr.open_dataset(reference_dataset)
    era_tp = xr.open_dataset(uncorrected_dataset)
    iso3, year = dataset_to_correct.split("-")
    year = int(year)
    data_path = get_dart_root() / "sources" / iso3 / "era5"
    pool = ReanalysisSingleLevels(
        gadm(iso3, 1), ["t2m", "tp"], path=data_path
    ).get_dataset_pool()
    output_file = data_path / f"{iso3}-{year}-era5.accum.tp_corrected.nc"
    accum_vars = pool[year].accum
    if "valid_time" in era_tp.coords:
        era_tp = era_tp.rename({"valid_time": "time"})
    tstart = max(tp_ref.time.min().values, era_tp.time.min().values)
    tend = min(tp_ref.time.max().values, era_tp.time.max().values)
    logger.info("Cropping time axis to common extents: %s --> %s", tstart, tend)
    tp_ref = tp_ref.sel(time=slice(tstart, tend))
    era_tp = era_tp.sel(time=slice(tstart, tend))

    tp_ref, era_tp = align_geo_extents(tp_ref, era_tp)
    accum_vars = accum_vars.rename({"valid_time": "time"})
    if is_hourly(accum_vars):
        accum_vars = accum_vars.resample(time="D").sum()
    corrected_tp = adjust_wrapper_tp(tp_ref.tp, era_tp.tp, accum_vars.tp).rename(
        {"time": "valid_time"}
    )
    corrected_tp.to_netcdf(output_file)
    logger.info("Output: %s", output_file)
    return output_file
