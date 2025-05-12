"""Bias correction module (forecast)"""

from typing import Literal

import numpy as np
import xarray as xr
from cmethods import adjust

# indicates range of percentiles and whether this is an extreme
# if extreme, will be handled by adjust_wrapper_extremes(), otherwise
# by adjust_wrapper()
PERCENTILES: list[tuple[int, int, bool]] = [
    (0, 5, True),
    (5, 10, True),
    (10, 90, False),
    (90, 95, True),
    (95, 100, True),
]


def adjust_wrapper_quantiles(
    n_quantiles: int,
    obs: xr.Dataset,
    simh: xr.Dataset,
    simp: xr.Dataset,
    kind: Literal["+", "*"],
) -> xr.Dataset:
    """Function to correct extreme values located in the tails of the distribution

    Parameters
    ----------
    n_quantiles
        Number of quantiles, passed as the `n_quantiles` parameter
        to cmethods.adjust
    obs
        Historical ERA5 data that will be used as reference for
        quantile mapping correction
    simh
        Historical weather forecast data
    simp
        Real-time forecast data that we want to correct
    kind
        Type of quantile delta mapping "+" is additive
        (for temperature and humidity) whereas "*" is for precipitation

    See Also
    --------
    cmethods.adjust
        This is a thin wrapper around this bias correction method
    """

    return adjust(
        method="quantile_delta_mapping",
        obs=obs,
        simh=simh,
        simp=simp,
        n_quantiles=n_quantiles,  # Default number of quantiles for extreme correction
        kind=kind,  # "+"" for non tp and "*"" for tp
    )


# def bias_correct_forecast(
#     obs, simh, simp, extreme_percentiles: list[tuple[int, int, bool]] = PERCENTILES
# ):
#     for step in np.arange(7, 20, 7):  # 2 weeks in advance 7,14 and 21 were the steps
#         forecast = (
#             data.sel(step=f"{step}.days").sel(number=slice(0, 10)).mean(dim="number")
#         )  # Selecting forecast step, and using the mean ensemble
#         reanalysis = {7: era_week1, 14: era_week2}[
#             step
#         ]  # Selecting reanalysis data with the same starting weeks as the forecast
#         masks = {var: [] for var in ["t2m", "r", "tp"]}  # variables to correct
#
#         for p in range(
#             len(first_percentile)
#         ):  # Here, we will create the masks for identifying timestamps where reanalysis data is within the percentil tresholds to correct
#             for var in masks:
#                 low_quantile = reanalysis[var].quantile(
#                     first_percentile[p] / 100, dim="time"
#                 )
#                 high_quantile = reanalysis[var].quantile(
#                     last_percentile[p] / 100, dim="time"
#                 )
#                 mask = (reanalysis[var] >= low_quantile) & (
#                     reanalysis[var] < high_quantile
#                 )
#                 masks[var].append(mask)
#
#         for m, extreme in enumerate(extreme_percentiles):
#             for la in range(forecast.tp.shape[2]):
#                 for lo in range(forecast.tp.shape[3]):
#                     for s in range(temp.tp.shape[1]):
#                         data_to_corr_or = temp.sel(
#                             number=s
#                         )  # Selecting section of data to correct
#                         for var in masks:
#                             kind = "*" if var == "tp" else "+"
#                             inter_forecast = (
#                                 forecast[var]
#                                 .sel(number=s)
#                                 .where(masks[var][m])
#                                 .sel(lat=forecast.lat[la], lon=forecast.lon[lo])
#                                 .dropna(dim="time")
#                             )  # selecting reference forecast data with the same timestamps as ERA5 to
#                             inter_reanalysis = (
#                                 reanalysis[var]
#                                 .where(masks[var][m])
#                                 .sel(lat=reanalysis.lat[la], lon=reanalysis.lon[lo])
#                                 .dropna(dim="time")
#                             )  # Selecting ERA5 data in the percentile range
#
#                             data_to_corr = data_to_corr_or[var].sel(
#                                 lat=data_to_corr_or.lat[la], lon=data_to_corr_or.lon[lo]
#                             )  # selecting data to correct
#                             data_to_corr = data_to_corr.where(
#                                 (data_to_corr >= inter_forecast.min())
#                                 & (data_to_corr < inter_forecast.max()),
#                                 drop=True,
#                             )  # Locating values in the forecast data that are within
#                             # the percentile treshold
#                             if (
#                                 data_to_corr.size != 0
#                             ):  # If the values to correct are within the selected interval
#                                 n_quantiles = 10 if extreme else 90
#                                 corr_data = adjust_wrapper_quantiles(
#                                     n_quantiles,
#                                     inter_reanalysis,
#                                     inter_forecast,
#                                     data_to_corr.rename({"time": "time_to_corr"}),
#                                     kind=kind,
#                                 )
#                                 corr_data = corr_data.rename({"time_to_corr": "time"})
#
#                                 # Now that we corrected the data, we need to identify whether the corrected values were already corrected in other percentile treshold
#                                 selected_data = bool_dataset[var].loc[
#                                     dict(
#                                         time=data_to_corr.time,
#                                         number=s,
#                                         lat=data_to_corr.lat,
#                                         lon=data_to_corr.lon,
#                                     )
#                                 ]
#                                 filtered_data = selected_data.where(
#                                     selected_data < 1, drop=True
#                                 )  # Retention of uncorrected values
#
#                                 if (
#                                     filtered_data.shape != corr_data[var].shape
#                                 ):  # if filtered_data and corr_data have different shapes, it means that some values were already corrected, so those will not
#                                     # we saved in the new matrix
#                                     corr_data = corr_data.sel(time=filtered_data.time)
#                                 corrected_forecast[var].loc[
#                                     dict(
#                                         time=corr_data.time,
#                                         number=s,
#                                         lat=corr_data.lat,
#                                         lon=corr_data.lon,
#                                     )
#                                 ] = corr_data[var].values
#                                 bool_dataset[var].loc[
#                                     dict(
#                                         time=corr_data.time,
#                                         number=s,
#                                         lat=corr_data.lat,
#                                         lon=corr_data.lon,
#                                     )
#                                 ] += 1  # Marking the corrected value in the dummy dataset
