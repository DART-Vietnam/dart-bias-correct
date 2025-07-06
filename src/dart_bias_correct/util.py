"""Utility module for dart-bias-correct"""

import os
import platform
from pathlib import Path

import xarray as xr

DATA_HOME = (
    Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    if platform.system() == "Windows"
    else Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local" / "share"))
)


def is_hourly(ds: xr.Dataset | xr.DataArray, time_dim: str = "time") -> bool:
    "Returns True if dataset is hourly"
    return xr.infer_freq(ds[time_dim]) == "h"

def get_dart_root() -> Path:
    return Path(os.getenv("DART_PIPELINE_DATA_HOME") or DATA_HOME / "dart-pipeline")
