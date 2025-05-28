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


def is_hourly(ds: xr.Dataset, time_dim: str = "time") -> bool:
    "Returns True if dataset is hourly"
    return sorted(set(ds[time_dim].dt.strftime("%H:%M").to_numpy())) == [
        f"{i:02d}:00" for i in range(24)
    ]


def get_dart_root() -> Path:
    return Path(os.getenv("DART_PIPELINE_DATA_HOME") or DATA_HOME / "dart-pipeline")
