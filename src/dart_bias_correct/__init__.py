import argparse
import sys
import logging

from .precipitation import bias_correct_precipitation
from .forecast import bias_correct_forecast_from_paths

LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(
        description="""
Bias correct either precipitation or forecast data using a
reference and uncorrected dataset

\033[1mMODES\033[0m

  \033[1mprecipitation\033[0m

    reference_dataset
      Authoritative historical reference dataset for precipitation (e.g., REMOCLIC 2016).

    uncorrected_dataset
      ~20 years of ERA5 data. This file can be generated using dart.

    dataset_to_correct
      Can be a NetCDF file, in which case an output file is created with _corrected
      and a tp_corrected variable appended; alternatively use dart::ISO3/era5/year
      to correct a specific ERA5 year. In the latter case, the default DART path
      will be used, or specify the root by setting the DART_PIPELINE_DATA_HOME
      environment variable

  \033[1mforecast\033[0m

    reference_dataset
      ~20 years of ERA5 data. This file can be generated using DART

    uncorrected_dataset
      ~20 years of historical forecast data. Generated with a script
      from DART-Pipeline.

    dataset_to_correct
       NetCDF file to correct
""",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Define the primary argument: mode
    parser.add_argument(
        "mode",
        choices=["precipitation", "forecast"],
        help="Select whether to run precipitation or forecast bias correction logic.",
    )

    # Positional arguments that follow
    parser.add_argument(
        "reference_dataset",
        help="Authoritative reference dataset (precipitation) or historical ERA5 data (forecast)",
    )
    parser.add_argument(
        "uncorrected_dataset",
        help="Uncorrected ERA5 dataset (precipitation) or historical forecast data (forecast)",
    )
    parser.add_argument(
        "dataset_to_correct",
        help="Either a NetCDF file to correct or a special syntax era5:ISO3-year to correct a specific year of ERA5 data.",
    )

    args = parser.parse_args()

    match args.mode:
        case "precipitation":
            bias_correct_precipitation(
                args.reference_dataset,
                args.uncorrected_dataset,
                args.dataset_to_correct,
            )
        case "forecast":
            bias_correct_forecast_from_paths(
                args.reference_dataset,
                args.uncorrected_dataset,
                args.dataset_to_correct,
            )
        case _:
            print(f"Unsupported mode: {args.mode}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
