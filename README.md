# dart-bias-correct

dart-bias-correct is a command-line utility designed to perform bias correction
on precipitation and forecast datasets using authoritative reference data. It accompanies
the broader [DART pipeline](https://github.com/kraemer-lab/DART-Pipeline).

## Installation

We recommend installation using either `uv` or `pipx`:

```shell
pipx install git+https://github.com/kraemer-lab/dart-bias-correct
uv tool install git+https://github.com/kraemer-lab/dart-bias-correct
```

## Usage

```shell
dart-bias-correct <precipitation|forecast> <reference_dataset> <uncorrected_dataset> <dataset_to_correct>
```

### precipitation

- **`reference_dataset`**: Authoritative historical precipitation dataset
  (e.g., REMOCLIC 2016 for Vietnam).
- **`uncorrected_dataset`**: ~20 years of ERA5 data, generated using
  dart-pipeline.
- **`dataset_to_correct`**: A NetCDF file — outputs a corrected copy with
  `_corrected` suffix and a `tp_corrected` variable, or `dart::ISO3/era5/year` 
  to select a specific year's data.

### forecast

- **`reference_dataset`**: ~20 years of ERA5 data (bias correction reference).
- **`uncorrected_dataset`**: ~20 years of historical forecast data (can be
  generated with a script from the DART-Pipeline repo).
- **`dataset_to_correct`**: A NetCDF file — corrected similarly as above, or
  `dart::ecmwf_forecast/ISO3/date` that processes forecast data for a specific
  date.

## Generating concatenated historical datasets

Bias correction relies on long-term (~20 years) reference and uncorrected
datasets. Use the dart-pipeline tool to generate these:

```shell
dart-pipeline process era5.prep_bias_correct <ISO3> <start_year>-<end_year> profile=<precipitation|forecast>
```

- Selecting the *precipitation* profile extracts only precipitation.
- The *forecast* profile includes 2m temperature, precipitation, and calculated
  relative humidity.

The files are output in the DART root folder as follows:

```shell
output/<ISO3>/era5/<ISO3>-<start_year>-<end_year>-era5.prep_bias_correct.{precipitation|forecast}.nc
```

## Metadata

Corrected files will include the following metadata fields:

- `DART_history`: Exact CLI command used to generate the file.
- `checksum_reference`: SHA256 checksum of the reference dataset.
- `checksum_uncorrected`: SHA256 checksum of the uncorrected dataset.
- `checksum_file`: SHA256 checksum of the corrected output file.

## Integration into DART-Pipeline

The DART processing pipeline automatically detects netCDF files with the
`_corrected` prefix and the appropriate metadata to output corrected
versions of appropriate metrics.
