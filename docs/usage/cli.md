---
title: Command Line Interface
---

## Running the CLI

Run the CLI with uv:

```bash
uv run physoce-datasets --help
```

Note that if you prefer the `pip` environment management, activate your environment according to the above and replace all `uv run` commands with `python`, e.g.:

```bash
python physoce-datasets --help
```

### Provider Commands

Top level commands are based on providing agency or program:

- `copernicus-marine`: Data from [Copernicus Marine Services](https://marine.copernicus.eu/?pk_vid=f1c2c33510b8f44b178301966049ffbe).
- `era5`: Data from [ERA5 single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview).
- `ooi-ea`: Data from [NSF Ocean Observatories Initiative Endurance Array](https://oceanobservatories.org/array/coastal-endurance/).
- `nasa`: Data from NASA Harmony API.
!!! danger
    Due to issues with the [NASA Harmony API](https://forum.earthdata.nasa.gov/viewtopic.php?t=7954&sid=bdeec61e589c16e9d8642040d2fb01ff), NASA datasets are currently unavailable with an unknown fix timeline. Running any of the `nasa` commands will fail with a `NotImplementedError` until this is fixed.

Show provider command help, which will include available datasets for downloading:

```bash
uv run physoce-datasets ooi-ea --help
```

### Dataset Commands

Available datasets are currently as follows:

[`copernicus-marine`](/api/copernicus-marine)
: - `eke`: Eddy kinetic energy derived from altimetric sea surface height anomalies.

[`era5`](/api/era5)
: - `wind-stress`: Wind stress derived from ERA5 10 m winds using the [COARE 3.5 algorithm](https://github.com/pyCOARE/coare).

[`nasa`](/api/nasa)
: - `sst`: Multi-scale Ultra-high Resolution (MUR) sea surface temperature.<br>
**Currently not implemented, see warning above.**

[`ooi-ea`](/api/ooi-ea)
: - `mooring-ctd`: Temperature, salinity, pressure, and density derived from mooring-mounted CTDs.
  - `profiler-ctd`: Mixed layer depth and stratification derived from temperature, salinity, pressure, and density from profiler-mounted CTDs.
  - `profiler-chl`: Chlorophyll *a* derived from profiler-mounted fluorometers.

The specific dataset to download is specified after the provider:

```bash
uv run physoce-datasets ooi-ea profiler-chl --help
```

### Options

All command line interfaces share the same base options:

- `--location`: Location for the dataset in the format 'lon,lat' (e.g., '-132.0,36.55'). **Required for all commands!**
- `--save-dir`: Directory where the dataset file is written. If not set, defaults to `.data/`, relative to the current working directory.
- `--save-file`: File name to save the dataset. If not set, a default file name based on the data to be downloaded will be used.
- `--start-date`: Start date in `YYYY-MM-DD` format. If not set, uses `2000-01-01`, or the earliest available date, whichever is later.
- `--end-date`: End date in `YYYY-MM-DD` format. If not set, uses the current date or the latest available date, whicher is earlier.

### Putting It All Together

Run `eke` download with defaults:

```bash
uv run physoce-datasets copernicus-marine eke
```

Run `wind-stress` with specified options:

```bash
uv run physoce-datasets era5 wind-stress --save-dir data --save-file wind-stress.nc --start-date 2020-01-01 --end-date 2020-01-31 --location -130,45
```
