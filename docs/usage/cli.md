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

Show provider command help, which will include available datasets for downloading:

```bash
uv run physoce-datasets ooi-ea --help
```

### Options

All command line interfaces share the same base options:

- `--location`: Location either in the format of 'lon,lat' (e.g., '-132.0,36.55') for global datasets or 'site' (e.g., CE01ISSP) for moored datasets. **Required for all commands!**
- `--instrument`: Download data from this instrument. **Required for all commands!**
- `--save-dir`: Directory where the dataset file is written. If not set, defaults to `.data/`, relative to the current working directory.
- `--save-file`: File name to save the dataset. If not set, a default file name based on the data to be downloaded will be used.
- `--start-date`: Start date in `YYYY-MM-DD` format. If not set, uses `2000-01-01`, or the earliest available date, whichever is later.
- `--end-date`: End date in `YYYY-MM-DD` format. If not set, uses the current date or the latest available date, whicher is earlier.

### Available Datasets

Available datasets are currently as follows:

[`copernicus-marine`](/api/copernicus-marine)
: - `eke`: Eddy kinetic energy derived from altimetric sea surface height anomalies.

[`era5`](/api/era5)
: - `wind-stress`: Wind stress derived from ERA5 10 m winds using the [COARE 3.5 algorithm](https://github.com/pyCOARE/coare).

[`ooi-ea`](/api/ooi-ea)
: - `ctd`: Temperature, salinity, pressure, and density.
  - `fluorometer`: Chlorophyll *a* and colored dissolved organic matter (CDOM).
  - `spectrophotometer`: Attenuation and absorption.
  - `oxygen`: Dissolved oxygen.
  - `nitrate`: Nitrate.
  - `irradiance`: Spectral downwelling irradiance.
  - `par`: Photosynthetically active radiation.

Datasets are specified with the required `--dataset` option:

```bash
uv run physoce-datasets ooi-ea --location CE01ISSP --dataset ctd
```

## Putting It All Together

Run `copernicus` download with defaults (location and dataset are always required!):

```bash
uv run physoce-datasets copernicus --location -130,45 --dataset eke
```

Run `era5` with specified options:

```bash
uv run physoce-datasets era5 --location -130,45 --dataset wind-stress --save-dir data --save-file wind-stress.nc --start-date 2020-01-01 --end-date 2020-01-31
```

Run `ooi-ea` with specified options:

```bash
uv run physoce-datasets ooi-ea --location CE02SHSM --dataset ctd
```
