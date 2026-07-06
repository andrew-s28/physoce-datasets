# physoce-datasets

A Python package and command line interface aimed at standardizing the access of various oceanographic datasets and calculating derived parameters according to modern best practices.

> [!WARNING]
> Due to issues with the [NASA Harmony API](https://forum.earthdata.nasa.gov/viewtopic.php?t=7954&sid=bdeec61e589c16e9d8642040d2fb01ff), the SST dataset is currently unavailable with an unknown fix timeline. Running any of the `sst` commands will fail with a `NotImplementedError` until this is fixed.

## Why physoce-datasets?

This package provides an *opinionated* interface which aims to simplify and align access to datasets across providing institutions (NASA, ECMWF, etc.); as such, it is aimed primarily at those looking for streamlined data access. If you are an expert user who wants a lot of control over the details of the download and analysis process, this may not be for you. However, if you want:

- a unified command line and Python interface across datasets
- expert-informed derived variables such as wind stress and eddy kinetic energy
- long time series at a single or a few locations
- the simplest downloading and opening process possible

then this package is designed for you!

## Install

### uv

`uv` is the preferred way to manage this package. Please refer to the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/). Once uv is installed, you can initialize a virtual environment with `uv sync` or you can run any commands directly with `uv run` and `uv` will handle the `venv` creation and activation auto-magically.

```bash
# add physoce-datasets to your project folder, equivalent to a pip install
uv add physoce-datasets
# you can now run physoce-datasets from your project folder
uv run physoce-datasets --help
```

Once you've added `physoce-datasets` to your project, you can import any of the download classes in your scripts:

```python
# note the import uses underscore in place of dash
from physoce_datasets import copernicus_marine, era5, ooi_ea
```

Alternatively, you can run the command line interface from anywhere using [uv tools](https://docs.astral.sh/uv/guides/tools/):

```bash
# run physoce-datasets from the command line anywhere!
uvx physoce-datasets --help
```

### pip

Of course, you can also use the classic `pip`, but you have to handle creating and activating the `venv` yourself:

```bash
python -m venv .venv
source .venv/bin/activate
pip install physoce-datasets
```

## Credentials

Upon usage, this package may prompt for credentials to various data stores. Please refer to their documentation for how to access credentials and how credentials are stored:

- [Copernicus Marine Services](https://toolbox-docs.marine.copernicus.eu/en/stable/usage/login-usage.html)
- [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/how-to-api)
- [NASA Earthdata](https://urs.earthdata.nasa.gov/users/new)

## Command Line Interface

Run the CLI with uv:

```bash
uv run physoce-datasets --help
```

Note that if you prefer the `pip` environment management, activate your environment according to the above and replace all `uv run` commands with `python`, e.g.:

```bash
python physoce-datasets --help
```

Top level commands are based on providing agency or program:

- `copernicus-marine`: Data from [Copernicus Marine Services](https://marine.copernicus.eu/?pk_vid=f1c2c33510b8f44b178301966049ffbe).
- `era5`: Data from [ERA5 single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview).
- `nasa`: Data from NASA Harmony API. **Currently not implemented due to issues with the API, see warning above**.
- `ooi-ea`: Data from [NSF Ocean Observatories Initiative Endurance Array](https://oceanobservatories.org/array/coastal-endurance/).

Show provider command help, which will include available datasets for downloading:

```bash
uv run physoce-datasets ooi-ea --help
```

The specific dataset to download is specified after the provider:

```bash
uv run physoce-datasets ooi-ea profiler-chl --help
```

Available datasets are currently as follows:

- `copernicus-marine`
  - `eke`: Eddy kinetic energy derived from altimetric sea surface height anomalies.
- `era5`
  - `wind-stress`: Wind stress derived from ERA5 10 m winds using the [COARE 3.5 algorithm](https://github.com/pyCOARE/coare).
- `nasa`
  - `sst`: Multi-scale Ultra-high Resolution (MUR) sea surface temperature. **Currently not implemented due to issues with the API, see warning above**.
- `ooi-ea`
  - `mooring-ctd`: Temperature, salinity, pressure, and density derived from mooring-mounted CTDs.
  - `profiler-ctd`: Mixed layer depth and stratification derived from temperature, salinity, pressure, and density from profiler-mounted CTDs.
  - `profiler-chl`: Chlorophyll *a* derived from profiler-mounted fluorometers.

### Options

All command line interfaces share the same base options:

- `--location`: Location for the dataset in the format 'lon,lat' (e.g., '-132.0,36.55'). **Required for all commands!**
- `--save-dir`: Directory where the dataset file is written. If not set, defaults to `.data/`, relative to the current working directory.
- `--save-file`: File name to save the dataset. If not set, a default file name based on the data to be downloaded will be used.
- `--start-date`: Start date in `YYYY-MM-DD` format. If not set, uses `2000-01-01`, or the earliest available date, whichever is later.
- `--end-date`: End date in `YYYY-MM-DD` format. If not set, uses the current date or the latest available date, whicher is earlier.

### Examples

Run `eke` download with defaults:

```bash
uv run physoce-datasets copernicus-marine eke
```

Run `wind-stress` with specified options:

```bash
uv run physoce-datasets era5 wind-stress --save-dir data --save-file wind-stress.nc --start-date 2020-01-01 --end-date 2020-01-31 --location -130,45
```

## Python Interface

Downloaders can also be used within Python scripts and Python notebooks as well.

Importing and initializing the classes takes similar arguments as the command line interface:

```python
from physoce_datasets import copernicus_marine

# initialize the downloader
eke_downloader = copernicus_marine.EddyKineticEnergy(
    latitude=45,
    longitude=-130,
    start_date="2020-01-01",
    end_date="2020-12-31",
    save_dir="data",
    save_file="eke.nc",
)

# perform the download
eke_downloader.download()

# open the dataset; **kwargs are passed to underlying xr.open_dataset() call
ds = eke_downloader.open_dataset(**kwargs)
```

`era5` and `nasa` downloaders follow the exact same interface.

The OOI Endurance Array downloaders follow a slightly different notation, using the `site` argument rather than `latitude` and `longitude`:

```python
from physoce_datasets import ooi_ea

# initialize the downloader
eke_downloader = ooi_ea.ProfilerCTD(
    site="CE02SHSP",  # case insensitive
    start_date="2020-01-01",
    end_date="2020-12-31",
    save_dir="data",
    save_file="eke.nc",
)
```

Available profiler sites are `CE01ISSP`, `CE02SHSP`, `CE04OSPS`, `CE04OSPD`, `CE06ISSP`, `CE07SHSP`, `CE09OSPM`, and `RS01SBPS`.

Available mooring sites are `CE01ISSM`, `CE02SHSM`, `CE04OSSM`, `CE06ISSM`, `CE07SHSM`, and `CE09OSSM`.

More information on these sites can be found at the [OOI Endurance Array](https://oceanobservatories.org/array/coastal-endurance/), the [OOI Cabled Endurance Array](https://oceanobservatories.org/array/cabled-and-endurance-arrays/), and the [OOI Cabled Continental Margin Array](https://oceanobservatories.org/array/cabled-continental-margin-array/) sites.

If you'd like to turn off logging in scripts, you can do so with the [Python standard library `logging` module](https://docs.python.org/3/library/logging.html):

```python
import logging

# supress info logging from physoce_datasets
logging.getLogger("physoce_datasets").setLevel(logging.WARNING)
```

## Contributions

Do you have a publicly available dataset that you've got a great download script for? Do you have specialist knowledge required for calculating derived data products? Please consider [opening an issue](https://github.com/andrew-s28/physoce-datasets/issues) or [submitting a pull request](https://github.com/andrew-s28/physoce-datasets/pulls)! We welcome any and all contributions to this project.

If you'd like to contribute to the code or documentation, please refer to the developer instructions below:

1. [Create a fork of the repository](https://github.com/andrew-s28/physoce-datasets/fork).
2. Clone your fork to your local machine:

    ```bash
    git clone https://github.com/your-username-here/physoce-datasets.git
    cd physoce-datasets
    ```

3. Setup the development environment by installing development and documentation dependencies and installing pre-commit hooks:

    ```bash
    uv sync --dev
    pre-commit install
    ```

    If you're only updating code, you don't need the docs group. If you're only updating docs, you *do* need the dev group.

4. Create a new branch with a helpful name:

    ```bash
    git checkout -b your-great-new-feature
    ```

5. Make your code changes, stage them via `git add`, and commit them with `git commit -m 'here's my great code changes'`.
6. Push your changes to your fork using:

    ```bash
    git push -u origin your-great-new-feature
    ```

7. Open a [pull request in the upstream repository](https://github.com/andrew-s28/physoce-datasets/compare).

Thanks so much for contributing to open source code!

### AI Contribution Policy

We share the same [AI Usage Policy as xarray](https://docs.xarray.dev/en/stable/contribute/ai-policy.html). In short, this allows developers to use AI tools as a part of their development workflow, but requires that contributors understand all submitted code and take full responosiblity for their changes.
