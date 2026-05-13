# physoce-datasets

A Python package and CLI for downloading various physical oceanographic datasets.

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
from physoce_datasets import EKEDownloader, SSTDownloader, WindStressDownloader
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

Available commands:

- `eke`: Download altimetry-derived geostrophic velocities and compute eddy kinetic energy from Copernicus Marine Services.
- `wind-stress`: Download ERA5 wind velocities and compute wind stress using the [COARE 3.5 algorithm](https://github.com/pyCOARE/coare). Please see the [notes on ERA5 downloads](#notes-for-era5-downloads) if using this command.
- `sst`: Download NASA Multi-scale Ultra-high Resolution (MUR) sea surface temperature.

Show command help:

```bash
uv run physoce-datasets eke --help
```

### Options

All commands share the same base options:

- `--save-dir`: Directory where the dataset file is written. If not set, defaults to `.data/`, relative to the current working directory.
- `--save-file`: File name to save the dataset. If not set, a default file name based on the data to be downloaded will be used.
- `--start-date`: Start date in `YYYY-MM-DD` format. If not set, uses `2000-01-01`, or the earliest available date, whichever is later.
- `--end-date`: End date in `YYYY-MM-DD` format. If not set, uses the current date or the latest available date, whicher is earlier.
- `--area`: Area string with the format `lon_min,lon_max,lat_min,lat_max` (note commas and no spaces). If not set, defaults to global extent.

### Examples

Run `eke` download with defaults:

```bash
uv run physoce-datasets eke
```

Run `wind-stress` with specified options:

```bash
uv run physoce-datasets wind-stress --save-dir data --save-file wind-stress.nc --start-date 2020-01-01 --end-date 2020-01-31 --area -140,-120,30,50
```

## Python Interface

Downloaders can also be used within Python scripts and Python notebooks as well.

Importing and initializing the classes takes the same arguments as the command line interface:

```python
from physoce_datasets import EKEDownloader

# initialize the downloader
eke_downloader = EKEDownloader(
    start_date="2020-01-01",
    end_date="2020-01-31",
    area="-140,-120,30,50",
    save_dir="data",
    save_file="eke.nc",
)

# perform the download
eke_downloader.download()

# open the dataset; **kwargs are passed to underlying xr.open_dataset() call
ds = eke_downloader.open_dataset(**kwargs)
```

`WindStressDownloader` and `SSTDownloader` follow the exact same interface.

If you'd like to turn off logging in scripts, you can do so with the [Python standard library `logging` module](https://docs.python.org/3/library/logging.html):

```python
import logging

# supress info logging from physoce_datasets
logging.getLogger("physoce_datasets").setLevel(logging.WARNING)
```

Note that progress bars for downloads will still appear when they are available.

## Notes for ERA5 Downloads

ERA5 data is accesed through the [Copernicus Climate Data Store (CDS)](https://cds.climate.copernicus.eu). Accessing data through this interface comes with several constraints on request size and numbers which have to be managed when downloading data from the CDS.

Notably, each user must submit "jobs" which are then processed one-by-one by the CDS backend. Each job has a maximum size based on the number of variables and areas. We require the original hourly data for calculating the wind stress (due to the non-linearity of the wind stress algorithm), which CDS recommends requesting no more than one month per job.

In addition to per-request limits, I found that CDS will pre-cancel any job if more than ~100 are submitted per account at any given time. Therefore, this package manually delays the submission of jobs past the 100 job cap to prevent arbitrary cancellations. This submission process can take some time for large subsets.

Since the submission and completion of jobs can take quite some time on the CDS backend, I wrote this package such that the user can exit the wind-stress program at any time and resume from where they left off. This is done by creating a "state file" (often `submitted_requests.json`) that is saved in the download directory. DO NOT DELETE THIS FILE!

One should keep these limits in mind as they are requesting wind stress data. I wrote this package to manage as much of this as possible in the code, but if you run into issues it might be due to the CDS backend (i.e., not an issue in this code). If you run into issues, take a look at your [existing CDS requests](https://cds.climate.copernicus.eu/requests?tab=all) and see if there are reasons given for any cancellations.
