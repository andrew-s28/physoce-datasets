# physoce-datasets

A Python package and CLI for downloading various physical oceanographic datasets.

## Install

Clone the package and navigate to the directory:

```bash
git clone (enter url here)
cd (enter package name here)
```

### uv

`uv` is the preferred way to manage this package. Please refer to the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/). Once uv is installed, you can initialize a virtual environment with `uv sync` or you can run any commands directly with `uv run` and `uv` will handle the `venv` creation and activation auto-magically.

### pip

Of course, you can also use the classic `pip`, but you have to handle creating and activating the `venv` yourself:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Credentials

Upon usage, this package may prompt for credentials to various data stores. Please refer to their documentation for how to access credentials and how credentials are stored:

- [Copernicus Marine Services](https://toolbox-docs.marine.copernicus.eu/en/stable/usage/login-usage.html)
- [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/how-to-api)

## CLI usage

Run the CLI with uv:

```bash
uv run datasets.py --help
```

Note that if you prefer the `pip` environment management, activate your environment according to the above and replace all `uv run` commands with `python`, e.g.:

```bash
python datasets.py --help
```

Available commands:

- `eke`: Download altimetry-derived geostrophic velocities and compute eddy kinetic energy from Copernicus Marine Services.
- `era5`: Download 10 m winds, surface temperature, and pressure data from the Copernicus Climate Data Store.

Show command help:

```bash
uv run datasets.py eke --help
```

### Options

All commands share the same options:

- `--save-dir`: Directory where the dataset file is written. If not set, defaults to the package `data/` directory.
- `--start-date`: Start date in `YYYY-MM-DD` format. If not set, uses the earliest available date.
- `--end-date`: End date in `YYYY-MM-DD` format. If not set, uses the latest available date

Run `eke` download with defaults:

```bash
uv run datasets.py eke
```

Run `era5` download with explicit options:

```bash
uv run datasets.py eke --save-dir data --start-date 2020-01-01 --end-date 2020-01-31
```
