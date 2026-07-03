---
title: Python Interface
---

Downloaders can also be used within Python scripts and Python notebooks as well. The Python interface follows one of two conventions, depending if the dataset is a [remote sensed or model](#remote-sensed-and-model-datasets) (e.g., ERA5) or [mooring/profiler based](#mooring-and-profiler-datasets) (e.g., OOI Endurance Array).

## Remote Sensed and Model Datasets

!!! info
    The Python interface uses `longitude` and `latitude` separately, rather than `--location lon,lat` as in the [command line interface](/usage/cli).

```python
from physoce_datasets import copernicus_marine

# initialize the downloader
eke_downloader = copernicus_marine.EddyKineticEnergy(
    longitude=-130,
    latitude=45,
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

!!! tip
    `era5` and `nasa` downloaders follow the exact same interface as above.

## Mooring and Profiler Datasets

Mooring and profiler downloaders follow a slightly different notation, using the `site` argument rather than `latitude` and `longitude`:

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

Available sites can be found at the respective API Reference pages.

Enum classes are also provided for any site-based downloaders for convenience. They can be used as follows:

```python
from physoce_datasets import ooi_ea

# initialize the downloader
eke_downloader = ooi_ea.ProfilerCTD(
    site=ooi_ea.ProfilerSites.CE02SHSP,  # using enum
    start_date="2020-01-01",
    end_date="2020-12-31",
    save_dir="data",
    save_file="eke.nc",
)
```

## Logging

If you'd like to turn off logging in scripts, you can do so with the [Python standard library `logging` module](https://docs.python.org/3/library/logging.html):

```python
import logging

# supress info logging from physoce_datasets
logging.getLogger("physoce_datasets").setLevel(logging.WARNING)
```
