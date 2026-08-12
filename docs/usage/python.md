---
title: Python Interface
---

Downloaders can also be used within Python scripts and Python notebooks as well. The Python interface follows one of two conventions, depending if the dataset is a [remote sensed or model](#remote-sensed-and-model-datasets) (e.g., ERA5) or [mooring based](#moored-datasets) (e.g., OOI Endurance Array).

## Remote Sensed and Model Datasets

!!! info
    The Python interface uses `longitude` and `latitude` separately, rather than `--location lon,lat` as in the [command line interface](/usage/cli).

```python
from physoce_datasets.copernicus import EddyKineticEnergy

# initialize the downloader
eke_downloader = EddyKineticEnergy(
    longitude=-130,
    latitude=45,
    # optional arguments
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
    `era5` downloaders follow the exact same interface as above.

## Moored Datasets

Moored downloaders follow a slightly different notation, using the `site` and `dataset` arguments rather than `latitude` and `longitude`:

```python
from physoce_datasets.ooi import EnduranceArray

# initialize the downloader
ea_downloader = EnduranceArray(
    site="CE02SHSP",  # case insensitive
    dataset="ctd",  # case insensitive
    start_date="2020-01-01",
    end_date="2020-12-31",
    save_dir="data",
    save_file="ctd.nc",
)

ea_downloader.download()
```

Available sites can be found at the respective API Reference pages.

Enum classes are also provided for convenience. They can be used as follows:

```python
# initialize the downloader
ea_downloader = EnduranceArray(
    site=EnduranceArray.ProfilerSites.CE02SHSP,  # using enum
    dataset="ctd",
    start_date="2020-01-01",
    end_date="2020-12-31",
    save_dir="data",
    save_file="ctd.nc",
)

ea_downloader.download()
```

You can view available OOI Endurance Array sites and datasets with two functions:

```python
print("\n".join(EnduranceArray.list_sites()))

print("\n".join(EnduranceArray.list_instruments(site=EnduranceArray.MooringSites.CE01ISSM)))
```

## Logging

If you'd like to turn off logging in scripts, you can do so with the [Python standard library `logging` module](https://docs.python.org/3/library/logging.html):

```python
import logging

# supress info logging from physoce_datasets
logging.getLogger("physoce_datasets").setLevel(logging.WARNING)
```
