"""Functions for downloading and processing ERA5 reanalysis data from the ECMWF Data Store."""

from __future__ import annotations

import contextlib
import os
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING, cast
from zipfile import ZipFile

import click
import numpy as np
import xarray as xr
from ecmwf.datastores import Client
from metpy.calc import relative_humidity_from_dewpoint
from metpy.units import units
from pycoare import coare_35

from physoce_datasets.logging import logger

from ._base import _Downloader

if TYPE_CHECKING:
    from physoce_datasets.util import LonLat

CONFIG_FILE = Path.home() / ".ecmwfdatastoresrc"

DATASET = "reanalysis-era5-single-levels-timeseries"
VARIABLES = [
    "2m_dewpoint_temperature",
    "surface_pressure",
    "sea_surface_temperature",
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]


def login_to_ecmwf_datastore() -> Client:
    """Perform a login to the ECMWF Data Store, prompting the user for credentials if not saved.

    Credentials will be stored in a file found at ~/.ecmwfdatastoresrc for future use.
    To obtain your key, go to https://cds.climate.copernicus.eu/how-to-api

    Overrides the default ECMWF Data Store login behavior to provide more user-friendly prompts and messages.

    Args:
        client (ecmwf.datastores.Client): An instance of the ECMWF Data Store client.

    Returns:
        ecmwf.datastores.Client: An authenticated instance of the ECMWF Data Store client.

    """
    logger.info("Attempting login...")
    try:
        # redirect output to avoid printing default login messages
        with contextlib.redirect_stdout(Path(os.devnull).open("w", encoding="utf-8")) and contextlib.redirect_stderr(
            Path(os.devnull).open("w", encoding="utf-8"),
        ):
            client = Client(progress=False, retry_after=1, maximum_tries=10)
            client.check_authentication()
            logger.info("Login successful!")
    except Exception as e:  # noqa: BLE001
        logger.info(f"Failed to authenticate with ECMWF Data Store: {e}")
        logger.info(
            "No valid credentials found. Please enter your Climate Data Store "
            "API key. These will be stored in a file found at "
            "~/.ecmwfdatastoresrc for future use. To obtain your key, go to "
            "https://cds.climate.copernicus.eu/how-to-api.",
        )
        key = click.prompt("Enter your key", type=str, hide_input=True)
        with CONFIG_FILE.open("w") as f:
            f.write(f"url: https://cds.climate.copernicus.eu/api\nkey: {key}\n")
        client = login_to_ecmwf_datastore()
    return client


class WindStressDownloader(_Downloader):
    """Downloader for ERA5-based wind stress datasets from the ECMWF Data Store."""

    def __init__(
        self,
        location: str,
        start_date: str | None = None,
        end_date: str | None = None,
        save_dir: str | None = None,
        save_file: str | None = None,
    ) -> None:
        """Initialize the downloader and set up the ECMWF Data Store client and request state manager."""
        super().__init__(
            location=location,
            location_type="lonlat",
            save_file_prefix="era5_reanalysis_wind_stress",
            save_dir=save_dir,
            save_file=save_file,
            start_date=start_date,
            end_date=end_date,
        )
        self.location = cast("LonLat", self.location)

        self.client = login_to_ecmwf_datastore()

        self.request = self.setup_request()

    def setup_request(self) -> dict:
        """Set up the request parameters for the ECMWF Data Store API.

        Returns:
            dict: A dictionary containing the request parameters for the ECMWF Data Store API.

        """
        request = {
            "variable": VARIABLES,
            "location": {"longitude": self.location.lon, "latitude": self.location.lat},
            "date": f"{self.start_date}/{self.end_date}",
            "data_format": "netcdf",
        }
        return request

    def download(self) -> None:
        """Download the ERA5 data, compute wind stress, and save to a NetCDF file."""
        logger.info("Submitting request to ECMWF Data Store...")
        remote = self.client.submit(
            collection_id=DATASET,
            request=self.request,
        )
        logger.info("Waiting for results...")
        while not remote.results_ready:
            sleep(1)

        logger.info("Results ready! Downloading...")
        zip_file_path = self.save_file_path.with_suffix(".zip")
        remote.download(str(zip_file_path))
        logger.info("Download complete! Processing data...")
        with ZipFile(zip_file_path, "r") as z:
            file = z.namelist()[0]  # expect only one file in the zip
            with z.open(file) as f:
                ds = xr.open_dataset(f, engine="h5netcdf")
                ds = self._process_data(ds)
                ds.to_netcdf(self.save_file_path)
        zip_file_path.unlink()  # remove the zip file after processing
        logger.info(f"Download complete! Dataset saved to {self.save_file_path}")

    def _process_data(self, ds: xr.Dataset) -> xr.Dataset:
        """Compute wind stress from the downloaded ERA5 data and return a new dataset containing the wind stress variables.

        Args:
            ds (xr.Dataset): The input dataset containing the raw ERA5 variables.

        Returns:
            xr.Dataset: A new dataset containing the computed wind stress variables.

        """
        ds = self._rename_vars(ds)
        # convert to degC
        ds["air_temperature"] -= 273.15
        ds["dew_point_temperature"] -= 273.15
        ds["sea_surface_temperature"] -= 273.15

        # compute relative humidity and wind stress
        ds["relative_humidity"] = xr.apply_ufunc(
            self._compute_relative_humidity,
            ds["air_temperature"],
            ds["dew_point_temperature"],
            input_core_dims=[[], []],
            output_core_dims=[[]],
        )
        ds["eastward_wind_stress"], ds["northward_wind_stress"] = xr.apply_ufunc(
            partial(self._compute_wind_stress, latitude=float(ds["latitude"].values)),
            ds["eastward_wind"],
            ds["northward_wind"],
            ds["air_temperature"],
            ds["relative_humidity"],
            ds["sea_surface_temperature"],
            ds["surface_air_pressure"],
            input_core_dims=[[], [], [], [], [], []],
            output_core_dims=[[], []],
        )

        # take the daily average
        ds = ds.resample(time="1D").mean()

        # update metadata
        ds = self._update_metadata(ds)

        return ds

    @staticmethod
    def _rename_vars(ds: xr.Dataset) -> xr.Dataset:
        """Rename variables in the dataset to more user-friendly names.

        Args:
            ds (xr.Dataset): The input dataset with original variable names.

        Returns:
            xr.Dataset: A new dataset with renamed variables.

        """
        var_renames = {
            "u10": "eastward_wind",
            "v10": "northward_wind",
            "t2m": "air_temperature",
            "d2m": "dew_point_temperature",
            "sst": "sea_surface_temperature",
            "sp": "surface_air_pressure",
            "valid_time": "time",
        }
        ds = ds.rename(var_renames)
        return ds

    @staticmethod
    def _update_metadata(ds: xr.Dataset) -> xr.Dataset:
        """Update the metadata of the dataset to be more informative and user-friendly.

        Args:
            ds (xr.Dataset): The input dataset with original metadata.

        Returns:
            xr.Dataset: A new dataset with updated metadata.

        """
        # remove unnecessary grib metadata
        ds.attrs = {k: v for k, v in ds.attrs.items() if "grib" not in k.lower()}
        for var in ds.data_vars:
            ds[var].attrs = {k: v for k, v in ds[var].attrs.items() if "grib" not in k.lower()}

        # update variable metadata
        ds["eastward_wind"].attrs.update(
            {
                "long_name": "10m Eastward Wind Component",
                "standard_name": "eastward_wind",
                "units": "m/s",
            },
        )
        ds["northward_wind"].attrs.update(
            {
                "long_name": "10m Northward Wind Component",
                "standard_name": "northward_wind",
                "units": "m/s",
            },
        )
        ds["air_temperature"].attrs.update(
            {
                "long_name": "2m Air Temperature",
                "standard_name": "air_temperature",
                "units": "degC",
            },
        )
        ds["dew_point_temperature"].attrs.update(
            {
                "long_name": "2m Dew Point Temperature",
                "standard_name": "dew_point_temperature",
                "units": "degC",
            },
        )
        ds["sea_surface_temperature"].attrs.update(
            {
                "long_name": "Sea Surface Temperature",
                "standard_name": "sea_surface_temperature",
                "units": "degC",
            },
        )
        ds["surface_air_pressure"].attrs.update(
            {
                "long_name": "Surface Air Pressure",
                "standard_name": "surface_air_pressure",
                "units": "Pa",
            },
        )
        ds["eastward_wind_stress"].attrs.update(
            {
                "long_name": "Eastward Wind Stress",
                "standard_name": "eastward_wind_stress",
                "units": "N/m^2",
            },
        )
        ds["northward_wind_stress"].attrs.update(
            {
                "long_name": "Northward Wind Stress",
                "standard_name": "northward_wind_stress",
                "units": "N/m^2",
            },
        )

        # update dataset metadata
        existing_history = ds.attrs.pop("history", "")
        ds.attrs.update(
            {
                "description": "Wind stress components computed from hourly ERA5 reanalysis data using the COARE 3.5 bulk flux algorithm and daily averaged. ",
                "last updated": datetime.now(UTC).isoformat(timespec="minutes"),
                "history": existing_history
                + "\n"
                + f"{datetime.now(UTC).isoformat(timespec='minutes')} Downloaded and processed data using physoce-datasets (https://github.com/physoce/physoce-datasets)",
            },
        )
        return ds

    @staticmethod
    def _compute_relative_humidity(t2m: np.ndarray, d2m: np.ndarray) -> np.ndarray:
        """Compute relative humidity from 2m temperature and 2m dewpoint temperature.

        Args:
            t2m (np.ndarray): 2m temperature in Kelvin.
            d2m (np.ndarray): 2m dewpoint temperature in Kelvin.

        Returns:
            np.ndarray: Relative humidity in percentage.

        """
        # drop metpy pint units and convert to percent
        rh = relative_humidity_from_dewpoint(t2m * units.degC, d2m * units.degC).m * 100
        return rh

    @staticmethod
    def _compute_wind_stress(
        u10: np.ndarray,
        v10: np.ndarray,
        t2m: np.ndarray,
        rh: np.ndarray,
        sst: np.ndarray,
        sp: np.ndarray,
        latitude: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute wind stress using standard ERA5 outputs and the COARE 3.5 bulk flux algorithm.

        Args:
            u10 (np.ndarray): 10m eastward wind component in m/s.
            v10 (np.ndarray): 10m northward wind component in m/s.
            t2m (np.ndarray): 2m temperature in degC.
            rh (np.ndarray): Relative humidity in percent.
            sst (np.ndarray): Sea surface temperature in degC.
            sp (np.ndarray): Surface pressure in Pa.
            latitude (float): Latitude of the location in degrees.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the eastward and northward wind stress components in N/m^2.

        """
        shape = u10.shape
        mag = np.sqrt(u10**2 + v10**2)
        angle = np.arctan2(v10, u10).flatten()
        # have to flatten inputs
        c35 = coare_35(
            u=mag.flatten(),
            t=t2m.flatten(),
            rh=rh.flatten(),
            ts=sst.flatten(),
            p=sp.flatten()
            / 100,  # convert to millibars for pycoare input, see also https://github.com/pyCOARE/coare/issues/57
            lat=latitude,
            zu=10,
            zt=2,
            zq=2,
            zrf=10,
        )
        tau_mag = c35.fluxes.tau
        tau_east = tau_mag * np.cos(angle)
        tau_north = tau_mag * np.sin(angle)
        return tau_east.reshape(shape), tau_north.reshape(shape)
