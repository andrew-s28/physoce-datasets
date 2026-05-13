"""Functions for downloading and processing satellite altimetry data from Copernicus Marine."""

from __future__ import annotations

import datetime
import logging
import warnings
from typing import TYPE_CHECKING

import click
import copernicusmarine
import xarray as xr

from physoce_datasets.logging import logger
from physoce_datasets.util import get_area_str

from ._base import _Downloader

if TYPE_CHECKING:
    from pathlib import Path

# supress info logging from copernicusmarine, will handle that ourselves
logging.getLogger("copernicusmarine").setLevel(logging.WARNING)


def login_to_copernicus_marine() -> None:
    """Perform a login to Copernicus Marine, prompting the user for credentials if not saved.

    Credentials will be stored in a file found at ~/.copernicusmarine/.copernicusmarine-credentials for future use.
    See https://toolbox-docs.marine.copernicus.eu/en/stable/usage/login-usage.html for more information.

    Overrides the default copernicusmarine login behavior to provide more user-friendly prompts and messages.

    """
    logger.info("Attempting login...")
    while not copernicusmarine.login(check_credentials_valid=True):
        logger.warning(
            "No valid credentials found. Please enter your Copernicus Marine "
            "credentials. These will be stored in a file found at "
            "~/.copernicusmarine/.copernicusmarine-credentials for future use. "
            "See https://toolbox-docs.marine.copernicus.eu/en/stable/usage/login-usage.html"
            "for more information.",
        )
        username = click.prompt("Enter your Copernicus Marine username", type=str)
        password = click.prompt("Enter your Copernicus Marine password", type=str, hide_input=True)
        copernicusmarine.login(username=username, password=password, force_overwrite=True)
    logger.info("Login successful!")


def update_metadata(dataset: xr.Dataset) -> xr.Dataset:
    """Update the metadata of the dataset to include standard names, long names, units, and other relevant attributes.

    Args:
        dataset (xr.Dataset): The dataset to update the metadata for.

    Returns:
        xr.Dataset: The dataset with updated metadata.

    """
    dataset["eke"].attrs["standard_name"] = "surface_geostrophic_eddy_kinetic_energy_assuming_sea_level_for_geoid"
    dataset["eke"].attrs["long_name"] = "eddy kinetic energy"
    dataset["eke"].attrs["units"] = "m^2/s^2"
    dataset["eke"].attrs["comment"] = (
        "Calculated as 1/2 * (ugosa^2 + vgosa^2), where ugosa and vgosa are "
        "the geostrophic surface velocities derived from sea level anomalies "
        "contained within this dataset. See "
        "https://www.aviso.altimetry.fr/fileadmin/documents/data/tools/"
        "monthly_mean_eke_hdbk.pdf for more information."
    )
    dataset.attrs["geospatial_lat_min"] = dataset["latitude"].min().item()
    dataset.attrs["geospatial_lat_max"] = dataset["latitude"].max().item()
    dataset.attrs["geospatial_lon_min"] = dataset["longitude"].min().item()
    dataset.attrs["geospatial_lon_max"] = dataset["longitude"].max().item()
    dataset.attrs["history"] = dataset.attrs.get("history", "") + (
        f"\n{datetime.datetime.now(tz=datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')} "
        "- Updated to include eke and err_eke variables."
    )
    dataset.attrs["time_coverage_start"] = dataset["time"].min().dt.strftime("%Y-%m-%dT%H:%M:%SZ").item()
    dataset.attrs["time_coverage_end"] = dataset["time"].max().dt.strftime("%Y-%m-%dT%H:%M:%SZ").item()
    return dataset


class EKEDownloader(_Downloader):
    """Downloader for geostrophic surface eddy kinetic energy (EKE) data from AVISO data through Copernicus Marine."""

    def __init__(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        area: str | None = None,
        save_dir: str | None = None,
        save_file: str | None = None,
    ) -> None:
        """Initialize the EKE downloader.

        Args:
            start_date (str | None): The start date for the dataset in "YYYY-MM-DD" format. If None, defaults to "2000-01-01".
            end_date (str | None): The end date for the dataset in "YYYY-MM-DD" format. If None, defaults to the current date.
            area (str | None): The area to download data for in "lon_min/lon_max/lat_min/lat_max" format. If None, defaults to global coverage.
            save_dir (str | None): The directory to save the downloaded dataset. If None, defaults to a "data" directory in the current working directory.
            save_file (str | None): The file name to save the downloaded dataset. If None, defaults to a name based on the dataset and date range.

        """
        super().__init__(start_date, end_date, area, save_dir, save_file)

        login_to_copernicus_marine()

        if save_file is None:
            area_str = get_area_str(self.area)
            save_file = f"copernicus_marine_eke_{self.start_date}_{self.end_date}_{area_str}.nc"
        self.save_file_path = self._create_save_file(
            self.save_dir,
            save_file,
        )
        self.existing_datetimes, self.existing_path = self._get_existing_datetimes()

    def _get_existing_datetimes(self) -> tuple[xr.DataArray, Path] | tuple[None, None]:
        """Get the datetime values from an existing file.

        Returns:
            tuple[xr.DataArray, pathlib.Path] | tuple[None, None]: The datetime values and file path from the existing file, or None if the file does not exist.

        """
        if not self.save_file_path.exists():
            # check for any file in the save directory that matches the pattern of the expected file name if it was based on the date range and area
            save_file_glob = self.save_dir.glob("copernicus_marine_eke_*.nc")
            matching_files = list(save_file_glob)
            if not matching_files:
                return None, None
            existing_path = matching_files[0]
        else:
            existing_path = self.save_file_path

        ds_existing = xr.open_dataset(existing_path)
        datetime_existing = ds_existing["time"]

        return datetime_existing, existing_path

    @staticmethod
    def _merge_datasets(existing_ds: xr.Dataset, new_ds: xr.Dataset) -> xr.Dataset:
        """Merge the existing dataset with the new dataset, ensuring no duplicate datetimes.

        Args:
            existing_ds (xr.Dataset): The existing dataset to update.
            new_ds (xr.Dataset): The new dataset to merge with the existing dataset.

        Returns:
            xr.Dataset: The merged dataset containing all unique datetimes from both datasets.

        """
        merged_ds = xr.merge([existing_ds, new_ds], compat="no_conflicts", join="outer")
        return merged_ds

    def download(self) -> None:
        """Download geostrophic velocities and compute eddy kinetic energy.

        Saves the resulting dataset to a netCDF file.
        """
        dataset = copernicusmarine.open_dataset(
            dataset_id="cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
            start_datetime=self.start_date,
            end_datetime=self.end_date,
            minimum_longitude=self.area["lon_min"],
            maximum_longitude=self.area["lon_max"],
            minimum_latitude=self.area["lat_min"],
            maximum_latitude=self.area["lat_max"],
        )

        # drop unnecessary variables to save space
        dataset = dataset.drop_vars(
            ["flag_ice", "adt", "sla", "err_sla", "ugos", "vgos", "tpa_correction", "err_ugosa", "err_vgosa"],
            errors="ignore",
        )

        # Calculate EKE based on guidance in:
        # https://www.aviso.altimetry.fr/fileadmin/documents/data/tools/monthly_mean_eke_hdbk.pdf
        dataset["eke"] = 1 / 2 * (dataset["ugosa"] ** 2 + dataset["vgosa"] ** 2)

        dataset = update_metadata(dataset)

        if self.existing_datetimes is not None:
            # drop any datetimes from the new dataset that are already in the existing file to avoid downloading duplicates
            dataset = dataset.where(~dataset["time"].isin(self.existing_datetimes), drop=True)

        logger.info(f"Downloading dataset with size {dataset.nbytes / 1e9:.2f} GB.")
        # strange warning being thrown by xarray when saving to netcdf4
        # seems to be related to endianness of the data and the netcdf4 engine
        # Suppress this warning since it doesn't seem to be causing any issues
        # and is likely out of our control since it's coming from the netcdf4 engine
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="endian-ness of dtype and endian kwarg do not match, using endian kwar",
            )
            if self.existing_path is not None:
                existing_ds = xr.open_dataset(self.existing_path)
                merged_ds = self._merge_datasets(existing_ds, dataset)
                merged_ds.to_netcdf(
                    self.save_file_path,
                    mode="w",
                    format="NETCDF4",
                    engine="netcdf4",
                )

                logger.info(f"Existing dataset {self.existing_path} merged with new data.")
                existing_ds.close()
                merged_ds.close()

                # delete the old file
                if self.existing_path != self.save_file_path:
                    self.existing_path.unlink()
            else:
                dataset.to_netcdf(
                    self.save_file_path,
                    mode="w",
                    format="NETCDF4",
                    engine="netcdf4",
                )
            logger.info(f"Download complete. Dataset saved to {self.save_file_path}.")
