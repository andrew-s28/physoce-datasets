"""Functions for downloading and processing satellite altimetry data from Copernicus Marine."""

from __future__ import annotations

import logging
import warnings
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

import click
import copernicusmarine
import xarray as xr

from physoce_datasets.logging import logger

from ._base import _Downloader

if TYPE_CHECKING:
    from pathlib import Path

    from physoce_datasets.util import LonLat

# supress info logging from copernicusmarine, will handle that ourselves
logging.getLogger("copernicusmarine").setLevel(logging.ERROR)


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


class EKEDownloader(_Downloader):
    """Downloader for geostrophic surface eddy kinetic energy (EKE) data from AVISO data through Copernicus Marine."""

    def __init__(
        self,
        location: str,
        start_date: str | None = None,
        end_date: str | None = None,
        save_dir: str | None = None,
        save_file: str | None = None,
    ) -> None:
        """Initialize the EKE downloader.

        Args:
            location (str): Location for the dataset in the format 'lon,lat' (e.g., '132.0,36.55'). Required.
            start_date (str | None): The start date for the dataset in "YYYY-MM-DD" format. If None, defaults to "2000-01-01".
            end_date (str | None): The end date for the dataset in "YYYY-MM-DD" format. If None, defaults to the current date.
            save_dir (str | None): The directory to save the downloaded dataset. If None, defaults to a "data" directory in the current working directory.
            save_file (str | None): The file name to save the downloaded dataset. If None, defaults to a name based on the dataset and date range.

        """
        super().__init__(
            location=location,
            location_type="lonlat",
            save_file_prefix="aviso_eke",
            save_dir=save_dir,
            save_file=save_file,
            start_date=start_date,
            end_date=end_date,
        )
        self.location = cast("LonLat", self.location)

        login_to_copernicus_marine()

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
        ds = copernicusmarine.open_dataset(
            dataset_id="cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
            start_datetime=self.start_date,
            end_datetime=self.end_date,
            minimum_longitude=self.location.lon,
            maximum_longitude=self.location.lon,
            minimum_latitude=self.location.lat,
            maximum_latitude=self.location.lat,
            coordinates_selection_method="nearest",
        )

        # drop unnecessary variables to save space
        ds = ds.drop_vars(
            ["flag_ice", "adt", "err_sla", "ugos", "vgos", "tpa_correction", "err_ugosa", "err_vgosa"],
            errors="ignore",
        )

        # Calculate EKE based on guidance in:
        # https://www.aviso.altimetry.fr/fileadmin/documents/data/tools/monthly_mean_eke_hdbk.pdf
        ds["eke"] = 1 / 2 * (ds["ugosa"] ** 2 + ds["vgosa"] ** 2)

        ds = self.update_metadata(ds)

        logger.info("Downloading...")
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
            ds.to_netcdf(
                self.save_file_path,
                mode="w",
                format="NETCDF4",
                engine="netcdf4",
            )
        logger.info(f"Download complete! Dataset saved to {self.save_file_path}")
        self.downloaded = True

    @staticmethod
    def update_metadata(ds: xr.Dataset) -> xr.Dataset:
        """Update the metadata of the dataset to include standard names, long names, units, and other relevant attributes.

        Args:
            ds (xr.Dataset): The dataset to update the metadata for.

        Returns:
            xr.Dataset: The dataset with updated metadata.

        """
        ds["eke"].attrs["standard_name"] = "surface_geostrophic_eddy_kinetic_energy_assuming_sea_level_for_geoid"
        ds["eke"].attrs["long_name"] = "eddy kinetic energy"
        ds["eke"].attrs["units"] = "m^2/s^2"
        ds["eke"].attrs["comment"] = (
            "Calculated as 1/2 * (ugosa^2 + vgosa^2), where ugosa and vgosa are "
            "the geostrophic surface velocities derived from sea level anomalies "
            "contained within this dataset. See "
            "https://www.aviso.altimetry.fr/fileadmin/documents/data/tools/"
            "monthly_mean_eke_hdbk.pdf for more information."
        )
        existing_history = ds.attrs.pop("history", "")
        ds.attrs.update(
            {
                "description": "Sea surface height anomalies, geostrophic velocity anomalies, and derived eddy kinetic energy from CNES/CLS DUACS obtained via Copernicus Marine Service. ",
                "last updated": datetime.now(UTC).isoformat(timespec="minutes"),
                "history": existing_history
                + "\n"
                + f"{datetime.now(UTC).isoformat(timespec='minutes')} Downloaded and processed data using physoce-datasets (https://github.com/physoce/physoce-datasets)",
                "doi": "10.48670/moi-00148",  # https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_L4_MY_008_047/description
                "geospatial_lat_min": ds["latitude"].min().item(),
                "geospatial_lat_max": ds["latitude"].max().item(),
                "geospatial_lon_min": ds["longitude"].min().item(),
                "geospatial_lon_max": ds["longitude"].max().item(),
                "time_coverage_start": ds["time"].min().dt.strftime("%Y-%m-%dT%H:%M:%SZ").item(),
                "time_coverage_end": ds["time"].max().dt.strftime("%Y-%m-%dT%H:%M:%SZ").item(),
            },
        )
        return ds
