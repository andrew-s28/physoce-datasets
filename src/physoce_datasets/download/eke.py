from __future__ import annotations

import datetime
import logging
import os
import warnings
from pathlib import Path

import click
import copernicusmarine
import xarray as xr

from physoce_datasets.logging import logger

# supress info logging from copernicusmarine, will handle that ourselves
logging.getLogger("copernicusmarine").setLevel(logging.WARNING)

MIN_LON = -150
MAX_LON = -120
MIN_LAT = 30
MAX_LAT = 60


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
            "See https://toolbox-docs.marine.copernicus.eu/en/stable/usage/"
            "login-usage.html for more information.",
        )
        username = click.prompt("Enter your Copernicus Marine username", type=str)
        password = click.prompt("Enter your Copernicus Marine password", type=str, hide_input=True)
        copernicusmarine.login(username=username, password=password, force_overwrite=True)
    logger.info("Login successful!")


def create_data_dir(save_dir: Path | None) -> Path:
    """Create the directory to save the downloaded dataset if it doesn't already exist.

    Args:
        save_dir (Path | None): The directory to save the downloaded dataset. If None,
            defaults to a "data" directory at the package level.

    Returns:
        Path: The directory to save the downloaded dataset.

    """
    # by default, save in a "data" directory at the package level
    data_dir = Path("data") if save_dir is None else save_dir
    data_dir.mkdir(exist_ok=True)
    return data_dir


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


def get_save_file(save_dir: Path, dataset: xr.Dataset) -> Path:
    """Create the save file name and checks if the file already exists and is writable.

    Args:
        save_dir (Path): The directory to save the downloaded dataset.
        dataset (xr.Dataset): The dataset to be saved, used to extract the time coverage for the file name and metadata.

    Returns:
        Path: The file path to save the dataset to.

    Raises:
        PermissionError: If the file already exists and is not writable.

    """
    start_time = datetime.datetime.strptime(dataset.attrs["time_coverage_start"], "%Y-%m-%dT%H:%M:%SZ").astimezone(
        datetime.UTC,
    )
    end_time = datetime.datetime.strptime(dataset.attrs["time_coverage_end"], "%Y-%m-%dT%H:%M:%SZ").astimezone(
        datetime.UTC,
    )
    save_file = save_dir / f"copernicus_marine_eke_{start_time.strftime('%Y-%m-%d')}_{end_time.strftime('%Y-%m-%d')}.nc"

    if save_file.exists() and not os.access(save_file, os.W_OK):
        msg = (
            f"File {save_file} already exists and is not writable. If this file "
            "is open in another application (e.g., Jupyter notebook), please "
            "close it and try again."
        )
        raise PermissionError(msg)
    return save_file


def _get_existing_datetimes(save_dir: Path, update_path: str) -> xr.DataArray:
    """Get the datetime values from the existing file.

    Args:
        save_dir (Path): The directory where the dataset files are saved.
        update_path (str): The path to the existing file to update.

    Returns:
        xr.DataArray: The datetime values from the existing file.

    Raises:
        FileNotFoundError: If the update file does not exist in the save directory.

    """
    if not Path(save_dir / update_path).exists():
        msg = f"Update file {update_path} not found in save directory {save_dir}."
        raise FileNotFoundError(msg)

    ds_existing = xr.open_dataset(save_dir / update_path)
    datetime_existing = ds_existing["time"]

    return datetime_existing


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


def download_eke(
    save_dir: Path | None = None,
    start_datetime: str | None = None,
    end_datetime: str | None = None,
    update_file: str | None = None,
) -> None:
    """Download geostrophic velocities and compute eddy kinetic energy.

    Saves the resulting dataset to a netCDF file.

    Args:
        save_dir (Path | None): The directory to save the downloaded dataset.
            If None, defaults to a "data" directory at the package level.
        start_datetime (str | None): The start datetime for the dataset in
            YYYY-MM-DD format. If None, defaults to the earliest available
            datetime for the dataset.
        end_datetime (str | None): The end datetime for the dataset in
            YYYY-MM-DD format. If None, defaults to the latest available
            datetime for the dataset.
        update_file (str | None): The path to an existing netCDF file to update. If None, a new file will be created.

    """
    login_to_copernicus_marine()

    dataset = copernicusmarine.open_dataset(
        dataset_id="cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        minimum_longitude=MIN_LON,
        maximum_longitude=MAX_LON,
        minimum_latitude=MIN_LAT,
        maximum_latitude=MAX_LAT,
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

    save_dir = create_data_dir(save_dir) if save_dir is None else save_dir
    save_file = get_save_file(save_dir, dataset)

    existing_datetimes = _get_existing_datetimes(save_dir, update_file) if update_file is not None else None

    if existing_datetimes is not None:
        # drop any datetimes from the new dataset that are already in the existing file to avoid downloading duplicates
        dataset = dataset.where(~dataset["time"].isin(existing_datetimes), drop=True)

    if click.confirm(f"Downloading dataset with size {dataset.nbytes / 1e9:.2f} GB. Proceed?"):
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
            if update_file is not None:
                existing_ds = xr.open_dataset(save_dir / update_file)
                merged_ds = _merge_datasets(existing_ds, dataset)
                merged_ds.to_netcdf(
                    save_file,
                    mode="w",
                    format="NETCDF4",
                    engine="netcdf4",
                )
                logger.info(f"Existing dataset {update_file} merged with new data and saved to {save_file}.")
                existing_ds.close()
                merged_ds.close()
                if click.prompt("Do you want to delete the old file?", default=True):
                    Path(save_dir / update_file).unlink()
                    logger.info(f"Deleted old file at {update_file}.")
            else:
                dataset.to_netcdf(
                    save_file,
                    mode="w",
                    format="NETCDF4",
                    engine="netcdf4",
                )
        logger.info(f"Download complete. Dataset saved to {save_file}.")
    else:
        logger.info("Download cancelled, exiting.")


if __name__ == "__main__":
    download_eke()
