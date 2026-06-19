"""Download NASA MUR SST datasets."""

import datetime
import getpass
import netrc
import warnings
from pathlib import Path

import earthaccess
import numpy as np
import xarray as xr
from harmony import Client

from physoce_datasets.logging import logger

from ._base import _Downloader

HARMONY_ROOT = "https://harmony.earthdata.nasa.gov"
COLLECTION_ID = (
    "C1996881146-POCLOUD"  # see https://podaac.jpl.nasa.gov/dataset/MUR-JPL-L4-GLOB-v4.1#capability-modal-subset
)
OPEN_DAP_ROOT = f"dap4://opendap.earthdata.nasa.gov/collections/{COLLECTION_ID}/granules/"


def setup_earthdata_login_auth(endpoint: str) -> Client:
    """Set up the authentication client.

    This looks in the .netrc file first and if no credentials are found, it prompts for them.

    From https://github.com/nasa/harmony/blob/main/docs/Harmony%20API%20introduction.ipynb

    Args:
        endpoint (str): The Earthdata Login endpoint to authenticate against.

    Returns:
        Client: An authenticated Harmony client for making requests to the API.

    """
    logger.info("Attempting login...")
    try:
        username, _, password = netrc.netrc().authenticators(endpoint)  # ty:ignore[not-iterable] this is caught by the except block
    except (FileNotFoundError, TypeError):
        # FileNotFound = There's no .netrc file
        # TypeError = The endpoint isn't in the netrc file, causing the above to try unpacking None
        logger.info(
            "Please provide your Earthdata Login credentials to allow data access. "
            "Credentials will be stored in a .netrc file in your home directory for future use."
        )
        username = input("Username: ")
        password = getpass.getpass()
        with Path("~/.netrc").expanduser().open("a") as netrc_file:
            netrc_file.write(f"\nmachine {endpoint}\n    login {username}\n    password {password}\n")

    harmony_client = Client(auth=(username, password))
    logger.info("Login successful!")
    return harmony_client


class SSTDownloader(_Downloader):
    """A class to download NASA MUR SST datasets."""

    def __init__(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        area: str | None = None,
        save_dir: str | None = None,
        save_file: str | None = None,
    ) -> None:
        """Initialize the SSTDownloader with parameters for downloading.

        Args:
            start_date (str | None): The start date for the dataset in 'YYYY-MM-DD' format. If None, defaults to '2000-01-01'.
            end_date (str | None): The end date for the dataset in 'YYYY-MM-DD' format. If None, defaults to the current date.
            area (str | None): A string representing the bounding box in the format 'lon_min,lon_max,lat_min,lat_max'. If None, defaults to global coverage.
            save_dir (str | None): The directory to save the downloaded dataset. If None, defaults to a 'data' directory in the current working directory.
            save_file (str | None): The file name to save the downloaded dataset. If None, defaults to a name based on the dataset and date range.

        """
        msg = "The 'sst' downloader is not implemented yet due to issues with the NASA Harmony API."
        raise NotImplementedError(msg)
        self.harmony_client = setup_earthdata_login_auth("urs.earthdata.nasa.gov")
        self.start_date = start_date if start_date is not None else "2000-01-01"
        self.end_date = (
            end_date if end_date is not None else datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d")
        )
        # area needs to be converted to location once other issues are fixed
        self.area = {"lon_min": -180, "lon_max": 180, "lat_min": -90, "lat_max": 90}
        area_str = area  # for file name
        self.save_dir = self._create_data_dir(save_dir)
        self.save_file = (
            save_file if save_file is not None else f"nasa_mur_sst_{self.start_date}_{self.end_date}_{area_str}.nc"
        )
        self.save_file_path = self._create_save_file(self.save_dir, self.save_file)
        logger.info("Getting granule URLs for the specified date range and area...")
        with warnings.catch_warnings():
            # annoying FutureWarnings in earthaccess about changing methods to attributes which we don't use anyways
            warnings.filterwarnings("ignore", category=FutureWarning, module="earthaccess")
            results = earthaccess.search_data(
                doi="10.5067/GHGMR-4FJ04",  # see https://podaac.jpl.nasa.gov/dataset/MUR-JPL-L4-GLOB-v4.1 for the DOI
                temporal=(f"{self.start_date}T00:00:00Z", f"{self.end_date}T23:59:59Z"),
            )
        granules = [r["meta"]["native-id"] for r in results]
        self.granule_urls = [OPEN_DAP_ROOT + g for g in granules]
        self.existing_datetimes, self.existing_path = self._get_existing_datetimes()

    def _get_existing_datetimes(self) -> tuple[xr.DataArray, Path] | tuple[None, None]:
        """Get the datetime values from an existing file.

        Returns:
            tuple[xr.DataArray, pathlib.Path] | tuple[None, None]: The datetime values and file path from the existing file, or None if the file does not exist.

        """
        if not self.save_file_path.exists():
            # check for any file in the save directory that matches the pattern of the expected file name if it was based on the date range and area
            save_file_glob = self.save_dir.glob("nasa_mur_sst_*.nc")
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
        """Download the MUR SST dataset for the specified date range and area, and save it to a NetCDF file."""
        # process follows from example at https://pydap.github.io/pydap/en/notebooks/PACE.html
        # first we open the dataset to get the dimensions
        logger.info("Accessing dataset to subset to specified area...")
        ds = xr.open_mfdataset(
            self.granule_urls,
            combine="by_coords",
            engine="pydap",
            session=self.harmony_client.session,
            parallel=True,
        )
        # now we get the indices for the specified area
        lon_idx = np.flatnonzero(np.asarray((ds["lon"] >= self.area["lon_min"]) & (ds["lon"] <= self.area["lon_max"])))
        lat_idx = np.flatnonzero(np.asarray((ds["lat"] >= self.area["lat_min"]) & (ds["lat"] <= self.area["lat_max"])))
        # setup the slices
        lon_slice = slice(lon_idx[0], lon_idx[-1] + 1)  # add 1 because slice end is exclusive
        lat_slice = slice(lat_idx[0], lat_idx[-1] + 1)
        # re-open the dataset with the appropriate chunks, which will rechunk on server side and enable us to only download the subset of data
        ds = xr.open_mfdataset(
            self.granule_urls,
            combine="by_coords",
            engine="pydap",
            session=self.harmony_client.session,
            parallel=True,
            chunks={"time": 1, "lat": len(lat_idx), "lon": len(lon_idx)},
        )
        ds["time"] = ds["time"].astype("datetime64[D]").astype("datetime64[ns]")  # convert time to daily resolution
        ds = ds.sel(time=slice(self.start_date, self.end_date))  # make sure we are only getting times we want
        if self.existing_datetimes is not None:
            # if there are existing datetimes, we want to make sure we don't download duplicates of those
            ds = ds.where(~ds["time"].isin(self.existing_datetimes), drop=True)
        # now get the slice we want
        ds = ds.isel(lon=lon_slice, lat=lat_slice)
        # saving will trigger the actual download of the data, which should be just the subset we want
        logger.info(f"Downloading dataset with size {ds.nbytes / 1e9:.2f} GB. This may take a while...")
        if self.existing_path is not None:
            # if there's an existing file, we want to merge the new data with the existing data and save that
            ds_existing = xr.open_dataset(self.existing_path)
            ds = self._merge_datasets(ds_existing, ds)
            ds_existing.close()
        ds.to_netcdf(self.save_file_path)
        if self.existing_path is not None and self.existing_path != self.save_file_path:
            # if we had to merge with an existing file that had a different name, we can remove the old file after saving the new merged file
            self.existing_path.unlink()
        logger.info(f"Dataset saved to {self.save_file_path}")
        self.downloaded = True
