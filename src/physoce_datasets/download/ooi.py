"""Module for downloading datasets from the OOI Endurance Array Mooring."""

import io
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

import requests
import xarray as xr
from bs4 import BeautifulSoup
from tqdm import tqdm

from physoce_datasets.logging import logger

from ._base import _Downloader

if TYPE_CHECKING:
    from physoce_datasets.util import OOISite


class EAMooringDownloader(_Downloader):
    """A class to download OOI Endurance Array Mooring CTD datasets."""

    def __init__(
        self,
        location: str,
        start_date: str | None = None,
        end_date: str | None = None,
        save_dir: str | None = None,
        save_file: str | None = None,
    ) -> None:
        """Initialize the EAMooringDownloader with parameters for downloading.

        Args:
            location (str): Location for the dataset in the form of a site identifier (e.g., "CE01ISSM"). Must be one of the following: "CE01ISSM", "CE02SHSM", "CE04OSSM", "CE06ISSM", "CE07SHSM", "CE09OSSM". Required.
            start_date (str | None): The start date for the dataset in "YYYY-MM-DD" format. If None, defaults to "2000-01-01".
            end_date (str | None): The end date for the dataset in "YYYY-MM-DD" format. If None, defaults to the current date.
            save_dir (str | None): The directory to save the downloaded dataset. If None, defaults to a "data" directory in the current working directory.
            save_file (str | None): The file name to save the downloaded dataset. If None, defaults to a name based on the dataset and date range.

        """
        super().__init__(
            location=location,
            location_type="site",
            save_file_prefix=f"ooi_{location.lower()}_ctd",
            save_dir=save_dir,
            save_file=save_file,
            start_date=start_date,
            end_date=end_date,
        )
        self.location = cast("OOISite", self.location)
        self.search_url = self.location.search_url
        self.base_url = "https://thredds.dataexplorer.oceanobservatories.org/thredds/fileServer/"
        self.tag = r"CTDBPC000.*.nc$"  # setup regex for files we want

    def _filter_dates(self, nc_files: list) -> list:
        """Filter the list of netCDF files based on the start and end dates.

        Args:
            nc_files (list): List of netCDF files to filter, with date information in the file names.

        Returns:
            list: A filtered list of netCDF files that fall within the specified date range.

        """
        date_regex = re.compile(r"(\d{8}T\d{6}(?:\.\d+)?)-(\d{8}T\d{6}(?:\.\d+)?)")
        start_date_dt = datetime.strptime(self.start_date, "%Y-%m-%d").replace(tzinfo=UTC)
        end_date_dt = datetime.strptime(self.end_date, "%Y-%m-%d").replace(tzinfo=UTC)
        # get start and end dates of each deployment from the file names
        date_search = [re.search(date_regex, f) for f in nc_files]
        nc_files_start_dates = [
            datetime.strptime(d.group(1).split(".")[0], "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
            for d in date_search
            if d is not None
        ]
        nc_files_end_dates = [
            datetime.strptime(d.group(2).split(".")[0], "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
            for d in date_search
            if d is not None
        ]
        # filter by start date, including any data files that end after the start date
        nc_files = [
            f
            for i, f in enumerate(nc_files)
            if nc_files_end_dates[i] >= start_date_dt and nc_files_start_dates[i] <= end_date_dt
        ]
        return nc_files

    def _list_files(self, url: str, tag: str = r".*\.nc$") -> list[str]:
        """List the netCDF data files in a THREDDS catalog.

        Args:
            url (str): URL to a THREDDS catalog specific to a data request
            tag (regexp, optional): Regex pattern used to distinguish files of interest. Defaults to all files ending in '.nc'.

        Returns:
            array: list of files in the catalog with the URL path set relative to the catalog

        """
        with requests.session() as s:
            page = s.get(url).text

        soup = BeautifulSoup(page, "html.parser")
        pattern = re.compile(tag)
        nc_files = []
        for node in soup.find_all("a"):
            href = node.get("href")
            if isinstance(href, str) and pattern.search(node.get_text()):
                nc_files.append(href)
        nc_files = [re.sub(r"catalog.html\?dataset=", "", file) for file in nc_files]
        nc_files = self._filter_dates(nc_files)
        return nc_files

    def download(self) -> None:
        """Download the netCDF data files from the THREDDS catalog and save them to a local directory."""
        logger.info(f"Getting list of data files for {self.location}...")
        nc_files = self._list_files(self.search_url, self.tag)
        if not nc_files:
            logger.warning(f"No files found for {self.location} in the specified date range.")
            return
        download_urls = [self.base_url + f + "#mode=bytes" for f in nc_files]

        logger.info(f"Downloading files for {self.location}...")
        ds = []
        for i, f in enumerate(tqdm(download_urls, desc="Downloading datasets")):
            r = requests.get(f, timeout=(3.05, 120))
            if r.ok:
                ds.append(xr.open_dataset(io.BytesIO(r.content)))
                ds[i].load()

        ds_merged = xr.merge(ds)
        ds_merged = ds_merged.sortby("time")  # ensure data is sorted by time after merging
        ds_merged = ds_merged.sel(
            time=slice(self.start_date, self.end_date)
        )  # subset to specified date range after merging

        ds_merged.to_netcdf(self.save_file_path)
        logger.info(f"Download complete! Dataset saved to {self.save_file_path}")
