import datetime
import os
from abc import ABC, abstractmethod
from pathlib import Path

import xarray as xr

from physoce_datasets.util import parse_location_or_site


class _Downloader(ABC):
    """Abstract base class for dataset downloaders."""

    def __init__(
        self,
        location: str,
        location_type: str,
        save_file_prefix: str,
        save_dir: str | None = None,
        save_file: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> None:
        """Initialize the downloader.

        Args:
            location (str): Location for the dataset in the format 'lon,lat' (e.g., '132.0,36.55'). Required.
            location_type (str): The type of the location, either 'lonlat' or 'site'. Used to determine how to parse the location string.
            save_file_prefix (str): A prefix to add to the saved file name.
            save_dir (str | None): The directory to save the downloaded dataset. If None, defaults to a "data" directory in the current working directory.
            save_file (str | None): The file name to save the downloaded dataset. If None, defaults to a name based on the dataset and date range.
            start_date (str | None): The start date for the dataset in "YYYY-MM-DD" format. If None, defaults to "2000-01-01".
            end_date (str | None): The end date for the dataset in "YYYY-MM-DD" format. If None, defaults to the current date.

        """
        # handle default parameters
        if start_date is None:
            start_date = "2000-01-01"
        if end_date is None:
            end_date = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d")

        self.start_date = start_date
        self.end_date = end_date
        self.location = parse_location_or_site(location, location_type)
        self.save_dir = self._create_data_dir(save_dir)
        if save_file is None:
            save_file = f"{save_file_prefix}_{self.start_date}_{self.end_date}_{self.location.file_name}.nc"
        self.save_file = save_file
        self.save_file_path = self._create_save_file(self.save_dir, self.save_file)
        self.downloaded = False

    @staticmethod
    def _create_data_dir(save_dir: str | None) -> Path:
        """Create the directory to save the downloaded dataset if it doesn't already exist.

        Args:
            save_dir (str | None): The directory to save the downloaded dataset. If None,
                defaults to a "data" directory in the current working directory.

        Returns:
            Path: The directory to save the downloaded dataset.

        """
        # by default, save in a "data" directory relative to current working directory
        data_dir = Path(save_dir) if save_dir is not None else Path.cwd() / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

    @staticmethod
    def _create_save_file(save_dir: Path, save_file: str) -> Path:
        """Create the file path to save the downloaded dataset.

        Args:
            save_dir (Path): The directory to save the downloaded dataset.
            save_file (str): The file name to save the downloaded dataset.

        Returns:
            Path: The file path to save the downloaded dataset.

        Raises:
            PermissionError: If the file already exists and is not writable.

        """
        save_file_path = Path(save_dir) / save_file
        if save_file_path.exists() and not os.access(save_file_path, os.W_OK):
            msg = (
                f"File {save_file} already exists and is not writable. If this file "
                "is open in another application (e.g., Jupyter notebook), please "
                "close it and try again."
            )
            raise PermissionError(msg)
        return save_file_path

    @abstractmethod
    def download(self) -> None:
        """Download the dataset."""

    def open_dataset(self, **kwargs: dict) -> xr.Dataset:
        """Open the downloaded dataset as an xarray Dataset.

        Args:
            **kwargs: Additional keyword arguments to pass to `xr.open_dataset()`.

        Returns:
            xr.Dataset: The downloaded dataset as an xarray Dataset.

        Raises:
            ValueError: If the `save_file` attribute is not set.

        """
        if self.save_file is None:
            msg = (
                "Attribute `save_file` not set. If you're seeing this, it's probably an issue in the implementation of the downloader subclass. \n"
                "Please consider opening an issue on GitHub to report this at https://github.com/andrew-s28/physoce-datasets/issues"
            )
            raise ValueError(msg)
        if not self.downloaded:
            msg = f"Dataset not downloaded. Please call the `{self.__class__.__name__}.download()` method first. If you've already called `download()`, there may have been an issue during the download process. "
            raise ValueError(msg)
        save_file_path = self._create_save_file(self.save_dir, self.save_file)
        return xr.open_dataset(save_file_path, **kwargs)  # ty:ignore[invalid-argument-type]
