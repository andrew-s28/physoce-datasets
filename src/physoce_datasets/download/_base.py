import datetime
import os
from abc import ABC, abstractmethod
from pathlib import Path

import xarray as xr

LON_MIN = -180
LON_MAX = 180
LAT_MIN = -90
LAT_MAX = 90


class LonLat:
    """A class representing a geographic point defined by longitude and latitude."""

    def __init__(self, lon: float, lat: float) -> None:
        """Initialize a LonLat object with longitude and latitude.

        Args:
            lon (float): Longitude of the point, must be between -180 and 180.
            lat (float): Latitude of the point, must be between -90 and 90.

        """
        self.lon = lon
        self.lat = lat
        self.lon_360 = False  # flag to indicate if longitude is in 0-360 range
        self.validate()

    def __repr__(self) -> str:
        """Represent the point in code outputs (e.g., Python REPL).

        Returns:
            str: A string representation of the point in the format 'LonLat(lon=..., lat=...)'.

        """
        lon_str = f"{-self.lon:.0f}W" if self.lon < 0 else f"{self.lon:.0f}E"
        lat_str = f"{-self.lat:.0f}S" if self.lat < 0 else f"{self.lat:.0f}N"
        return f"LonLat(lon={lon_str}, lat={lat_str})"

    def __str__(self) -> str:
        """Convert the point to a formatted string. Accessed with str(point).

        Returns:
            str: A string representation of the point in the format '({lon[E/W]}, {lat[N/S]})'.

        """
        lon_str = f"{-self.lon:.0f}W" if self.lon < 0 else f"{self.lon:.0f}E"
        lat_str = f"{-self.lat:.0f}S" if self.lat < 0 else f"{self.lat:.0f}N"
        return f"({lon_str}, {lat_str})"

    def validate(self) -> None:
        """Validate that the longitude and latitude values are within acceptable bounds.

        Raises:
            ValueError: If longitude or latitude values are out of bounds.

        """
        lon_msg = ""
        lat_msg = ""
        if not LON_MIN <= self.lon <= LON_MAX:
            lon_msg = f"Longitude must be between {LON_MIN} and {LON_MAX}. Received: {self.lon}."
        if not LAT_MIN <= self.lat <= LAT_MAX:
            lat_msg = f"Latitude must be between {LAT_MIN} and {LAT_MAX}. Received: {self.lat}."
        if lon_msg or lat_msg:
            raise ValueError(f"{lon_msg} {lat_msg}".strip())

    @property
    def file_name(self) -> str:
        """Convert the point to a string format suitable for filenames.

        Returns:
            str: A string representation of the point in the format 'lon{E/W}_lat{N/S}'.

        """
        lon_str = f"{-self.lon:.0f}W" if self.lon < 0 else f"{self.lon:.0f}E"
        lat_str = f"{-self.lat:.0f}S" if self.lat < 0 else f"{self.lat:.0f}N"
        return f"{lon_str}_{lat_str}"

    def _convert_to_360(self) -> None:
        """Convert longitude from the -180 to 180 range to the 0 to 360 range."""
        if not self.lon_360:
            self.lon %= 360
            self.lon_360 = True

    def _convert_to_180(self) -> None:
        """Convert longitude from the 0 to 360 range to the -180 to 180 range."""
        if self.lon_360:
            self.lon = (self.lon + 180) % 360 - 180
            self.lon_360 = False

    def convert_longitude(self) -> None:
        """Convert longitude between the -180 to 180 range and the 0 to 360 range and vice versa.

        Depends on the current state of the longitude. If the longitude is currently in the -180 to 180 range, it will be converted to the 0 to 360 range by adding 360 to negative values. If the longitude is currently in the 0 to 360 range, it will be converted back to the -180 to 180 range by subtracting 360 from values greater than 180. Uses the lon_360 flag to track the current state of the longitude.

        """
        if self.lon_360:
            self._convert_to_180()
        else:
            self._convert_to_360()


def parse_lonlat(location_str: str) -> LonLat:
    """Parse a location string into a LonLat object.

    Args:
        location_str (str): A string representing the location in the format 'lon,lat' (e.g., '132.0,36.55').

    Returns:
        LonLat: A LonLat instance.

    Raises:
        ValueError: If the input string is not in the correct format or if the values are out of bounds.

    """
    try:
        lon_str, lat_str = location_str.split(",")
        lon = float(lon_str)
        lat = float(lat_str)
    except ValueError as e:
        msg = (
            "Location must be a string in the format 'lon,lat' with valid float values, "
            f"or a valid OOI site identifier. Received: '{location_str}'."
        )
        raise ValueError(msg) from e
    return LonLat(lon=lon, lat=lat)


class _Downloader(ABC):
    """Abstract base class for dataset downloaders."""

    def __init__(
        self,
        save_file_prefix: str,
        save_dir: str | None = None,
        save_file: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> None:
        """Initialize the downloader.

        Args:
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
        self.save_dir = self._create_data_dir(save_dir)
        if save_file is None:
            save_file = f"{save_file_prefix}_{self.start_date}_{self.end_date}.nc"
        self.save_file = save_file
        self.save_file_path = self._create_save_file(self.save_dir, self.save_file)

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
        if not self.save_file_path.exists():
            msg = f"Dataset does not exist at {self.save_file_path}. Check that the file exists and is readable. If not, call the `{self.__class__.__name__}.download()` method first. If you've already called `download()`, there may have been an issue during the download process. "
            raise ValueError(msg)
        save_file_path = self._create_save_file(self.save_dir, self.save_file)
        return xr.open_dataset(save_file_path, **kwargs)  # ty:ignore[invalid-argument-type]
