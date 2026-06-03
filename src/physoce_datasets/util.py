"""Utility functions for physoce_datasets."""

from typing import TypedDict

LON_MIN = -180
LON_MAX = 180
LAT_MIN = -90
LAT_MAX = 90


class Location:
    """A class representing a geographic location defined by longitude and latitude."""

    def __init__(self, lon: float, lat: float) -> None:
        """Initialize a Location object with longitude and latitude.

        Args:
            lon (float): Longitude of the location, must be between -180 and 180.
            lat (float): Latitude of the location, must be between -90 and 90.

        """
        self.lon = lon
        self.lat = lat
        self.lon_360 = False  # flag to indicate if longitude is in 0-360 range
        self.validate()

    def __repr__(self) -> str:
        """Represent the location in code outputs (e.g., Python REPL).

        Returns:
            str: A string representation of the location in the format 'Location(lon=..., lat=...)'.

        """
        lon_str = f"{-self.lon:.0f}W" if self.lon < 0 else f"{self.lon:.0f}E"
        lat_str = f"{-self.lat:.0f}S" if self.lat < 0 else f"{self.lat:.0f}N"
        return f"Location(lon={lon_str}, lat={lat_str})"

    def __str__(self) -> str:
        """Convert the location to a formatted string. Accessed with str(location).

        Returns:
            str: A string representation of the location in the format '(lon{E/W}, lat{N/S})'.

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
        """Convert the location to a string format suitable for filenames.

        Returns:
            str: A string representation of the location in the format 'lon{E/W}_lat{N/S}'.

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


def parse_location(location_str: str) -> Location:
    """Parse a location string into a Location object.

    Args:
        location_str (str): A string representing the location in the format 'lon,lat' (e.g., '132.0,36.55').

    Returns:
        Location: A Location object with the parsed longitude and latitude.

    Raises:
        ValueError: If the input string is not in the correct format or if the values are out of bounds.

    """
    try:
        lon_str, lat_str = location_str.split(",")
        lon = float(lon_str)
        lat = float(lat_str)
    except ValueError as e:
        msg = f"Location must be a string in the format 'lon,lat' with valid float values. Received: '{location_str}'."
        raise ValueError(msg) from e

    return Location(lon=lon, lat=lat)


class AreaDict(TypedDict):
    """A dictionary representing a geographic bounding box with longitude and latitude limits."""

    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float


def parse_area(area_str: str | None) -> AreaDict:
    """Parse an area string into a dictionary of floats.

    Args:
        area_str (str | None): A string representing the bounding box in the format 'lon_min,lon_max,lat_min,lat_max'
            If None, defaults to global coverage.

    Returns:
        AreaDict: An AreaDict TypedDict with keys 'lon_min', 'lon_max', 'lat_min', 'lat_max'.

    Raises:
        ValueError: If the input string is not in the correct format or if the values are out of bounds.

    """
    if area_str is None:
        return {
            "lon_min": float(LON_MIN),
            "lon_max": float(LON_MAX),
            "lat_min": float(LAT_MIN),
            "lat_max": float(LAT_MAX),
        }  # default to global coverage

    # split the string and convert to floats, with error handling for invalid formats
    try:
        lon_min, lon_max, lat_min, lat_max = map(float, area_str.split(","))
    except ValueError as e:
        msg = (
            f"Area must be a string in the format 'lon_min,lon_max,lat_min,lat_max' with valid float values. "
            f"Received: '{area_str}'."
        )
        raise ValueError(msg) from e

    # validate longitude and latitude values
    msg = ""
    if not LON_MIN <= lon_min <= lon_max <= LON_MAX:
        msg += (
            f"Longitude values must be between {LON_MIN} and {LON_MAX}, "
            f"with lon_min <= lon_max. Received: lon_min={lon_min}, lon_max={lon_max}. "
        )
    if not LAT_MIN <= lat_min <= lat_max <= LAT_MAX:
        msg += (
            f"Latitude values must be between {LAT_MIN} and {LAT_MAX}, "
            f"with lat_min <= lat_max. Received: lat_min={lat_min}, lat_max={lat_max}. "
        )
    if msg:
        raise ValueError(msg.strip())

    return {
        "lon_min": lon_min,
        "lon_max": lon_max,
        "lat_min": lat_min,
        "lat_max": lat_max,
    }


def get_area_str(area: AreaDict) -> str:
    """Convert an AreaDict back into a string format for use in filenames.

    Args:
        area (AreaDict): An AreaDict TypedDict with keys 'lon_min', 'lon_max', 'lat_min', 'lat_max'.

    Returns:
        str: A string representing the bounding box in the format 'lon_min-lon_max_lat_min-lat_max', with longitude values suffixed by 'E' or 'W' and latitude values suffixed by 'N' or 'S'.

    """
    lon_min = f"{-area['lon_min']:.0f}W" if area["lon_min"] < 0 else f"{area['lon_min']:.0f}E"
    lon_max = f"{-area['lon_max']:.0f}W" if area["lon_max"] < 0 else f"{area['lon_max']:.0f}E"
    lat_min = f"{-area['lat_min']:.0f}S" if area["lat_min"] < 0 else f"{area['lat_min']:.0f}N"
    lat_max = f"{-area['lat_max']:.0f}S" if area["lat_max"] < 0 else f"{area['lat_max']:.0f}N"

    return f"{lon_min}-{lon_max}_{lat_min}-{lat_max}"
