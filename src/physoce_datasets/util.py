"""Utility functions for physoce_datasets."""

from typing import TypedDict

LON_MIN = -180
LON_MAX = 180
LAT_MIN = -90
LAT_MAX = 90


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
