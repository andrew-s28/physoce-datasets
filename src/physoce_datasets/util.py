"""Utility functions for physoce_datasets."""

from typing import Protocol, TypedDict

LON_MIN = -180
LON_MAX = 180
LAT_MIN = -90
LAT_MAX = 90


class OOISiteInfo(TypedDict):
    """A dictionary representing the information for an OOI site, including its reference designator, method, instrument, and geographic coordinates."""

    refdes: str
    method: str
    instrument: str
    short_name: str
    lat: float
    lon: float


OOI_SITES_INFO: dict[str, OOISiteInfo] = {
    "ce01issm": {
        "refdes": "CE01ISSM-RID16-03-CTDBPC000",
        "method": "recovered_inst",
        "instrument": "ctdbp_cdef_instrument_recovered",
        "short_name": "CTD",
        "lat": 44.6598,
        "lon": -124.095,
    },
    "ce02shsm": {
        "refdes": "CE02SHSM-RID27-03-CTDBPC000",
        "method": "recovered_inst",
        "instrument": "ctdbp_cdef_instrument_recovered",
        "short_name": "CTD",
        "lat": 44.6393,
        "lon": -124.304,
    },
    "ce04ossm": {
        "refdes": "CE04OSSM-RID27-03-CTDBPC000",
        "method": "recovered_inst",
        "instrument": "ctdbp_cdef_instrument_recovered",
        "short_name": "CTD",
        "lat": 44.3811,
        "lon": -124.956,
    },
    "ce06issm": {
        "refdes": "CE06ISSM-RID16-03-CTDBPC000",
        "method": "recovered_inst",
        "instrument": "ctdbp_cdef_instrument_recovered",
        "short_name": "CTD",
        "lat": 47.1336,
        "lon": 124.272,
    },
    "ce07shsm": {
        "refdes": "CE07SHSM-RID27-03-CTDBPC000",
        "method": "recovered_inst",
        "instrument": "ctdbp_cdef_instrument_recovered",
        "short_name": "CTD",
        "lat": 46.9859,
        "lon": 124.566,
    },
    "ce09ossm": {
        "refdes": "CE09OSSM-RID27-03-CTDBPC000",
        "method": "recovered_inst",
        "instrument": "ctdbp_cdef_instrument_recovered",
        "short_name": "CTD",
        "lat": 46.8517,
        "lon": 124.982,
    },
}


class LocationLike(Protocol):
    """A minimal interface for location-like objects."""

    lat: float
    lon: float

    @property
    def file_name(self) -> str: ...  # noqa: D102


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


class OOISite:
    """A class representing an OOI site with its reference designator, method, instrument, and geographic coordinates."""

    def __init__(self, site: str) -> None:
        """Initialize an OOISite object with the given parameters.

        Args:
            site (str): The site identifier. Must be one of the following: "CE01ISSM", "CE02SHSM", "CE04OSSM", "CE06ISSM", "CE07SHSM", "CE09OSSM". Required.

        """
        self.site = site.lower()
        self.validate()

        self.site_info = OOI_SITES_INFO[self.site]
        self.refdes = self.site_info["refdes"]
        self.method = self.site_info["method"]
        self.instrument = self.site_info["instrument"]
        self.lat = self.site_info["lat"]
        self.lon = self.site_info["lon"]
        self.short_name = self.site_info["short_name"]

    def validate(self) -> None:
        """Validate that the site identifier is valid and that the latitude and longitude values are within acceptable bounds.

        Raises:
            ValueError: If the site identifier is not one of the following: "CE01ISSM", "CE02SHSM", "CE04OSSM", "CE06ISSM", "CE07SHSM", "CE09OSSM".

        """
        if self.site not in OOI_SITES_INFO:
            msg = f"Invalid site identifier: '{self.site}'. Must be one of the following (case insensitive): {', '.join(OOI_SITES_INFO.keys())}."
            raise ValueError(msg)

    def __repr__(self) -> str:
        """Represent the OOISite in code outputs (e.g., Python REPL).

        Returns:
            str: A string representation of the OOISite in the format 'OOISite(site=..., short_name=..., refdes=..., method=..., instrument=..., latitude=..., longitude=...)'.

        """
        lon_str = f"{-self.lon:.0f}W" if self.lon < 0 else f"{self.lon:.0f}E"
        lat_str = f"{-self.lat:.0f}S" if self.lat < 0 else f"{self.lat:.0f}N"
        return (
            f"OOISite(site='{self.site.upper()}', short_name='{self.short_name}', refdes='{self.refdes}', method='{self.method}', "
            f"instrument='{self.instrument}', latitude={lat_str}, longitude={lon_str})"
        )

    def __str__(self) -> str:
        """Convert the OOISite to a formatted string. Accessed with str(ooisite).

        Returns:
            str: A string representation of the OOISite in the format 'OOISite(site=..., short_name=..., refdes=..., method=..., instrument=..., latitude=..., longitude=...)'.

        """
        return f"OOI EA Site {self.site.upper()} {self.short_name}"

    @property
    def search_url(self) -> str:
        """Construct the search URL for the OOI site based on its reference designator, method, and instrument.

        Returns:
            str: The search URL for the OOI site.

        """
        search_url_base = "https://thredds.dataexplorer.oceanobservatories.org/thredds/catalog/ooigoldcopy/public/"
        return search_url_base + f"{self.refdes}-{self.method}-{self.instrument}" + "/catalog.html"

    @property
    def file_name(self) -> str:
        """Convert the OOISite to a string format suitable for filenames.

        Returns:
            str: A string representation of the OOISite in the format '{site}_{short_name}'.

        """
        return f"{self.site.upper()}_{self.short_name}"


def parse_location_or_site(location_str: str, location_type: str) -> LocationLike:
    """Parse a location string into a Location or OOISite object.

    Args:
        location_str (str): A string representing the location in the format 'lon,lat' (e.g., '132.0,36.55').
        location_type (str): The type of location to parse. Must be either 'lonlat' or 'site'.

    Returns:
        LocationLike: A Location or OOISite instance.

    Raises:
        ValueError: If the input string is not in the correct format or if the values are out of bounds.

    """
    if location_type == "site":
        if location_str.lower() not in OOI_SITES_INFO:
            msg = f"Invalid site identifier: '{location_str}'. Must be one of the following: {', '.join(OOI_SITES_INFO.keys())}."
            raise ValueError(msg)
        return OOISite(location_str)

    if location_type == "lonlat":
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

    msg = f"Invalid location type: '{location_type}'. Must be either 'lonlat' or 'site'."
    raise ValueError(msg)
