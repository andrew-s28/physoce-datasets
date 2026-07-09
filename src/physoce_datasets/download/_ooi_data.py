from enum import StrEnum
from typing import TypedDict


class _OOISite:
    """A class representing an OOI site with its reference designator, method, instrument, and geographic coordinates."""

    def __init__(self, site: str, instrument: str, location_type: str) -> None:
        """Initialize an OOISite object with the given parameters.

        Args:
            site (str): The site identifier. Must be one of the following: `CE01ISSP`, `CE02SHSP`, `CE04OSPS`, `CE04OSPD`, `CE06ISSP`, `CE07SHSP`, `CE09OSSP`, `RS01SBPS`. Required.
            instrument (str): The instrument identifier. Required.
            location_type (str): The type of location for the site. Required.

        """
        self.site = site.upper()
        self.instrument = instrument
        self.location_type = location_type

        self.validate()

        # we've already validated type of self.site to be in valid sites
        if instrument == "ctd" and location_type == "profiler":
            self.site_info = OOI_PROFILERS_CTD[self.site]  # ty:ignore[invalid-argument-type]
        elif instrument == "chl" and location_type == "profiler":
            self.site_info = OOI_PROFILERS_CHL[self.site]  # ty:ignore[invalid-argument-type]
        elif instrument == "ctd" and location_type == "mooring":
            self.site_info = OOI_MOORINGS_CTD[self.site]  # ty:ignore[invalid-argument-type]

        self.refdes = self.site_info["refdes"]
        self.method = self.site_info["method"]
        self.instrument = self.site_info["instrument"]
        self.lat = self.site_info["lat"]
        self.lon = self.site_info["lon"]
        self.depth = self.site_info["depth"]
        self.short_name = self.site_info["short_name"]

    def validate(self) -> None:
        """Validate that the site identifier is valid and that the latitude and longitude values are within acceptable bounds.

        Raises:
            ValueError: If the site identifier is not one of the following: `CE01ISSP`, `CE02SHSP`, `CE04OSPS`, `CE04OSPD`, `CE06ISSP`, `CE07SHSP`, `CE09OSSP`, `RS01SBPS`.

        """
        if self.location_type not in {"profiler", "mooring"}:
            msg = f"Invalid location type: '{self.location_type}'. Must be either 'profiler' or 'mooring'."
            raise ValueError(msg)
        if self.location_type == "mooring" and self.instrument != "ctd":
            msg = f"Invalid instrument identifier: '{self.instrument}' for location type 'mooring'. Must be 'ctd'."
            raise ValueError(msg)
        if self.location_type == "profiler" and self.instrument not in {"ctd", "chl"}:
            msg = f"Invalid instrument identifier: '{self.instrument}'. Must be either 'ctd' or 'chl'."
            raise ValueError(msg)
        valid_sites = (
            ", ".join(ProfilerSites.__members__)
            if self.location_type == "profiler"
            else ", ".join(MooringSites.__members__)
        )
        if (
            (
                self.location_type == "profiler"
                and self.instrument == "ctd"
                and self.site not in ProfilerSites.__members__
            )
            or (
                self.location_type == "profiler"
                and self.instrument == "chl"
                and self.site not in ProfilerSites.__members__
            )
            or (
                self.location_type == "mooring"
                and self.instrument == "ctd"
                and self.site not in MooringSites.__members__
            )
        ):
            msg = f"Invalid profiler identifier: '{self.site}' for instrument '{self.instrument}'. Must be one of the following (case insensitive): {valid_sites}."
            raise ValueError(msg)

    def __repr__(self) -> str:
        """Represent the OOISite in code outputs (e.g., Python REPL).

        Returns:
            str: A string representation of the OOISite in the format 'OOISite(site=..., short_name=..., refdes=..., method=..., instrument=..., latitude=..., longitude=...)'.

        """
        lon_str = f"{-self.lon:.0f}W" if self.lon < 0 else f"{self.lon:.0f}E"
        lat_str = f"{-self.lat:.0f}S" if self.lat < 0 else f"{self.lat:.0f}N"
        return (
            f"OOISite(site='{self.site}', short_name='{self.short_name}', refdes='{self.refdes}', method='{self.method}', "
            f"instrument='{self.instrument}', latitude={lat_str}, longitude={lon_str})"
        )

    def __str__(self) -> str:
        """Convert the OOISite to a formatted string. Accessed with str(ooisite).

        Returns:
            str: A string representation of the OOISite in the format 'OOI EA Site {site} {short_name}'.

        """
        return f"OOI EA Site {self.site} {self.short_name}"

    @property
    def search_url(self) -> str:
        """Construct the search URL for the OOI site based on its reference designator, method, and instrument."""
        search_url_base = "https://thredds.dataexplorer.oceanobservatories.org/thredds/catalog/ooigoldcopy/public/"
        return search_url_base + f"{self.refdes}-{self.method}-{self.instrument}" + "/catalog.html"

    @property
    def file_name(self) -> str:
        """Convert the profiler site and name to a string format suitable for filenames in the format '{site}_{short_name}'."""
        return f"{self.site}_{self.short_name}"


class ProfilerSites(StrEnum):
    """Representation for valid OOI profiler sites.

    Valid options are:

    - [`CE01ISSP`](https://oceanobservatories.org/site/ce01issp/)
    - [`CE02SHSP`](https://oceanobservatories.org/site/ce02shsp/)
    - [`CE04OSPS`](https://oceanobservatories.org/site/ce04osps/)
    - [`CE04OSPD`](https://oceanobservatories.org/site/ce04ospd/)
    - [`CE06ISSP`](https://oceanobservatories.org/site/ce06issp/)
    - [`CE07SHSP`](https://oceanobservatories.org/site/ce07shsp/)
    - [`CE09OSPM`](https://oceanobservatories.org/site/ce09ospm/)
    - [`RS01SBPS`](https://oceanobservatories.org/site/rs01sbps/)

    """

    CE01ISSP = "CE01ISSP"
    CE02SHSP = "CE02SHSP"
    CE04OSPS = "CE04OSPS"
    CE04OSPD = "CE04OSPD"
    CE06ISSP = "CE06ISSP"
    CE07SHSP = "CE07SHSP"
    CE09OSPM = "CE09OSPM"
    RS01SBPS = "RS01SBPS"


class MooringSites(StrEnum):
    """Representation for valid OOI mooring sites.

    Valid options are:

    - [`CE01ISSM`](https://oceanobservatories.org/site/ce01issm/)
    - [`CE02SHSM`](https://oceanobservatories.org/site/ce02shsm/)
    - [`CE04OSSM`](https://oceanobservatories.org/site/ce04ossm/)
    - [`CE06ISSM`](https://oceanobservatories.org/site/ce06issm/)
    - [`CE07SHSM`](https://oceanobservatories.org/site/ce07shsm/)
    - [`CE09OSSM`](https://oceanobservatories.org/site/ce09ossm/)

    """

    CE01ISSM = "CE01ISSM"
    CE02SHSM = "CE02SHSM"
    CE04OSSM = "CE04OSSM"
    CE06ISSM = "CE06ISSM"
    CE07SHSM = "CE07SHSM"
    CE09OSSM = "CE09OSSM"


class _OOISiteInfo(TypedDict):
    """A dictionary representing the information for an OOI site, including its reference designator, method, instrument, and geographic coordinates."""

    refdes: str
    method: str
    instrument: str
    short_name: str
    lat: float
    lon: float
    depth: float


OOI_MOORINGS_CTD: dict[MooringSites, _OOISiteInfo] = {
    MooringSites.CE01ISSM: {
        "refdes": "CE01ISSM-RID16-03-CTDBPC000",
        "method": "recovered_inst",
        "instrument": "ctdbp_cdef_instrument_recovered",
        "short_name": "CTD",
        "lat": 44.6598,
        "lon": -124.095,
        "depth": 25,
    },
    MooringSites.CE02SHSM: {
        "refdes": "CE02SHSM-RID27-03-CTDBPC000",
        "method": "recovered_inst",
        "instrument": "ctdbp_cdef_instrument_recovered",
        "short_name": "CTD",
        "lat": 44.6393,
        "lon": -124.304,
        "depth": 80,
    },
    MooringSites.CE04OSSM: {
        "refdes": "CE04OSSM-RID27-03-CTDBPC000",
        "method": "recovered_inst",
        "instrument": "ctdbp_cdef_instrument_recovered",
        "short_name": "CTD",
        "lat": 44.3811,
        "lon": -124.956,
        "depth": 588,
    },
    MooringSites.CE06ISSM: {
        "refdes": "CE06ISSM-RID16-03-CTDBPC000",
        "method": "recovered_inst",
        "instrument": "ctdbp_cdef_instrument_recovered",
        "short_name": "CTD",
        "lat": 47.1336,
        "lon": 124.272,
        "depth": 29,
    },
    MooringSites.CE07SHSM: {
        "refdes": "CE07SHSM-RID27-03-CTDBPC000",
        "method": "recovered_inst",
        "instrument": "ctdbp_cdef_instrument_recovered",
        "short_name": "CTD",
        "lat": 46.9859,
        "lon": 124.566,
        "depth": 87,
    },
    MooringSites.CE09OSSM: {
        "refdes": "CE09OSSM-RID27-03-CTDBPC000",
        "method": "recovered_inst",
        "instrument": "ctdbp_cdef_instrument_recovered",
        "short_name": "CTD",
        "lat": 46.8517,
        "lon": 124.982,
        "depth": 544,
    },
}

OOI_PROFILERS_CTD: dict[ProfilerSites, _OOISiteInfo] = {
    ProfilerSites.CE01ISSP: {
        "refdes": "CE01ISSP-SP001-09-CTDPFJ000",
        "method": "recovered_cspp",
        "instrument": "ctdpf_j_cspp_instrument_recovered",
        "short_name": "CTD",
        "lat": 44.662,
        "lon": -124.096,
        "depth": 25,
    },
    ProfilerSites.CE02SHSP: {
        "refdes": "CE02SHSP-SP001-08-CTDPFJ000",
        "method": "recovered_cspp",
        "instrument": "ctdpf_j_cspp_instrument_recovered",
        "short_name": "CTD",
        "lat": 44.6372,
        "lon": -124.299,
        "depth": 80,
    },
    ProfilerSites.CE04OSPS: {
        "refdes": "CE04OSPS-SF01B-2A-CTDPFA107",
        "method": "streamed",
        "instrument": "ctdpf_sbe43_sample",
        "short_name": "CTD",
        "lat": 44.3683,
        "lon": -124.953,
        "depth": 588,
    },
    ProfilerSites.RS01SBPS: {
        "refdes": "RS01SBPS-SF01A-2A-CTDPFA102",
        "method": "streamed",
        "instrument": "ctdpf_sbe43_sample",
        "short_name": "CTD",
        "lat": 44.529,
        "lon": -125.3893,
        "depth": 2906,
    },
    ProfilerSites.CE06ISSP: {
        "refdes": "CE06ISSP-SP001-09-CTDPFJ000",
        "method": "recovered_cspp",
        "instrument": "ctdpf_j_cspp_instrument_recovered",
        "short_name": "CTD",
        "lat": 47.136,
        "lon": 124.269,
        "depth": 29,
    },
    ProfilerSites.CE07SHSP: {
        "refdes": "CE07SHSP-SP001-08-CTDPFJ000",
        "method": "recovered_cspp",
        "instrument": "ctdpf_j_cspp_instrument_recovered",
        "short_name": "CTD",
        "lat": 46.9843,
        "lon": 124.565,
        "depth": 87,
    },
    ProfilerSites.CE09OSPM: {
        "refdes": "CE09OSPM-WFP01-03-CTDPFK000",
        "method": "recovered_wfp",
        "instrument": "wfp-ctdpf_ckl_wfp_instrument_recovered",
        "short_name": "CTD",
        "lat": 46.8517,
        "lon": 124.982,
        "depth": 544,
    },
}

OOI_PROFILERS_CHL: dict[ProfilerSites, _OOISiteInfo] = {
    ProfilerSites.CE01ISSP: {
        "refdes": "CE01ISSP-SP001-08-FLORTJ000",
        "method": "recovered_cspp",
        "instrument": "flort_sample",
        "short_name": "Fluorometer Chlorophyll",
        "lat": 44.662,
        "lon": -124.096,
        "depth": 25,
    },
    ProfilerSites.CE02SHSP: {
        "refdes": "CE02SHSP-SP001-07-FLORTJ000",
        "method": "recovered_cspp",
        "instrument": "flort_sample",
        "short_name": "Fluorometer Chlorophyll",
        "lat": 44.6372,
        "lon": -124.299,
        "depth": 80,
    },
    ProfilerSites.CE04OSPS: {
        "refdes": "CE04OSPS-SF01B-2A-FLORTD104",
        "method": "streamed",
        "instrument": "flort_d_data_record",
        "short_name": "Fluorometer Chlorophyll",
        "lat": 44.3683,
        "lon": -124.953,
        "depth": 588,
    },
    ProfilerSites.RS01SBPS: {
        "refdes": "RS01SBPS-SF01A-3A-FLORTD101",
        "method": "streamed",
        "instrument": "flort_d_data_record",
        "short_name": "Fluorometer Chlorophyll",
        "lat": 44.529,
        "lon": -125.3893,
        "depth": 2906,
    },
    ProfilerSites.CE06ISSP: {
        "refdes": "CE06ISSP-SP001-08-FLORTJ000",
        "method": "recovered_cspp",
        "instrument": "flort_sample",
        "short_name": "Fluorometer Chlorophyll",
        "lat": 47.136,
        "lon": 124.269,
        "depth": 29,
    },
    ProfilerSites.CE07SHSP: {
        "refdes": "CE07SHSP-SP001-08-FLORTJ000",
        "method": "recovered_cspp",
        "instrument": "flort_sample",
        "short_name": "Fluorometer Chlorophyll",
        "lat": 46.9843,
        "lon": 124.565,
        "depth": 87,
    },
    ProfilerSites.CE09OSPM: {
        "refdes": "CE09OSPM-WFP01-03-FLORTK000",
        "method": "recovered_wfp",
        "instrument": "flort_sample",
        "short_name": "Fluorometer Chlorophyll",
        "lat": 46.8517,
        "lon": 124.982,
        "depth": 544,
    },
}
