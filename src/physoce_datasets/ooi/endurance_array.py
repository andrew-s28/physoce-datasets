"""Download and process [OOI Endurance Array data](https://oceanobservatories.org/array/coastal-endurance/)."""

import io
import re
import warnings
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TypedDict

import gsw
import numpy as np
import requests
import xarray as xr
import yaml
from bs4 import BeautifulSoup
from flox.xarray import xarray_reduce
from tqdm import tqdm

from physoce_datasets._util import Downloader, logger

__all__ = ["EnduranceArray"]


class InstrumentInfo(TypedDict):
    """A TypedDict representing the information for an OOI instrument."""

    refdes: str
    method: str
    stream: str
    variables: set[str]
    description: str
    title: str


class SiteInfo(TypedDict):
    """A TypedDict representing the information for an OOI site."""

    name: str
    site: str
    description: str
    doi: str
    lat: float
    lon: float
    water_depth: int
    valid_depth: int
    type: str
    instruments: dict[str, InstrumentInfo]


class OOISite:
    """A class representing an OOI site with its reference designator, method, instrument, and geographic coordinates.

    Mostly intended for internal use, but can be used externally to access site information and metadata for OOI sites if desired.
    """

    def __init__(self, site: str, instrument: str, location_type: str) -> None:
        """Initialize an OOISite object with the given parameters.

        Args:
            site (str): The site identifier. Required.
            instrument (str): The instrument identifier. Required.
            location_type (str): The type of location for the site. Required.

        """
        self.site = site.upper()
        self.instrument = instrument.lower()
        self.location_type = location_type.lower()

        self.site_info = OOISite.get_site_info(site=self.site, location_type=self.location_type)
        self.instrument_info = OOISite.get_instrument_info(self.site_info, self.instrument)
        self.instrument_metadata = OOISite.get_instrument_metadata(self.instrument)

    @staticmethod
    def get_site_info(site: str, location_type: str) -> SiteInfo:
        """Validate that the site identifier and location type are valid and return the site information.

        Args:
            site (str): The site identifier. Required.
            location_type (str): The type of location for the site. Required.

        Returns:
            SiteInfo: A SiteInfo object containing the site information.

        Raises:
            ValueError: If the validation fails.

        """
        with (Path(__file__).parent / "site_info.yaml").open("r", encoding="utf-8") as f:
            sites_info: dict[str, SiteInfo] = yaml.safe_load(f)

        site_info = sites_info.get(site)
        if not site_info:
            valid_sites = ", ".join(sites_info.keys())
            msg = f"Invalid site identifier: '{site}'. Must be one of the following (case insensitive): {valid_sites}."
            raise ValueError(msg)
        if location_type != site_info["type"]:
            msg = f"Invalid location type: '{location_type}' for site '{site}'. Must be '{site_info['type']}'."
            raise ValueError(msg)

        return SiteInfo(
            name=site_info["name"],
            site=site_info["site"],
            description=site_info["description"],
            doi=site_info["doi"],
            lat=site_info["lat"],
            lon=site_info["lon"],
            water_depth=site_info["water_depth"],
            valid_depth=site_info["valid_depth"],
            type=site_info["type"],
            instruments=site_info["instruments"],
        )

    @staticmethod
    def get_instrument_info(site_info: SiteInfo, instrument: str) -> InstrumentInfo:
        """Validate that the instrument identifier is valid for the given site and return the reference designator, method, and stream for the instrument.

        Returns:
            InstrumentInfo: An InstrumentInfo object containing the instrument information.

        Raises:
            ValueError: If the validation fails.

        """
        instrument_info = site_info["instruments"].get(instrument)
        if not instrument_info:
            msg = f"No instruments found for site '{site_info['site']}'."
            raise ValueError(msg)
        if instrument not in site_info.get("instruments"):
            valid_instruments = ", ".join(list(site_info.get("instruments").keys()))
            msg = f"Invalid instrument identifier: '{instrument}' for site '{site_info['site']}'. Must be one of the following (case insensitive): {valid_instruments}."
            raise ValueError(msg)

        refdes = instrument_info["refdes"]
        method = instrument_info["method"]
        stream = instrument_info["stream"]
        variables = set(instrument_info["variables"])  # convert to set to de-duplicate just in case
        description = instrument_info["description"]
        title = instrument_info["title"]

        return InstrumentInfo(
            refdes=refdes, method=method, stream=stream, variables=variables, description=description, title=title
        )

    @staticmethod
    def get_instrument_metadata(instrument: str) -> dict[str, dict[str, str]]:
        with (Path(__file__).parent / "instrument_metadata.yaml").open("r", encoding="utf-8") as f:
            instruments_metadata: dict[str, dict[str, dict[str, str]]] = yaml.safe_load(f)
        instrument_metadata = instruments_metadata.get(instrument, {})
        common_metadata = instruments_metadata.get("common", {})
        instrument_metadata.update(common_metadata)
        return instrument_metadata

    def __repr__(self) -> str:
        """Represent the OOISite in code outputs (e.g., Python REPL).

        Returns:
            str: A string representation of the OOISite in the format 'OOISite(site=..., instrument=..., latitude=..., longitude=...)'.

        """
        lon_str = f"{-self.site_info['lon']:.0f}W" if self.site_info["lon"] < 0 else f"{self.site_info['lon']:.0f}E"
        lat_str = f"{-self.site_info['lat']:.0f}S" if self.site_info["lat"] < 0 else f"{self.site_info['lat']:.0f}N"
        return f"OOISite(site='{self.site}', instrument='{self.instrument}', latitude={lat_str}, longitude={lon_str})"

    def __str__(self) -> str:
        """Convert the OOISite to a formatted string. Accessed with str(ooisite).

        Returns:
            str: A string representation of the OOISite in the format 'OOI EA Site {site} {instrument}'.

        """
        return f"OOI EA Site {self.site} {self.instrument}"

    @property
    def search_url(self) -> str:
        """Construct the search URL for the OOI site based on its reference designator, method, and stream."""
        search_url_base = "https://thredds.dataexplorer.oceanobservatories.org/thredds/catalog/ooigoldcopy/public/"
        return (
            search_url_base
            + f"{self.instrument_info['refdes']}-{self.instrument_info['method']}-{self.instrument_info['stream']}"
            + "/catalog.html"
        )

    @property
    def file_name(self) -> str:
        """Convert the profiler site and name to a string format suitable for filenames in the format '{site}_{instrument}'."""
        return f"{self.site}_{self.instrument}"


class EnduranceArray(Downloader):
    """Class for downloading and processing OOI Endurance Array data from the THREDDS catalog."""

    QARTOD_DROP_FLAG = 4

    class ProfilerSites(StrEnum):
        """Representation for valid OOI profiler sites.

        Valid options are:

        - [`CE01ISSP`](https://oceanobservatories.org/site/ce01issp/)
        - [`CE02SHSP`](https://oceanobservatories.org/site/ce02shsp/)
        - [`CE04OSPS`](https://oceanobservatories.org/site/ce04osps/)
        - [`CE06ISSP`](https://oceanobservatories.org/site/ce06issp/)
        - [`CE07SHSP`](https://oceanobservatories.org/site/ce07shsp/)
        - [`CE09OSPM`](https://oceanobservatories.org/site/ce09ospm/)

        """

        CE01ISSP = "CE01ISSP"
        CE02SHSP = "CE02SHSP"
        CE04OSPS = "CE04OSPS"
        CE06ISSP = "CE06ISSP"
        CE07SHSP = "CE07SHSP"
        CE09OSPM = "CE09OSPM"

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

    def __init__(
        self,
        site: str | ProfilerSites | MooringSites,
        dataset: str,
        save_dir: str | None = None,
        save_file: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> None:
        """Initialize the EnduranceArray downloader.

        Args:
            site (str | ProfilerSites | MooringSites): The OOI site identifier. Can be a string or an instance of the ProfilerSites or MooringSites enums.
            dataset (str): The dataset name to download (e.g., "ctd", "oxygen", etc.).
            save_dir (str | None): Directory to save the downloaded dataset. If None, defaults to the current working directory.
            save_file (str | None): Filename to save the dataset. If None, defaults to a filename based on the site and dataset.
            start_date (str | None): Start date for filtering data in "YYYY-MM-DD" format. If None, defaults to the earliest available date.
            end_date (str | None): End date for filtering data in "YYYY-MM-DD" format. If None, defaults to the latest available date.

        """
        site = site.value if isinstance(site, self.ProfilerSites | self.MooringSites) else site.upper()
        if site in self.ProfilerSites:
            location_type = "profiler"
        elif site in self.MooringSites:
            location_type = "mooring"
        self.site = OOISite(site=site, instrument=dataset.lower(), location_type=location_type)

        self.search_url = self.site.search_url
        self.base_url = "https://thredds.dataexplorer.oceanobservatories.org/thredds/fileServer/"
        self.tag = self.site.instrument_info["refdes"] + r".*.nc$"  # setup regex for files we want

        if save_file is None:
            save_file = f"ooi_{location_type}_{site}_{dataset}.nc"

        super().__init__(
            save_file=save_file,
            save_dir=save_dir,
            start_date=start_date,
            end_date=end_date,
        )

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

    def _list_files(self) -> list[str]:
        """List the netCDF data files in a THREDDS catalog.

        Returns:
            array: list of files in the catalog with the URL path set relative to the catalog

        Raises:
            ValueError: If no files are found for the specified location and date range.

        """
        with requests.session() as s:
            page = s.get(self.search_url).text

        soup = BeautifulSoup(page, "html.parser")
        pattern = re.compile(self.tag)
        nc_files = []
        for node in soup.find_all("a"):
            href = node.get("href")
            if isinstance(href, str) and pattern.search(node.get_text()):
                nc_files.append(href)
        nc_files = [re.sub(r"catalog.html\?dataset=", "", file) for file in nc_files]
        nc_files = self._filter_dates(nc_files)
        if not nc_files:
            msg = f"No files found for {self.site} in the specified date range. Exiting download."
            raise ValueError(msg)
        return nc_files

    @staticmethod
    def _qc_check(ds: xr.Dataset, variables: set[str]) -> xr.Dataset:
        """Apply QC checks to the dataset, masking out values that fail the checks.

        Args:
            ds (xr.Dataset): The xarray Dataset containing the OOI EA data.
            variables (set[str]): A set of variable names to apply QC checks to.

        Returns:
            xr.Dataset: The xarray Dataset with QC checks applied, where values that fail the checks are masked out.

        """
        for var in variables:
            if var in ds and f"{var}_qartod_results" in ds:
                ds[var] = ds[var].where(ds[f"{var}_qartod_results"] != EnduranceArray.QARTOD_DROP_FLAG)
        return ds

    @staticmethod
    def _drop_unused_vars(ds: xr.Dataset, variables: set[str]) -> xr.Dataset:
        """Drop variables from the dataset that are not needed for processing or analysis.

        Args:
            ds (xr.Dataset): The xarray Dataset containing the OOI EA data.
            variables (set[str]): A set of variable names to keep in the dataset. All other variables will be dropped.

        Returns:
            xr.Dataset: The xarray Dataset with unused variables dropped.

        """
        vars_to_drop = set(ds.variables) - variables
        ds = ds.drop_vars(vars_to_drop, errors="ignore")
        return ds

    @staticmethod
    def _calculate_density(ds: xr.Dataset, lat: float, lon: float) -> xr.Dataset:
        """Calculate density from salinity and temperature for an OOI EA dataset.

        Args:
            ds (xr.Dataset): The raw xarray Dataset containing the OOI EA data.
            lat (float): The latitude of the data.
            lon (float): The longitude of the data.

        Returns:
            xr.Dataset: The processed xarray Dataset with calculated density and added metadata.

        """
        ds["sea_water_absolute_salinity"] = gsw.SA_from_SP(
            ds["sea_water_practical_salinity"], ds["sea_water_pressure"], lon, lat
        )
        ds["sea_water_conservative_temperature"] = gsw.CT_from_t(
            ds["sea_water_absolute_salinity"], ds["sea_water_temperature"], ds["sea_water_pressure"]
        )
        ds["sea_water_density"] = gsw.rho(
            ds["sea_water_absolute_salinity"], ds["sea_water_conservative_temperature"], ds["sea_water_pressure"]
        )

        return ds

    @staticmethod
    def _split_profiles(ds: xr.Dataset) -> list:
        """Split the data set into a list of individual profiles, where each profile is a collection of data from a single deployment and profile sequence.

        Args:
            ds (xr.Dataset): The xarray Dataset containing the profiler data.

        Returns:
            list[xr.Dataset]: A list of xarray Datasets, where each Dataset corresponds to a single profile.

        """
        # split the data into profiles, assuming at least 120 seconds between profiles
        dt = ds.where(ds["time"].diff("time") > np.timedelta64(120, "s"), drop=True).get_index("time")

        # process each profile, adding the results to a list of profiles
        profiles = []
        jback = np.timedelta64(30, "s")  # 30 second jump back to avoid collecting data from the following profile
        for i, d in enumerate(dt):
            # pull out the profile
            if i == 0:
                profile = ds.sel(time=slice(ds["time"].values[0], d - jback))
            else:
                profile = ds.sel(time=slice(dt[i - 1], d - jback))

            # add the profile to the list
            profiles.append(profile)

        # grab the last profile and append it to the list
        profile = ds.sel(time=slice(d, ds["time"].values[-1]))
        profiles.append(profile)
        return profiles

    @staticmethod
    def _bin_dataset(
        ds: xr.Dataset, maximum_depth: float, z_lab: str = "depth", t_lab: str = "time"
    ) -> xr.Dataset | None:
        """Bins a profiler time series into depth bins.

        Args:
            ds (xr.dataset): OOI profiler dataset
            maximum_depth (float): The maximum depth to include in the binned dataset.
            z_lab (str, optional): name of depth/pressure in dataset. Defaults to 'depth'.
            t_lab (str, optional): name of time in dataset. Defaults to 'time'.

        Returns:
            xr.dataset: binned dataset

        """
        if z_lab not in ds or t_lab not in ds:
            return None
        maximum_depth = np.min(
            [maximum_depth, ds[z_lab].max().values]
        )  # lesser of maximum valid depth and maximum depth in dataset
        # use 1 meter depth bins
        step = 1
        # find minimum and maximum depth bins over all data
        depth_min = -step / 2  # want centers to be integer depths, so need to start edges at step / 2 before 0
        depth_max = maximum_depth + step / 2
        # need to go past stop by step due to exclusive end range
        depth_bins = np.arange(depth_min, depth_max + step, step)

        time_min = ds[t_lab].min().values.astype("datetime64[D]")
        # add one day to include the last day in the range
        time_max = ds[t_lab].max().values.astype("datetime64[D]") + np.timedelta64(1, "D")
        time_bins = xr.date_range(start=time_min, end=time_max, freq="1D")

        # one last catch for datetime or string types which will break the binning, though these should be removed in _drop_unused_vars
        types = [ds[i].dtype for i in ds]
        var_names = list(ds.keys())
        exclude = []
        for i, t in enumerate(types):
            if not (np.issubdtype(t, np.number)):
                exclude.append(var_names[i])
        ds = ds.drop_vars(exclude, errors="ignore")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # very fast binning
            ds: xr.Dataset = xarray_reduce(
                ds,
                ds[t_lab],
                ds[z_lab],
                func="mean",
                expected_groups=(time_bins, depth_bins),
                isbin=[True, True],
                method="map-reduce",
                skipna=True,
            )

        # manually re-assign coordinates to remove _bin coordinates and replace them with bin centers for depth and left edges for time
        ds[z_lab] = xr.DataArray(
            [interval.mid for interval in ds.depth_bins.values],
            dims=[z_lab + "_bins"],
            attrs={"long_name": "Depth", "standard_name": "depth", "units": "m"},
        )
        ds[t_lab] = xr.DataArray(
            [interval.left for interval in ds.time_bins.values],
            dims=[t_lab + "_bins"],
            attrs={"long_name": "Time", "standard_name": "time"},
        )
        ds = ds.swap_dims({z_lab + "_bins": z_lab, t_lab + "_bins": t_lab})
        ds = ds.drop_vars([z_lab + "_bins", t_lab + "_bins"], errors="ignore")

        return ds

    def _update_variable_metadata(self, ds: xr.Dataset) -> xr.Dataset:
        for var in ds.variables:
            comment = ds[var].attrs.get("comment", "")
            ds[var].attrs.clear()
            if var in self.site.instrument_metadata:
                ds[var].attrs.update(self.site.instrument_metadata.get(var, {}))
                if comment:
                    ds[var].attrs["comment"] = comment
        return ds

    def _update_global_metadata(self, ds: xr.Dataset) -> xr.Dataset:
        existing_history = ds.attrs.pop("history", "")
        title = f"{self.site.instrument_info['title']} {self.site.site_info['name']} {self.site.site_info['site']}"
        instrument_description = self.site.instrument_info["description"]
        site_description = f"Data is collected by a {self.site.site_info['description']}."
        if self.site.location_type == "mooring":
            description = f"{instrument_description} {site_description} Data associated with QARTOD flags indicating bad data (4) is removed and remaining data is binned in 1 day time bins."
        else:
            description = f"{instrument_description} {site_description} Data associated with QARTOD flags indicating bad data (4) is removed and remaining data is binned in 1 meter depth bins and 1 day time bins."
        ds.attrs.update(
            {
                "title": title,
                "description": description,
                "doi": self.site.site_info["doi"],
                "institution": "NSF Ocean Observatories Initiative Endurance Array",
                "Conventions": "CF-1.6",
                "date_modified": datetime.now(UTC).isoformat(timespec="minutes"),
                "history": existing_history
                + "\n"
                + f"{datetime.now(UTC).isoformat(timespec='minutes')} Downloaded and processed data using physoce-datasets (https://github.com/physoce/physoce-datasets)",
                "source": f"{self.site.instrument_info['refdes']}-{self.site.instrument_info['method']}-{self.site.instrument_info['stream']}",
                "time_coverage_start": ds["time"].min().dt.strftime("%Y-%m-%dT%H:%M:%SZ").item(),
                "time_coverage_end": ds["time"].max().dt.strftime("%Y-%m-%dT%H:%M:%SZ").item(),
                "geospatial_lat_min": self.site.site_info["lat"],
                "geospatial_lat_max": self.site.site_info["lat"],
                "geospatial_lat_units": "degrees_north",
                "geospatial_lon_min": self.site.site_info["lon"],
                "geospatial_lon_max": self.site.site_info["lon"],
                "geospatial_lon_units": "degrees_east",
            },
        )
        if self.site.location_type == "profiler":
            ds.attrs.update(
                {
                    "geospatial_vertical_min": 0,
                    "geospatial_vertical_max": np.min([self.site.site_info["valid_depth"], ds["depth"].max().values]),
                    "geospatial_vertical_positive": "down",
                    "geospatial_vertical_units": "m",
                }
            )
        return ds

    def _update_coords(self, ds: xr.Dataset) -> xr.Dataset:
        """Update the coordinates of the dataset to include latitude, longitude, and site information.

        Args:
            ds (xr.Dataset): The xarray Dataset containing the OOI EA data.

        Returns:
            xr.Dataset: The xarray Dataset with updated coordinates.

        """
        ds = ds.assign_coords(
            {
                "latitude": self.site.site_info["lat"],
                "longitude": self.site.site_info["lon"],
                "site": self.site.site_info["site"],
            }
        )
        return ds

    def download(self) -> None:
        """Download the OOI Endurance Array dataset for the specified site, instrument, and date range."""
        download_urls = [self.base_url + f + "#mode=bytes" for f in self._list_files()]

        logger.info(f"Downloading files for {self.site}...")
        ds_list: list[xr.Dataset] = []
        for f in tqdm(download_urls, desc="Downloading and processing datasets"):
            r = requests.get(f, timeout=(3.05, 120))
            if r.ok:
                ds = xr.open_dataset(io.BytesIO(r.content))
                ds = ds.swap_dims({"obs": "time"}).squeeze()
                ds = self._qc_check(ds, variables=self.site.instrument_info["variables"])
                ds = self._drop_unused_vars(ds, self.site.instrument_info["variables"])
                if self.site.instrument == "ctd":
                    ds = self._calculate_density(ds, lat=self.site.site_info["lat"], lon=self.site.site_info["lon"])
                if self.site.location_type == "profiler":
                    ds = self._bin_dataset(ds, maximum_depth=self.site.site_info["valid_depth"])
                    # only happens if dataset does not have a valid depth dimension for binning, which surprisingly happens!
                    if ds is None:
                        continue
                ds_list.append(ds)

        ds_concat = xr.concat(ds_list, dim="time", join="outer")
        ds_concat = ds_concat.sortby("time")  # ensure data is sorted by time after merging
        # take daily mean after merging to handle deployments that overlap in end time date
        ds_concat = ds_concat.resample(time="1D").mean()
        ds_concat = ds_concat.sel(
            time=slice(self.start_date, self.end_date)
        )  # subset to specified date range after merging

        ds_concat = self._update_variable_metadata(ds_concat)
        ds_concat = self._update_global_metadata(ds_concat)
        ds_concat = self._update_coords(ds_concat)

        ds_concat.to_netcdf(self.save_file_path)
        logger.info(f"Download complete! Dataset saved to {self.save_file_path}")

    @staticmethod
    def list_sites() -> list[str]:
        """List all valid OOI Endurance Array sites, including both profiler and mooring sites.

        Returns:
            list[str]: A list of valid OOI site identifiers.

        """
        return [site.value for site in EnduranceArray.ProfilerSites] + [
            site.value for site in EnduranceArray.MooringSites
        ]

    @staticmethod
    def list_datasets(site: str | ProfilerSites | MooringSites) -> list[str]:
        """List the instruments available for a given OOI Endurance Array site.

        Args:
            site (str | ProfilerSites | MooringSites): The OOI site identifier. Can be a string or an instance of ProfilerSites or MooringSites.

        Returns:
            list[str]: A list of instrument identifiers available for the specified site.

        Raises:
            ValueError: If the site identifier is invalid.

        """
        with (Path(__file__).parent / "site_info.yaml").open("r", encoding="utf-8") as f:
            sites_info: dict[str, SiteInfo] = yaml.safe_load(f)
        site = (
            site.value if isinstance(site, EnduranceArray.ProfilerSites | EnduranceArray.MooringSites) else site.upper()
        )

        site_info = sites_info.get(site)
        if not site_info:
            valid_sites = ", ".join(sites_info.keys())
            msg = f"Invalid site identifier: '{site}'. Must be one of the following (case insensitive): {valid_sites}."
            raise ValueError(msg)

        return list(site_info.get("instruments", {}).keys())
