"""Module for downloading datasets from the OOI Endurance Array Mooring."""

import io
import re
import warnings
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, cast

import gsw
import numpy as np
import requests
import xarray as xr
from bs4 import BeautifulSoup
from flox.xarray import xarray_reduce
from tqdm import tqdm

from physoce_datasets.logging import logger

from ._base import _Downloader

if TYPE_CHECKING:
    from physoce_datasets.util import OOIMooring, OOIProfiler

# only drop fully failing flags as suspect flag 3 can be valid data
QARTOD_DROP_FLAG = 4

# variables to drop from OOI EA CTD datasets, used in processing after download
VARIABLES_TO_DROP = [
    "obs",
    # these times should all be the same as the time dimension
    "driver_timestamp",
    "ctd_time",
    "internal_timestamp",
    "ingestion_timestamp",
    "port_timestamp",
    "preferred_timestamp",
    "suspect_timestamp",
    "id",
    # don't need conductivity
    "conductivity",
    "sea_water_electrical_conductivity",
    # will calculate density ourselves
    "density",
    "sea_water_density",
    # qc executed
    "pressure_qc_executed",
    "conductivity_qc_executed",
    "sea_water_pressure_qc_executed",
    "sea_water_electrical_conductivity_qc_executed",
    "sea_water_practical_salinity_qc_executed",
    "sea_water_temperature_qc_executed",
    "sea_water_density_qc_executed",
    # qc results
    "pressure_qc_results",
    "sea_water_pressure_qc_results",
    "sea_water_electrical_conductivity_qc_results",
    "sea_water_practical_salinity_qc_results",
    "sea_water_temperature_qc_results",
    "sea_water_density_qc_results",
    # qartod results
    "sea_water_pressure_qartod_results",
    "sea_water_electrical_conductivity_qartod_results",
    "sea_water_practical_salinity_qartod_results",
    "sea_water_temperature_qartod_results",
    # qartod executed
    "sea_water_pressure_qartod_executed",
    "sea_water_electrical_conductivity_qartod_executed",
    "sea_water_practical_salinity_qartod_executed",
    "sea_water_temperature_qartod_executed",
    # unproccessed L0 variables
    "conductivity",
    "temperature",
    "provenance",
    # unprocessed temperature from the pressure sensor, used by OOI to calculate output params but not necessary here
    "pressure_temp",
]


class _OOIBase(_Downloader):
    """Base class for downloading OOI Endurance Array datasets, containing shared methods and attributes for both mooring and profiler datasets."""

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
        super().__init__(
            location=location,
            location_type=location_type,
            save_file_prefix=save_file_prefix,
            save_dir=save_dir,
            save_file=save_file,
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


class EAProfilerDownloader(_OOIBase):
    """A class to download OOI Endurance Array Profiler CTD datasets with calculated stratification and mixed layer depth."""

    def __init__(
        self,
        location: str,
        start_date: str | None = None,
        end_date: str | None = None,
        save_dir: str | None = None,
        save_file: str | None = None,
    ) -> None:
        """Initialize the EAProfilerDownloader with parameters for downloading.

        Args:
            location (str): Location for the dataset in the form of a site identifier (e.g., "CE01ISSM"). Must be one of the following: "CE01ISSM", "CE02SHSM", "CE04OSSM", "CE06ISSM", "CE07SHSM", "CE09OSSM". Required.
            start_date (str | None): The start date for the dataset in "YYYY-MM-DD" format. If None, defaults to "2000-01-01".
            end_date (str | None): The end date for the dataset in "YYYY-MM-DD" format. If None, defaults to the current date.
            save_dir (str | None): The directory to save the downloaded dataset. If None, defaults to a "data" directory in the current working directory.
            save_file (str | None): The file name to save the downloaded dataset. If None, defaults to a name based on the dataset and date range.

        """
        super().__init__(
            location=location,
            location_type="profiler",
            save_file_prefix=f"ooi_profiler_{location.lower()}_ctd",
            save_dir=save_dir,
            save_file=save_file,
            start_date=start_date,
            end_date=end_date,
        )
        self.location = cast("OOIProfiler", self.location)
        self.search_url = self.location.search_url
        self.base_url = "https://thredds.dataexplorer.oceanobservatories.org/thredds/fileServer/"
        self.tag = self.location.refdes + r".*.nc$"  # setup regex for files we want

    @staticmethod
    def _update_metadata(ds: xr.Dataset) -> xr.Dataset:
        """Update the metadata of the OOI EA dataset to be CF-compliant and include necessary attributes.

        Args:
            ds (xr.Dataset): The xarray Dataset containing the OOI EA data.

        Returns:
            xr.Dataset: The xarray Dataset with updated metadata.

        """
        ds["sea_water_pressure"].attrs.update(
            {
                "long_name": "Sea Water Pressure",
                "standard_name": "sea_water_pressure",
                "units": "dbar",
            }
        )
        ds["sea_water_temperature"].attrs.update(
            {
                "long_name": "Sea Water Temperature",
                "standard_name": "sea_water_temperature",
                "units": "degree_C",
            }
        )
        ds["sea_water_practical_salinity"].attrs.update(
            {
                "long_name": "Sea Water Practical Salinity",
                "standard_name": "sea_water_practical_salinity",
                "units": "dimensionless",
            }
        )
        ds["sea_water_absolute_salinity"].attrs.update(
            {
                "long_name": "Sea Water Absolute Salinity",
                "standard_name": "sea_water_absolute_salinity",
                "units": "g/kg",
            }
        )
        ds["sea_water_conservative_temperature"].attrs.update(
            {
                "long_name": "Sea Water Conservative Temperature",
                "standard_name": "sea_water_conservative_temperature",
                "units": "degree_C",
            }
        )
        ds["sea_water_density"].attrs.update(
            {
                "long_name": "Sea Water Density",
                "standard_name": "sea_water_density",
                "units": "kg/m^3",
            }
        )
        ds["mixed_layer_depth_from_density"].attrs.update(
            {
                "long_name": "Mixed Layer Depth From Density",
                "standard_name": "sea_water_mixed_layer_depth_from_density",
                "units": "m",
                "notes": "Calculated using a threshold method based on the depth that is 0.03 kg/m^3 denser than the surface value, where the surface value is defined by the mean of the upper 5 meters.",
            }
        )
        ds["mixed_layer_depth_from_temperature"].attrs.update(
            {
                "long_name": "Mixed Layer Depth From Temperature",
                "standard_name": "sea_water_mixed_layer_depth_from_temperature",
                "units": "m",
                "notes": "Calculated using a threshold method based on the depth that is 0.2 degree C colder than the surface value, where the surface value is defined by the mean of the upper 5 meters.",
            }
        )
        ds["n_squared"].attrs.update(
            {
                "long_name": "N Squared",
                "standard_name": "square_of_brunt_vaisala_frequency_in_sea_water",
                "units": "1/s^2",
                "description": "A measure of the stratification of the water column, calculated from the vertical gradient of density.",
            }
        )
        existing_history = ds.attrs.pop("history", "")
        ds.attrs.update(
            {
                "description": "CTD data from OOI Endurance Array surface moorings, obtained via the OOI THREDDS catalog. This dataset includes processed variables such as absolute salinity, conservative temperature, and density calculated from the L2 derived variables of practical salinity, temperature, and pressure provided from OOI.",
                "last updated": datetime.now(UTC).isoformat(timespec="minutes"),
                "history": existing_history
                + "\n"
                + f"{datetime.now(UTC).isoformat(timespec='minutes')} Downloaded and processed data using physoce-datasets (https://github.com/physoce/physoce-datasets)",
                "time_coverage_start": ds["time"].min().dt.strftime("%Y-%m-%dT%H:%M:%SZ").item(),
                "time_coverage_end": ds["time"].max().dt.strftime("%Y-%m-%dT%H:%M:%SZ").item(),
            },
        )
        return ds

    def _calculate_density(self, ds: xr.Dataset) -> xr.Dataset:
        """Calculate density from salinity and temperature for an OOI EA dataset.

        Args:
            ds (xr.Dataset): The raw xarray Dataset containing the OOI EA data.

        Returns:
            xr.Dataset: The processed xarray Dataset with calculated density and added metadata.

        """
        ds = ds.swap_dims({"obs": "time"})

        ds["sea_water_pressure"] = ds["sea_water_pressure"].where(
            ds["sea_water_pressure_qartod_results"] != QARTOD_DROP_FLAG
        )
        ds["sea_water_temperature"] = ds["sea_water_temperature"].where(
            ds["sea_water_temperature_qartod_results"] != QARTOD_DROP_FLAG
        )
        ds["sea_water_practical_salinity"] = ds["sea_water_practical_salinity"].where(
            ds["sea_water_practical_salinity_qartod_results"] != QARTOD_DROP_FLAG
        )

        ds = ds.drop_vars(VARIABLES_TO_DROP, errors="ignore")

        ds["sea_water_absolute_salinity"] = gsw.SA_from_SP(
            ds["sea_water_practical_salinity"], ds["sea_water_pressure"], self.location.lon, self.location.lat
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
    def _bin_profiles(ds: xr.Dataset, z_lab: str = "depth", t_lab: str = "time") -> xr.DataArray | xr.Dataset:
        """Bins a profiler time series into depth bins.

        Args:
            ds (xr.dataset): OOI profiler dataset
            z (array): edges of depth/pressure bins
            z_lab (str, optional): name of depth/pressure in dataset. Defaults to 'depth'.
            t_lab (str, optional): name of time in dataset. Defaults to 'time'.

        Returns:
            xr.dataset: binned dataset

        """
        # setup 1 meter depth bins
        step = 1
        # find minimum and maximum depth bins over all data
        depth_min = (
            np.floor(np.min(ds["depth"])) - step / 2
        )  # want centers to be integer depths, so need to start edges at step / 2 before min depth
        depth_max = np.ceil(np.max(ds["depth"])) + step / 2  # same as above
        # depth_bins is edges of bins
        depth_bins = np.arange(
            depth_min, depth_max + step, step
        )  # need to go past by step for stop due to exclusive end range

        # one last catch for datetime or string types which will break the binning, though these should be removed in processing
        types = [ds[i].dtype for i in ds]
        var_names = list(ds.keys())
        exclude = []
        for i, t in enumerate(types):
            if not (np.issubdtype(t, np.number)):
                exclude.append(var_names[i])
        ds = ds.drop_vars(exclude)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # handle ctd data without offset
            ds: xr.Dataset = xarray_reduce(
                ds,
                ds[t_lab],
                ds[z_lab],
                func="nanmean",
                expected_groups=(None, depth_bins),
                isbin=[False, True],
                method="map-reduce",
                skipna=True,
            )

        # manually re-assign coordinates to remove _bin coordinates and replace them with bin centers
        depth = np.array([x.mid for x in ds.depth_bins.values])
        ds[z_lab] = ([z_lab + "_bins"], depth)
        ds = ds.swap_dims({z_lab + "_bins": z_lab})
        ds = ds.drop_vars([z_lab + "_bins"])
        time_mean = ds[t_lab].mean().values
        ds = ds.mean(dim=t_lab)  # average over time dimension to get one profile per deployment
        ds = ds.expand_dims({t_lab: [time_mean]})  # add time dimension back in with mean time for the profile

        return ds

    @staticmethod
    def _bin_dataset(ds: xr.Dataset, z_lab: str = "depth", t_lab: str = "time") -> xr.Dataset:
        """Split a dataset into profiles, bin them, and recombine.

        Args:
            ds (xr.Dataset): The xarray Dataset containing the profiler data.
            z_lab (str, optional): The name of the depth variable in the dataset. Defaults to "depth".
            t_lab (str, optional): The name of the time variable in the dataset. Defaults to "time".

        Returns:
            xr.Dataset: The binned xarray Dataset.

        """
        profiles = EAProfilerDownloader._split_profiles(ds)
        binned_profiles = [EAProfilerDownloader._bin_profiles(p, z_lab=z_lab, t_lab=t_lab) for p in profiles]
        ds_binned = xr.concat(binned_profiles, dim="time", join="outer")
        return ds_binned

    @staticmethod
    def threshold_mld(
        variable: xr.DataArray, threshold_type: Literal["temperature", "density"], threshold: float
    ) -> xr.DataArray:
        """Interpolate depth to a threshold value of the variable using linear interpolation between the two bounding depth levels.

        Args:
            variable (xr.DataArray): 2D array of variable values with dimensions (time, depth)
            threshold_type (str): "temperature" or "density", determines whether to look for first value less than or greater than threshold
            threshold (float): difference from surface value to define threshold for MLD calculation, always positive

        Returns:
            xr.DataArray: 1D array of interpolated depth values at the threshold for each time step

        """
        # need to treat temp and density differently since temp decreases with depth and density increases with depth
        # ensure threshold is positive and flip sign for density since it increases with depth
        threshold = abs(threshold)
        threshold = threshold if threshold_type == "density" else -threshold
        # find threshold value using the top 5 meters of the profile as the surface value
        threshold_target = variable.isel(depth=slice(0, 5)).mean(dim="depth") + threshold

        # find indices of bounding depth levels for interpolation
        if threshold_type == "temperature":
            # index of first greater than target, since argmax finds first True
            hi = np.argmax(variable.values <= threshold_target.values[:, None], axis=1)
        elif threshold_type == "density":
            # index of first less than target, since argmax finds first True
            hi = np.argmax(variable.values >= threshold_target.values[:, None], axis=1)

        hi = np.clip(hi, 1, variable["depth"].size - 1)  # ensure hi is at least 1 and at most the last index
        # hi is high in the index sense, not the real depth space sense, so low index is hi - 1
        lo = hi - 1

        # Get values at bounding indices
        d0 = variable["depth"].values[lo]
        d1 = variable["depth"].values[hi]
        v0 = variable.values[np.arange(variable.values.shape[0]), lo]
        v1 = variable.values[np.arange(variable.values.shape[0]), hi]

        # Slope is rise over run
        slope = (d1 - d0) / (v1 - v0)
        out = d0 + slope * (threshold_target.values - v0)

        out = xr.DataArray(out, coords={"time": variable.coords["time"]}, dims=["time"])

        # mask out profiles that have all nans in the upper 5 meters since we can't calculate a threshold for those
        all_surface_nans = variable.isel(depth=slice(0, 5)).isnull().all(dim="depth")

        return out[~all_surface_nans]

    def _calculate_stratification(self, ds: xr.Dataset) -> xr.Dataset:
        """Calculate the Brunt-Vaisala frequency (n_squared) from density profiles in the dataset.

        Args:
            ds (xr.Dataset): The xarray Dataset containing the profiler data with a "sea_water_density" and "sea_water_pressure" variable.

        Returns:
            xr.DataArray: The xarray DataArray containing stratification data.

        """
        n_squared = np.sqrt(
            (gsw.grav(self.location.lat, ds["sea_water_pressure"]) / ds["sea_water_density"])
            * ds["sea_water_density"].differentiate("depth", edge_order=2)
        )
        return n_squared

    def download(self) -> None:
        """Download the netCDF data files from the THREDDS catalog and save them to a local directory."""
        logger.info(f"Getting list of data files for {self.location}...")
        nc_files = self._list_files(self.search_url, self.tag)
        if not nc_files:
            logger.warning(f"No files found for {self.location} in the specified date range. Exiting download.")
            return
        download_urls = [self.base_url + f + "#mode=bytes" for f in nc_files]

        logger.info(f"Downloading files for {self.location}...")
        ds = []
        for i, f in enumerate(tqdm(download_urls, desc="Downloading datasets")):
            r = requests.get(f, timeout=(3.05, 120))
            if r.ok:
                ds.append(xr.open_dataset(io.BytesIO(r.content)))
                ds[i].load()

        ds = [self._calculate_density(di) for di in ds]
        ds_merged = xr.merge(ds, compat="no_conflicts", join="outer")
        ds_merged = ds_merged.sortby("time")  # ensure data is sorted by time after merging
        ds_merged = ds_merged.sel(
            time=slice(self.start_date, self.end_date)
        )  # subset to specified date range after merging

        ds_binned = self._bin_dataset(ds_merged)

        # interpolate up to 5 meters
        ds_binned = ds_binned.interpolate_na(dim="depth", method="linear", use_coordinate=True, max_gap=5)
        # interpolate up to 1 day
        ds_binned = ds_binned.interpolate_na(
            dim="time", method="linear", use_coordinate=True, max_gap=np.timedelta64(1, "D")
        )
        # calculate mixed layer depth using a density threshold of 0.03 kg/m^3
        ds_binned["mixed_layer_depth_from_density"] = self.threshold_mld(
            ds_binned["sea_water_density"], threshold_type="density", threshold=0.03
        )
        # calculate mixed layer depth using a temperature threshold of 0.2 degree C
        ds_binned["mixed_layer_depth_from_temperature"] = self.threshold_mld(
            ds_binned["sea_water_temperature"], threshold_type="temperature", threshold=0.2
        )
        # calculate stratification
        ds_binned["n_squared"] = self._calculate_stratification(ds_binned)
        # now take daily mean
        ds_binned = ds_binned.resample(time="1D").mean()

        ds_binned = ds_binned.assign_coords(
            {
                "latitude": self.location.lat,
                "longitude": self.location.lon,
                "site": self.location.site,  # ty:ignore[unresolved-attribute] location is OOIMooring or OOIProfiler, both of which have a site property
            }
        )
        ds_binned = self._update_metadata(ds_binned)

        ds_binned.to_netcdf(self.save_file_path)
        logger.info(f"Download complete! Dataset saved to {self.save_file_path}")


class EAMooringDownloader(_OOIBase):
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
            location_type="mooring",
            save_file_prefix=f"ooi_mooring_{location.lower()}_ctd",
            save_dir=save_dir,
            save_file=save_file,
            start_date=start_date,
            end_date=end_date,
        )
        self.location = cast("OOIMooring", self.location)
        self.search_url = self.location.search_url
        self.base_url = "https://thredds.dataexplorer.oceanobservatories.org/thredds/fileServer/"
        self.tag = self.location.refdes + r".*.nc$"  # setup regex for files we want

    @staticmethod
    def _update_metadata(ds: xr.Dataset) -> xr.Dataset:
        """Update the metadata of the OOI EA dataset to be CF-compliant and include necessary attributes.

        Args:
            ds (xr.Dataset): The xarray Dataset containing the OOI EA data.

        Returns:
            xr.Dataset: The xarray Dataset with updated metadata.

        """
        ds["sea_water_pressure"].attrs.update(
            {
                "long_name": "Sea Water Pressure",
                "standard_name": "sea_water_pressure",
                "units": "dbar",
            }
        )
        ds["sea_water_temperature"].attrs.update(
            {
                "long_name": "Sea Water Temperature",
                "standard_name": "sea_water_temperature",
                "units": "degree_C",
            }
        )
        ds["sea_water_practical_salinity"].attrs.update(
            {
                "long_name": "Sea Water Practical Salinity",
                "standard_name": "sea_water_practical_salinity",
                "units": "dimensionless",
            }
        )
        ds["sea_water_absolute_salinity"].attrs.update(
            {
                "long_name": "Sea Water Absolute Salinity",
                "standard_name": "sea_water_absolute_salinity",
                "units": "g/kg",
            }
        )
        ds["sea_water_conservative_temperature"].attrs.update(
            {
                "long_name": "Sea Water Conservative Temperature",
                "standard_name": "sea_water_conservative_temperature",
                "units": "degree_C",
            }
        )
        ds["sea_water_density"].attrs.update(
            {
                "long_name": "Sea Water Density",
                "standard_name": "sea_water_density",
                "units": "kg/m^3",
            }
        )
        existing_history = ds.attrs.pop("history", "")
        ds.attrs.update(
            {
                "description": "CTD data from OOI Endurance Array surface moorings, obtained via the OOI THREDDS catalog. This dataset includes processed variables such as absolute salinity, conservative temperature, and density calculated from the L2 derived variables of practical salinity, temperature, and pressure provided from OOI.",
                "last updated": datetime.now(UTC).isoformat(timespec="minutes"),
                "history": existing_history
                + "\n"
                + f"{datetime.now(UTC).isoformat(timespec='minutes')} Downloaded and processed data using physoce-datasets (https://github.com/physoce/physoce-datasets)",
                "time_coverage_start": ds["time"].min().dt.strftime("%Y-%m-%dT%H:%M:%SZ").item(),
                "time_coverage_end": ds["time"].max().dt.strftime("%Y-%m-%dT%H:%M:%SZ").item(),
            },
        )
        return ds

    def _process(self, ds: xr.Dataset) -> xr.Dataset:
        """Process the raw OOI EA dataset by calculating density and adding metadata.

        Args:
            ds (xr.Dataset): The raw xarray Dataset containing the OOI EA data.

        Returns:
            xr.Dataset: The processed xarray Dataset with calculated density and added metadata.

        """
        ds = ds.swap_dims({"obs": "time"})

        ds["sea_water_pressure"] = ds["sea_water_pressure"].where(
            ds["sea_water_pressure_qartod_results"] != QARTOD_DROP_FLAG
        )
        ds["sea_water_temperature"] = ds["sea_water_temperature"].where(
            ds["sea_water_temperature_qartod_results"] != QARTOD_DROP_FLAG
        )
        ds["sea_water_practical_salinity"] = ds["sea_water_practical_salinity"].where(
            ds["sea_water_practical_salinity_qartod_results"] != QARTOD_DROP_FLAG
        )

        ds = ds.drop_vars(VARIABLES_TO_DROP, errors="ignore")

        # interpolate over gaps of up to one day
        ds = ds.interpolate_na("time", method="linear", use_coordinate=True, max_gap=np.timedelta64(1, "D"))

        ds["sea_water_absolute_salinity"] = gsw.SA_from_SP(
            ds["sea_water_practical_salinity"], ds["sea_water_pressure"], self.location.lon, self.location.lat
        )
        ds["sea_water_conservative_temperature"] = gsw.CT_from_t(
            ds["sea_water_temperature"], ds["sea_water_pressure"], ds["sea_water_absolute_salinity"]
        )
        ds["sea_water_density"] = gsw.rho(
            ds["sea_water_absolute_salinity"], ds["sea_water_conservative_temperature"], ds["sea_water_pressure"]
        )

        # take daily mean
        ds = ds.resample(time="1D").mean()

        return ds

    def download(self) -> None:
        """Download the netCDF data files from the THREDDS catalog and save them to a local directory."""
        logger.info(f"Getting list of data files for {self.location}...")
        nc_files = self._list_files(self.search_url, self.tag)
        if not nc_files:
            logger.warning(f"No files found for {self.location} in the specified date range. Exiting download.")
            return
        download_urls = [self.base_url + f + "#mode=bytes" for f in nc_files]

        logger.info(f"Downloading files for {self.location}...")
        ds = []
        for i, f in enumerate(tqdm(download_urls, desc="Downloading datasets")):
            r = requests.get(f, timeout=(3.05, 120))
            if r.ok:
                ds.append(xr.open_dataset(io.BytesIO(r.content)))
                ds[i].load()

        ds = [self._process(di) for di in ds]
        ds_merged = xr.merge(ds, compat="no_conflicts", join="outer")
        ds_merged = ds_merged.sortby("time")  # ensure data is sorted by time after merging
        ds_merged = ds_merged.sel(
            time=slice(self.start_date, self.end_date)
        )  # subset to specified date range after merging
        ds_merged = ds_merged.assign_coords(
            {
                "latitude": self.location.lat,
                "longitude": self.location.lon,
                "site": self.location.site,  # ty:ignore[unresolved-attribute] location is OOIMooring or OOIProfiler, both of which have a site property
            }
        )
        ds_merged = self._update_metadata(ds_merged)

        ds_merged.to_netcdf(self.save_file_path)
        logger.info(f"Download complete! Dataset saved to {self.save_file_path}")
