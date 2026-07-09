"""Download and process [OOI Endurance Array data](https://oceanobservatories.org/array/coastal-endurance/)."""

import io
import re
import warnings
from datetime import UTC, datetime
from typing import Literal

import gsw
import numpy as np
import requests
import xarray as xr
from bs4 import BeautifulSoup
from flox.xarray import xarray_reduce
from tqdm import tqdm

from physoce_datasets.logging import logger

from ._base import _Downloader
from ._ooi_data import MooringSites, ProfilerSites, _OOISite

__all__ = ["MooringCTD", "ProfilerCTD", "ProfilerChlorophyll"]

# only drop fully failing flags as suspect flag 3 can be valid data
QARTOD_DROP_FLAG = 4

# variables to drop from OOI EA CTD datasets, used in processing after download
VARIABLES_TO_DROP = [
    "obs",
    # will manually add back latitude and longitude
    "lat",
    "lon",
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
    # qc executed
    "pressure_qc_executed",
    "conductivity_qc_executed",
    "sea_water_pressure_qc_executed",
    "sea_water_electrical_conductivity_qc_executed",
    "sea_water_practical_salinity_qc_executed",
    "sea_water_temperature_qc_executed",
    "sea_water_density_qc_executed",
    "fluorometric_chlorophyll_a_qc_executed",
    "fluorometric_cdom_qc_executed",
    "optical_backscatter_qc_executed",
    "total_volume_scattering_coefficient_qc_executed",
    # qc results
    "pressure_qc_results",
    "sea_water_pressure_qc_results",
    "sea_water_electrical_conductivity_qc_results",
    "sea_water_practical_salinity_qc_results",
    "sea_water_temperature_qc_results",
    "sea_water_density_qc_results",
    "fluorometric_chlorophyll_a_qc_results",
    "fluorometric_cdom_qc_results",
    "optical_backscatter_qc_results",
    "total_volume_scattering_coefficient_qc_results",
    # qartod results
    "sea_water_pressure_qartod_results",
    "sea_water_electrical_conductivity_qartod_results",
    "sea_water_practical_salinity_qartod_results",
    "sea_water_temperature_qartod_results",
    "fluorometric_chlorophyll_a_qartod_results",
    "fluorometric_cdom_qartod_results",
    "optical_backscatter_qartod_results",
    # qartod executed
    "sea_water_pressure_qartod_executed",
    "sea_water_electrical_conductivity_qartod_executed",
    "sea_water_practical_salinity_qartod_executed",
    "sea_water_temperature_qartod_executed",
    "fluorometric_chlorophyll_a_qartod_executed",
    "fluorometric_cdom_qartod_executed",
    "optical_backscatter_qartod_executed",
    # unproccessed L0 variables
    "conductivity",
    "temperature",
    "provenance",
    "pressure",
    "raw_signal_chl",
    "raw_signal_cdom",
    "raw_signal_beta",
    "raw_internal_temp",
    # unprocessed temperature from the pressure sensor, used by OOI to calculate output params but not necessary here
    "pressure_temp",
]


class _OOIBase(_Downloader):
    """Base class for downloading OOI Endurance Array datasets, containing shared methods and attributes for both mooring and profiler datasets."""

    def __init__(
        self,
        site: str | ProfilerSites | MooringSites,
        location_type: str,
        instrument: str,
        save_dir: str | None = None,
        save_file: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> None:
        site = site.value if isinstance(site, ProfilerSites | MooringSites) else site.upper()
        self.location = _OOISite(site=site, instrument=instrument, location_type=location_type)

        self.search_url = self.location.search_url
        self.base_url = "https://thredds.dataexplorer.oceanobservatories.org/thredds/fileServer/"
        self.tag = self.location.refdes + r".*.nc$"  # setup regex for files we want

        super().__init__(
            save_file_prefix=f"ooi_{location_type}_{site}_{instrument}",
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

    @staticmethod
    def _qc_check(ds: xr.Dataset, variables: list) -> xr.Dataset:
        """Apply QC checks to the dataset, masking out values that fail the checks.

        Args:
            ds (xr.Dataset): The xarray Dataset containing the OOI EA data.
            variables (list): A list of variable names to apply QC checks to.

        Returns:
            xr.Dataset: The xarray Dataset with QC checks applied, where values that fail the checks are masked out.

        """
        for var in variables:
            if var in ds:
                ds[var] = ds[var].where(ds[f"{var}_qartod_results"] != QARTOD_DROP_FLAG)
        return ds

    @staticmethod
    def _drop_unused_vars(ds: xr.Dataset) -> xr.Dataset:
        """Drop variables from the dataset that are not needed for processing or analysis.

        Args:
            ds (xr.Dataset): The xarray Dataset containing the OOI EA data.

        Returns:
            xr.Dataset: The xarray Dataset with unused variables dropped.

        """
        ds = ds.drop_vars(VARIABLES_TO_DROP, errors="ignore")
        return ds


class _ProfilerBase(_OOIBase):
    """Base class for downloading OOI Endurance Array Profiler datasets, containing shared methods and attributes for profiler datasets."""

    def __init__(
        self,
        site: str | ProfilerSites,
        instrument: str,
        save_dir: str | None = None,
        save_file: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> None:
        super().__init__(
            site=site,
            location_type="profiler",
            instrument=instrument,
            save_dir=save_dir,
            save_file=save_file,
            start_date=start_date,
            end_date=end_date,
        )

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

    def _bin_dataset(self, ds: xr.Dataset, z_lab: str = "depth", t_lab: str = "time") -> xr.Dataset:
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
        depth_min = -step / 2  # want centers to be integer depths, so need to start edges at step / 2 before 0
        depth_max = self.location.depth + step / 2
        depth_bins = np.arange(
            depth_min, depth_max + step, step
        )  # need to go past by step for stop due to exclusive end range

        time_min = ds[t_lab].min().values.astype("datetime64[D]")
        # add one day to include the last day in the range
        time_max = ds[t_lab].max().values.astype("datetime64[D]") + np.timedelta64(1, "D")
        time_bins = xr.date_range(start=time_min, end=time_max, freq="1D")

        # one last catch for datetime or string types which will break the binning, though these should be removed in processing
        types = [ds[i].dtype for i in ds]
        var_names = list(ds.keys())
        exclude = []
        for i, t in enumerate(types):
            if not (np.issubdtype(t, np.number)):
                exclude.append(var_names[i])
        ds = ds.drop_vars(exclude, errors="ignore")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # handle ctd data without offset
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

        # manually re-assign coordinates to remove _bin coordinates and replace them with bin centers
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


class ProfilerCTD(_ProfilerBase):
    """A class to download OOI Endurance Array Profiler CTD datasets with calculated stratification and mixed layer depth."""

    def __init__(
        self,
        site: str | ProfilerSites,
        start_date: str | None = None,
        end_date: str | None = None,
        save_dir: str | None = None,
        save_file: str | None = None,
    ) -> None:
        """Initialize the EAProfilerDownloader with parameters for downloading.

        Args:
            site: Site identifier for the dataset. Required.
            start_date: The start date for the dataset in "YYYY-MM-DD" format. If None, defaults to "2000-01-01".
            end_date: The end date for the dataset in "YYYY-MM-DD" format. If None, defaults to the current date.
            save_dir: The directory to save the downloaded dataset. If None, defaults to a "data" directory in the current working directory.
            save_file: The file name to save the downloaded dataset. If None, defaults to a name based on the dataset and date range.

        """
        super().__init__(
            site=site,
            instrument="ctd",
            save_dir=save_dir,
            save_file=save_file,
            start_date=start_date,
            end_date=end_date,
        )

    @staticmethod
    def _update_metadata(ds: xr.Dataset) -> xr.Dataset:
        """Update the metadata of the OOI EA dataset to be CF-compliant and include necessary attributes.

        Args:
            ds: The xarray Dataset containing the OOI EA data.

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
    def _interpolate_along_axis(
        variable: xr.DataArray, target: xr.DataArray, lo: xr.DataArray, hi: xr.DataArray
    ) -> xr.DataArray:
        """Interpolate depth to a target value of the variable using linear interpolation between two bounding depth indices.

        Args:
            variable (xr.DataArray): 2D array of variable values with dimensions (time, depth)
            target (xr.DataArray): 1D array of target values for each time step
            lo (xr.DataArray): 1D array of lower bounding depth indices for each time step
            hi (xr.DataArray): 1D array of upper bounding depth indices for each time step

        Returns:
            1D array of interpolated depth values at the threshold for each time step

        """
        # Get values at bounding indices
        d0 = variable["depth"].values[lo.values]
        d1 = variable["depth"].values[hi.values]
        v0 = variable.values[np.arange(variable.values.shape[0]), lo.values]
        v1 = variable.values[np.arange(variable.values.shape[0]), hi.values]

        # check if target_values is within the bounds of v0 and v1 for each profile
        threshold_in_bounds = (np.min(np.stack([v0, v1], axis=0), axis=0) < target.values) & (
            target.values < np.max(np.stack([v0, v1], axis=0), axis=0)
        )

        # Slope is rise over run
        slope = (d1 - d0) / (v1 - v0)
        out = d0 + slope * (target.values - v0)

        out = xr.DataArray(out, coords={"time": variable.coords["time"]}, dims=["time"])

        out = out.where(threshold_in_bounds, np.nan)  # set to nan if threshold is out of bounds

        return out

    def _threshold_mld(
        self, variable: xr.DataArray, threshold_type: Literal["temperature", "density"], threshold: float
    ) -> xr.DataArray:
        """Interpolate depth to a threshold value of the variable using linear interpolation between the two bounding depth levels.

        Args:
            variable (xr.DataArray): 2D array of variable values with dimensions (time, depth)
            threshold_type (str): "temperature" or "density", determines whether to look for first value less than or greater than threshold
            threshold (float): difference from surface value to define threshold for MLD calculation, always positive

        Returns:
            1D array of interpolated depth values at the threshold for each time step

        """
        # need to treat temp and density differently since temp decreases with depth and density increases with depth
        # ensure threshold is positive and flip sign for density since it increases with depth
        threshold = abs(threshold)
        threshold = threshold if threshold_type == "density" else -threshold
        # find threshold value using the top 2 to 7 meters of the profile as the surface value
        # this removes sometimes problematic surface spikes in the data that can throw off the MLD calculation
        threshold_target = variable.sel(depth=10, method="bfill") + threshold

        # find indices of bounding depth levels for interpolation
        if threshold_type == "temperature":
            # index of first greater than target, since argmax finds first True
            hi = (variable <= threshold_target).argmax("depth")
            all_false = (variable.sel(depth=slice(10, None)) > threshold_target).all(
                "depth"
            )  # find profiles with all values greater than target
        elif threshold_type == "density":
            # index of first less than target, since argmax finds first True
            hi = (variable >= threshold_target).argmax("depth")
            all_false = (variable.sel(depth=slice(10, None)) < threshold_target).all(
                "depth"
            )  # find profiles with all values less than target

        hi = hi.clip(min=1, max=variable["depth"].size - 1)  # ensure hi is at least 1 and at most the last index
        # hi is high in the index sense, not the real depth space sense, so low index is hi - 1
        lo = hi - 1

        out = self._interpolate_along_axis(variable, threshold_target, lo, hi)

        # for profiles where all values are greater than (temp) or less than (density) the threshold, set MLD to last valid depth value (deepest depth)
        out = out.where(~all_false, variable["depth"].max(dim="depth"))

        return out

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
        ds: list[xr.Dataset] = []
        for f in tqdm(download_urls, desc="Downloading datasets"):
            r = requests.get(f, timeout=(3.05, 120))
            if r.ok:
                ds.append(xr.open_dataset(io.BytesIO(r.content)))
                ds[-1] = ds[-1].swap_dims({"obs": "time"}).squeeze()
                ds[-1] = self._qc_check(
                    ds[-1], variables=["sea_water_pressure", "sea_water_temperature", "sea_water_practical_salinity"]
                )
                ds[-1] = self._drop_unused_vars(ds[-1])
                ds[-1] = self._calculate_density(ds[-1])
                ds[-1] = self._bin_dataset(ds[-1]).compute()

        ds_concat = xr.concat(ds, dim="time", join="outer")
        ds_concat = ds_concat.sortby("time")  # ensure data is sorted by time after merging
        ds_concat = ds_concat.resample(
            time="1D"
        ).mean()  # take daily mean after merging for deployments that overlap in end time date
        ds_concat = ds_concat.sel(
            time=slice(self.start_date, self.end_date)
        )  # subset to specified date range after merging

        # interpolate up to 5 meters
        ds_concat = ds_concat.interpolate_na(dim="depth", method="linear", use_coordinate=True, max_gap=5)
        # interpolate up to 1 day
        ds_concat = ds_concat.interpolate_na(
            dim="time", method="linear", use_coordinate=True, max_gap=np.timedelta64(1, "D")
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # calculate mixed layer depth using a density threshold of 0.03 kg/m^3
            ds_concat["mixed_layer_depth_from_density"] = self._threshold_mld(
                ds_concat["sea_water_density"], threshold_type="density", threshold=0.03
            )
            # calculate mixed layer depth using a temperature threshold of 0.2 degree C
            ds_concat["mixed_layer_depth_from_temperature"] = self._threshold_mld(
                ds_concat["sea_water_temperature"], threshold_type="temperature", threshold=0.2
            )
            # calculate stratification
            ds_concat["n_squared"] = self._calculate_stratification(ds_concat)

        ds_concat = ds_concat.assign_coords(
            {
                "latitude": self.location.lat,
                "longitude": self.location.lon,
                "site": self.location.site,
            }
        )
        ds_concat = self._update_metadata(ds_concat)

        ds_concat.to_netcdf(self.save_file_path)
        logger.info(f"Download complete! Dataset saved to {self.save_file_path}")


class ProfilerChlorophyll(_ProfilerBase):
    """A class to download OOI Endurance Array Profiler Chlorophyll datasets."""

    def __init__(
        self,
        site: str | ProfilerSites,
        start_date: str | None = None,
        end_date: str | None = None,
        save_dir: str | None = None,
        save_file: str | None = None,
    ) -> None:
        """Initialize the EAProfilerDownloader with parameters for downloading.

        Args:
            site: Site identifier for the dataset. Must be one of the following: `CE01ISSP`, `CE02SHSP`, `CE04OSPS`, `CE06ISSP`, `CE07SHSP`, `CE09OSPM`, `RS01SBPS`. Required.
            start_date: The start date for the dataset in "YYYY-MM-DD" format. If None, defaults to "2000-01-01".
            end_date: The end date for the dataset in "YYYY-MM-DD" format. If None, defaults to the current date.
            save_dir: The directory to save the downloaded dataset. If None, defaults to a "data" directory in the current working directory.
            save_file: The file name to save the downloaded dataset. If None, defaults to a name based on the dataset and date range.

        """
        super().__init__(
            site=site,
            instrument="chl",
            save_dir=save_dir,
            save_file=save_file,
            start_date=start_date,
            end_date=end_date,
        )

    @staticmethod
    def _update_metadata(ds: xr.Dataset) -> xr.Dataset:
        """Update the metadata of the OOI EA dataset to be CF-compliant and include necessary attributes.

        Args:
            ds (xr.Dataset): The xarray Dataset containing the OOI EA data.

        Returns:
            xr.Dataset: The xarray Dataset with updated metadata.

        """
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

    def download(self) -> None:
        """Download the netCDF data files from the THREDDS catalog and save them to a local directory."""
        logger.info(f"Getting list of data files for {self.location}...")
        nc_files = self._list_files(self.search_url, self.tag)
        if not nc_files:
            logger.warning(f"No files found for {self.location} in the specified date range. Exiting download.")
            return
        download_urls = [self.base_url + f + "#mode=bytes" for f in nc_files]

        logger.info(f"Downloading files for {self.location}...")
        ds: list[xr.Dataset] = []
        for f in tqdm(download_urls, desc="Downloading datasets"):
            r = requests.get(f, timeout=(3.05, 120))
            if r.ok:
                ds.append(xr.open_dataset(io.BytesIO(r.content)))
                ds[-1] = ds[-1].swap_dims({"obs": "time"}).squeeze()
                ds[-1] = ds[-1].reset_coords(["lat", "lon", "depth"])
                ds[-1] = ds[-1].where(ds[-1]["depth"] <= self.location.depth, drop=True)
                ds[-1] = self._bin_dataset(ds[-1]).compute()

        ds_concat = xr.concat(ds, dim="time", join="outer")
        ds_concat = ds_concat.sortby("time")  # ensure data is sorted by time after merging
        ds_concat = ds_concat.sel(
            time=slice(self.start_date, self.end_date)
        )  # subset to specified date range after merging

        ds_concat = self._qc_check(
            ds_concat,
            ["fluorometric_cdom", "fluorometric_chlorophyll", "optical_backscatter"],
        )
        ds_concat = self._drop_unused_vars(ds_concat)

        # interpolate up to 5 meters
        ds_concat = ds_concat.interpolate_na(dim="depth", method="linear", use_coordinate=True, max_gap=5)
        # interpolate up to 1 day
        ds_concat = ds_concat.interpolate_na(
            dim="time", method="linear", use_coordinate=True, max_gap=np.timedelta64(1, "D")
        )

        ds_concat = ds_concat.resample(time="1D").mean()

        ds_concat = ds_concat.assign_coords(
            {
                "latitude": self.location.lat,
                "longitude": self.location.lon,
                "site": self.location.site,
            }
        )
        ds_concat = self._update_metadata(ds_concat)

        ds_concat.to_netcdf(self.save_file_path)
        logger.info(f"Download complete! Dataset saved to {self.save_file_path}")


class _MooringBase(_OOIBase):
    """Base class for downloading OOI Endurance Array Mooring datasets, containing shared methods and attributes for mooring datasets."""

    def __init__(
        self,
        site: str | MooringSites,
        save_dir: str | None = None,
        save_file: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> None:
        super().__init__(
            site=site,
            location_type="mooring",
            instrument="ctd",
            save_dir=save_dir,
            save_file=save_file,
            start_date=start_date,
            end_date=end_date,
        )


class MooringCTD(_MooringBase):
    """A class to download OOI Endurance Array Mooring CTD datasets."""

    def __init__(
        self,
        site: str | MooringSites,
        start_date: str | None = None,
        end_date: str | None = None,
        save_dir: str | None = None,
        save_file: str | None = None,
    ) -> None:
        """Initialize the EAMooringDownloader with parameters for downloading.

        Args:
            site: Site identifier for the dataset (e.g., "CE01ISSM"). Must be one of the following: "CE01ISSM", "CE02SHSM", "CE04OSSM", "CE06ISSM", "CE07SHSM", "CE09OSSM". Required.
            start_date: The start date for the dataset in "YYYY-MM-DD" format. If None, defaults to "2000-01-01".
            end_date: The end date for the dataset in "YYYY-MM-DD" format. If None, defaults to the current date.
            save_dir: The directory to save the downloaded dataset. If None, defaults to a "data" directory in the current working directory.
            save_file: The file name to save the downloaded dataset. If None, defaults to a name based on the dataset and date range.

        """
        super().__init__(
            site=site,
            save_dir=save_dir,
            save_file=save_file,
            start_date=start_date,
            end_date=end_date,
        )

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
        ds = self._qc_check(
            ds, variables=["sea_water_pressure", "sea_water_temperature", "sea_water_practical_salinity"]
        )
        ds = self._drop_unused_vars(ds)

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
        ds: list[xr.Dataset] = []
        for f in tqdm(download_urls, desc="Downloading datasets"):
            r = requests.get(f, timeout=(3.05, 120))
            if r.ok:
                ds.append(xr.open_dataset(io.BytesIO(r.content)))
                ds[-1] = ds[-1].swap_dims({"obs": "time"}).squeeze()

        ds = [self._process(di) for di in ds]
        ds_concat = xr.concat(ds, dim="time", join="outer")
        ds_concat = ds_concat.sortby("time")  # ensure data is sorted by time after merging
        ds_concat = ds_concat.sel(
            time=slice(self.start_date, self.end_date)
        )  # subset to specified date range after merging
        ds_concat = ds_concat.assign_coords(
            {
                "latitude": self.location.lat,
                "longitude": self.location.lon,
                "site": self.location.site,
            }
        )
        ds_concat = self._update_metadata(ds_concat)

        ds_concat.to_netcdf(self.save_file_path)
        logger.info(f"Download complete! Dataset saved to {self.save_file_path}")
