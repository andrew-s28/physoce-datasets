"""Functions for downloading and processing ERA5 reanalysis data from the ECMWF Data Store."""

from __future__ import annotations

import contextlib
import datetime
import json
import operator
import os
import time
import warnings
from pathlib import Path
from typing import Final, Literal, TypedDict, cast

import click
import numpy as np
import xarray as xr
from ecmwf.datastores import Client, Remote, Results
from metpy.calc import relative_humidity_from_dewpoint
from metpy.units import units
from pycoare import coare_35
from tqdm import tqdm

from physoce_datasets.logging import logger
from physoce_datasets.util import AreaDict, parse_area

CONFIG_FILE = Path.home() / ".ecmwfdatastoresrc"

REQUEST_STATE_FILE = "submitted_requests.json"
MAX_ACTIVE_REQUESTS = 10
POLL_INTERVAL_SECONDS = 1
ECMWF_JOB_LIMIT = 1000
DATASET = "reanalysis-era5-single-levels"

type _JobStatus = Literal["accepted", "running", "successful", "failed", "rejected"]


class _RequestParams(TypedDict):
    """The parameters for an ERA5 data request to the ECMWF Data Store API."""

    product_type: list[str]
    variable: list[str]
    date: str
    time: list[str]
    area: list[float]
    data_format: str


class _State(TypedDict):
    """Internal state representation for tracking the status of ERA5 data requests and processing steps."""

    start_date: str
    end_date: str
    request_id: str | None
    remote_status: str | None
    download_status: str | None
    processing_status: str | None
    request: _RequestParams | None


class _RemoteJobStatus:
    """Remote job status categories for ECMWF Data Store jobs."""

    ACCEPTED: Final[_JobStatus] = "accepted"
    RUNNING: Final[_JobStatus] = "running"
    SUCCESSFUL: Final[_JobStatus] = "successful"
    FAILED: Final[_JobStatus] = "failed"
    REJECTED: Final[_JobStatus] = "rejected"
    UNKNOWN: Final[str] = "unknown"

    ACTIVE: Final[tuple[_JobStatus, ...]] = (ACCEPTED, RUNNING)
    SAVED: Final[tuple[_JobStatus, ...]] = (ACCEPTED, RUNNING, SUCCESSFUL, FAILED, REJECTED)
    PREFILL: Final[tuple[_JobStatus, ...]] = (SUCCESSFUL, ACCEPTED, RUNNING)
    FINISHED: Final[tuple[_JobStatus, ...]] = (SUCCESSFUL, FAILED, REJECTED)


class _LocalJobStatus:
    """Local job status categories for tracking downloading and processing of ERA5 files."""

    PENDING = "pending"
    DOWNLOADED = "downloaded"
    PROCESSED = "processed"
    FAILED = "failed"


def _get_state_path(save_dir: Path) -> Path:
    return save_dir / REQUEST_STATE_FILE


def login_to_ecmwf_datastore() -> Client:
    """Perform a login to the ECMWF Data Store, prompting the user for credentials if not saved.

    Credentials will be stored in a file found at ~/.ecmwfdatastoresrc for future use.
    To obtain your key, go to https://cds.climate.copernicus.eu/how-to-api

    Overrides the default ECMWF Data Store login behavior to provide more user-friendly prompts and messages.

    Args:
        client (ecmwf.datastores.Client): An instance of the ECMWF Data Store client.

    Returns:
        ecmwf.datastores.Client: An authenticated instance of the ECMWF Data Store client.

    """
    logger.info("Attempting login...")
    try:
        with contextlib.redirect_stdout(Path(os.devnull).open("w", encoding="utf-8")) and contextlib.redirect_stderr(
            Path(os.devnull).open("w", encoding="utf-8"),
        ):
            client = Client(progress=False, retry_after=1, maximum_tries=10)
            client.check_authentication()
            logger.info("Login successful!")
    except Exception as e:  # noqa: BLE001
        logger.info(f"Failed to authenticate with ECMWF Data Store: {e}")
        logger.info(
            "No valid credentials found. Please enter your Climate Data Store "
            "API key. These will be stored in a file found at "
            "~/.ecmwfdatastoresrc for future use. To obtain your key, go to "
            "https://cds.climate.copernicus.eu/how-to-api.",
        )
        key = click.prompt("Enter your key", type=str, hide_input=True)
        with CONFIG_FILE.open("w") as f:
            f.write(f"url: https://cds.climate.copernicus.eu/api\nkey: {key}\n")
        client = login_to_ecmwf_datastore()
    return client


def _create_data_dir(save_dir: Path | None) -> Path:
    """Create the directory to save the downloaded dataset if it doesn't already exist.

    Args:
        save_dir (Path | None): The directory to save the downloaded dataset. If None,
            defaults to a "data" directory in the current working directory.

    Returns:
        Path: The directory to save the downloaded dataset.

    """
    # by default, save in a "data" directory relative to current working directory
    data_dir = Path("data") if save_dir is None else save_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _retrieve_results(remote: Remote) -> Results:
    """Retrieve the results for a completed job from the ECMWF Data Store.

    Args:
        remote (Remote): The remote job object for the completed job.

    Returns:
        Results: The results object containing the data for the completed job.

    Raises:
        RuntimeError: If the results are not ready for download, or if there is
            an error during the retrieval process.

    """
    if not remote.results_ready:
        msg = (
            f"Results for request ID {remote.request_id} are not ready for "
            f"download. Current status: {remote.status}. Please wait and try "
            "again later."
        )
        raise RuntimeError(msg)
    return remote.get_results()


def _process_data(input_file: Path, output_file: Path) -> Path:
    """Convert a downloaded hourly ERA5 month into a daily-mean NetCDF file.

    Args:
        input_file (Path): The file path to the downloaded hourly ERA5 NetCDF file.
        output_file (Path): The file path to save the processed daily-mean ERA5 NetCDF file.

    Returns:
        Path: The file path to the processed daily-mean ERA5 NetCDF file.

    """
    ds = xr.open_dataset(input_file)
    ds = ds.rename({"valid_time": "time"})
    # convert temperatures to degC
    ds["t2m"] -= 273.15
    ds["d2m"] -= 273.15
    ds["sst"] -= 273.15
    ds["latitude_broadcast"] = ds["latitude"].broadcast_like(ds["u10"])
    # we can expect warnings in the wind stress calc (e.g., when sst is nan, over land), so silence warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ds["eastward_wind_stress"], ds["northward_wind_stress"] = xr.apply_ufunc(
            compute_wind_stress,
            ds["u10"],
            ds["v10"],
            ds["t2m"],
            ds["d2m"],
            ds["sst"],
            ds["sp"],
            ds["latitude_broadcast"],
            input_core_dims=[["time"], ["time"], ["time"], ["time"], ["time"], ["time"], ["time"]],
            output_core_dims=[["time"], ["time"]],
            vectorize=True,
        )
    ds = ds.resample(time="1D").mean(keep_attrs=True)
    ds.attrs.update(ds.attrs)
    ds.attrs["history"] = (
        ds.attrs.get("history", "")
        + (
            f"\n{datetime.datetime.now(tz=datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')} "
            "- Resampled hourly ERA5 data to daily means."
        )
    ).strip()
    ds.to_netcdf(output_file)
    ds.close()

    return output_file


def compute_relative_humidity(t2m: np.ndarray, d2m: np.ndarray) -> np.ndarray:
    """Compute relative humidity from 2m temperature and 2m dewpoint temperature.

    Args:
        t2m (np.ndarray): 2m temperature in Kelvin.
        d2m (np.ndarray): 2m dewpoint temperature in Kelvin.

    Returns:
        np.ndarray: Relative humidity in percentage.

    """
    try:
        # drop metpy pint units and convert to percent
        rh = relative_humidity_from_dewpoint(t2m * units.degC, d2m * units.degC).m * 100
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error computing relative humidity: {e}")
        rh = np.full_like(t2m, fill_value=75.0)  # fill with a default value of 75% if there is an error
    return rh


def compute_wind_stress(
    u10: np.ndarray,
    v10: np.ndarray,
    t2m: np.ndarray,
    d2m: np.ndarray,
    sst: np.ndarray,
    sp: np.ndarray,
    latitude: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute wind stress using standard ERA5 outputs and the COARE 3.5 bulk flux algorithm.

    Args:
        u10 (np.ndarray): 10m eastward wind component in m/s.
        v10 (np.ndarray): 10m northward wind component in m/s.
        t2m (np.ndarray): 2m temperature in degC.
        d2m (np.ndarray): 2m dewpoint temperature in degC.
        sst (np.ndarray): Sea surface temperature in degC.
        sp (np.ndarray): Surface pressure in Pa.
        latitude (np.ndarray): Latitude coordinates.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the eastward and northward wind stress components in N/m^2.

    """
    rh = compute_relative_humidity(t2m, d2m)
    mag = np.sqrt(u10**2 + v10**2)
    angle = np.arctan2(v10, u10)
    c35 = coare_35(
        u=mag,
        t=t2m,
        rh=rh,
        ts=sst,
        p=sp / 100,  # convert to millibars for pycoare input, see also https://github.com/pyCOARE/coare/issues/57
        lat=latitude,
        zu=10,
        zt=2,
        zq=2,
        zrf=10,
    )
    tau_mag = c35.fluxes.tau
    tau_east = tau_mag * np.cos(angle)
    tau_north = tau_mag * np.sin(angle)
    return tau_east, tau_north


def _split_monthly_jobs(start_date: str, end_date: str) -> tuple[list[str], list[str]]:
    """Split a request with a date range into multiple requests with monthly date ranges.

    Args:
        start_date (str): The start date for the request. Format should be YYYY-MM-DD.
        end_date (str): The end date for the request. Format should be YYYY-MM-DD.

    Returns:
        list[dict]: A list of request dictionaries with monthly date ranges.

    """
    start_date_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d").astimezone(datetime.UTC)
    end_date_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d").astimezone(datetime.UTC)

    start_dates, end_dates = [], []
    current_start = start_date_dt
    while current_start < end_date_dt:
        current_end = (current_start + datetime.timedelta(days=32)).replace(day=1) - datetime.timedelta(days=1)
        current_end = min(current_end, end_date_dt)
        start_dates.append(f"{current_start.strftime('%Y-%m-%d')}")
        end_dates.append(f"{current_end.strftime('%Y-%m-%d')}")
        current_start = current_end + datetime.timedelta(days=1)
    return start_dates, end_dates


def _create_requests(states: list[_State], area: AreaDict) -> list[_State]:
    for s in states:
        if s["request"] is None:
            s["request"] = {
                "product_type": ["reanalysis"],
                "variable": [
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                    "2m_dewpoint_temperature",
                    "2m_temperature",
                    "sea_surface_temperature",
                    "surface_pressure",
                ],
                "date": f"{s['start_date']}/{s['end_date']}",
                "time": [f"{hour:02d}:00" for hour in range(0, 24, 1)],
                "area": [area["lat_max"], area["lon_min"], area["lat_min"], area["lon_max"]],
                "data_format": "netcdf",
            }
    return states


def _build_request_state(start_dates: list[str], end_dates: list[str], area: AreaDict) -> list[_State]:
    states: list[_State] = [
        {
            "start_date": s,
            "end_date": e,
            "request_id": None,
            "remote_status": None,
            "download_status": None,
            "processing_status": None,
            "request": None,
        }
        for s, e in zip(start_dates, end_dates, strict=True)
    ]
    states = _create_requests(states, area)
    return states


def _load_request_state(state_file: Path) -> list[_State]:
    with state_file.open() as f:
        states = json.load(f)
    return states


def _save_request_state(states: list[_State], state_file: Path) -> None:
    tmp_file = state_file.with_suffix(".tmp")
    with tmp_file.open("w") as f:
        json.dump(states, f, indent=4)
    tmp_file.replace(state_file)


def _update_request_state(
    states: list[_State],
    state_file: Path,
    start_dates: list[str],
    end_dates: list[str],
    area: AreaDict,
) -> list[_State]:
    """Update the request states with the new date range from the user.

    Args:
        states (list[_State]): The current request states loaded from the states file.
        state_file (Path): The path to the request states file for saving updates.
        start_dates (list[str]): The list of start dates for the new date range.
        end_dates (list[str]): The list of end dates for the new date range.
        area (AreaDict): The dictionary of area coordinates.

    Returns:
        list[_State]: The updated request states with any new date ranges added.

    """
    existing_date_ranges = {(s["start_date"], s["end_date"]) for s in states}
    for start_date, end_date in zip(start_dates, end_dates, strict=True):
        if (start_date, end_date) not in existing_date_ranges:
            states.append(
                {
                    "start_date": start_date,
                    "end_date": end_date,
                    "request_id": None,
                    "remote_status": None,
                    "download_status": None,
                    "processing_status": None,
                    "request": None,
                },
            )
    states = _create_requests(states, area)
    states.sort(key=operator.itemgetter("start_date"))
    _save_request_state(states, state_file)
    return states


def _delete_expired_requests(client: Client) -> None:
    jobs = client.get_jobs(
        ECMWF_JOB_LIMIT,
        sortby="-created",
        status=_RemoteJobStatus.SUCCESSFUL,
    ).json.get("jobs", [])
    expired_job_ids = [
        job.get("jobID") for job in jobs if job.get("metadata").get("results").get("type") == "results expired"
    ]
    client.delete(*expired_job_ids)


def _get_active_job_count(client: Client) -> int:
    """Count accepted/running jobs for active-cap enforcement.

    Args:
        client (Client): An authenticated ECMWF Data Store client.

    Returns:
        int: The number of active jobs currently in the ECMWF Data Store queue for the user.

    """
    jobs = client.get_jobs(
        ECMWF_JOB_LIMIT,
        sortby="-created",
        status=list(_RemoteJobStatus.ACTIVE),
    ).json.get("jobs", [])
    return len(jobs)


def _under_total_jobs_limit(client: Client, number_to_submit: int) -> int:
    """Check existing jobs and enforce total saved job cap before submitting new jobs.

    Args:
        client (Client): An authenticated ECMWF Data Store client.
        number_to_submit (int): The number of new jobs intended to be submitted.

    Returns:
        bool: True if under the total jobs limit and safe to submit, False if submitting would exceed the limit.

    """
    jobs = client.get_jobs(
        ECMWF_JOB_LIMIT,
        sortby="-created",
        status=list(_RemoteJobStatus.SAVED),
    ).json.get("jobs", [])
    total_saved_jobs = len(jobs)
    projected_total_jobs = total_saved_jobs + number_to_submit
    if projected_total_jobs >= ECMWF_JOB_LIMIT:
        jobs_to_delete = projected_total_jobs - (ECMWF_JOB_LIMIT - 1)
        logger.error(
            "Cannot proceed with submission: "
            f"you currently have {total_saved_jobs} saved jobs and this run needs {number_to_submit} new submissions, "
            f"which would bring your total to {projected_total_jobs} (ECMWF limit is {ECMWF_JOB_LIMIT}). "
            f"Delete at least {jobs_to_delete} jobs in the CDS/ECMWF portal at "
            "https://cds.climate.copernicus.eu/requests, then rerun era5 "
            "submit.",
        )
        return False
    return True


def _extract_job_status(job: dict) -> str:
    """Extract a normalized job status string from a get_jobs entry.

    Args:
        job (dict): A single job entry from the list returned by client.get_jobs().

    Returns:
        str: The normalized job status, or an empty string if it cannot be determined.

    """
    status = job.get("status", {})
    if isinstance(status, str) and status.strip():
        return status.strip().lower()
    return ""


def _request_matches(receipt_request: dict, target_request: dict) -> bool:
    """Return True when all target request parameters match the receipt request.

    Args:
        receipt_request (dict): The request dictionary extracted from a job receipt.
        target_request (dict): The request dictionary for a monthly request we want to match against.

    Returns:
        bool: True if all parameters in the target request match those in the receipt request, False otherwise.

    """
    for key, value in target_request.items():
        if key not in receipt_request:
            return False
        if receipt_request[key] != value:
            return False
    return True


def _prefill_submitted_requests_from_recent_jobs(
    client: Client,
    states: list[_State],
    state_file: Path,
) -> tuple[list[_State], int]:
    """Populate states rows from recent matching jobs to avoid duplicate submissions.

    Args:
        client (Client): An authenticated ECMWF Data Store client.
        monthly_requests (list[dict]): The list of monthly request dictionaries to match against recent jobs.
        states (list[dict[str, str]]): The current request states dataframe to update with matched request IDs.
        state_file (Path): The path to the request states file for saving updates.

    Returns:
        tuple[pd.DataFrame, int]: The updated request states dataframe and the count of matched existing jobs.

    """
    jobs = client.get_jobs(
        100,  # number of recent jobs to check, 100 is ~30s for receipt retrieval
        sortby="-created",
        status=list(_RemoteJobStatus.PREFILL),
    ).json.get("jobs", [])

    # if there are no recent jobs, return early to avoid unnecessary receipt retrieval step
    if not jobs:
        return states, 0

    # load all remotes from api, this is the step that takes the longest
    remotes = [client.get_remote(job.get("jobID")) for job in tqdm(jobs, desc="Fetching receipts for recent jobs")]

    # get requests and statuses for all existing jobs
    existing_requests = {remote.request_id: remote.request for remote in remotes}
    remote_status = {remote.request_id: remote.status for remote in remotes}

    # build a lookup table of recent requests to receipts for quick matching against monthly requests
    lookup = {
        (
            request["date"],
            tuple(request["time"]),
            tuple(request["area"]),
            tuple(request["variable"]),
            tuple(request["product_type"]),
        ): request_id
        for request_id, request in existing_requests.items()
    }

    # iterate through monthly requests and fill in request_id and remote_status from lookup table when a match is found
    matched_existing_jobs = 0
    for s in states:
        if s.get("remote_status") in _RemoteJobStatus.FINISHED:
            continue  # skip already finished jobs in states
        request = s.get("request")
        if request is None:
            continue
        key = (
            request.get("date"),
            tuple(request.get("time")),
            tuple(request.get("area")),
            tuple(request.get("variable")),
            tuple(request.get("product_type")),
        )
        request_id = lookup.get(key)
        if request_id is None:
            continue
        s["request_id"] = request_id
        s["remote_status"] = remote_status[request_id]
        matched_existing_jobs += 1

    _save_request_state(states, state_file)
    return states, matched_existing_jobs


def _submit_one_pending_request(
    client: Client,
    states: list[_State],
    state_file: Path,
) -> tuple[list[_State], str]:
    """Submit one pending request to the ECMWF Data Store and updates the request states dataframe accordingly.

    Args:
        client (Client): An authenticated ECMWF Data Store client.
        monthly_requests (list[dict]): The list of monthly request dictionaries to submit from.
        states (list[_State]): The current request states list to update with the submitted request ID and status.
        state_file (Path): The path to the request states file for saving updates.

    Returns:
        tuple[list[_State], str]: The updated request states list and the submitted request ID.

    """
    # find the first index in the states where the request_id is None, meaning it has not been submitted yet
    pending_indices = [i for i, s in enumerate(states) if s.get("request_id") is None]
    # if everything has been submitted, return the states as is
    if not pending_indices:
        return states, ""

    # grab the first pending request and submit it
    i = pending_indices[0]
    # cast type for submission, this is safe because we always fill in the request before submission
    request = cast("dict", states[i]["request"])
    remote = client.submit(DATASET, request)

    # update the states with the returned request ID and initial status
    states[i]["request_id"] = remote.request_id
    states[i]["remote_status"] = remote.status
    _save_request_state(states, state_file)

    return states, remote.request_id


def _submit_requests(
    client: Client,
    states: list[_State],
    state_file: Path,
    remaining_to_submit: int,
    matched_existing_jobs: int,
) -> list[_State]:
    """Submit pending requests to the ECMWF Data Store while enforcing active and total job caps.

    Args:
        client (Client): An authenticated ECMWF Data Store client.
        states (list[_State]): The current request states list to update with submitted request IDs and statuses.
        state_file (Path): The path to the request states file for saving updates.
        remaining_to_submit (int): The number of requests that still need to be submitted.
        matched_existing_jobs (int): The number of requests that were pre-filled with existing jobs.

    Returns:
        list[_State]: The updated request states list with submitted request IDs and statuses.

    """
    submitted = 0
    with tqdm(total=remaining_to_submit + matched_existing_jobs, desc="Submission progress") as progress:
        progress.n = matched_existing_jobs
        progress.refresh()
        while True:
            # update progress bar with counts
            progress.n = matched_existing_jobs + submitted
            progress.refresh()

            # break the loop if all requests are submitted (all requests have ids)
            if all(s.get("request_id") for s in states):
                break

            # enforce active job cap by waiting to submit if we are at or above the max active requests limit
            active_before = _get_active_job_count(client)
            if active_before >= MAX_ACTIVE_REQUESTS:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            # if we're under the active job cap, submit one pending request
            states, request_id = _submit_one_pending_request(
                client=client,
                states=states,
                state_file=state_file,
            )

            # after submission, check if the request was successful and update the status
            if request_id:
                submitted += 1
                states = _update_status_for_submitted_request(
                    client=client,
                    states=states,
                    state_file=state_file,
                    request_id=request_id,
                )

            # wait just a bit longer to be sure the newly submitted jobs have registered before we check again
            time.sleep(POLL_INTERVAL_SECONDS)

    _save_request_state(states, state_file)
    return states


def _update_status_for_submitted_request(
    client: Client,
    states: list[_State],
    state_file: Path,
    request_id: str,
) -> list[_State]:
    """Check the status of a submitted request and update the request states list accordingly.

    Args:
        client (Client): An authenticated ECMWF Data Store client.
        states (list[_State]): The current request states list to update with the latest remote statuses.
        state_file (Path): The path to the request states file for saving updates.
        request_id (str): The request ID for the submitted job to check the status of.

    Returns:
        list[_State]: The updated request states list with the latest remote statuses.

    """
    s = next((s for s in states if s.get("request_id") == request_id), None)
    if s is None:
        logger.warning(f"Request for {request_id} not found.")
        return states
    try:
        remote = client.get_remote(request_id)
        s["remote_status"] = remote.status
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error checking status for request ID {request_id}: {e}")
        s["remote_status"] = "unknown"
    _save_request_state(states, state_file)
    return states


def _get_existing_datetimes(save_dir: Path, update_path: str) -> xr.DataArray | None:
    """Get the datetime values from the existing file.

    Args:
        save_dir (Path): The directory where the dataset files are saved.
        update_path (str): The path to the existing file to update.

    Returns:
        xr.DataArray: The datetime values from the existing file.

    """
    if not Path(save_dir / update_path).exists():
        return None

    ds_existing = xr.open_dataset(save_dir / update_path)
    datetime_existing = ds_existing["time"]

    return datetime_existing


def _download_file(client: Client, state: _State, save_file_path: Path) -> Path:
    """Download the raw file for a completed request.

    Args:
        client (Client): An authenticated ECMWF Data Store client.
        state (_State): The request state for the job to download.
        save_file_path (Path): The path to the file to save the downloaded data as.

    Returns:
        Path: The path to the downloaded raw file.

    """
    # setup file names
    raw_file = save_file_path.with_name(
        save_file_path.stem + f"_raw_{state['start_date']}_{state['end_date']}.nc",
    )

    # download results if not already downloaded
    if (
        state["download_status"] != _LocalJobStatus.DOWNLOADED
        and state["remote_status"] == _RemoteJobStatus.SUCCESSFUL
        and state["request_id"] is not None
    ):
        remote = client.get_remote(state["request_id"])
        results = _retrieve_results(remote)
        results.download(str(raw_file))
    return raw_file


def _process_file(state: _State, raw_file: Path, save_file_path: Path) -> Path:
    """Process a downloaded raw file into a cleaned daily-mean file.

    Args:
        state (_State): The request state for the job to process.
        raw_file (Path): The path to the downloaded raw file.
        save_file_path (Path): The path to save the processed file as.

    Returns:
        Path: The path to the processed file.

    """
    # process data if downloaded but not yet processed
    if state["download_status"] == _LocalJobStatus.DOWNLOADED:
        try:
            processed_file = save_file_path.with_name(
                save_file_path.stem + f"_processed_{state['start_date']}_{state['end_date']}.nc",
            )
            _process_data(raw_file, processed_file)
            # remove raw file after processing
            if raw_file.exists():
                raw_file.unlink()
        # If anything goes wrong, mark local status as failed and log the error
        # but do not raise so other requests can continue
        except Exception as e:  # noqa: BLE001
            logger.exception(f"Error processing file for request ID {state['request_id']}: {e}")
    return processed_file


def _download_and_process_ready_requests(
    client: Client,
    states: list[_State],
    save_dir: Path,
    state_file: Path,
    save_file: str | None = None,
) -> list[_State]:
    """Download and process datasets for requests with successful remote jobs.

    Only requests not yet processed locally are handled.

    Args:
        client (Client): An authenticated ECMWF Data Store client.
        states (list[_State]): The current request states list to update with
            downloaded file paths and processing statuses.
        save_dir (Path): The directory to save the downloaded and processed datasets.
        state_file (Path): The path to the request states file for saving updates.
        save_file (str | None): The name of an existing file in the save directory to update with new data.
            If provided, any dates in the existing file will be skipped during downloading and processing.

    Returns:
        list[_State]: The updated request states list with downloaded file paths and processing statuses.

    """
    if save_file is None:
        save_file = f"era5_reanalysis_combined_{states[0]['start_date']}_{states[-1]['end_date']}.nc"
        save_file_path = save_dir / save_file
    else:
        save_file_path = save_dir / save_file

    # iterate through requests and download/process those that are successful
    processed_files: list[Path] = []
    for state in tqdm(states, desc="Downloading and processing ready requests"):
        # skip jobs that don't have a successful remote status or have already been processed locally
        if (
            state["remote_status"] != _RemoteJobStatus.SUCCESSFUL
            or state["download_status"] == _LocalJobStatus.PROCESSED
        ):
            continue

        try:
            raw_file = _download_file(client, state, save_file_path)
            state["download_status"] = _LocalJobStatus.DOWNLOADED
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error downloading file for request ID {state['request_id']}: {e}")
            state["download_status"] = _LocalJobStatus.FAILED
            continue

        try:
            processed_file = _process_file(state, raw_file, save_file_path)
            state["processing_status"] = _LocalJobStatus.PROCESSED
            processed_files.append(processed_file)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error processing file for request ID {state['request_id']}: {e}")
            state["processing_status"] = _LocalJobStatus.FAILED

        _save_request_state(states, state_file)

    # this whole section feels fragile and a bit sloppy
    # now combine analyzed files
    # only get processed files that exist
    processed_files = [f for f in processed_files if Path(f).exists()]
    processed_ds = xr.open_mfdataset(processed_files, chunks="auto", compat="equals")

    if save_file_path.exists():
        # if a combined file already exists, try to add to it
        existing_ds = xr.open_dataset(save_file_path)
        ds = xr.merge([existing_ds, processed_ds], compat="equals", join="outer")
        existing_ds.close()
        ds.to_netcdf(save_file_path)
        ds.close()
    else:
        # otherwise save all processed datasets as coombined file
        processed_ds.to_netcdf(save_file_path)
    processed_ds.close()

    # now we can remove the individual processed datasets too
    for f in processed_files:
        if f.exists():
            f.unlink()
    return states


def submit_era5(
    save_dir: Path | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    area_str: str | None = None,
) -> None:
    """Submit ERA5 remote jobs without downloading local files.

    Args:
        save_dir (Path | None): The directory to save the downloaded dataset.
            If None, defaults to a "data" directory at the package level.
        start_date (str | None): The start date for the dataset in
            YYYY-MM-DD format. If None, defaults to the earliest available
            date for the dataset.
        end_date (str | None): The end date for the dataset in
            YYYY-MM-DD format. If None, defaults to the latest available
            date for the dataset.
        area_str (str | None): The area to subset the dataset to, in the format "lat_min,lon_min,lat_max,lon_max".
            If None, defaults to the full global extent.

    """
    # handle default parameters
    if start_date is None:
        start_date = "2000-01-01"
    if end_date is None:
        end_date = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d")

    area = parse_area(area_str)

    # setup client, request, and save directory
    client = login_to_ecmwf_datastore()
    _delete_expired_requests(client)
    save_dir = _create_data_dir(save_dir)
    state_file = _get_state_path(save_dir)

    # get request states, either by building a new one or loading from an existing states file
    start_dates, end_dates = _split_monthly_jobs(start_date, end_date)
    if not state_file.exists():
        states = _build_request_state(start_dates, end_dates, area)
        _save_request_state(states, state_file)
    else:
        states = _load_request_state(state_file)
        states = _update_request_state(states, state_file, start_dates, end_dates, area)

    # Check existing jobs to prefill request IDs and statuses for matching
    # recent jobs and avoid unnecessary duplicate submissions.
    states, matched_existing_jobs = _prefill_submitted_requests_from_recent_jobs(
        client,
        states,
        state_file,
    )

    # check how many remaining jobs we will have to submit and let the user know
    remaining_to_submit = sum(1 for s in states if not s.get("request_id"))
    logger.info(
        f"Using {matched_existing_jobs} existing jobs; submitting remaining {remaining_to_submit}.",
    )

    # check to see if the user will end up over 1000 requests before submitting any new jobs
    if not _under_total_jobs_limit(client, remaining_to_submit):
        # save and exit if user would be over the total jobs limit
        _save_request_state(states, state_file)
        return

    # If all requests already have IDs, they are already submitted
    # exit early without starting the submission manager loop.
    if all(s.get("request_id") for s in states):
        logger.info(
            "All requests are already submitted. View progress at "
            "https://cds.climate.copernicus.eu/requests. Run `uv run "
            "datasets.py era5 download` when jobs are ready.",
        )
        return

    try:
        logger.info("Starting ERA5 submission. If exited, re-run the command to resume.")
        states = _submit_requests(
            client=client,
            states=states,
            state_file=state_file,
            remaining_to_submit=remaining_to_submit,
            matched_existing_jobs=matched_existing_jobs,
        )

        # all done, save request states and exit
        _save_request_state(states, state_file)
        logger.info(
            "All requests have been submitted. View progress at "
            "https://cds.climate.copernicus.eu/requests. Run `uv run "
            "datasets.py era5 download` once all requests are ready to "
            "download files.",
        )
    except KeyboardInterrupt:
        # make sure request states is saved on interrupt
        _save_request_state(states, state_file)
        logger.warning(
            f"Submission interrupted. Progress saved to {state_file}. Re-run the command to resume.",
        )
        return
    return


def download_era5(
    save_dir: Path | None = None,
    save_file: str | None = None,
) -> None:
    """Download and process ERA5 files after all remote jobs are successful.

    Args:
        save_dir (Path | None): The directory to save the downloaded dataset.
            If None, defaults to a "data" directory at the package level.
            There must be a request states file in this directory by running `uv run datasets.py era5 submit` first.
        save_file (str | None): The name of an existing file in the save directory to update with new data.
            If provided, any dates in the existing file will be skipped during downloading and processing.


    """
    # setup client and save directory
    client = login_to_ecmwf_datastore()
    save_dir = _create_data_dir(save_dir)
    state_file = _get_state_path(save_dir)

    if not state_file.exists():
        logger.error(f"No request states found at {state_file}. Run `uv run datasets.py era5 submit` first.")
        return

    states = _load_request_state(state_file)

    # update and save remote statuses one last time before starting downloads, in case there have been any changes
    for request_id in [s.get("request_id") for s in states]:
        if request_id is not None:
            states = _update_status_for_submitted_request(client, states, state_file, request_id)
    _save_request_state(states, state_file)

    if any(s.get("remote_status") not in _RemoteJobStatus.FINISHED for s in states):
        logger.error(
            "Not all requests are finished yet. Please wait for all requests to be successful before downloading. "
            "View progress at https://cds.climate.copernicus.eu/requests "
            "and run `uv run datasets.py era5 download` once all requests are ready.",
        )
        return

    # check for any failed remote jobs before starting downloads, and warn the user before downloading
    # but allow them to continue if they want to download any successful requests
    if any(s.get("remote_status") == _RemoteJobStatus.FAILED for s in states):
        failed = sum(1 for s in states if s.get("remote_status") == _RemoteJobStatus.FAILED)
        logger.warning(
            f"{failed} jobs failed or were cancelled on the CDS server."
            "Please check your CDS account at https://cds.climate.copernicus.eu/requests for more details.",
        )
        if not click.confirm("Do you want to continue with downloading any successful requests?", default=True):
            return

    try:
        logger.info("Starting ERA5 download...")
        # this function is where the actual downloading and processing happens, all the rest is just checks
        states = _download_and_process_ready_requests(client, states, save_dir, state_file, save_file)

        _save_request_state(states, state_file)

        # make sure they know everything is successful :)
        if all(s.get("processing_status") == _LocalJobStatus.PROCESSED for s in states):
            logger.info("All downloads complete and processed successfully.")
            return

        logger.warning(
            "Unknown download issue; some months are not processed yet. "
            "Examine your CDS account at https://cds.climate.copernicus.eu/requests "
            f"and the states file at {state_file} for more details.",
        )

    except KeyboardInterrupt:
        _save_request_state(states, state_file)
        # make sure request states is saved on interrupt
        logger.warning(
            f"Download interrupted. Progress saved to {state_file}. Re-run the command to resume.",
        )
        return
    return
