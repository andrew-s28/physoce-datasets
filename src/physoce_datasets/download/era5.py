from __future__ import annotations

import contextlib
import datetime
import os
import time
from pathlib import Path
from typing import Final, Literal

import click
import pandas as pd
import xarray as xr
from ecmwf.datastores import Client, Remote, Results
from metpy.calc import relative_humidity_from_dewpoint
from requests.exceptions import HTTPError
from tqdm import tqdm

from physoce_datasets.logging import logger

CONFIG_FILE = Path.home() / ".ecmwfdatastoresrc"

MIN_LON = -150
MAX_LON = -120
MIN_LAT = 30
MAX_LAT = 60
REQUEST_STATE_FILE = "submitted_requests.csv"
DEFAULT_MAX_ACTIVE_REQUESTS = 10
DEFAULT_POLL_INTERVAL_SECONDS = 1
ECMWF_JOB_LIMIT = 1000

type JobStatus = Literal["accepted", "running", "successful", "failed", "rejected"]


class RemoteJobStatus:
    """Remote job status categories for ECMWF Data Store jobs."""

    ACCEPTED: Final[JobStatus] = "accepted"
    RUNNING: Final[JobStatus] = "running"
    SUCCESSFUL: Final[JobStatus] = "successful"
    FAILED: Final[JobStatus] = "failed"
    REJECTED: Final[JobStatus] = "rejected"

    ACTIVE: Final[tuple[JobStatus, ...]] = (ACCEPTED, RUNNING)
    SAVED: Final[tuple[JobStatus, ...]] = (ACCEPTED, RUNNING, SUCCESSFUL, FAILED, REJECTED)
    PREFILL: Final[tuple[JobStatus, ...]] = (SUCCESSFUL, ACCEPTED, RUNNING)
    NON_ACTIVE: Final[tuple[JobStatus, ...]] = (SUCCESSFUL, FAILED, REJECTED)


class LocalJobStatus:
    """Local job status categories for tracking downloading and processing of ERA5 files."""

    PENDING = "pending"
    DOWNLOADED = "downloaded"
    PROCESSED = "processed"
    FAILED = "failed"


def _request_state_path(save_dir: Path) -> Path:
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
            client = Client(progress=False)
            client.check_authentication()
            logger.info("Login successful!")
    except Exception as e:  # noqa: BLE001
        logger.info(f"Failed to authenticate with ECMWF Data Store: {e:s}")
        logger.info(
            "No valid credentials found. Please enter your Climate Data Store "
            "API key. These will be stored in a file found at "
            "~/.ecmwfdatastoresrc for future use. To obtain your key, go to "
            "https://cds.climate.copernicus.eu/how-to-api.",
        )
        key = click.prompt("Enter your key", type=str, hide_input=True)
        with CONFIG_FILE.open("w") as f:
            f.write(f"url: https://cds.climate.copernicus.eu/api\nkey: {key}\n")
        login_to_ecmwf_datastore()
    return client


def create_data_dir(save_dir: Path | None) -> Path:
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


def setup_request(start_datetime: str | None, end_datetime: str | None) -> tuple[str, dict]:
    """Set up the dataset and request parameters for downloading ERA5 reanalysis data.

    Args:
        start_datetime (str | None): The start datetime for the dataset in YYYY-MM-DD format.
            If None, defaults to the earliest available datetime for the dataset.
        end_datetime (str | None): The end datetime for the dataset in YYYY-MM-DD format.
            If None, defaults to the latest available datetime for the dataset.

    Returns:
        tuple[str, dict]: A tuple containing the dataset name and the request parameters.

    """
    if start_datetime is None:
        start_datetime = "2001-01-01"
    if end_datetime is None:
        end_datetime = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d")
    dataset = "reanalysis-era5-single-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_dewpoint_temperature",
            "2m_temperature",
            "sea_surface_temperature",
            "surface_pressure",
        ],
        "date": f"{start_datetime}/{end_datetime}",
        "time": [f"{hour:02d}:00" for hour in range(0, 24, 1)],
        "area": [MAX_LAT, MIN_LON, MIN_LAT, MAX_LON],
        "data_format": "netcdf",
    }

    return dataset, request


def check_submitted_job(client: Client, request_id: str) -> Remote:
    """Check the status of a submitted job to the ECMWF Data Store.

    Args:
        client (ecmwf.datastores.Client): An instance of the ECMWF Data Store client.
        request_id (str): The request ID for the submitted job.

    Returns:
        Remote: The remote job object.

    Raises:
        RuntimeError: If there is an error checking the job status, such as if
            the request ID is invalid or if there is a problem with the ECMWF
            Data Store service.

    """
    try:
        remote = client.get_remote(request_id)
    except HTTPError as e:
        msg = (
            f"Error checking job with request ID {request_id}. Try to re-run "
            "with the --new-request flag to submit a new request for the "
            "specified date range."
        )
        raise RuntimeError(msg) from e
    return remote


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


def process_data(input_file: Path, output_file: Path) -> Path:
    """Convert a downloaded hourly ERA5 month into a daily-mean NetCDF file.

    Args:
        input_file (Path): The file path to the downloaded hourly ERA5 NetCDF file.
        output_file (Path): The file path to save the processed daily-mean ERA5 NetCDF file.

    Returns:
        Path: The file path to the processed daily-mean ERA5 NetCDF file.

    """
    ds = xr.open_dataset(input_file)
    ds = ds.rename({"valid_time": "time"})
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


def compute_relative_humidity(t2m: xr.DataArray, d2m: xr.DataArray) -> xr.DataArray:
    """Compute relative humidity from 2m temperature and 2m dewpoint temperature.

    Args:
        t2m (xr.DataArray): 2m temperature in Kelvin.
        d2m (xr.DataArray): 2m dewpoint temperature in Kelvin.

    Returns:
        xr.DataArray: Relative humidity in percentage.

    """
    try:
        # drop metpy pint units and convert to percent
        rh = relative_humidity_from_dewpoint(t2m, d2m).metpy.dequantify() * 100
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error computing relative humidity: {e:s}")
        rh = xr.full_like(t2m, fill_value=75.0)  # fill with a default value of 75% if there is an error
    return rh


# def compute_wind_stress(
#     u10: xr.DataArray,
#     v10: xr.DataArray,
#     t2m: xr.DataArray,
#     d2m: xr.DataArray,
#     sst: xr.DataArray,
#     sp: xr.DataArray,
# ) -> xr.DataArray:
#     rh = compute_relative_humidity(t2m, d2m)
#     mag = xr.ufuncs.sqrt(u10**2 + v10**2)
#     c35 = coare_35(
#         u=mag.values,
#         t=t2m.values,
#         rh=rh.values,
#         ts=sst.values,
#         p=sp.values,
#         lat=45,
#         zu=10,
#         zt=2,
#         zq=2,
#         zrf=10,
#     )
#     tau


def monthly_jobs(request: dict) -> list[dict]:
    """Split a request with a date range into multiple requests with monthly date ranges.

    Args:
        request (dict): The original request dictionary with a date range in the "date" key.

    Returns:
        list[dict]: A list of request dictionaries with monthly date ranges.

    """
    start_date_str, end_date_str = request["date"].split("/")
    start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d").astimezone(datetime.UTC)
    end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d").astimezone(datetime.UTC)

    monthly_requests = []
    current_start = start_date
    while current_start < end_date:
        current_end = (current_start + datetime.timedelta(days=32)).replace(day=1) - datetime.timedelta(days=1)
        current_end = min(current_end, end_date)
        monthly_request = request.copy()
        monthly_request["date"] = f"{current_start.strftime('%Y-%m-%d')}/{current_end.strftime('%Y-%m-%d')}"
        monthly_requests.append(monthly_request)
        current_start = current_end + datetime.timedelta(days=1)
    return monthly_requests


def _build_request_state(monthly_requests: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "start_date": pd.Series([r["date"].split("/")[0] for r in monthly_requests], dtype="string"),
            "end_date": pd.Series([r["date"].split("/")[1] for r in monthly_requests], dtype="string"),
            "request_id": pd.Series(["" for _ in monthly_requests], dtype="string"),
            "remote_status": pd.Series(["" for _ in monthly_requests], dtype="string"),
            "local_status": pd.Series([LocalJobStatus.PENDING for _ in monthly_requests], dtype="string"),
            "raw_file": pd.Series(["" for _ in monthly_requests], dtype="string"),
            "processed_file": pd.Series(["" for _ in monthly_requests], dtype="string"),
            "error": pd.Series(["" for _ in monthly_requests], dtype="string"),
        },
    )


def _normalize_request_state(df: pd.DataFrame) -> pd.DataFrame:
    if "status" in df.columns and "remote_status" not in df.columns:
        df = df.rename(columns={"status": "remote_status"})

    expected_columns = [
        "start_date",
        "end_date",
        "request_id",
        "remote_status",
        "local_status",
        "raw_file",
        "processed_file",
        "error",
    ]
    for column in expected_columns:
        if column not in df.columns:
            df[column] = ""
    df = df[expected_columns]
    for column in ["request_id", "remote_status", "local_status", "raw_file", "processed_file", "error"]:
        df[column] = df[column].astype("string").fillna("")
    df["start_date"] = df["start_date"].astype("string").fillna("")
    df["end_date"] = df["end_date"].astype("string").fillna("")
    df["local_status"] = df["local_status"].replace("", LocalJobStatus.PENDING)
    return df


def _load_request_state_file(state_file: Path) -> pd.DataFrame:
    if not state_file.exists():
        msg = f"No request state found at {state_file}. Run era5 submit first."
        raise RuntimeError(msg)
    df = pd.read_csv(state_file, dtype="string")
    return _normalize_request_state(df)


def _load_request_state(state_file: Path, monthly_requests: list[dict]) -> pd.DataFrame:
    if not state_file.exists():
        return _build_request_state(monthly_requests)

    df = _load_request_state_file(state_file)

    expected_dates = [r["date"] for r in monthly_requests]
    existing_dates = [f"{row['start_date']}/{row['end_date']}" for _, row in df.iterrows()]
    if existing_dates != expected_dates:
        msg = "Existing request file does not match the submitted request. Use --new-request to overwrite this request."
        raise RuntimeError(msg)
    return df


def _save_request_state(df: pd.DataFrame, state_file: Path) -> None:
    tmp_file = state_file.with_suffix(".tmp")
    df.to_csv(tmp_file, index=False)
    tmp_file.replace(state_file)


def _active_request_count(df: pd.DataFrame) -> int:
    mask = df["request_id"].ne("") & ~df["remote_status"].str.lower().isin(RemoteJobStatus.NON_ACTIVE)
    return int(mask.sum())


def _state_counts(df: pd.DataFrame) -> dict[str, int]:
    """Return a dictionary with counts of requests in each state category.

    Args:
        df (pd.DataFrame): The request state dataframe.

    Returns:
        dict[str, int]: A dictionary with counts of requests in each state category.

    """
    remote_status = df["remote_status"].str.lower()
    local_status = df["local_status"].str.lower()
    return {
        "total": len(df),
        "pending": int(df["request_id"].eq("").sum()),
        "active": _active_request_count(df),
        "successful": int(remote_status.eq(RemoteJobStatus.SUCCESSFUL).sum()),
        "processed": int(local_status.eq(LocalJobStatus.PROCESSED).sum()),
        "remote_failed": int(
            remote_status.isin({RemoteJobStatus.FAILED, RemoteJobStatus.REJECTED}).sum(),
        ),
        "local_failed": int(local_status.eq(LocalJobStatus.FAILED).sum()),
    }


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
        status=list(RemoteJobStatus.ACTIVE),
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
        status=list(RemoteJobStatus.SAVED),
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
    monthly_requests: list[dict],
    df: pd.DataFrame,
    state_file: Path,
) -> tuple[pd.DataFrame, int]:
    """Populate state rows from recent matching jobs to avoid duplicate submissions.

    Args:
        client (Client): An authenticated ECMWF Data Store client.
        monthly_requests (list[dict]): The list of monthly request dictionaries to match against recent jobs.
        df (pd.DataFrame): The current request state dataframe to update with matched request IDs and statuses.
        state_file (Path): The path to the request state file for saving updates.

    Returns:
        tuple[pd.DataFrame, int]: The updated request state dataframe and the count of matched existing jobs.

    """
    jobs = client.get_jobs(
        100,  # number of recent jobs to check, 100 is ~30s for receipt retrieval
        sortby="-created",
        status=list(RemoteJobStatus.PREFILL),
    ).json.get("jobs", [])

    # if there are no recent jobs, return early to avoid unnecessary receipt retrieval step
    if not jobs:
        return df, 0

    # get receipts for all existing jobs, this is the step that takes the longest
    receipts = [
        client.get_receipt(str(job.get("jobID", ""))) for job in tqdm(jobs, desc="Fetching receipts for recent jobs")
    ]

    # get requests and convert all lists to tuples for use as dict keys in lookup table
    existing_requests = [
        {
            key: tuple(value) if isinstance(value, list) else value
            for key, value in receipt.get("request").items()  # ty:ignore[unresolved-attribute]
        }
        for receipt in receipts
    ]

    # build a lookup table of recent requests to receipts for quick matching against monthly requests
    lookup = {
        (request["date"], request["time"], request["area"], request["variable"], request["product_type"]): request
        if request is not None
        and isinstance(request, dict)
        and "date" in request
        and "time" in request
        and "area" in request
        and "variable" in request
        and "product_type" in request
        else None
        for request in existing_requests
    }

    # iterate through monthly requests and fill in request_id and remote_status from lookup table when a match is found
    matched_existing_jobs = 0
    for i, request in enumerate(monthly_requests):
        key = (
            request.get("date"),
            tuple(request.get("time", [])),
            tuple(request.get("area", [])),
            tuple(request.get("variable", [])),
            tuple(request.get("product_type", [])),
        )
        receipt = lookup.get(key)
        if receipt is None:
            continue
        df.loc[i, "request_id"] = receipt.get("jobID", "")
        df.loc[i, "remote_status"] = receipt.get("status", "")
        df.loc[i, "error"] = ""
        matched_existing_jobs += 1

    _save_request_state(df, state_file)
    return df, matched_existing_jobs


def _poll_remote_statuses(client: Client, df: pd.DataFrame, state_file: Path) -> pd.DataFrame:
    """Poll the ECMWF Data Store for status updates on active jobs and updates the request state dataframe accordingly.

    Args:
        client (Client): An authenticated ECMWF Data Store client.
        df (pd.DataFrame): The current request state dataframe to update with polled statuses.
        state_file (Path): The path to the request state file for saving updates.

    Returns:
        pd.DataFrame: The updated request state dataframe with the latest remote statuses.

    """
    # get indices where there is a request_id and the remote_status is still active (accepted/running) or unknown
    indices = [
        i
        for i, row in df.iterrows()
        if str(row["request_id"]).strip() and str(row["remote_status"]).lower() not in RemoteJobStatus.NON_ACTIVE
    ]

    # if there are no active jobs to poll, return early
    if not indices:
        return df

    # get most recent 1000 jobs
    jobs_payload = client.get_jobs(limit=ECMWF_JOB_LIMIT, sortby="-created").json
    jobs = jobs_payload.get("jobs", [])
    jobs_by_id = {job.get("jobID"): job for job in jobs if job.get("jobID") is not None}

    # iterate through active/unknown jobs and update statuses in the
    # dataframe according to the polled job statuses
    for _, row in df.loc[indices].iterrows():
        # extract relevant job from recent jobs list, if it exists, otherwise continue
        job = jobs_by_id.get(row["request_id"])
        if job is None:
            continue
        job_status = _extract_job_status(job)
        if not job_status:
            continue

        # update the job status in the dataframe for this request
        row["remote_status"] = job_status

        # add an error if the remote job failed
        if job_status == RemoteJobStatus.FAILED:
            row["local_status"] = LocalJobStatus.FAILED
            row["error"] = f"Remote job failed for request ID {row['request_id']}"

    _save_request_state(df, state_file)
    return df


def _submit_one_pending_request(
    client: Client,
    dataset: str,
    monthly_requests: list[dict],
    df: pd.DataFrame,
    state_file: Path,
) -> pd.DataFrame:
    """Submit one pending request to the ECMWF Data Store and updates the request state dataframe accordingly.

    Args:
        client (Client): An authenticated ECMWF Data Store client.
        dataset (str): The name of the dataset to request.
        monthly_requests (list[dict]): The list of monthly request dictionaries to submit from.
        df (pd.DataFrame): The current request state dataframe to update with the submitted request ID and status.
        state_file (Path): The path to the request state file for saving updates.

    Returns:
        pd.DataFrame: The updated request state dataframe.

    """
    pending_indices = df.index[df["request_id"] == ""].tolist()
    if not pending_indices:
        return df

    i = pending_indices[0]
    request = monthly_requests[i]
    remote = client.submit(dataset, request)
    df.loc[i, "request_id"] = remote.request_id
    df.loc[i, "remote_status"] = remote.status
    df.loc[i, "error"] = ""
    _save_request_state(df, state_file)
    return df


def _get_existing_datetimes(save_dir: Path, update_path: str) -> xr.DataArray:
    """Get the datetime values from the existing file.

    Args:
        save_dir (Path): The directory where the dataset files are saved.
        update_path (str): The path to the existing file to update.

    Returns:
        xr.DataArray: The datetime values from the existing file.

    Raises:
        FileNotFoundError: If the update file does not exist in the save directory.

    """
    if not Path(save_dir / update_path).exists():
        msg = f"Update file {update_path} not found in save directory {save_dir}."
        raise FileNotFoundError(msg)

    ds_existing = xr.open_dataset(save_dir / update_path)
    datetime_existing = ds_existing["time"]

    return datetime_existing


def _download_and_process_ready_requests(
    client: Client,
    df: pd.DataFrame,
    save_dir: Path,
    state_file: Path,
    update_file: str | None = None,
) -> pd.DataFrame:
    """Download and process datasets for requests with successful remote jobs.

    Only requests not yet processed locally are handled.

    Args:
        client (Client): An authenticated ECMWF Data Store client.
        df (pd.DataFrame): The current request state dataframe to update with
            downloaded file paths and processing statuses.
        save_dir (Path): The directory to save the downloaded and processed datasets.
        state_file (Path): The path to the request state file for saving updates.
        update_file (str | None): The path to an existing file to update with new data.

    Returns:
        pd.DataFrame: The updated request state dataframe with downloaded file paths and processing statuses.

    """
    existing_datetimes = _get_existing_datetimes(save_dir, update_file) if update_file is not None else None
    if existing_datetimes is not None:
        logger.info(f"Checking existing file {update_file} for already-downloaded dates to avoid duplicates...")
        rows_to_skip = []
        for i, row in df.iterrows():
            dates = existing_datetimes.sel(time=slice(row["start_date"], row["end_date"]))
            if len(dates) == 0:
                continue
            rows_to_skip.append(i)
        if rows_to_skip:
            logger.info(
                f"Found existing data for {len(rows_to_skip)} requests in the provided update file. "
                "These requests will be skipped during downloading and processing.",
            )
            df.loc[rows_to_skip, "local_status"] = LocalJobStatus.PROCESSED
            _save_request_state(df, state_file)

    # iterate through requests and download/process those that are successful
    for _, row in tqdm(df.iterrows(), desc="Downloading and processing ready requests", total=len(df)):
        # skip rows that don't have a successful remote status or have already been processed locally
        if row["remote_status"] != RemoteJobStatus.SUCCESSFUL or row["local_status"] == LocalJobStatus.PROCESSED:
            continue

        # setup file names
        raw_file = save_dir / f"era5_reanalysis_{row['start_date']}_{row['end_date']}_raw.nc"
        processed_file = save_dir / f"era5_reanalysis_{row['start_date']}_{row['end_date']}.nc"

        # download results if not already downloaded
        if row["local_status"] == LocalJobStatus.PENDING:
            remote = check_submitted_job(client, row["request_id"])
            results = _retrieve_results(remote)
            results.download(str(raw_file))
            row["raw_file"] = str(raw_file)
            row["local_status"] = LocalJobStatus.DOWNLOADED
            row["error"] = None

        # process data if downloaded but not yet processed
        if row["local_status"] == LocalJobStatus.DOWNLOADED:
            try:
                process_data(raw_file, processed_file)
                # remove raw file after processing
                if raw_file.exists():
                    raw_file.unlink()
            # If anything goes wrong, mark local status as failed and log the error
            # but do not raise so other requests can continue
            except Exception as e:  # noqa: BLE001
                logger.error(f"Error processing file for request ID {row['request_id']}: {e:s}")
                row["local_status"] = LocalJobStatus.FAILED
                row["error"] = str(e)
                continue
    _save_request_state(df, state_file)
    return df


def submit_era5(
    save_dir: Path | None = None,
    start_datetime: str | None = None,
    end_datetime: str | None = None,
    new_request: bool = False,  # noqa: FBT001, FBT002
    max_active_requests: int = DEFAULT_MAX_ACTIVE_REQUESTS,
) -> None:
    """Submit ERA5 remote jobs without downloading local files.

    Args:
        save_dir (Path | None): The directory to save the downloaded dataset.
            If None, defaults to a "data" directory at the package level.
        start_datetime (str | None): The start datetime for the dataset in
            YYYY-MM-DD format. If None, defaults to the earliest available
            datetime for the dataset.
        end_datetime (str | None): The end datetime for the dataset in
            YYYY-MM-DD format. If None, defaults to the latest available
            datetime for the dataset.
        new_request (bool): Whether to start a new request and overwrite any existing request state file. Default False.
        max_active_requests (int): Maximum number of active ECMWF requests to keep in flight.

    """
    # setup client, request, and save directory
    client = login_to_ecmwf_datastore()
    dataset, request = setup_request(start_datetime, end_datetime)
    monthly_requests = monthly_jobs(request)
    save_dir = create_data_dir(save_dir)
    state_file = _request_state_path(save_dir)
    poll_interval_seconds = DEFAULT_POLL_INTERVAL_SECONDS

    # get request state, either by building a new one or loading from an existing state file
    if new_request or not state_file.exists():
        df = _build_request_state(monthly_requests)
        _save_request_state(df, state_file)
    else:
        df = _load_request_state(state_file, monthly_requests)

    # Check existing jobs to prefill request IDs and statuses for matching
    # recent jobs and avoid unnecessary duplicate submissions.
    df, matched_existing_jobs = _prefill_submitted_requests_from_recent_jobs(
        client=client,
        monthly_requests=monthly_requests,
        df=df,
        state_file=state_file,
    )

    # check how many remaining jobs we will have to submit and let the user know
    remaining_to_submit = int(df["request_id"].eq("").sum())
    logger.info(
        f"Prefill summary: using {matched_existing_jobs} existing jobs; submitting remaining {remaining_to_submit}.",
    )

    # check to see if the user will end up over 1000 requests before submitting any new jobs
    if not _under_total_jobs_limit(client, remaining_to_submit):
        # save and exit if user would be over the total jobs limit
        _save_request_state(df, state_file)
        return

    # If all requests already have IDs, they are already submitted, so exit
    # early without starting the submission manager loop.
    if not df["request_id"].eq("").any():
        logger.info(
            "All requests are already submitted. View progress at "
            "https://cds.climate.copernicus.eu/requests. Run `uv run "
            "datasets.py era5 download` when jobs are ready.",
        )
        return

    try:
        logger.info("Starting ERA5 submission...")
        logger.info(
            f"Submission uses active-job cap {max_active_requests} (accepted + running). "
            "If exited, run `uv run datasets.py era5 submit` later to resume.",
        )

        with tqdm(total=len(df), desc="Submission progress") as progress:
            progress.refresh()
            while True:
                # get pending, submitted, and active counts for progress bar
                counts = _state_counts(df)

                # update progress bar with counts
                progress.n = counts["successful"]
                progress.set_postfix(
                    active=f"{counts['active']}/{max_active_requests}",
                    pending=counts["pending"],
                    refresh=False,
                )
                progress.refresh()

                # break the loop if there are no pending requests left (all requests have been submitted)
                if counts["pending"] == 0:
                    break

                # enforce active job cap by waiting to submit if we are at or above the max active requests limit
                if counts["active"] >= max_active_requests:
                    time.sleep(poll_interval_seconds)
                    continue

                # if we're under the active job cap, submit one pending request
                df = _submit_one_pending_request(
                    client=client,
                    dataset=dataset,
                    monthly_requests=monthly_requests,
                    df=df,
                    state_file=state_file,
                )

                # re-check immediately after each submit so we never exceed limit
                active_after = _get_active_job_count(client)
                if active_after >= max_active_requests:
                    time.sleep(poll_interval_seconds)
                    continue

                # wait just a bit longer to be sure the newly submitted jobs have registered before we check again
                time.sleep(poll_interval_seconds)

        # all done, save request state and exit
        _save_request_state(df, state_file)
        logger.info(
            "All requests have been submitted. View progress at "
            "https://cds.climate.copernicus.eu/requests. Run `uv run "
            "datasets.py era5 download` once all requests are ready to "
            "download files.",
        )
    except KeyboardInterrupt:
        # make sure request state is saved on interrupt
        _save_request_state(df, state_file)
        logger.warning(
            f"Submission interrupted. Progress saved to {state_file}. Run `uv run datasets.py era5 submit` to resume.",
        )
        return
    return


def download_era5(
    save_dir: Path | None = None,
    update_file: str | None = None,
) -> None:
    """Download and process ERA5 files after all remote jobs are successful.

    Args:
        save_dir (Path | None): The directory to save the downloaded dataset.
            If None, defaults to a "data" directory at the package level.
            There must be a request state file in this directory by running `uv run datasets.py era5 submit` first.
        update_file (str | None): Path to an existing netCDF file to update. If None, a new file will be created.


    """
    # setup client and save directory
    client = login_to_ecmwf_datastore()
    save_dir = create_data_dir(save_dir)
    state_file = _request_state_path(save_dir)

    if not state_file.exists():
        logger.warning(f"No request state found at {state_file}. Run `uv run datasets.py era5 submit` first.")
    else:
        df = _load_request_state_file(state_file)

    # update and save remote statuses one last time before starting downloads, in case there have been any changes
    df = _poll_remote_statuses(client, df, state_file)
    _save_request_state(df, state_file)
    # get counts of requests in each state category for logging
    counts = _state_counts(df)

    # check for any active remote jobs before starting downloads, and exit if there are still active jobs
    if counts["active"] > 0:
        logger.info(
            f"Cannot start download yet, waiting on {counts['active']} active jobs. "
            "Please check your CDS account at https://cds.climate.copernicus.eu/requests for more details. "
            "Re-run this command once all jobs are completed to download the datasets.",
        )
        _save_request_state(df, state_file)
        return

    # check for any failed remote jobs before starting downloads, and warn the user before downloading
    # but allow them to continue if they want to download any successful requests
    if counts["remote_failed"] > 0:
        logger.warning(
            f"{counts['remote_failed']} jobs failed or were cancelled on the CDS server."
            "Please check your CDS account at https://cds.climate.copernicus.eu/requests for more details.",
        )
        if not click.confirm("Do you want to continue with downloading any successful requests?", default=True):
            return

    try:
        logger.info("Starting ERA5 download...")
        # this function is where the actual downloading and processing happens, all the rest is just checks
        df = _download_and_process_ready_requests(client, df, save_dir, state_file, update_file)

        counts = _state_counts(df)
        _save_request_state(df, state_file)

        # warn on any local failures, but still allow for successful downloads to be used
        if counts["local_failed"] > 0:
            logger.warning(
                f"{counts['local_failed']} downloads failed. Fix the issue and rerun era5 download to resume.",
            )
            return

        # make sure they know everything is successful :)
        if counts["processed"] == counts["total"]:
            logger.info("All downloads complete and processed successfully.")
            return

        logger.warning(
            "Unknown download issue; some months are not processed yet. "
            "Examine your CDS account at https://cds.climate.copernicus.eu/requests "
            f"and the state file at {state_file} for more details.",
        )
    except KeyboardInterrupt:
        _save_request_state(df, state_file)
        # make sure request state is saved on interrupt
        logger.warning(
            f"Download interrupted. Progress saved to {state_file}.  Run `uv run datasets.py era5 download` to resume.",
        )
        return
    return
