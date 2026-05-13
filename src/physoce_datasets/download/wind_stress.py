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
from typing import TYPE_CHECKING, Final, Literal, TypedDict, cast

import click
import numpy as np
import xarray as xr
from ecmwf.datastores import Client
from metpy.calc import relative_humidity_from_dewpoint
from metpy.units import units
from pycoare import coare_35
from tqdm import tqdm

from physoce_datasets.logging import logger
from physoce_datasets.util import get_area_str

from ._base import _Downloader

if TYPE_CHECKING:
    from physoce_datasets.util import AreaDict

CONFIG_FILE = Path.home() / ".ecmwfdatastoresrc"

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
    combined_file: str | None
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


class _RequestStateManager:
    """Class for managing the state of ERA5 data requests, including tracking request parameters, statuses, and handling updates."""

    REQUEST_STATE_FILE = "submitted_requests.json"

    def __init__(self, save_dir: Path, start_dates: list[str], end_dates: list[str], area: AreaDict) -> None:
        """Initialize the request state manager and load or build the request states."""
        self.save_dir = save_dir
        self.start_dates = start_dates
        self.end_dates = end_dates
        self.area = area
        self.state_file = save_dir / self.REQUEST_STATE_FILE
        if self.state_file.exists():
            self.states = self.load_request_state()
            self.states = self.update_request_state()
        else:
            self.states = self.build_request_state()
            self.save_request_state()
        self._update_completion_status_flags()

    def load_request_state(self) -> list[_State]:
        """Load the request states from a JSON file.

        Returns:
            list[_State]: The list of request states loaded from the file.

        """
        with self.state_file.open() as f:
            states = json.load(f)
        return states

    def save_request_state(self) -> None:
        """Save the current request states to a JSON file."""
        tmp_file = self.state_file.with_suffix(".tmp")
        with tmp_file.open("w") as f:
            json.dump(self.states, f, indent=4)
        tmp_file.replace(self.state_file)

    def build_request_state(
        self,
    ) -> list[_State]:
        """Build the initial request states list from the start and end dates provided by the user.

        Returns:
            list[_State]: The initial request states with request parameters filled in for each date range.

        """
        self.states: list[_State] = [
            {
                "start_date": s,
                "end_date": e,
                "request_id": None,
                "remote_status": None,
                "download_status": None,
                "processing_status": None,
                "combined_file": None,
                "request": None,
            }
            for s, e in zip(self.start_dates, self.end_dates, strict=True)
        ]
        states = self.create_requests()
        return states

    def update_request_state(
        self,
    ) -> list[_State]:
        """Update the request states with the new date range from the user.

        Returns:
            list[_State]: The updated request states with any new date ranges added.

        """
        existing_date_ranges = {(s["start_date"], s["end_date"]) for s in self.states}
        for start_date, end_date in zip(self.start_dates, self.end_dates, strict=True):
            if (start_date, end_date) not in existing_date_ranges:
                self.states.append(
                    {
                        "start_date": start_date,
                        "end_date": end_date,
                        "request_id": None,
                        "remote_status": None,
                        "download_status": None,
                        "processing_status": None,
                        "combined_file": None,
                        "request": None,
                    },
                )
        states = self.create_requests()
        states.sort(key=operator.itemgetter("start_date"))
        self.save_request_state()
        return states

    def create_requests(self) -> list[_State]:
        """Fill in the request parameters for any states that don't already have them.

        Returns:
            list[_State]: The updated request states with request parameters filled in for any states that were missing them.

        """
        for s in self.states:
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
                    "area": [self.area["lat_max"], self.area["lon_min"], self.area["lat_min"], self.area["lon_max"]],
                    "data_format": "netcdf",
                }
        return self.states

    def update_dates(
        self,
        start_dates: list[str],
        end_dates: list[str],
    ) -> None:
        """Update the start and end dates for the request states.

        Args:
            start_dates (list[str]): The list of new start dates for the request states.
            end_dates (list[str]): The list of new end dates for the request states.

        """
        self.start_dates = start_dates
        self.end_dates = end_dates
        self.states = self.update_request_state()

    def update_status_for_request(
        self,
        client: Client,
        request_id: str,
    ) -> None:
        """Check the status of a submitted request and update the request states list accordingly.

        Args:
            client (Client): An authenticated ECMWF Data Store client.
            request_id (str): The request ID for the submitted job to check the status of.

        """
        s = next((s for s in self.states if s.get("request_id") == request_id), None)
        if s is None:
            logger.warning(f"Request for {request_id} not found.")
            return
        try:
            remote = client.get_remote(request_id)
            s["remote_status"] = remote.status
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error checking status for request ID {request_id}: {e}")
            s["remote_status"] = "unknown"
        self._update_completion_status_flags()

    def update_statuses_for_all_requests(
        self,
        client: Client,
    ) -> None:
        """Check the status of all submitted requests and update the request states list accordingly.

        Args:
            client (Client): An authenticated ECMWF Data Store client.

        """
        for request_id in [s.get("request_id") for s in self.states]:
            if request_id is not None:
                self.update_status_for_request(client, request_id)
        self._update_completion_status_flags()

    def _update_completion_status_flags(self) -> None:
        """Update the overall completion status flags based on the current states."""
        self.all_submitted = all(s.get("request_id") is not None for s in self.states)
        self.all_completed = all(s.get("remote_status") in _RemoteJobStatus.FINISHED for s in self.states)
        self.all_downloaded = all(s.get("download_status") == _LocalJobStatus.DOWNLOADED for s in self.states)
        self.all_processed = all(s.get("processing_status") == _LocalJobStatus.PROCESSED for s in self.states)
        self.all_combined = all(s.get("combined_file") is not None for s in self.states)


class WindStressDownloader(_Downloader):
    """Downloader for ERA5-based wind stress datasets from the ECMWF Data Store."""

    def __init__(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        area: str | None = None,
        save_dir: str | None = None,
        save_file: str | None = None,
    ) -> None:
        """Initialize the downloader and set up the ECMWF Data Store client and request state manager."""
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            area=area,
            save_dir=save_dir,
            save_file=save_file,
        )
        self.client = login_to_ecmwf_datastore()
        self.start_dates, self.end_dates = self._split_monthly_jobs(self.start_date, self.end_date)
        self.request_manager = _RequestStateManager(
            save_dir=self.save_dir,
            start_dates=self.start_dates,
            end_dates=self.end_dates,
            area=self.area,
        )
        self.request_manager.update_statuses_for_all_requests(self.client)
        self.request_manager.save_request_state()

        if not self.request_manager.all_submitted:
            self.matched_existing_jobs = self._prefill_submitted_requests_from_recent_jobs()

        # if save_file is not provided, default to a file name based on the date range, otherwise use the provided file name
        if save_file is None:
            area_str = get_area_str(self.area)
            save_file = f"era5_reanalysis_combined_{self.start_date}_{self.end_date}_{area_str}.nc"
        self.save_file_path = self._create_save_file(
            self.save_dir,
            save_file,
        )

    def _delete_expired_requests(self) -> None:
        jobs = self.client.get_jobs(
            ECMWF_JOB_LIMIT,
            sortby="-created",
            status=_RemoteJobStatus.SUCCESSFUL,
        ).json.get("jobs", [])
        expired_job_ids = [
            job.get("jobID") for job in jobs if job.get("metadata").get("results").get("type") == "results expired"
        ]
        self.client.delete(*expired_job_ids)

    def _get_active_job_count(self) -> int:
        """Count accepted/running jobs for active-cap enforcement.

        Args:
            client (Client): An authenticated ECMWF Data Store client.

        Returns:
            int: The number of active jobs currently in the ECMWF Data Store queue for the user.

        """
        jobs = self.client.get_jobs(
            ECMWF_JOB_LIMIT,
            sortby="-created",
            status=list(_RemoteJobStatus.ACTIVE),
        ).json.get("jobs", [])
        return len(jobs)

    def _check_under_total_jobs_limit(self, number_to_submit: int) -> bool:
        """Check existing jobs and enforce total saved job cap before submitting new jobs.

        Args:
            number_to_submit (int): The number of new jobs intended to be submitted.

        Returns:
            bool: True if under the total jobs limit and safe to submit, False if submitting would exceed the limit.

        """
        jobs = self.client.get_jobs(
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

    def _prefill_submitted_requests_from_recent_jobs(self) -> int:
        """Populate states rows from recent matching jobs to avoid duplicate submissions.

        Returns:
            int: The number of states that were successfully pre-filled with existing jobs.

        """
        jobs = self.client.get_jobs(
            100,  # number of recent jobs to check, 100 is ~30s for receipt retrieval
            sortby="-created",
            status=list(_RemoteJobStatus.PREFILL),
        ).json.get("jobs", [])

        # if there are no recent jobs, return early to avoid unnecessary receipt retrieval step
        if not jobs:
            return 0

        # load all remotes from api, this is the step that takes the longest
        remotes = [
            self.client.get_remote(job.get("jobID")) for job in tqdm(jobs, desc="Fetching receipts for recent jobs")
        ]

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
        for s in self.request_manager.states:
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
        self.request_manager.update_statuses_for_all_requests(self.client)
        self.request_manager.save_request_state()
        return matched_existing_jobs

    def _submit_one_pending_request(self) -> str:
        """Submit one pending request to the ECMWF Data Store and updates the request states dataframe accordingly.

        Returns:
            str: The submitted request ID.

        """
        # find the first index in the states where the request_id is None, meaning it has not been submitted yet
        pending_indices = [i for i, s in enumerate(self.request_manager.states) if s.get("request_id") is None]
        # if everything has been submitted, return the states as is
        if not pending_indices:
            return ""

        # grab the first pending request and submit it
        i = pending_indices[0]
        # cast type for submission, this is safe because we always fill in the request before submission
        request = cast("dict", self.request_manager.states[i]["request"])
        remote = self.client.submit(DATASET, request)

        # update the states with the returned request ID and initial status
        self.request_manager.states[i]["request_id"] = remote.request_id
        self.request_manager.states[i]["remote_status"] = remote.status
        self.request_manager.save_request_state()

        return remote.request_id

    def _submit_requests(
        self,
        remaining_to_submit: int,
    ) -> None:
        """Submit pending requests to the ECMWF Data Store while enforcing active and total job caps.

        Args:
            remaining_to_submit (int): The number of requests that still need to be submitted.
            matched_existing_jobs (int): The number of requests that were pre-filled with existing jobs.

        """
        submitted = 0
        with tqdm(total=remaining_to_submit + self.matched_existing_jobs, desc="Submission progress") as progress:
            progress.n = self.matched_existing_jobs
            progress.refresh()
            while True:
                # update progress bar with counts
                progress.n = self.matched_existing_jobs + submitted
                progress.refresh()

                # break the loop if all requests are submitted (all requests have ids)
                if self.request_manager.all_submitted:
                    break

                # enforce active job cap by waiting to submit if we are at or above the max active requests limit
                active_before = self._get_active_job_count()
                if active_before >= MAX_ACTIVE_REQUESTS:
                    time.sleep(POLL_INTERVAL_SECONDS)
                    continue

                # if we're under the active job cap, submit one pending request
                request_id = self._submit_one_pending_request()

                # after submission, check if the request was successful and update the status
                if request_id:
                    submitted += 1
                    self.request_manager.update_status_for_request(
                        client=self.client,
                        request_id=request_id,
                    )
                self.request_manager.save_request_state()

                # wait just a bit longer to be sure the newly submitted jobs have registered before we check again
                time.sleep(POLL_INTERVAL_SECONDS)

        self.request_manager.save_request_state()

    def _download_file(self, state: _State, raw_file: Path) -> None:
        """Download the raw file for a completed request.

        Args:
            state (_State): The request state for the job to download.
            raw_file (Path): The path to the raw file to download.

        """
        # download results if not already downloaded
        if (
            state["download_status"] != _LocalJobStatus.DOWNLOADED
            and state["remote_status"] == _RemoteJobStatus.SUCCESSFUL
            and state["request_id"] is not None
        ):
            try:
                remote = self.client.get_remote(state["request_id"])
                results = remote.get_results()
                results.download(str(raw_file))
                state["download_status"] = _LocalJobStatus.DOWNLOADED
            except Exception as e:  # noqa: BLE001
                logger.error(f"Error downloading file for request ID {state['request_id']}: {e}")
                state["download_status"] = _LocalJobStatus.FAILED

    def _process_file(self, state: _State, raw_file: Path, processed_file: Path) -> None:
        """Process a downloaded raw file into a cleaned daily-mean file.

        Args:
            state (_State): The request state for the job to process.
            raw_file (Path): The path to the downloaded raw file.
            processed_file (Path): The path to the processed file.

        """
        # process data if downloaded but not yet processed
        if (
            state["download_status"] == _LocalJobStatus.DOWNLOADED
            and state["processing_status"] != _LocalJobStatus.PROCESSED
        ):
            try:
                self._process_data(raw_file, processed_file)
                state["processing_status"] = _LocalJobStatus.PROCESSED
                # remove raw file after processing
                if raw_file.exists():
                    raw_file.unlink()
            # If anything goes wrong, mark local status as failed and log the error
            # but do not raise so other requests can continue
            except Exception as e:  # noqa: BLE001
                logger.exception(f"Error processing file for request ID {state['request_id']}: {e}")
                state["processing_status"] = _LocalJobStatus.FAILED

    def _download_and_process_ready_requests(self) -> None:  # noqa: C901
        """Download and process datasets for requests with successful remote jobs.

        Only requests not yet processed locally are handled.
        """
        # iterate through requests and download/process those that are successful
        processed_files: list[Path] = []
        for state in tqdm(self.request_manager.states, desc="Downloading and processing ready requests"):
            self.request_manager.save_request_state()

            # skip jobs that don't have a successful remote status or already have a combined file (meaning they were processed in a previous run)
            if state["remote_status"] != _RemoteJobStatus.SUCCESSFUL or state["combined_file"] is not None:
                continue

            # setup file names
            raw_file = self.save_file_path.with_name(
                self.save_file_path.stem + f"_raw_{state['start_date']}_{state['end_date']}.nc",
            )
            processed_file = self.save_file_path.with_name(
                self.save_file_path.stem + f"_processed_{state['start_date']}_{state['end_date']}.nc",
            )

            if state["download_status"] != _LocalJobStatus.DOWNLOADED:
                self._download_file(state, raw_file)
            if state["download_status"] != _LocalJobStatus.DOWNLOADED:
                continue

            if state["processing_status"] != _LocalJobStatus.PROCESSED:
                self._process_file(state, raw_file, processed_file)
            if state["processing_status"] != _LocalJobStatus.PROCESSED:
                continue

            processed_files.append(processed_file)

        logger.info("Combining processed files into one dataset.")
        # this whole section for combining with existing files feels fragile and a bit sloppy
        # only get processed files that exist
        processed_files = [f for f in processed_files if Path(f).exists()]
        processed_ds = xr.open_mfdataset(processed_files, chunks="auto", compat="no_conflicts", join="outer")

        combined_files = {
            Path(s["combined_file"]) for s in self.request_manager.states if s["combined_file"] is not None
        }

        if combined_files:
            existing_ds = xr.open_mfdataset(list(combined_files), chunks="auto", compat="no_conflicts", join="outer")
            ds = xr.merge([existing_ds, processed_ds], compat="no_conflicts", join="outer")
            existing_ds.close()
            ds.to_netcdf(self.save_file_path)
            ds.close()
        else:
            # otherwise save all processed datasets as coombined file
            processed_ds.to_netcdf(self.save_file_path)
        processed_ds.close()
        self.request_manager.states = [
            {
                **s,
                "combined_file": str(self.save_file_path)
                if s["processing_status"] == _LocalJobStatus.PROCESSED
                else None,
            }
            for s in self.request_manager.states
        ]
        self.request_manager.update_statuses_for_all_requests(self.client)
        self.request_manager.save_request_state()

        # now we can remove the individual processed datasets too
        for f in processed_files:
            if f.exists():
                f.unlink()
        # and the old combined files if they exist and are different from the new combined file
        for f in combined_files:
            if f != self.save_file_path and f.exists():
                f.unlink()

    def _submit_manager(self) -> None:
        """Submit ERA5 remote jobs without downloading local files."""
        # If all requests already have IDs, they are already submitted
        # exit early without starting the submission manager loop.
        if self.request_manager.all_submitted:
            return

        # check how many remaining jobs we will have to submit and let the user know
        remaining_to_submit = sum(1 for s in self.request_manager.states if not s.get("request_id"))
        logger.info(
            f"Using {self.matched_existing_jobs} existing jobs; submitting remaining {remaining_to_submit}.",
        )

        # check to see if the user will end up over 1000 requests before submitting any new jobs
        if not self._check_under_total_jobs_limit(remaining_to_submit):
            # save and exit if user would be over the total jobs limit
            self.request_manager.save_request_state()
            return

        try:
            logger.info("Starting ERA5 submission. If exited, re-run the command to resume.")
            self._submit_requests(remaining_to_submit=remaining_to_submit)

            # all done, save request states and exit
            self.request_manager.save_request_state()
            logger.info(
                "All requests have been submitted.",
            )
        except KeyboardInterrupt:
            # make sure request states is saved on interrupt
            self.request_manager.save_request_state()
            logger.warning(
                f"Submission interrupted. Progress saved to {self.request_manager.state_file}. Re-run the command to resume.",
            )
            return
        return

    def download(self) -> None:
        """Submit, download and process wind stress files."""
        if not self.request_manager.all_submitted:
            self._submit_manager()

        if self.request_manager.all_combined:
            logger.info("All processing already completed.")
            return

        self.request_manager.update_statuses_for_all_requests(self.client)

        if not self.request_manager.all_completed:
            logger.info(
                "Waiting for all requests to finish on the CDS server before downloading. "
                "View progress at https://cds.climate.copernicus.eu/requests. "
                "You can safely exit and re-run this command once your requests have finished."
            )
            while True:
                self.request_manager.update_statuses_for_all_requests(self.client)
                # all_completed is updated in the above call, so we can check it here to break the loop when everything is finished
                if self.request_manager.all_completed:
                    break
                time.sleep(POLL_INTERVAL_SECONDS)

        self.request_manager.save_request_state()

        # check for any failed remote jobs before starting downloads, and warn the user before downloading
        # but allow them to continue if they want to download any successful requests
        if any(s.get("remote_status") == _RemoteJobStatus.FAILED for s in self.request_manager.states):
            failed = sum(1 for s in self.request_manager.states if s.get("remote_status") == _RemoteJobStatus.FAILED)
            logger.warning(
                f"{failed} jobs failed or were cancelled on the CDS server."
                "Please check your CDS account at https://cds.climate.copernicus.eu/requests for more details.",
            )
            if not click.confirm("Do you want to continue with downloading any successful requests?", default=True):
                return

        try:
            logger.info("Starting download and processing.")
            # this function is where the actual downloading and processing happens, all the rest is just checks
            self._download_and_process_ready_requests()

            self.request_manager.update_statuses_for_all_requests(self.client)
            self.request_manager.save_request_state()

            # make sure they know everything is successful :)
            if self.request_manager.all_combined:
                logger.info("All downloads complete and combined successfully.")
                return

            logger.warning(
                "Unknown download issue; some months are not processed yet. "
                "Examine your CDS account at https://cds.climate.copernicus.eu/requests "
                f"and the states file at {self.request_manager.state_file} for more details.",
            )

        except KeyboardInterrupt:
            self.request_manager.save_request_state()
            # make sure request states is saved on interrupt
            logger.warning(
                f"Download interrupted. Progress saved to {self.request_manager.state_file}. Re-run the command to resume.",
            )
            return
        return

    @staticmethod
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

    @staticmethod
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
                WindStressDownloader._compute_wind_stress,
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

    @staticmethod
    def _compute_relative_humidity(t2m: np.ndarray, d2m: np.ndarray) -> np.ndarray:
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

    @staticmethod
    def _compute_wind_stress(
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
        rh = WindStressDownloader._compute_relative_humidity(t2m, d2m)
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
