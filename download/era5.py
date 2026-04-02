from __future__ import annotations

import contextlib
import datetime
import os
import time
from pathlib import Path

import click
from ecmwf.datastores import Client

CONFIG_FILE = Path.home() / ".ecmwfdatastoresrc"


FILE_DIR = Path(__file__).parent
MIN_LON = 210
MAX_LON = 240
MIN_LAT = 30
MAX_LAT = 60


def login_to_ecmwf_datastore() -> Client:
    """Performs a login to the ECMWF Data Store, prompting the user for credentials if not saved.

    Credentials will be stored in a file found at ~/.ecmwfdatastoresrc for future use.
    To obtain your key, go to https://cds.climate.copernicus.eu/how-to-api

    Overrides the default ECMWF Data Store login behavior to provide more user-friendly prompts and messages.

    Args:
        client (ecmwf.datastores.Client): An instance of the ECMWF Data Store client.

    Returns:
        ecmwf.datastores.Client: An authenticated instance of the ECMWF Data Store client.

    """
    click.echo("Attempting login...")
    try:
        with contextlib.redirect_stdout(Path(os.devnull).open("w", encoding="utf-8")) and contextlib.redirect_stderr(
            Path(os.devnull).open("w", encoding="utf-8"),
        ):
            client = Client()
            client.check_authentication()
            click.echo("Login successful!")
    except Exception as _e:  # noqa: BLE001
        click.echo(
            "No valid credentials found. Please enter your Climate Data Store API key. These will be stored in a file found at ~/.ecmwfdatastoresrc for future use. To obtain your key, go to https://cds.climate.copernicus.eu/how-to-api.",
        )
        key = click.prompt("Enter your key", type=str, hide_input=True)
        with CONFIG_FILE.open("w") as f:
            f.write(f"url: https://cds.climate.copernicus.eu/api\nkey: {key}\n")
        login_to_ecmwf_datastore()
    return client


def create_data_dir(save_dir: Path | None) -> Path:
    """Creates the directory to save the downloaded dataset if it doesn't already exist.

    Args:
        save_dir (Path | None): The directory to save the downloaded dataset. If None,
            defaults to a "data" directory at the package level.

    Returns:
        Path: The directory to save the downloaded dataset.

    """
    # by default, save in a "data" directory at the package level
    data_dir = FILE_DIR.parent / "data" if save_dir is None else save_dir
    data_dir.mkdir(exist_ok=True)
    return data_dir


def setup_request(start_datetime: str | None, end_datetime: str | None) -> tuple[str, dict]:
    """Sets up the dataset and request parameters for downloading ERA5 reanalysis data.

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
        "time": [f"{hour:02d}:00" for hour in range(0, 24, 6)],
        "area": [MAX_LAT, MIN_LON, MIN_LAT, MAX_LON],
        "data_format": "netcdf",
    }

    return dataset, request


def download_era5(
    save_dir: Path | None = None,
    start_datetime: str | None = None,
    end_datetime: str | None = None,
) -> None:
    """Downloads ERA5 reanalysis data from the Copernicus Climate Data Store and saves it to a netCDF file.

    Args:
        save_dir (Path | None): The directory to save the downloaded dataset. If None, defaults to a "data" directory at the package level.
        start_datetime (str | None): The start datetime for the dataset in YYYY-MM-DD format. If None, defaults to the earliest available datetime for the dataset.
        end_datetime (str | None): The end datetime for the dataset in YYYY-MM-DD format. If None, defaults to the latest available datetime for the dataset.

    """
    client = login_to_ecmwf_datastore()
    dataset, request = setup_request(start_datetime, end_datetime)

    remote = client.submit(dataset, request)
    save_file = create_data_dir(save_dir) / f"era5_reanalysis_{start_datetime}_{end_datetime}.nc"

    msg = "Request submitted. Waiting for results"
    while not remote.results_ready:
        for i in range(4):
            print(f"\r{msg}{'.' * i}".ljust(len(msg) + 4), end="", flush=True)
            time.sleep(1)
        time.sleep(2)
    click.echo("Results ready!")

    results = remote.get_results()

    if click.confirm(f"Downloading dataset with size {results.content_length / 1e9:.2f} GB. Proceed?"):
        results.download(str(save_file))
        click.echo(f"Download complete! Dataset saved to {save_file}")
    else:
        click.echo("Download cancelled, exiting.")
