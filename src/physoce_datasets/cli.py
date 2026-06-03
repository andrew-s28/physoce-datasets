"""CLI commands for physoce_datasets."""

from __future__ import annotations

import click

from .download import EKEDownloader, WindStressDownloader

save_dir_option = click.option(
    "--save-dir",
    type=str,
    default=None,
    help=(
        "Directory to save the downloaded dataset. If not specified, defaults "
        "to a 'data' directory in the current working directory."
    ),
)
save_file_option = click.option(
    "--save-file",
    type=str,
    default=None,
    help=(
        "Filename to save the dataset. If not specified, defaults to a filename "
        "based on the dataset name and date range (e.g., 'eke_2000-01-01_to_2020-12-31.nc')."
    ),
)
start_datetime_option = click.option(
    "--start-date",
    type=str,
    default=None,
    help=(
        "Start datetime for the dataset. Format should be YYYY-MM-DD. If not "
        "specified, defaults to the earliest available datetime for the dataset."
    ),
)
end_datetime_option = click.option(
    "--end-date",
    type=str,
    default=None,
    help=(
        "End datetime for the dataset. Format should be YYYY-MM-DD. If not "
        "specified, defaults to the latest available datetime for the dataset."
    ),
)
update_option = click.option(
    "--update-file",
    type=str,
    default=None,
    help=(
        "Whether to update an existing dataset with new data. If specified, should be "
        "the path to an existing netCDF file found in the directory specified by --save-dir (default './data')."
    ),
)
location_option = click.option(
    "--location",
    type=str,
    required=True,
    help=("Location for the dataset in the format 'lon,lat' (e.g., '132.0,36.55'). Required."),
)


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Command-line interface for downloading datasets."""
    if ctx.invoked_subcommand is None:
        click.echo("No subcommand specified. Use --help for more information.")


@cli.command(
    "eke",
    help="Download geostrophic velocities and compute eddy kinetic energy from Copernicus Marine Services.",
)
@location_option
@save_dir_option
@save_file_option
@start_datetime_option
@end_datetime_option
def _eke(
    location: str,
    save_dir: str | None,
    save_file: str | None,
    start_date: str | None,
    end_date: str | None,
) -> None:
    """Download geostrophic velocities and compute eddy kinetic energy from Copernicus Marine Services.

    Args:
        location (str): Location for the dataset in the format 'lon,lat' (e.g., '132.0,36.55').
        save_dir (str | None): Directory to save the downloaded dataset. If not specified,
            defaults to a "data" directory in the current working directory.
        start_date (str | None): Start date for the dataset. Format should be YYYY-MM-DD.
            If not specified, defaults to the earliest available date for the dataset.
        end_date (str | None): End date for the dataset. Format should be YYYY-MM-DD.
            If not specified, defaults to the latest available date for the dataset.
        save_file (str | None): Filename to save the dataset. If not specified, defaults to a filename
            based on the dataset name and date range (e.g., "eke_2000-01-01_to_2020-12-31.nc").

    """
    downloader = EKEDownloader(
        location=location,
        save_dir=save_dir,
        start_date=start_date,
        end_date=end_date,
        save_file=save_file,
    )
    downloader.download()


@cli.command("sst", help="Download NASA MUR SST datasets.")
@location_option
@save_dir_option
@save_file_option
@start_datetime_option
@end_datetime_option
def _sst(
    location: str,
    save_dir: str | None,
    save_file: str | None,
    start_date: str | None,
    end_date: str | None,
) -> None:
    """Download NASA MUR SST datasets.

    Args:
        location (str): Location for the dataset in the format 'lon,lat' (e.g., '132.0,36.55').
        save_dir (str | None): Directory to save the downloaded dataset. If not specified,
            defaults to a "data" directory in the current working directory.
        start_date (str | None): Start date for the dataset. Format should be YYYY-MM-DD.
            If not specified, defaults to the earliest available date for the dataset.
        end_date (str | None): End date for the dataset. Format should be YYYY-MM-DD.
            If not specified, defaults to the latest available date for the dataset.
        save_file (str | None): Filename to save the dataset. If not specified, defaults to a filename
            based on the dataset name and date range (e.g., "eke_2000-01-01_to_2020-12-31.nc").

    """
    msg = "The 'sst' command is not implemented yet due to issues with the NASA Harmony API."
    raise NotImplementedError(msg)


@cli.command("wind-stress", help="Download wind velocity and compute wind stress from ERA5.")
@location_option
@save_dir_option
@save_file_option
@start_datetime_option
@end_datetime_option
def _wind_stress(
    location: str,
    save_dir: str | None,
    save_file: str | None,
    start_date: str | None,
    end_date: str | None,
) -> None:
    """Download wind velocity and compute wind stress from ERA5.

    Args:
        location (str): Location for the dataset in the format 'lon,lat' (e.g., '132.0,36.55').
        save_dir (str | None): Directory to save the downloaded dataset. If not specified,
            defaults to a "data" directory in the current working directory.
        start_date (str | None): Start date for the dataset. Format should be YYYY-MM-DD.
            If not specified, defaults to the earliest available date for the dataset.
        end_date (str | None): End date for the dataset. Format should be YYYY-MM-DD.
            If not specified, defaults to the latest available date for the dataset.
        save_file (str | None): Filename to save the dataset. If not specified, defaults to a filename
            based on the dataset name and date range (e.g., "eke_2000-01-01_to_2020-12-31.nc").

    """
    downloader = WindStressDownloader(
        location=location,
        save_dir=save_dir,
        save_file=save_file,
        start_date=start_date,
        end_date=end_date,
    )
    downloader.download()


if __name__ == "__main__":
    cli()
