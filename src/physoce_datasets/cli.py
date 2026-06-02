"""CLI commands for physoce_datasets."""

from __future__ import annotations

import click

from .download import EKEDownloader, SSTDownloader, WindStressDownloader

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
area_option = click.option(
    "--area",
    type=str,
    default=None,
    help=(
        "Bounding box for the dataset in the format 'lon_min,lon_max,lat_min,lat_max'. "
        "If not specified, defaults to global coverage."
    ),
)
location_option = click.option(
    "--location",
    type=str,
    default=None,
    help=(
        "Location for the dataset in the format 'lon,lat' (e.g., '132.0,36.55'). "
        "If not specified, defaults to (0,0). This is a temporary option until the switch from area-based to point-based data access is fully implemented."
    ),
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
@save_dir_option
@save_file_option
@start_datetime_option
@end_datetime_option
@area_option
def _eke(
    save_dir: str | None,
    save_file: str | None,
    start_date: str | None,
    end_date: str | None,
    area: str | None,
) -> None:
    """Download geostrophic velocities and compute eddy kinetic energy from Copernicus Marine Services.

    Args:
        save_dir (str | None): Directory to save the downloaded dataset. If not specified,
            defaults to a "data" directory in the current working directory.
        start_date (str | None): Start date for the dataset. Format should be YYYY-MM-DD.
            If not specified, defaults to the earliest available date for the dataset.
        end_date (str | None): End date for the dataset. Format should be YYYY-MM-DD.
            If not specified, defaults to the latest available date for the dataset.
        save_file (str | None): Filename to save the dataset. If not specified, defaults to a filename
            based on the dataset name and date range (e.g., "eke_2000-01-01_to_2020-12-31.nc").
        area (str | None): Bounding box for the dataset in the format 'lon_min,lat_min,lon_max,lat_max'.
            If not specified, defaults to global coverage.

    """
    downloader = EKEDownloader(
        save_dir=save_dir,
        start_date=start_date,
        end_date=end_date,
        save_file=save_file,
        area=area,
    )
    downloader.download()


@cli.command("sst", help="Download NASA MUR SST datasets.")
@save_dir_option
@save_file_option
@start_datetime_option
@end_datetime_option
@area_option
def _sst(
    save_dir: str | None,
    save_file: str | None,
    start_date: str | None,
    end_date: str | None,
    area: str | None,
) -> None:
    """Download NASA MUR SST datasets.

    Args:
        save_dir (str | None): Directory to save the downloaded dataset. If not specified,
            defaults to a "data" directory in the current working directory.
        start_date (str | None): Start date for the dataset. Format should be YYYY-MM-DD.
            If not specified, defaults to the earliest available date for the dataset.
        end_date (str | None): End date for the dataset. Format should be YYYY-MM-DD.
            If not specified, defaults to the latest available date for the dataset.
        save_file (str | None): Filename to save the dataset. If not specified, defaults to a filename
            based on the dataset name and date range (e.g., "eke_2000-01-01_to_2020-12-31.nc").
        area (str | None): Bounding box for the dataset in the format 'lon_min,lat_min,lon_max,lat_max'.
            If not specified, defaults to global coverage.

    """
    downloader = SSTDownloader(
        save_dir=save_dir,
        save_file=save_file,
        start_date=start_date,
        end_date=end_date,
        area=area,
    )
    downloader.download()


@cli.command("wind-stress", help="Download wind velocity and compute wind stress from ERA5.")
@save_dir_option
@save_file_option
@start_datetime_option
@end_datetime_option
@area_option
@location_option
def _wind_stress(
    save_dir: str | None,
    save_file: str | None,
    start_date: str | None,
    end_date: str | None,
    area: str | None,
    location: str | None,
) -> None:
    """Download wind velocity and compute wind stress from ERA5.

    Args:
        save_dir (str | None): Directory to save the downloaded dataset. If not specified,
            defaults to a "data" directory in the current working directory.
        start_date (str | None): Start date for the dataset. Format should be YYYY-MM-DD.
            If not specified, defaults to the earliest available date for the dataset.
        end_date (str | None): End date for the dataset. Format should be YYYY-MM-DD.
            If not specified, defaults to the latest available date for the dataset.
        save_file (str | None): Filename to save the dataset. If not specified, defaults to a filename
            based on the dataset name and date range (e.g., "eke_2000-01-01_to_2020-12-31.nc").
        area (str | None): Bounding box for the dataset in the format 'lon_min,lat_min,lon_max,lat_max'.
            If not specified, defaults to global coverage. Unused currently and to be removed.
        location (str | None): Location for the dataset in the format 'lon,lat' (e.g., '132.0,36.55'). Used instead of area.

    """
    downloader = WindStressDownloader(
        save_dir=save_dir,
        save_file=save_file,
        start_date=start_date,
        end_date=end_date,
        area=area,
        location=location,
    )
    downloader.download()


if __name__ == "__main__":
    cli()
