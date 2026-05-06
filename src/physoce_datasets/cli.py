"""CLI commands for physoce_datasets."""

from __future__ import annotations

from pathlib import Path

import click

from .download import download_eke, download_era5, submit_era5

save_dir_option = click.option(
    "--save-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
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
        "Bounding box for the dataset in the format 'lon_min,lat_min,lon_max,lat_max'. "
        "If not specified, defaults to global coverage."
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
@start_datetime_option
@end_datetime_option
@save_file_option
@area_option
def _eke(
    save_dir: Path | None,
    start_date: str | None,
    end_date: str | None,
    save_file: str | None,
    area_str: str | None,
) -> None:
    """Download geostrophic velocities and compute eddy kinetic energy from Copernicus Marine Services.

    Args:
        save_dir (Path | None): Directory to save the downloaded dataset. If not specified,
            defaults to a "data" directory in the current working directory.
        start_date (str | None): Start date for the dataset. Format should be YYYY-MM-DD.
            If not specified, defaults to the earliest available date for the dataset.
        end_date (str | None): End date for the dataset. Format should be YYYY-MM-DD.
            If not specified, defaults to the latest available date for the dataset.
        save_file (str | None): Filename to save the dataset. If not specified, defaults to a filename
            based on the dataset name and date range (e.g., "eke_2000-01-01_to_2020-12-31.nc").
        area_str (str | None): Bounding box for the dataset in the format 'lon_min,lat_min,lon_max,lat_max'.
            If not specified, defaults to global coverage.

    """
    download_eke(
        save_dir=save_dir,
        start_datetime=start_date,
        end_datetime=end_date,
        save_file=save_file,
        area_str=area_str,
    )


@cli.group("era5", help="ERA5 workflow commands.")
def _era5() -> None:
    """ERA5 workflow commands group."""


@_era5.command("submit", help="Submit ERA5 jobs until all requested months are queued.")
@save_dir_option
@start_datetime_option
@end_datetime_option
@area_option
def _era5_submit(
    save_dir: Path | None,
    start_date: str | None,
    end_date: str | None,
    area_str: str | None,
) -> None:
    """Submit ERA5 jobs only.

    Args:
        save_dir (Path | None): Directory to save request state. If not
            specified, defaults to a "data" directory in the current working directory.
        start_date (str | None): Start date for the dataset. Format should be YYYY-MM-DD.
        end_date (str | None): End date for the dataset. Format should be YYYY-MM-DD.
        area_str (str | None): Bounding box for the dataset in the format 'lon_min,lat_min,lon_max,lat_max'.
            If not specified, defaults to global coverage.

    """
    submit_era5(
        save_dir=save_dir,
        start_date=start_date,
        end_date=end_date,
        area_str=area_str,
    )


@_era5.command("download", help="Download and process ERA5 files after all remote jobs are successful.")
@save_dir_option
@save_file_option
def _era5_download(save_dir: Path | None, save_file: str | None) -> None:
    """Download ERA5 data for successful remote jobs and process locally.

    Args:
        save_dir (Path | None): Directory containing saved request state and output files.
                If not specified, defaults to a "data" directory in the current working directory.
        save_file (str | None): Filename to save the dataset. If not specified, defaults to a filename
            based on the dataset name and date range.

    """
    download_era5(
        save_dir=save_dir,
        save_file=save_file,
    )


if __name__ == "__main__":
    cli()
