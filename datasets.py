from __future__ import annotations

from pathlib import Path

import click

from download import download_eke, download_era5

save_dir_option = click.option(
    "--save-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Directory to save the downloaded dataset. If not specified, defaults to a 'data' directory at the package level.",
)
start_datetime_option = click.option(
    "--start-date",
    type=str,
    default=None,
    help="Start datetime for the dataset. Format should be YYYY-MM-DD. If not specified, defaults to the earliest available datetime for the dataset.",
)
end_datetime_option = click.option(
    "--end-date",
    type=str,
    default=None,
    help="End datetime for the dataset. Format should be YYYY-MM-DD. If not specified, defaults to the latest available datetime for the dataset.",
)


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Command-line interface for downloading datasets."""
    if ctx.invoked_subcommand is None:
        click.echo("No subcommand specified. Use --help for more information.")


@cli.command(help="Download geostrophic velocities and compute eddy kinetic energy from Copernicus Marine Services.")
@save_dir_option
@start_datetime_option
@end_datetime_option
def eke(save_dir: Path | None, start_date: str | None, end_date: str | None) -> None:
    """Download geostrophic velocities and compute eddy kinetic energy from Copernicus Marine Services.

    Args:
        save_dir (Path | None): Directory to save the downloaded dataset. If not specified, defaults to a "data" directory at the package level.
        start_date (str | None): Start date for the dataset. Format should be YYYY-MM-DD. If not specified, defaults to the earliest available date for the dataset.
        end_date (str | None): End date for the dataset. Format should be YYYY-MM-DD. If not specified, defaults to the latest available date for the dataset.

    """
    download_eke(
        save_dir=save_dir,
        start_datetime=start_date,
        end_datetime=end_date,
    )


@cli.command(help="Download ERA5 reanalysis data from the Copernicus Climate Data Store.")
@save_dir_option
@start_datetime_option
@end_datetime_option
def era5(save_dir: Path | None, start_date: str | None, end_date: str | None) -> None:
    """Download ERA5 reanalysis data from the Copernicus Climate Data Store.

    Args:
        save_dir (Path | None): Directory to save the downloaded dataset. If not specified, defaults to a "data" directory at the package level.
        start_date (str | None): Start date for the dataset. Format should be YYYY-MM-DD. If not specified, defaults to the earliest available date for the dataset.
        end_date (str | None): End date for the dataset. Format should be YYYY-MM-DD. If not specified, defaults to the latest available date for the dataset.

    """
    download_era5(
        save_dir=save_dir,
        start_datetime=start_date,
        end_datetime=end_date,
    )


if __name__ == "__main__":
    cli()
