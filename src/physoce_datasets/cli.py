"""CLI commands for physoce_datasets."""

from __future__ import annotations

import click

from physoce_datasets._util import parse_lonlat
from physoce_datasets.copernicus import EddyKineticEnergy
from physoce_datasets.era5 import WindStress
from physoce_datasets.ooi import EnduranceArray


def _validate_site(ctx: click.Context, param: click.Option, value: str | None) -> str | None:
    """Enforce that the --location option is provided if --list-sites is not specified.

    Args:
        ctx (click.Context): The Click context object.
        param (click.Option): The Click option object.
        value (str | None): The value of the option.

    Returns:
        str | None: The value of the option, if provided.

    Raises:
        click.MissingParameter: If the option is required and not provided, and --list-sites is not specified.

    """
    if ctx.params.get("list_sites"):
        return value
    if not value:
        raise click.MissingParameter(ctx=ctx, param=param)
    return value


def _validate_instrument(ctx: click.Context, param: click.Option, value: str | None) -> str | None:
    """Enforce that the --instrument option is provided if --list-instruments and --list-sites are not specified.

    Args:
        ctx (click.Context): The Click context object.
        param (click.Option): The Click option object.
        value (str | None): The value of the option.

    Returns:
        str | None: The value of the option, if provided.

    Raises:
        click.MissingParameter: If the option is required and not provided, and --list-instruments and --list-sites are not specified.

    """
    if ctx.params.get("list_sites") or ctx.params.get("list_instruments"):
        return value
    if not value:
        raise click.MissingParameter(ctx=ctx, param=param)
    return value


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
    help=(
        "Location for the dataset, either in the format 'lon,lat' (e.g., '132.0,36.55') for global datasets or as a site name (see sub-command documentation for available sites) for moored datasets. Required."
    ),
    callback=_validate_site,  # only validate if --list-sites is not specified
)
dataset_option = click.option(
    "--dataset",
    type=str,
    help=("Dataset to be downloaded. Required."),
    callback=_validate_instrument,  # only validate if --list-sites or --list-instruments is not specified
)


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Command-line interface for downloading datasets."""
    if ctx.invoked_subcommand is None:
        click.echo("No subcommand specified. Use --help for more information.")


@cli.command("copernicus", help="Commands for downloading Copernicus Marine datasets.")
@location_option
@dataset_option
@save_dir_option
@save_file_option
@start_datetime_option
@end_datetime_option
def _eke(
    location: str,
    dataset: str,
    save_dir: str | None,
    save_file: str | None,
    start_date: str | None,
    end_date: str | None,
) -> None:
    """Download geostrophic velocities and compute eddy kinetic energy from Copernicus Marine Services.

    Args:
        location (str): Location for the dataset in the format 'lon,lat' (e.g., '132.0,36.55').
        dataset (str): Dataset to be downloaded. Required.
        save_dir (str | None): Directory to save the downloaded dataset. If not specified,
            defaults to a "data" directory in the current working directory.
        start_date (str | None): Start date for the dataset. Format should be YYYY-MM-DD.
            If not specified, defaults to the earliest available date for the dataset.
        end_date (str | None): End date for the dataset. Format should be YYYY-MM-DD.
            If not specified, defaults to the latest available date for the dataset.
        save_file (str | None): Filename to save the dataset. If not specified, defaults to a filename
            based on the dataset name and date range (e.g., "eke_2000-01-01_to_2020-12-31.nc").

    Raises:
        click.BadParameter: If the specified dataset is not supported.

    """
    lonlat = parse_lonlat(location)
    if dataset.lower() == "eke":
        downloader = EddyKineticEnergy(
            longitude=lonlat.lon,
            latitude=lonlat.lat,
            save_dir=save_dir,
            start_date=start_date,
            end_date=end_date,
            save_file=save_file,
        )
    else:
        msg = f"Invalid dataset '{dataset}'. Supported datasets are: 'eke'."
        raise click.BadParameter(msg)
    downloader.download()


@cli.command("era5", help="Commands for downloading ERA5 datasets.")
@location_option
@dataset_option
@save_dir_option
@save_file_option
@start_datetime_option
@end_datetime_option
def _wind_stress(
    location: str,
    dataset: str,
    save_dir: str | None,
    save_file: str | None,
    start_date: str | None,
    end_date: str | None,
) -> None:
    """Download wind velocity and compute wind stress from ERA5.

    Args:
        location (str): Location for the dataset in the format 'lon,lat' (e.g., '132.0,36.55').
        dataset (str): The dataset to download.
        save_dir (str | None): Directory to save the downloaded dataset. If not specified,
            defaults to a "data" directory in the current working directory.
        start_date (str | None): Start date for the dataset. Format should be YYYY-MM-DD.
            If not specified, defaults to the earliest available date for the dataset.
        end_date (str | None): End date for the dataset. Format should be YYYY-MM-DD.
            If not specified, defaults to the latest available date for the dataset.
        save_file (str | None): Filename to save the dataset. If not specified, defaults to a filename
            based on the dataset name and date range (e.g., "eke_2000-01-01_to_2020-12-31.nc").

    Raises:
        click.BadParameter: If the specified dataset is not supported.

    """
    lonlat = parse_lonlat(location)
    if dataset.lower() == "wind-stress":
        downloader = WindStress(
            longitude=lonlat.lon,
            latitude=lonlat.lat,
            save_dir=save_dir,
            save_file=save_file,
            start_date=start_date,
            end_date=end_date,
        )
    else:
        msg = f"Invalid dataset '{dataset}'. Supported datasets are: 'wind-stress'."
        raise click.BadParameter(msg)
    downloader.download()


@cli.group("ooi-ea", help="Commands for downloading OOI Endurance Array datasets.")
@click.pass_context
def _ooi_ea(ctx: click.Context) -> None:
    """Commands for downloading OOI Endurance Array datasets."""
    if ctx.invoked_subcommand is None:
        click.echo("No subcommand specified. Use --help for more information.")


@cli.command(
    "ooi-ea",
    help="""
    Download OOI Endurance Array datasets.

    Available sites for the --location argument include (case insensitive):\n
    \t- 'CE01ISSM' (Oregon Inshore Surface Mooring)\n
    \t- 'CE02SHSM' (Oregon Shelf Surface Mooring)\n
    \t- 'CE04OSSM' (Oregon Offshore Surface Mooring)\n
    \t- 'CE06ISSM' (Washington Inshore Surface Mooring)\n
    \t- 'CE07SHSM' (Washington Shelf Surface Mooring)\n
    \t- 'CE09OSSM' (Washington Offshore Surface Mooring)\n
    \t- 'CE01ISSP' (Oregon Inshore Surface Piercing Profiler Mooring)\n
    \t- 'CE02SHSP' (Oregon Shelf Surface Piercing Profiler Mooring)\n
    \t- 'CE06ISSP' (Washington Inshore Surface Piercing Profiler Mooring)\n
    \t- 'CE07SHSP' (Washington Shelf Surface Piercing Profiler Mooring)\n
    \t- 'CE09OSPM' (Washington Offshore Profiler Mooring)\n

    Available instruments for the --instrument argument include (case insensitive):\n
    \t- 'ctd' (temperature, salinity, pressure)\n
    \t- 'fluorometer' (chlorophyll and cdom)\n
    \t- 'oxygen' (dissolved oxygen)\n
    \t- 'nitrate' (nitrate)\n
    \t- 'spectrophotometer' (optical absorption and attenuation)\n
    \t- 'par' (photosynthetically active radiation)\n
    \t- 'irradiance' (downwelling irradiance)\n

    Not all instruments are available at all sites.

    Please see the OOI Endurance Array documentation for more information on these sites: https://oceanobservatories.org/array/coastal-endurance/.
    """,
)
@location_option
@dataset_option
@save_dir_option
@save_file_option
@start_datetime_option
@end_datetime_option
@click.option(
    "--list-sites",
    is_flag=True,
    is_eager=True,
    help="List available sites for the OOI Endurance Array datasets. Takes precedence over --list-instruments if both are specified.",
)
@click.option(
    "--list-datasets",
    is_flag=True,
    is_eager=True,
    help="List available datasets for the OOI Endurance Array datasets.",
)
def _ooi_ea_mooring_ctd(
    location: str,
    dataset: str,
    save_dir: str | None,
    save_file: str | None,
    start_date: str | None,
    end_date: str | None,
    list_sites: bool = False,  # noqa: FBT001, FBT002
    list_datasets: bool = False,  # noqa: FBT001, FBT002
) -> None:
    """Download moored OOI Endurance Array temperature, salinity, and density datasets.

    Args:
        location (str): Location for the dataset, as a site name.
        dataset (str): The dataset to download.
        save_dir (str | None): Directory to save the downloaded dataset. If not specified,
            defaults to a "data" directory in the current working directory.
        start_date (str | None): Start date for the dataset. Format should be YYYY-MM-DD.
            If not specified, defaults to the earliest available date for the dataset.
        end_date (str | None): End date for the dataset. Format should be YYYY-MM-DD.
            If not specified, defaults to the latest available date for the dataset.
        save_file (str | None): Filename to save the dataset. If not specified, defaults to a filename
            based on the dataset name and date range (e.g., "ooi_ea_mooring_2000-01-01_to_2020-12-31.nc").
        list_sites (bool): If True, list available sites for the OOI Endurance Array datasets and exit. Takes precedence over --list-datasets if both are specified.
        list_datasets (bool): If True, list available datasets for the OOI Endurance Array datasets and exit. If both --list-sites and --list-datasets are specified, --list-sites takes precedence.

    """
    if list_sites:
        click.echo("\n".join(EnduranceArray.list_sites()))
        return
    if list_datasets:
        click.echo("\n".join(EnduranceArray.list_datasets(site=location)))
        return
    downloader = EnduranceArray(
        site=location,
        dataset=dataset,
        save_dir=save_dir,
        save_file=save_file,
        start_date=start_date,
        end_date=end_date,
    )
    downloader.download()
