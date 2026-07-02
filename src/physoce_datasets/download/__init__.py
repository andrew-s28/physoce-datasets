"""Module for downloading physical oceanographic datasets and calculating derived quantities."""

from . import copernicus_marine, era5, ooi_ea

__all__ = ["copernicus_marine", "era5", "ooi_ea"]
