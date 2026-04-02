"""Module for downloading physical oceanographic datasets and processing them into xarray Datasets with appropriate metadata."""

from .eke import download_eke
from .era5 import download_era5

__all__ = ["download_eke", "download_era5"]
