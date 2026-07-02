"""The physoce_datasets package provides tools for downloading and processing physical oceanography datasets."""

from .download import copernicus_marine, era5, nasa, ooi_ea

__all__ = ["copernicus_marine", "era5", "nasa", "ooi_ea"]
