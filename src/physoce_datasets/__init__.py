"""The physoce_datasets package provides tools for downloading and processing physical oceanography datasets."""

from .download import download_eke, download_era5, submit_era5

__all__ = ["download_eke", "download_era5", "submit_era5"]
