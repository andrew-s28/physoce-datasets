"""The physoce_datasets package provides tools for downloading and processing physical oceanography datasets."""

from .download import EKEDownloader, WindStressDownloader

__all__ = ["EKEDownloader", "WindStressDownloader"]
