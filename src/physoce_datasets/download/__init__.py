"""Module for downloading physical oceanographic datasets.

This module also processes downloaded data into xarray Datasets with
appropriate metadata.
"""

from .eke import EKEDownloader
from .sst import SSTDownloader
from .wind_stress import WindStressDownloader

__all__ = ["EKEDownloader", "SSTDownloader", "WindStressDownloader"]
