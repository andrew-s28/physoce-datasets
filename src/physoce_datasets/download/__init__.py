"""Module for downloading physical oceanographic datasets.

This module also processes downloaded data into xarray Datasets with
appropriate metadata.
"""

from .eke import EKEDownloader
from .ooi import EAMooringDownloader
from .sst import SSTDownloader
from .wind_stress import WindStressDownloader

__all__ = ["EAMooringDownloader", "EKEDownloader", "SSTDownloader", "WindStressDownloader"]
