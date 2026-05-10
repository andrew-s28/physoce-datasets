"""Module for downloading physical oceanographic datasets.

This module also processes downloaded data into xarray Datasets with
appropriate metadata.
"""

from .eke import download_eke
from .wind_stress import download_wind_stress, submit_wind_stress

__all__ = ["download_eke", "download_wind_stress", "submit_wind_stress"]
