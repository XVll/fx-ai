"""
Data configuration for data sources and lifecycle management.
"""

from typing import List, Optional
from dataclasses import dataclass


@dataclass
class DataConfig:
    """Data source and processing configuration"""
    provider: str = "databento"  # Options: databento  # Data provider
    # NOTE: Paths now managed by PathManager - these fields deprecated
    # Use PathManager.databento_dir and PathManager.indices_cache_dir instead
    auto_build_index: bool = True                 # Auto-build index
    include_weekends: bool = False                # Include weekends in data