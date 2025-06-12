"""
Data configuration for data sources and lifecycle management.
"""

from typing import List, Optional
from dataclasses import dataclass


@dataclass
class DataConfig:
    """Data source and processing configuration"""
    shutdown_timeout: Optional[int] = 30         # Timeout for data source shutdown in seconds
    provider: str = "databento"  # Options: databento  # Data provider
    data_dir: str = "dnb"                         # Data directory
    index_dir: str = "cache/indices"              # Index directory
    auto_build_index: bool = True                 # Auto-build index
    include_weekends: bool = False                # Include weekends in data