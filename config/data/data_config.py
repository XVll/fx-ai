"""
Data configuration for data sources and lifecycle management.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Data source and processing configuration"""
    shutdown_timeout: Optional[int] = Field(30, description="Timeout for data source shutdown in seconds")
    provider: Literal["databento"] = Field("databento", description="Data provider")
    data_dir: str = Field("dnb", description="Data directory")
    index_dir: str = Field("cache/indices", description="Index directory")
    auto_build_index: bool = Field(True, description="Auto-build index")
    include_weekends: bool = Field(False, description="Include weekends in data")