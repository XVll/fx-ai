"""
Data configuration for data sources and lifecycle management.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Data source and processing configuration"""

    # Provider settings
    provider: Literal["databento"] = Field("databento", description="Data provider")
    data_dir: str = Field("dnb", description="Data directory")
    symbols: List[str] = Field(["MLGO"], description="Trading symbols")

    # Data types
    load_trades: bool = Field(True, description="Load trade data")
    load_quotes: bool = Field(True, description="Load quote data")
    load_order_book: bool = Field(True, description="Load order book data")
    load_ohlcv: bool = Field(True, description="Load OHLCV data")

    # Caching
    cache_enabled: bool = Field(True, description="Enable data caching")
    cache_dir: str = Field("cache", description="Cache directory")
    preload_days: int = Field(2, description="Days to preload")

    # Index configuration
    index_dir: str = Field("cache/indices", description="Index directory")
    auto_build_index: bool = Field(True, description="Auto-build index")