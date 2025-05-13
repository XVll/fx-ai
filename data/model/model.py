# data/models/data_models.py
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

@dataclass
class TradeData:
    """Standardized trade data model."""
    timestamp: datetime  # From ts_event
    price: float
    size: float
    side: str  # 'B', 'S', or 'N'
    exchange: str
    conditions: List[str]  # No longer optional
    trade_id: str  # No longer optional

@dataclass
class QuoteData:
    """Standardized quote data model."""
    timestamp: datetime  # From ts_event
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    bid_count: int  # No longer optional
    ask_count: int  # No longer optional
    exchange: str  # No longer optional

@dataclass
class BarData:
    """Standardized OHLCV bar data model."""
    timestamp: datetime  # From ts_event
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str  # '1s', '1m', '5m', '1d'

@dataclass
class StatusData:
    """Standardized trading status data model."""
    timestamp: datetime  # From ts_event
    status: str  # 'TRADING', 'HALTED', etc.
    reason: str  # No longer optional
    is_trading: bool
    is_halted: bool
    is_short_sell_restricted: bool