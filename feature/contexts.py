# feature/v2/base/contexts.py

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional


@dataclass
class MarketContext:
    """Market context for feature extraction"""

    timestamp: datetime
    current_price: float

    # Data windows
    hf_window: List[Dict[str, Any]]  # High frequency (1s) data
    mf_1m_window: List[Dict[str, Any]]  # Medium frequency 1m bars
    lf_5m_window: List[Dict[str, Any]]  # Low frequency 5m bars

    # Previous day data
    prev_day_close: float
    prev_day_high: float
    prev_day_low: float

    # Session data
    session_high: float
    session_low: float
    session: str  # PREMARKET, REGULAR, POSTMARKET

    # Static data
    market_cap: float

    # Session statistics
    session_volume: float = 0.0
    session_trades: int = 0
    session_vwap: float = 0.0

    # Portfolio state (optional, set by environment)
    portfolio_state: Optional[Dict[str, Any]] = None
