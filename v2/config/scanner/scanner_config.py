"""
Scanner configuration for momentum scanning and quality scoring.
"""

from typing import Optional
from pydantic import BaseModel, Field


class ScannerConfig(BaseModel):
    """Consolidated scanner configuration"""

    # Momentum scanning
    min_daily_move: float = Field(0.10, description="Min 10% intraday movement")
    min_volume_multiplier: float = Field(2.0, description="Min 2x average volume")
    max_daily_move: Optional[float] = Field(
        None, description="Max daily move (uncapped)"
    )
    max_volume_multiplier: Optional[float] = Field(
        None, description="Max volume (uncapped)"
    )

    # Momentum scoring
    roc_lookback_minutes: int = Field(5, description="ROC calculation window")
    activity_lookback_minutes: int = Field(
        10, description="Activity calculation window"
    )
    min_reset_points: int = Field(60, description="Min reset points per day")
    roc_weight: float = Field(0.6, description="ROC score weight")
    activity_weight: float = Field(0.4, description="Activity score weight")

    # Session volume calculations
    premarket_start: str = Field("04:00", description="Pre-market start")
    premarket_end: str = Field("09:30", description="Pre-market end")
    regular_start: str = Field("09:30", description="Regular market start")
    regular_end: str = Field("16:00", description="Regular market end")
    postmarket_start: str = Field("16:00", description="Post-market start")
    postmarket_end: str = Field("20:00", description="Post-market end")
    volume_window_days: int = Field(10, description="Volume baseline window")
    use_session_specific_baselines: bool = Field(
        True, description="Session-specific baselines"
    )