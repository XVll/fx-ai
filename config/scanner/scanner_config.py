"""
Scanner configuration for momentum scanning and quality scoring.
"""

from typing import Optional
from dataclasses import dataclass


@dataclass
class ScannerConfig:
    """Consolidated scanner configuration"""

    # NOTE: Paths now managed by PathManager - these fields deprecated
    # Use PathManager.databento_dir and PathManager.scanner_cache_dir instead
    # -------------------------------------------------------------------------------------#

    # Momentum scanning
    min_daily_move: float = 0.10                  # Min 10% intraday movement
    min_volume_multiplier: float = 2.0            # Min 2x average volume
    max_daily_move: Optional[float] = None        # Max daily move (uncapped)
    max_volume_multiplier: Optional[float] = None # Max volume (uncapped)

    # Momentum scoring
    roc_lookback_minutes: int = 5                 # ROC calculation window
    activity_lookback_minutes: int = 10           # Activity calculation window
    min_reset_points: int = 60                    # Min reset points per day
    roc_weight: float = 0.6                       # ROC score weight
    activity_weight: float = 0.4                  # Activity score weight

    # Session volume calculations
    premarket_start: str = "04:00"                # Pre-market start
    premarket_end: str = "09:30"                  # Pre-market end
    regular_start: str = "09:30"                  # Regular market start
    regular_end: str = "16:00"                    # Regular market end
    postmarket_start: str = "16:00"               # Post-market start
    postmarket_end: str = "20:00"                 # Post-market end
    volume_window_days: int = 10                  # Volume baseline window
    use_session_specific_baselines: bool = True   # Session-specific baselines