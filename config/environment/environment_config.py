"""
Environment configuration for trading environment settings.
"""

from typing import Optional, Any
from dataclasses import dataclass, field
from config.rewards import RewardConfig


@dataclass
class EnvironmentConfig:
    """Trading environment settings"""
    shutdown_timeout: Optional[int] = 30        # Timeout for environment shutdown in seconds

    # Termination conditions
    max_loss_percent: float = 0.15              # Max episode loss
    bankruptcy_threshold_factor: float = 0.01   # Bankruptcy threshold

    # Environment settings
    render_mode: str = "none"  # Options: human, logs, none  # Render mode

    # Reward system
    reward: RewardConfig = field(default_factory=RewardConfig)