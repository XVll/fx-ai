"""
Environment configuration for trading environment settings.
"""

from typing import Optional, Literal, Any
from pydantic import BaseModel, Field
from config.rewards import RewardConfig


class EnvironmentConfig(BaseModel):
    """Trading environment settings"""
    shutdown_timeout: Optional[int] = Field(30, description="Timeout for environment shutdown in seconds")

    # Termination conditions
    max_loss_percent: float = Field(0.15, description="Max episode loss")
    bankruptcy_threshold_factor: float = Field(0.01, description="Bankruptcy threshold")

    # Environment settings
    render_mode: Literal["human", "logs", "none"] = Field("none", description="Render mode")

    # Reward system
    reward: RewardConfig = Field(default_factory=lambda: RewardConfig())