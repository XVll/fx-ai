"""
Environment configuration for trading environment settings.
"""

from typing import Optional, Literal, Any
from pydantic import BaseModel, Field
from v2.config.rewards import RewardConfig


class EnvironmentConfig(BaseModel):
    """Trading environment settings"""

    # Episode control
    max_steps: int = Field(256, gt=0, description="Maximum steps per episode")
    max_training_steps: Optional[int] = Field(
        None, description="Maximum training steps"
    )

    # Training mode
    use_momentum_training: bool = Field(True, description="Use momentum-based training")

    # Termination conditions
    early_stop_loss_threshold: float = Field(0.85, description="Early stop threshold")
    max_episode_loss_percent: float = Field(0.15, description="Max episode loss")
    bankruptcy_threshold_factor: float = Field(0.01, description="Bankruptcy threshold")

    # Environment settings
    random_reset: bool = Field(True, description="Random episode start")
    render_mode: Literal["human", "logs", "none"] = Field(
        "none", description="Render mode"
    )
    feature_update_interval: int = Field(1, description="Feature update interval")

    # Reward system
    reward: RewardConfig = Field(default_factory=lambda: RewardConfig())

    # Training management (will be set in main config to avoid circular imports)
    training_manager: Any = Field(default=None, description="Training manager config")