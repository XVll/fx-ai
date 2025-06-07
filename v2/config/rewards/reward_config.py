"""
Reward system configuration for modular reward components.
"""

from pydantic import BaseModel, Field


class RewardConfig(BaseModel):
    """Modular reward system configuration"""

    # Core PnL rewards
    pnl_coefficient: float = Field(100.0, description="P&L scaling coefficient")

    # Risk management
    holding_penalty_coefficient: float = Field(2.0, description="Holding time penalty")
    drawdown_penalty_coefficient: float = Field(5.0, description="Drawdown penalty")
    bankruptcy_penalty_coefficient: float = Field(
        50.0, description="Bankruptcy penalty"
    )

    # MFE/MAE penalties
    profit_giveback_penalty_coefficient: float = Field(
        2.0, description="Profit giveback penalty"
    )
    profit_giveback_threshold: float = Field(0.3, description="Giveback threshold")
    max_drawdown_penalty_coefficient: float = Field(
        15.0, description="Max drawdown penalty"
    )
    max_drawdown_threshold_percent: float = Field(0.01, description="MAE threshold")

    # Trading bonuses
    profit_closing_bonus_coefficient: float = Field(
        100.0, description="Profit closing bonus"
    )
    base_multiplier: float = Field(5000, description="Clean trade base multiplier")
    max_mae_threshold: float = Field(0.02, description="Max allowed MAE")
    min_gain_threshold: float = Field(0.01, description="Min gain for clean trade")

    # Activity incentives
    activity_bonus_per_trade: float = Field(0.025, description="Trading activity bonus")
    hold_penalty_per_step: float = Field(0.01, description="Hold action penalty")
    action_penalty_coefficient: float = Field(
        0.1, description="Action penalty coefficient"
    )
    quick_profit_bonus_coefficient: float = Field(
        1.0, description="Quick profit bonus coefficient"
    )

    # Limits
    max_holding_time_steps: int = Field(180, description="Max holding time")

    # Component toggles
    enable_pnl_reward: bool = Field(True, description="Enable P&L reward")
    enable_holding_penalty: bool = Field(True, description="Enable holding penalty")
    enable_drawdown_penalty: bool = Field(True, description="Enable drawdown penalty")
    enable_profit_giveback_penalty: bool = Field(
        True, description="Enable giveback penalty"
    )
    enable_max_drawdown_penalty: bool = Field(
        True, description="Enable max drawdown penalty"
    )
    enable_profit_closing_bonus: bool = Field(True, description="Enable profit bonus")
    enable_clean_trade_bonus: bool = Field(True, description="Enable clean trade bonus")
    enable_trading_activity_bonus: bool = Field(
        True, description="Enable activity bonus"
    )
    enable_inactivity_penalty: bool = Field(
        True, description="Enable inactivity penalty"
    )