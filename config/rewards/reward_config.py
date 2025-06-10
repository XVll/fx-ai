"""
Reward system configuration for modular reward components - Hydra version.
"""

from dataclasses import dataclass


@dataclass
class RewardConfig:
    """Modular reward system configuration"""

    # Core PnL rewards
    pnl_coefficient: float = 100.0                    # P&L scaling coefficient

    # Risk management
    holding_penalty_coefficient: float = 2.0          # Holding time penalty
    drawdown_penalty_coefficient: float = 5.0         # Drawdown penalty  
    bankruptcy_penalty_coefficient: float = 50.0      # Bankruptcy penalty

    # MFE/MAE penalties
    profit_giveback_penalty_coefficient: float = 2.0  # Profit giveback penalty
    profit_giveback_threshold: float = 0.3            # Giveback threshold (0.0-1.0)
    max_drawdown_penalty_coefficient: float = 15.0    # Max drawdown penalty
    max_drawdown_threshold_percent: float = 0.01      # MAE threshold (0.01 = 1%)

    # Trading bonuses
    profit_closing_bonus_coefficient: float = 100.0   # Profit closing bonus
    base_multiplier: float = 5000                     # Clean trade base multiplier
    max_mae_threshold: float = 0.02                   # Max allowed MAE (0.02 = 2%)
    min_gain_threshold: float = 0.01                  # Min gain for clean trade (0.01 = 1%)

    # Activity incentives
    activity_bonus_per_trade: float = 0.025           # Trading activity bonus
    hold_penalty_per_step: float = 0.01               # Hold action penalty
    action_penalty_coefficient: float = 0.1           # Action penalty coefficient
    quick_profit_bonus_coefficient: float = 1.0       # Quick profit bonus coefficient

    # Limits
    max_holding_time_steps: int = 180                 # Max holding time in steps

    # Component toggles
    enable_pnl_reward: bool = True                    # Enable P&L reward component
    enable_holding_penalty: bool = True               # Enable holding penalty component
    enable_drawdown_penalty: bool = True              # Enable drawdown penalty component
    enable_profit_giveback_penalty: bool = True       # Enable giveback penalty component
    enable_max_drawdown_penalty: bool = True          # Enable max drawdown penalty component
    enable_profit_closing_bonus: bool = True          # Enable profit bonus component
    enable_clean_trade_bonus: bool = True             # Enable clean trade bonus component
    enable_trading_activity_bonus: bool = True        # Enable activity bonus component
    enable_inactivity_penalty: bool = True            # Enable inactivity penalty component

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate thresholds are percentages
        if not 0.0 <= self.profit_giveback_threshold <= 1.0:
            raise ValueError(f"profit_giveback_threshold {self.profit_giveback_threshold} must be in range [0.0, 1.0]")
        
        if not 0.0 <= self.max_drawdown_threshold_percent <= 1.0:
            raise ValueError(f"max_drawdown_threshold_percent {self.max_drawdown_threshold_percent} must be in range [0.0, 1.0]")
        
        if not 0.0 <= self.max_mae_threshold <= 1.0:
            raise ValueError(f"max_mae_threshold {self.max_mae_threshold} must be in range [0.0, 1.0]")
        
        if not 0.0 <= self.min_gain_threshold <= 1.0:
            raise ValueError(f"min_gain_threshold {self.min_gain_threshold} must be in range [0.0, 1.0]")