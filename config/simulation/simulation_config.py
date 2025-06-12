"""
Simulation configuration for market simulation and trading parameters.
"""

from typing import Optional
from dataclasses import dataclass


@dataclass
class SimulationConfig:
    """Market simulation and trading parameters"""

    # Capital settings
    initial_capital: float = 25000.0              # Starting capital
    max_position_value_ratio: float = 1.0         # Max position as fraction of equity
    leverage: float = 1.0                         # Trading leverage

    # Trading costs
    commission_rate: float = 0.001                # Commission rate
    slippage_rate: float = 0.0005                 # Slippage rate
    min_transaction_amount: float = 100.0         # Min trade size

    # Risk limits
    max_drawdown: float = 0.3                     # Max allowed drawdown
    stop_loss_pct: float = 0.15                   # Stop loss percentage
    daily_loss_limit: float = 0.25                # Daily loss limit

    # Execution settings
    execution_delay_ms: int = 100                 # Order execution delay
    partial_fill_probability: float = 0.0         # Partial fill probability
    allow_shorting: bool = False                  # Allow short selling

    # Latency simulation
    mean_latency_ms: float = 100.0                # Mean execution latency
    latency_std_dev_ms: float = 20.0              # Latency std dev

    # Slippage parameters
    base_slippage_bps: float = 10.0               # Base slippage (bps)
    size_impact_slippage_bps_per_unit: float = 0.2  # Size impact slippage
    max_total_slippage_bps: float = 100.0         # Max total slippage

    # Cost parameters
    commission_per_share: float = 0.005           # Commission per share
    fee_per_share: float = 0.001                  # Fee per share
    min_commission_per_order: float = 1.0         # Min commission
    max_commission_pct_of_value: float = 0.5      # Max commission %

    # Market impact
    market_impact_model: str = "linear"  # Options: linear, square_root, none  # Market impact model
    market_impact_coefficient: float = 0.0001     # Market impact coeff

    # Spread modeling
    spread_model: str = "historical"  # Options: fixed, dynamic, historical
    fixed_spread_bps: float = 10.0                # Fixed spread (bps)

    # Episode randomization
    random_start_prob: float = 0.95               # Random start probability
    warmup_steps: int = 60                        # Warmup steps

    # Portfolio settings
    max_position_holding_seconds: Optional[int] = None  # Max holding time