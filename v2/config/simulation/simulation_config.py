"""
Simulation configuration for market simulation and trading parameters.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field


class SimulationConfig(BaseModel):
    """Market simulation and trading parameters"""

    # Capital settings
    initial_capital: float = Field(25000.0, gt=0.0, description="Starting capital")
    max_position_value_ratio: float = Field(
        1.0, ge=0.0, le=1.0, description="Max position as fraction of equity"
    )
    leverage: float = Field(1.0, gt=0.0, description="Trading leverage")

    # Trading costs
    commission_rate: float = Field(0.001, ge=0.0, description="Commission rate")
    slippage_rate: float = Field(0.0005, ge=0.0, description="Slippage rate")
    min_transaction_amount: float = Field(100.0, ge=0.0, description="Min trade size")

    # Risk limits
    max_drawdown: float = Field(0.3, ge=0.0, le=1.0, description="Max allowed drawdown")
    stop_loss_pct: float = Field(
        0.15, ge=0.0, le=1.0, description="Stop loss percentage"
    )
    daily_loss_limit: float = Field(
        0.25, ge=0.0, le=1.0, description="Daily loss limit"
    )

    # Execution settings
    execution_delay_ms: int = Field(100, description="Order execution delay")
    partial_fill_probability: float = Field(0.0, description="Partial fill probability")
    allow_shorting: bool = Field(False, description="Allow short selling")

    # Latency simulation
    mean_latency_ms: float = Field(100.0, description="Mean execution latency")
    latency_std_dev_ms: float = Field(20.0, description="Latency std dev")

    # Slippage parameters
    base_slippage_bps: float = Field(10.0, description="Base slippage (bps)")
    size_impact_slippage_bps_per_unit: float = Field(
        0.2, description="Size impact slippage"
    )
    max_total_slippage_bps: float = Field(100.0, description="Max total slippage")

    # Cost parameters
    commission_per_share: float = Field(0.005, description="Commission per share")
    fee_per_share: float = Field(0.001, description="Fee per share")
    min_commission_per_order: float = Field(1.0, description="Min commission")
    max_commission_pct_of_value: float = Field(0.5, description="Max commission %")

    # Market impact
    market_impact_model: Literal["linear", "square_root", "none"] = Field("linear")
    market_impact_coefficient: float = Field(0.0001, description="Market impact coeff")

    # Spread modeling
    spread_model: Literal["fixed", "dynamic", "historical"] = Field("historical")
    fixed_spread_bps: float = Field(10.0, description="Fixed spread (bps)")

    # Episode randomization
    random_start_prob: float = Field(0.95, description="Random start probability")
    warmup_steps: int = Field(60, description="Warmup steps")

    # Portfolio settings
    max_position_holding_seconds: Optional[int] = Field(
        None, description="Max holding time"
    )