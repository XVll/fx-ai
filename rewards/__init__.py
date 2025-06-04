# rewards/__init__.py - Clean percentage-based reward system

from rewards.core import RewardComponent, RewardState, RewardType, RewardMetadata
from rewards.components import (
    PnLReward,
    HoldingTimePenalty,
    DrawdownPenalty,
    ProfitGivebackPenalty,
    MaxDrawdownPenalty,
    ProfitClosingBonus,
    BankruptcyPenalty,
)
from rewards.calculator import RewardSystem

__all__ = [
    "RewardComponent",
    "RewardState",
    "RewardType",
    "RewardMetadata",
    "PnLReward",
    "HoldingTimePenalty",
    "DrawdownPenalty",
    "ProfitGivebackPenalty",
    "MaxDrawdownPenalty",
    "ProfitClosingBonus",
    "BankruptcyPenalty",
    "RewardSystem",
]
