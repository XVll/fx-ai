# v2/rewards/__init__.py - Clean percentage-based reward system

from v2.rewards.core import RewardComponent, RewardState, RewardType, RewardMetadata
from v2.rewards.components import (
    PnLReward,
    HoldingTimePenalty,
    DrawdownPenalty,
    ProfitGivebackPenalty,
    MaxDrawdownPenalty,
    ProfitClosingBonus,
    BankruptcyPenalty,
)
from v2.rewards.calculator import RewardSystem

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