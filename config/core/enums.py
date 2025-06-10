"""
Core enums used across configuration modules.
"""

from enum import Enum


class ActionType(str, Enum):
    """Trading action types"""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class SessionType(str, Enum):
    """Market session types"""

    PREMARKET = "PREMARKET"
    REGULAR = "REGULAR"
    POSTMARKET = "POSTMARKET"
    CLOSED = "CLOSED"