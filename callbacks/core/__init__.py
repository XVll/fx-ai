"""
Core callback system components.

Simple but powerful callback architecture:
- 15 essential event hooks
- Simple state management 
- Error isolation
- Easy to understand and extend
"""

from .base import BaseCallback
from .callback_manager import CallbackManager

__all__ = [
    "BaseCallback",
    "CallbackManager",
]