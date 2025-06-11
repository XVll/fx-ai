"""
Evaluation configuration for model performance assessment.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation runs."""
    
    # Basic settings
    enabled: bool = True
    frequency: int = 50  # Updates between evaluations
    episodes: int = 20   # Episodes per evaluation
    
    # Determinism
    seed: int = 42
    deterministic_actions: bool = True
    
    # Episode selection
    use_fixed_episodes: bool = True  # Use same episodes every time
    episode_selection: Literal["diverse", "best", "worst", "random"] = "diverse"