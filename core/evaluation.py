"""
Evaluation result data structures and exports.
"""

from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime
from config.evaluation.evaluation_config import EvaluationConfig


@dataclass 
class EvaluationEpisodeResult:
    """Result from a single evaluation episode - just the reward."""
    episode_num: int
    reward: float
    
    
@dataclass
class EvaluationResult:
    """Results from a complete evaluation run."""
    timestamp: datetime
    model_version: Optional[str]
    config: EvaluationConfig
    
    # Episode results
    episodes: List[EvaluationEpisodeResult]
    
    # Aggregate metrics (reward only for now)
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    total_episodes: int


# Export the data structures for backward compatibility
__all__ = [
    "EvaluationEpisodeResult",
    "EvaluationResult"
]