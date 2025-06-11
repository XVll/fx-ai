"""
Evaluation configuration for model performance assessment.
"""

from dataclasses import dataclass
from typing import Literal, Optional


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
    
    # Benchmark-specific settings
    benchmark_episodes: int = 100    # More episodes for comprehensive benchmarking
    save_episode_details: bool = True  # Save individual episode results
    benchmark_output_dir: Optional[str] = None  # Directory for benchmark results
    model = Optional[str] = None  # Model version to evaluate, if None use latest
    
    # Advanced evaluation settings
    max_steps_per_episode: int = 1000  # Safety limit for episode length
    warm_up_episodes: int = 0  # Episodes to run before measurement (for cache warming)
    
    # Performance measurement
    measure_inference_time: bool = False  # Measure model inference time
    measure_memory_usage: bool = False    # Track memory usage during evaluation