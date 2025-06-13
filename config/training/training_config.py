"""
Training configuration for PPO and training management - Hydra version.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class TrainingManagerConfig:
    """Training Manager - Central authority for training lifecycle"""

    # Core mode selection
    mode: str = "training"  # Training mode: training, optuna, benchmark

    # Training Termination (always enforced)
    termination_max_episodes: Optional[int] = None  # Max total episodes before training ends
    termination_max_updates: Optional[int] = None   # Max total updates before training ends
    termination_max_cycles: Optional[int] = None    # Max total data cycles before training ends

    intelligent_termination: bool = True             # Enable intelligent termination via callbacks
    plateau_patience: Optional[int] = 50             # Updates without improvement before plateau termination
    degradation_threshold: Optional[float] = 0.05   # Performance degradation threshold (5%)

    # Episode limits
    episode_max_steps: Optional[int] = None          # Max steps per episode before reset

    # Daily limits (when to switch to next day)
    daily_max_episodes: Optional[int] = None         # Max episodes per day before day switch
    daily_max_updates: Optional[int] = None          # Max updates per day before day switch
    daily_max_cycles: Optional[int] = 3              # Max cycles through reset points before day switch

    # Data selection
    symbols: List[str] = field(default_factory=lambda: [])                    # Trading symbols to use
    date_range: List[Optional[str]] = field(default_factory=lambda: [None, None])  # Date range [start, end] or [None, None] for all

    day_score_range: List[float] = field(default_factory=lambda: [0.0, 1.0])       # Day quality score filter range [min, max]
    reset_roc_range: List[float] = field(default_factory=lambda: [0.0, 1.0])      # Reset point ROC score filter range
    reset_activity_range: List[float] = field(default_factory=lambda: [0.0, 1.0])  # Reset point activity score filter range

    day_selection_mode: str = "sequential"          # Day ordering: sequential/quality/random
    reset_point_selection_mode: str = "sequential"  # Reset point ordering: sequential/quality/random

    # Data management
    preload_enabled: bool = True                     # Enable data preloading for performance

    # Evaluation
    eval_frequency: Optional[int] = 5                # Updates between evaluations (None to disable)
    eval_episodes: int = 10                          # Episodes per evaluation

    # Continuous training
    performance_window: Optional[int] = 50           # Performance history window size
    recommendation_frequency: Optional[int] = 10     # Episodes between recommendations
    adaptation_enabled: bool = True                  # Enable adaptive data difficulty
    early_stop_patience: int = 300                   # Updates without improvement for early stop
    early_stop_min_delta: float = 0.01              # Minimum improvement threshold

    # Model management
    best_model_metric: str = "mean_reward"           # Metric for best model selection
    continue_with_best_model: bool = True                 # Continue from best saved model
    keep_best_n_models: int = 5                      # Number of best models to keep
    checkpoint_frequency: Optional[int] = 25         # Updates between checkpoints (None to disable)

    rollout_steps: int = 2048                        # Steps per rollout collection


@dataclass
class TrainingConfig:
    """PPO training configuration"""
    
    # Core settings
    seed: int = 42                                   # Random seed for reproducibility
    shutdown_timeout: int = 10                       # Shutdown timeout in seconds
    device: str = "mps"   # Training device: cuda/mps/cpu

    # PPO hyperparameters
    learning_rate: float = 1.5e-4                   # Learning rate (typical: 1e-5 to 1e-3)
    batch_size: int = 64                            # Batch size (typical: 32, 64, 128, 256)
    n_epochs: int = 8                               # Training epochs per update (typical: 4-10)
    gamma: float = 0.99                             # Discount factor (0.9-0.999)
    gae_lambda: float = 0.95                        # GAE lambda (0.9-0.98)
    clip_epsilon: float = 0.15                      # PPO clip range (0.1-0.3)
    value_coef: float = 0.5                         # Value function coefficient (0.1-1.0)
    entropy_coef: float = 0.01                      # Entropy coefficient (0.001-0.1)
    max_grad_norm: float = 0.3                      # Gradient clipping (0.1-1.0)

    training_manager: TrainingManagerConfig = field(default_factory=TrainingManagerConfig)