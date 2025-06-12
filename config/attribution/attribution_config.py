"""
Attribution configuration using Hydra structured configs.

Defines configuration for feature attribution analysis using Captum.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class AttributionConfig:
    """Configuration for Captum feature attribution analysis."""
    
    # Enable/disable attribution analysis
    enabled: bool = True
    
    # Attribution methods to use
    methods: List[str] = field(default_factory=lambda: ["integrated_gradients", "deep_lift"])
    
    # Method-specific parameters
    n_steps: int = 50                                 # Steps for integrated gradients
    n_samples: int = 25                               # Samples for gradient SHAP
    
    # Analysis settings
    analyze_branches: bool = True                     # Analyze each transformer branch separately
    analyze_fusion: bool = True                       # Analyze attention fusion layer
    analyze_actions: bool = True                      # Analyze action head attributions
    
    # Baseline configuration
    baseline_type: str = "zero"                       # Baseline type: zero, mean, random
    
    # Visualization settings
    save_visualizations: bool = True                  # Save attribution visualizations
    # NOTE: Paths now managed by PathManager - use PathManager.experiment_analysis_dir instead
    heatmap_threshold: float = 0.01                   # Minimum attribution value to show
    
    # Control which visualizations to create
    create_branch_heatmap: bool = True                # Create branch comparison heatmap
    create_timeseries_plot: bool = True               # Create time series attribution plots
    create_aggregated_plot: bool = True               # Create aggregated importance plot
    timeseries_branches: List[str] = field(default_factory=lambda: ["hf", "mf", "lf", "portfolio", "fusion"])
    
    # Performance settings
    batch_analysis: bool = False                      # Analyze multiple samples at once
    max_batch_size: int = 32                          # Maximum batch size for analysis
    
    # Feature importance aggregation
    aggregate_features: bool = True                   # Aggregate attributions by feature groups
    feature_groups: Optional[Dict[str, List[str]]] = field(default_factory=lambda: {
        "price_action": ["price", "returns", "volatility"],
        "volume": ["volume", "vwap", "relative_volume"],
        "microstructure": ["spread", "imbalance", "order_flow"],
        "technical": ["ema", "rsi", "patterns"],
        "portfolio": ["position", "pnl", "risk"],
    })
    
    # Callback settings (scheduling and integration)
    callback_enabled: bool = True                     # Enable callback integration
    analyze_every_n_episodes: Optional[int] = 10     # Episodes between analyses (None to disable)
    analyze_every_n_updates: Optional[int] = 5       # Updates between analyses (None to disable)
    save_to_wandb: bool = True                        # Log results to WandB