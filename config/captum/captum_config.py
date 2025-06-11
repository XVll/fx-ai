"""
Captum configuration for feature attribution analysis.
"""

from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class CaptumCallbackConfig:
    """Captum callback configuration"""
    
    analyze_every_n_episodes: Optional[int] = 10     # Run analysis every N episodes (null to disable)
    analyze_every_n_updates: Optional[int] = 5       # Run analysis every N PPO updates (null to disable)
    save_to_wandb: bool = True                        # Log results to Weights & Biases
    save_to_dashboard: bool = False                   # Send results to dashboard (deprecated)
    output_dir: str = "outputs/captum"                # Directory for analysis reports


@dataclass
class CaptumConfig:
    """Captum feature attribution configuration"""
    
    # Enable/disable Captum analysis
    enabled: bool = True                              # Enable/disable Captum feature attribution analysis
    
    # Attribution methods to use
    methods: List[str] = field(default_factory=lambda: ["integrated_gradients", "deep_lift"])  # Attribution methods
    
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
    visualization_dir: str = "outputs/captum"        # Visualization directory
    heatmap_threshold: float = 0.01                   # Minimum attribution value to show
    
    # Control which visualizations to create
    create_branch_heatmap: bool = True                # Create branch comparison heatmap
    create_timeseries_plot: bool = True               # Create time series attribution plots
    create_aggregated_plot: bool = True               # Create aggregated importance plot
    timeseries_branches: List[str] = field(default_factory=lambda: ["hf", "mf", "lf", "portfolio"])  # Branches for timeseries plots
    
    # Performance settings
    batch_analysis: bool = False                      # Analyze multiple samples at once
    max_batch_size: int = 32                          # Maximum batch size for analysis
    
    # Callback settings
    callback: CaptumCallbackConfig = field(default_factory=CaptumCallbackConfig)