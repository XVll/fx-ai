"""
Captum configuration for feature attribution analysis.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class CaptumCallbackConfig(BaseModel):
    """Captum callback configuration"""
    
    analyze_every_n_episodes: Optional[int] = Field(10, description="Run analysis every N episodes (null to disable)")
    analyze_every_n_updates: Optional[int] = Field(5, description="Run analysis every N PPO updates (null to disable)")
    save_to_wandb: bool = Field(True, description="Log results to Weights & Biases")
    save_to_dashboard: bool = Field(False, description="Send results to dashboard (deprecated)")
    output_dir: str = Field("outputs/captum", description="Directory for analysis reports")


class CaptumConfig(BaseModel):
    """Captum feature attribution configuration"""
    
    # Enable/disable Captum analysis
    enabled: bool = Field(True, description="Enable/disable Captum feature attribution analysis")
    
    # Attribution methods to use
    methods: List[str] = Field(
        default_factory=lambda: ["integrated_gradients", "deep_lift"],
        description="Attribution methods: integrated_gradients, deep_lift, gradient_shap, saliency, input_x_gradient"
    )
    
    # Method-specific parameters
    n_steps: int = Field(50, description="Steps for integrated gradients")
    n_samples: int = Field(25, description="Samples for gradient SHAP")
    
    # Analysis settings
    analyze_branches: bool = Field(True, description="Analyze each transformer branch separately")
    analyze_fusion: bool = Field(True, description="Analyze attention fusion layer")
    analyze_actions: bool = Field(True, description="Analyze action head attributions")
    
    # Baseline configuration
    baseline_type: str = Field("zero", description="Baseline type: zero, mean, random")
    
    # Visualization settings
    save_visualizations: bool = Field(True, description="Save attribution visualizations")
    visualization_dir: str = Field("outputs/captum", description="Visualization directory")
    heatmap_threshold: float = Field(0.01, description="Minimum attribution value to show")
    
    # Control which visualizations to create
    create_branch_heatmap: bool = Field(True, description="Create branch comparison heatmap")
    create_timeseries_plot: bool = Field(True, description="Create time series attribution plots")
    create_aggregated_plot: bool = Field(True, description="Create aggregated importance plot")
    timeseries_branches: List[str] = Field(
        default_factory=lambda: ["hf", "mf", "lf", "portfolio"],
        description="Which branches to create timeseries plots for"
    )
    
    # Performance settings
    batch_analysis: bool = Field(False, description="Analyze multiple samples at once")
    max_batch_size: int = Field(32, description="Maximum batch size for analysis")
    
    # Callback settings
    callback: CaptumCallbackConfig = Field(default_factory=CaptumCallbackConfig)