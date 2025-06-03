# Comprehensive SHAP Feature Attribution System

## Overview

The Comprehensive SHAP Feature Attribution System is a production-ready implementation designed to handle models with 2,330+ features efficiently. It provides deep insights into feature importance, interactions, and model behavior through multiple attribution methods and advanced visualizations.

## Key Features

### 1. Multiple Attribution Methods
- **GradientSHAP**: Fast approximation using gradient sampling
- **Integrated Gradients**: Path-based attribution from baseline to input
- **DeepLift**: Compares activations to reference activations
- **Feature Ablation**: Measures impact of removing features
- **Noise Tunnel**: Adds robustness through noisy sampling

### 2. Feature-Level Attribution
- Precise attribution for each of the 2,330 features
- Branch-aware analysis (HF, MF, LF, Portfolio)
- Sequence position tracking
- Feature name mapping and metadata

### 3. Feature Interactions & Dependencies
- Correlation-based interaction detection
- Top interaction pairs identification
- Interaction strength metrics
- Network visualization of dependencies

### 4. Dead Feature Detection
- Configurable importance thresholds
- Historical tracking for confirmation
- Branch-level dead feature analysis
- Automatic flagging and reporting

### 5. Feature Importance Trends
- Exponential moving averages
- Trend detection (increasing/decreasing/stable)
- Variance tracking for stability
- Historical importance visualization

### 6. Individual Sample Explanations
- Top positive/negative contributors per decision
- Action-specific explanations
- Reward correlation analysis
- Waterfall visualizations

### 7. Statistical Analysis
- Distribution statistics (mean, std, skewness, kurtosis)
- Gini coefficient for importance inequality
- Concentration ratios
- Outlier detection with z-scores

### 8. Advanced Visualizations
- Feature importance heatmaps by branch
- Branch importance pie charts
- Top features bar charts
- Interaction matrices
- Sample explanation waterfalls
- Importance trend plots

### 9. Performance Optimizations
- Smart sampling (first, last, random middle)
- Background data caching
- GPU acceleration
- Batch processing
- Dynamic feature selection
- Configurable update frequency

### 10. Dashboard Integration
- Real-time feature importance display
- Dead feature monitoring
- Branch importance tracking
- Interactive visualizations
- Performance metrics

## Configuration

```python
from feature.attribution import AttributionConfig

config = AttributionConfig(
    # Analysis frequency
    update_frequency=10,  # Run every 10 updates
    max_samples_per_analysis=5,  # Limited for performance
    background_samples=10,
    
    # Methods
    methods=["gradient_shap", "integrated_gradients"],
    primary_method="gradient_shap",
    
    # Feature selection
    top_k_features=50,
    dead_feature_threshold=0.001,
    interaction_top_k=20,
    
    # Performance
    use_gpu=True,
    batch_size=8,
    cache_background=True,
    
    # Visualization
    save_plots=True,
    plot_dir="outputs/shap_plots",
    plot_dpi=150,
    
    # Tracking
    history_length=100,
    trend_window=10,
    dashboard_update_freq=5
)
```

## Usage Example

```python
from feature.attribution import ComprehensiveSHAPAnalyzer

# Initialize analyzer
analyzer = ComprehensiveSHAPAnalyzer(
    model=model,
    feature_names={
        'hf': ['price_velocity', 'tape_imbalance', ...],
        'mf': ['ema_distance', 'vwap_slope', ...],
        'lf': ['daily_range', 'session_progress', ...],
        'portfolio': ['position', 'pnl', ...]
    },
    branch_configs={
        'hf': (60, 7),      # 60 timesteps, 7 features
        'mf': (30, 43),     # 30 timesteps, 43 features
        'lf': (30, 19),     # 30 timesteps, 19 features
        'portfolio': (5, 10) # 5 timesteps, 10 features
    },
    config=config
)

# Setup background data
analyzer.setup_background(background_states)

# Run analysis
results = analyzer.analyze_features(
    states=recent_states,
    actions=actions_taken,
    rewards=rewards_received
)

# Get dashboard summary
summary = analyzer.get_summary_for_logging()

# Log to WandB
analyzer.log_to_wandb(results, step=current_step)
```

## Performance Considerations

### With 2,330 Features:
- **Analysis Time**: ~2-10 seconds per analysis (5 samples)
- **Memory Usage**: ~500MB-1GB during analysis
- **GPU Recommended**: 10x faster with GPU acceleration
- **Update Frequency**: Default 10, can increase to 50 for lighter analysis

### Optimization Strategies:
1. **Reduce sample count**: Use 2-5 samples instead of more
2. **Increase update frequency**: Run less often (every 50 updates)
3. **Use faster methods**: GradientSHAP is faster than IntegratedGradients
4. **Limit background samples**: Use 5-10 instead of more
5. **Disable visualizations**: Skip plots during training
6. **Use primary method only**: Don't run multiple methods

## Output Structure

```python
{
    "method": "gradient_shap",
    "num_samples": 5,
    "analysis_time": 3.45,
    
    "feature_importance": {
        "hf.price_velocity": {
            "current": 0.0234,
            "ema": 0.0221,
            "variance": 0.0003,
            "stability": 0.9997
        },
        # ... all features
    },
    
    "branch_importance": {
        "hf": {
            "mean": 0.0156,
            "std": 0.0089,
            "proportion": 0.35
        },
        # ... all branches
    },
    
    "top_features": [
        {
            "rank": 1,
            "name": "vwap_distance",
            "branch": "mf",
            "importance": 0.0452,
            "importance_ema": 0.0448,
            "attribution_range": [-0.23, 0.31]
        },
        # ... top K features
    ],
    
    "dead_features": {
        "total_dead": 234,
        "dead_percentage": 10.04,
        "dead_by_branch": {
            "hf": [...],
            "mf": [...],
            "lf": [...],
            "portfolio": [...]
        }
    },
    
    "feature_interactions": {
        "top_interactions": [
            {
                "feature_1": "hf.price_velocity",
                "feature_2": "hf.volume_velocity",
                "correlation": 0.82,
                "strength": "strong"
            },
            # ... top interactions
        ]
    },
    
    "sample_explanations": [
        {
            "sample_index": 0,
            "action": 1,  # BUY
            "reward": 0.0023,
            "top_positive_features": [...],
            "top_negative_features": [...],
            "branch_contributions": {...}
        },
        # ... per sample
    ],
    
    "statistics": {
        "attribution_stats": {...},
        "importance_stats": {...},
        "sparsity_stats": {...}
    },
    
    "importance_trends": {
        "top_features_trends": [...],
        "global_trend": {...}
    }
}
```

## Integration with Training Pipeline

See `shap_integration_example.py` for a complete example of integrating the analyzer with PPO training callbacks.

## Visualization Gallery

### 1. Feature Importance Heatmap
Shows top features per branch with color-coded importance values.

### 2. Branch Importance Pie Chart
Displays relative importance of each feature branch with statistics.

### 3. Top Features Bar Chart
Horizontal bar chart of top K features colored by branch.

### 4. Feature Interaction Matrix
Correlation heatmap showing feature dependencies.

### 5. Sample Explanation Waterfall
Shows positive and negative contributors for individual predictions.

### 6. Importance Trends
Line plots showing feature importance evolution over time.

## Best Practices

1. **Start with higher update frequency**: Begin with analysis every 50 updates, then decrease
2. **Monitor analysis time**: If >10s, reduce samples or increase frequency
3. **Check dead features regularly**: Remove or fix features with <0.1% importance
4. **Use GPU when available**: Significant speedup for attribution calculations
5. **Cache background data**: Reuse background samples across analyses
6. **Focus on top features**: Most insights come from top 50-100 features
7. **Compare methods occasionally**: Run full comparison every 100 updates
8. **Save analyzer state**: Checkpoint for later analysis

## Troubleshooting

### Issue: Analysis takes too long
- Reduce `max_samples_per_analysis` to 2-3
- Increase `update_frequency` to 50+
- Use only `gradient_shap` method
- Disable visualization generation

### Issue: Out of memory
- Reduce `background_samples` to 5
- Lower `batch_size` to 4
- Clear cache more frequently
- Use CPU instead of GPU

### Issue: Dead features not detected
- Lower `dead_feature_threshold` to 0.0001
- Increase `history_length` for more data
- Check feature normalization

### Issue: Unclear attributions
- Add more background samples
- Try different attribution methods
- Check model convergence first
- Verify feature preprocessing

## Future Enhancements

1. **Hierarchical Analysis**: Aggregate features by semantic groups
2. **Temporal Attribution**: Track attribution changes within episodes
3. **Causal Analysis**: Use interventional methods
4. **Automated Feature Pruning**: Remove dead features automatically
5. **Attribution-Guided Training**: Use insights to improve model
6. **Real-time Dashboard**: Interactive web-based visualization
7. **Feature Importance Reports**: Automated PDF/HTML reports
8. **Cross-Episode Analysis**: Compare attributions across different scenarios