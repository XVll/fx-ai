# Comprehensive SHAP Feature Attribution Usage Guide

## Overview

The FxAIv2 trading system now includes a state-of-the-art SHAP (SHapley Additive exPlanations) feature attribution system that provides deep insights into how the model makes trading decisions. This guide explains how to use, configure, and interpret the SHAP analysis results.

## Key Features

### 1. **Multiple Attribution Methods**
- **GradientSHAP**: Fast and stable, good for real-time analysis
- **IntegratedGradients**: More accurate, captures non-linear relationships
- **DeepLift**: Compares to reference inputs
- **Feature Ablation**: Tests feature removal impact
- **Noise Tunnel**: Adds robustness to attributions

### 2. **Comprehensive Analysis**
- Feature-level importance across 2,330+ features
- Branch-level importance (HF, MF, LF, Portfolio)
- Feature interactions and dependencies
- Dead feature detection
- Temporal trends and stability
- Individual sample explanations
- Statistical analysis and correlations

### 3. **Performance Optimizations**
- Smart sampling strategies
- GPU acceleration
- Background data caching
- Batch processing
- Dynamic feature selection

### 4. **Rich Visualizations**
- Feature importance heatmaps
- Branch importance pie charts
- Top features bar charts
- Feature interaction matrices
- Sample explanation waterfalls
- Importance trend plots

### 5. **Integration**
- Automatic WandB logging
- Real-time dashboard updates
- Configurable update frequency
- State persistence

## Configuration

### Basic Configuration

```yaml
# In your training config (e.g., config/config.yaml)
defaults:
  - training: shap_config  # Include SHAP configuration

# Set attribution frequency
attribution_update_frequency: 10  # Run every 10 updates
```

### Advanced Configuration

```yaml
model:
  enable_attribution: true  # Enable SHAP system
  
  shap_config:
    # Master control
    enabled: true  # Set to false to completely disable
    
    # Frequency control
    update_frequency: 10  # Run every N updates (10-50 recommended)
    
    # Performance settings
    max_samples_per_analysis: 5  # Samples per analysis (3-10)
    background_samples: 10  # Background for baseline (5-20)
    batch_size: 4  # GPU batch size
    
    # Methods to use
    methods:
      - gradient_shap  # Primary method
      - integrated_gradients  # Secondary
    primary_method: gradient_shap
    
    # Feature selection
    top_k_features: 50  # Track top features
    dead_feature_threshold: 0.001
    interaction_top_k: 20
    
    # Analysis options
    analyze_interactions: true
    analyze_gradients: true
    track_attention: true
    detect_outliers: true
    
    # Output settings
    save_plots: true
    log_to_wandb: true
    dashboard_update: true
```

## Usage Examples

### 1. **Default Usage (Automatic)**

The SHAP analyzer runs automatically during training:

```bash
# Standard training with SHAP analysis every 10 updates
poetry run python main.py --config config
```

### 2. **Light Analysis Mode**

For faster training with less frequent analysis:

```bash
# SHAP analysis every 50 updates
poetry run python main.py --config config attribution_update_frequency=50
```

### 3. **Heavy Analysis Mode**

For detailed insights:

```bash
# Comprehensive SHAP every 10 updates with all methods
poetry run python main.py --config config \
  attribution_update_frequency=10 \
  model.shap_config.methods=[gradient_shap,integrated_gradients,deep_lift] \
  model.shap_config.max_samples_per_analysis=10
```

### 4. **Disable SHAP**

To completely disable SHAP analysis:

```bash
# No SHAP analysis
poetry run python main.py --config config \
  model.enable_attribution=false \
  model.shap_config.enabled=false
```

### 5. **Programmatic Control**

```python
# In your code
from metrics.factory import create_complete_metrics_system

# Get metrics system
metrics = create_complete_metrics_system(...)

# Configure SHAP frequency
metrics.model_internals_collector.configure_shap_frequency(50)

# Disable SHAP
metrics.model_internals_collector.disable_shap_analysis()

# Re-enable SHAP
metrics.model_internals_collector.enable_shap_analysis()
```

## Interpreting Results

### 1. **WandB Dashboard**

Navigate to your WandB project to see:

- **Feature Importance Charts**: Shows top contributing features
- **Branch Importance Pie**: Distribution across HF/MF/LF/Portfolio
- **Interaction Matrices**: Feature dependencies
- **Trend Plots**: How importance changes over time
- **Dead Feature Tracking**: Unused features

### 2. **Console Output**

During training, you'll see:

```
🔍 Starting comprehensive SHAP analysis with 50 states
✅ SHAP attribution analysis completed successfully
🏆 Top features: [('hf.price_momentum', 0.234), ('mf_1m.rsi', 0.189), ('portfolio.position_pnl', 0.156)]
```

### 3. **Dashboard Integration**

The live dashboard (http://localhost:8051) shows:

- Real-time feature importance
- Dead feature counts
- Branch importance distribution
- Feature health metrics
- Top trending features

### 4. **Analysis Metrics**

Key metrics to monitor:

- **Mean Absolute Importance**: Overall feature contribution
- **Feature Sparsity**: Percentage of low-importance features
- **Dead Features Count**: Features not contributing
- **Gini Coefficient**: Feature importance concentration
- **Interaction Strength**: Feature dependencies

## Performance Considerations

### Timing Guidelines

| Update Frequency | Analysis Time | Use Case |
|-----------------|---------------|----------|
| Every 10 updates | 2-10 seconds | Development, debugging |
| Every 25 updates | 2-10 seconds | Standard training |
| Every 50 updates | 2-10 seconds | Production training |
| Every 100 updates | 2-10 seconds | Long runs |

### Memory Usage

- Base: ~500MB for analyzer
- During analysis: +500MB-1GB
- With visualizations: +200MB

### GPU Acceleration

- 10x speedup with GPU enabled
- Ensure CUDA is available
- Set `use_gpu: true` in config

## Best Practices

### 1. **Start Light**
Begin with `update_frequency: 50` and increase detail as needed.

### 2. **Monitor Dead Features**
If >20% features are dead, consider feature engineering improvements.

### 3. **Check Branch Balance**
Ensure no single branch dominates (>80% importance).

### 4. **Track Trends**
Look for features with increasing/decreasing importance over time.

### 5. **Use Interactions**
Identify feature pairs that work together for better feature engineering.

### 6. **Sample Explanations**
Review individual predictions to understand decision-making.

## Troubleshooting

### Issue: SHAP Analysis Too Slow

**Solution**: 
- Reduce `max_samples_per_analysis` (try 3)
- Increase `update_frequency` (try 50)
- Use only `gradient_shap` method
- Disable `analyze_interactions`

### Issue: Out of Memory

**Solution**:
- Reduce `background_samples` (try 5)
- Disable `save_plots`
- Increase `update_frequency`
- Use CPU instead of GPU

### Issue: No Attribution Results

**Check**:
- Is `enabled: true` in config?
- Are there enough states in buffer?
- Check logs for error messages
- Verify model compatibility

### Issue: Poor Attribution Quality

**Solution**:
- Increase `background_samples` (try 20)
- Add more attribution methods
- Increase `max_samples_per_analysis`
- Check for numerical instabilities

## Advanced Features

### 1. **Custom Attribution Methods**

```python
# Add custom method
analyzer.explainers["custom"] = YourCustomExplainer(model)
analyzer.config.methods.append("custom")
```

### 2. **Export Results**

```python
# Save analyzer state
analyzer.save_state("shap_state.pkl")

# Load for analysis
analyzer.load_state("shap_state.pkl")
```

### 3. **Offline Analysis**

```python
# Analyze saved states
states = load_states("episode_states.pkl")
results = analyzer.analyze_features(states)
```

### 4. **Feature Importance API**

```python
# Get current importance
summary = analyzer.get_summary_for_logging()
top_features = summary["top_features"]
dead_count = summary["dead_features"]["count"]
```

## Summary

The comprehensive SHAP system provides unparalleled insights into your trading model's decision-making process. By understanding which features drive predictions, you can:

- Improve feature engineering
- Remove dead features
- Understand model behavior
- Debug unexpected trades
- Build trust in the system

Start with default settings and adjust based on your needs. The system is designed to provide valuable insights without significantly impacting training performance.