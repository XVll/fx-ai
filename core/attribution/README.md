# Captum Feature Attribution for Trading Models

This module provides comprehensive gradient-based feature attribution analysis using Facebook's Captum library, replacing the previous SHAP implementation.

## Key Features

- **Multiple Attribution Methods**: Integrated Gradients, GradientShap, DeepLift, Saliency, and more
- **Gradient Flow Analysis**: Track gradient magnitudes through layers to detect vanishing/exploding gradients
- **Real-time Attribution**: Automatic analysis during training every 10 updates
- **Consensus Analysis**: Compare multiple attribution methods for robustness
- **Dead Feature Detection**: Identify features with consistently low importance
- **Visualization**: Comprehensive reports with attribution heatmaps and gradient flow charts

## Why Captum over SHAP?

1. **Native PyTorch Integration**: Captum is designed specifically for PyTorch models
2. **Gradient-Based Methods**: More efficient and accurate for deep neural networks
3. **Layer Attribution**: Can analyze importance at different network layers
4. **Better Performance**: Faster computation, especially for transformer architectures
5. **Rich Method Selection**: Multiple state-of-the-art attribution algorithms

## Usage

The attribution analyzer is automatically integrated into the training pipeline:

```python
from core.attribution import CaptumFeatureAnalyzer, AttributionConfig, AttributionMethod

# Configuration
config = AttributionConfig(
    primary_method=AttributionMethod.INTEGRATED_GRADIENTS,
    secondary_methods=[AttributionMethod.GRADIENT_SHAP, AttributionMethod.DEEP_LIFT],
    n_steps=50,
    use_noise_tunnel=True,
    track_gradients=True
)

# Initialize analyzer
analyzer = CaptumFeatureAnalyzer(
    model=your_model,
    feature_names=feature_names_dict,
    config=config
)

# Calculate attributions
results = analyzer.calculate_attributions(sample_states)

# Get feature rankings
rankings = analyzer.get_feature_importance_ranking(n_top=10)

# Analyze gradient flow
gradient_analysis = analyzer.analyze_gradient_flow()

# Generate comprehensive report
report = analyzer.generate_attribution_report(
    sample_states,
    save_path="attribution_report.png"
)
```

## Attribution Methods

### Integrated Gradients (Default)
- Computes attribution by integrating gradients along a straight path from baseline to input
- Most reliable and theoretically grounded method
- Satisfies completeness axiom

### GradientShap
- Combines ideas from Integrated Gradients and SHAP
- Uses multiple baselines for more robust attributions
- Good for understanding average behavior

### DeepLift
- Compares activation of each neuron to its 'reference activation'
- Efficient computation
- Good for ReLU networks

### Saliency
- Simple gradient-based method
- Very fast but less accurate
- Good for quick analysis

## Integration with Training

The analyzer automatically runs during training:

1. **Every 10 Updates**: Comprehensive attribution analysis
2. **Real-time Tracking**: Attention weights and gradients monitored continuously
3. **Dashboard Integration**: Results displayed in live dashboard
4. **W&B Logging**: Attribution metrics logged to Weights & Biases

## Interpreting Results

### Feature Rankings
- Higher attribution scores indicate more important features
- Scores are aggregated across time steps for sequential features
- Both positive and negative attributions are considered (absolute values)

### Gradient Flow
- **Vanishing Gradients**: Layers with gradient norm < 1e-7
- **Exploding Gradients**: Layers with gradient norm > 1e3
- **Healthy Flow**: Gradient norms between 0.01 and 100

### Consensus Scores
- **High Consensus** (>0.7): Different methods agree on feature importance
- **Medium Consensus** (0.4-0.7): Some agreement between methods
- **Low Consensus** (<0.4): Methods disagree, results less reliable

## Performance Considerations

- Attribution analysis runs asynchronously to minimize training impact
- Batch processing for efficiency
- Configurable analysis frequency
- GPU acceleration when available

## Troubleshooting

### High Memory Usage
- Reduce `n_steps` for Integrated Gradients
- Disable `track_activations` if not needed
- Use smaller sample batches for analysis

### Slow Analysis
- Use faster methods like Saliency for quick checks
- Reduce analysis frequency
- Disable secondary methods if not needed

### Attribution Errors
- Ensure model is in eval mode during analysis
- Check that input tensors require gradients
- Verify feature dimensions match model expectations