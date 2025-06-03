# Feature Attribution with Captum: Complete Guide

This guide covers the comprehensive gradient-based feature attribution system integrated into FxAI trading models using Facebook's Captum library.

## Table of Contents
- [Overview](#overview)
- [Integration & Setup](#integration--setup)
- [Where to Find Metrics](#where-to-find-metrics)
- [Key Metrics Explained](#key-metrics-explained)
- [Interpreting Results](#interpreting-results)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

## Overview

### What is Feature Attribution?
Feature attribution tells you **which features your model considers most important** for making trading decisions. Instead of having a "black box" model, you get clear insights into:
- Which price features drive buy/sell decisions
- How technical indicators influence the model
- Whether portfolio state properly guides actions
- Which features are unused (dead features)

### Why Captum over SHAP?
- **Native PyTorch Integration**: Designed specifically for PyTorch models
- **Gradient-Based Methods**: More accurate for deep neural networks
- **Better Performance**: 3-5x faster than SHAP for transformer models
- **Richer Analysis**: Layer-wise attribution, gradient flow tracking
- **Multiple Methods**: Compare different attribution algorithms for robustness

## Integration & Setup

### Automatic Integration
The feature attribution system is **fully integrated** into your training pipeline. No separate setup required!

```bash
# Normal training - attribution runs automatically
poetry run poe momentum

# Continue training - attribution continues from previous state
poetry run poe momentum-continue
```

### How It Works
1. **Every 10 updates**: Comprehensive attribution analysis (configurable)
2. **Every step**: Real-time gradient and attention tracking
3. **Continuous**: Feature importance scoring and dead feature detection
4. **Automatic**: Integration with dashboard and W&B logging

### Dependencies
Already included in your `pyproject.toml`:
```toml
captum = "^0.7.0"  # Feature attribution library
```

## Where to Find Metrics

### 1. Live Dashboard (Primary Interface)
**URL**: `http://localhost:8051` (auto-opens during training)

**What you'll see**:
- Real-time feature importance scores
- Gradient flow health indicators
- Attribution consensus metrics
- Dead feature counts

**Location in Dashboard**:
- **PPO Metrics Section**: Attribution quality indicators
- **Model Internals**: Feature importance by branch
- **Training Progress**: Attribution analysis status

### 2. Console Logs
```bash
🔍 Captum attribution analysis completed - feature importance updated
Attribution consensus: 0.847
Attribution quality - Sparsity: 0.234, Concentration: 0.156
Gradient flow: 0 vanishing, 0 exploding layers
```

### 3. Weights & Biases Integration
**Metrics Logged**:
- `model.internals.top_feature_importance_*`: Top feature scores per branch
- `model.internals.dead_features_count`: Number of unused features
- `model.internals.attention_stability`: Attention pattern consistency
- `model.internals.attribution_analysis_count`: Number of analyses run

### 4. Generated Reports
**Location**: `outputs/` directory
**Files**:
- `feature_importance_analysis.png`: Comprehensive attribution visualizations
- Attribution heatmaps by branch
- Gradient flow charts
- Feature ranking tables

## Key Metrics Explained

### 1. Feature Importance Rankings

**What it shows**: Most influential features for model decisions

```python
# Example output structure
{
  "hf": [
    ("price_velocity", 0.845),      # Price change rate
    ("volume_ratio", 0.723),        # Volume vs average
    ("spread", 0.654)               # Bid-ask spread
  ],
  "mf": [
    ("ema_cross_signal", 0.892),    # EMA crossover
    ("rsi_divergence", 0.743),      # RSI momentum
    ("bollinger_position", 0.621)   # Price vs Bollinger bands
  ],
  "lf": [
    ("support_distance", 0.743),    # Distance to support
    ("resistance_strength", 0.598), # Resistance level quality
    ("trend_strength", 0.487)       # Overall trend momentum
  ],
  "portfolio": [
    ("position_pnl_percent", 0.934), # Current P&L %
    ("cash_ratio", 0.721),           # Cash vs total equity
    ("position_hold_time", 0.543)    # How long holding position
  ]
}
```

**Healthy Patterns**:
- ✅ **Portfolio features highest** (0.6-0.9): Model considers current position
- ✅ **Consistent rankings**: Same features stay important over time
- ✅ **Balanced distribution**: No single feature >0.95

**Warning Signs**:
- ⚠️ **Wild fluctuations**: Rankings change dramatically between updates
- ⚠️ **All scores low** (<0.3): Model not using features effectively
- ❌ **Single feature dominance** (>0.95): Over-reliance on one input

### 2. Attribution Consensus

**What it measures**: Agreement between different attribution methods

```python
{
  "mean_correlation": 0.847,    # Average correlation between methods
  "min_correlation": 0.723,     # Worst agreement
  "std_correlation": 0.089,     # Consistency of agreement
  "agreement_score": 0.75       # % methods with high correlation (>0.7)
}
```

**Interpretation**:
- ✅ **>0.7**: High confidence - different methods agree on importance
- ⚠️ **0.4-0.7**: Moderate confidence - some disagreement
- ❌ **<0.4**: Low confidence - methods disagree significantly

**What low consensus means**:
- Model may be unstable
- Feature preprocessing issues
- Architecture problems
- Need for model simplification

### 3. Attribution Quality Metrics

```python
{
  "sparsity": 0.234,        # % features with very low importance
  "concentration": 0.156,    # How concentrated attributions are
  "infidelity": 0.023,      # Attribution accuracy (lower better)
  "sensitivity": 0.045      # Attribution stability (lower better)
}
```

**Sparsity** (0.0-1.0):
- ✅ **0.2-0.4**: Healthy - some features more important than others
- ⚠️ **<0.1**: Too uniform - all features equally important (suspicious)
- ⚠️ **>0.6**: Too sparse - few features doing all the work

**Concentration** (0.0-1.0):
- ✅ **0.1-0.3**: Balanced importance distribution
- ⚠️ **<0.05**: Too distributed - no clear important features
- ❌ **>0.5**: Over-concentrated on few features

**Infidelity** (lower is better):
- ✅ **<0.05**: Attributions accurately reflect model behavior
- ⚠️ **0.05-0.1**: Moderate accuracy
- ❌ **>0.1**: Poor attribution quality

**Sensitivity** (lower is better):
- ✅ **<0.1**: Stable attributions
- ⚠️ **0.1-0.3**: Some instability
- ❌ **>0.3**: Very unstable attributions

### 4. Gradient Flow Analysis

**What it tracks**: Gradient magnitudes through model layers

```python
{
  "layer_gradients": {
    "hf_transformer_0": {"norm": 0.234, "mean": 0.023},
    "mf_transformer_1": {"norm": 0.156, "mean": 0.019},
    "fusion": {"norm": 0.089, "mean": 0.012}
  },
  "vanishing_gradients": [],      # Layers with norm < 1e-7
  "exploding_gradients": []       # Layers with norm > 1e3
}
```

**Healthy Gradient Flow**:
- ✅ **Norms 0.01-10**: Good gradient magnitudes
- ✅ **Empty vanishing/exploding lists**: No gradient problems
- ✅ **Gradual decrease**: Norms decrease slightly toward input

**Problem Indicators**:
- ❌ **Vanishing gradients**: Norms <1e-7, learning rate too low
- ❌ **Exploding gradients**: Norms >1e3, learning rate too high
- ⚠️ **Sudden drops**: Sharp gradient decreases between layers

### 5. Dead Feature Detection

**What it finds**: Features with consistently low importance

```python
{
  "dead_features": {
    "hf": ["unused_tick_feature", "redundant_volume_calc"],
    "mf": [],
    "lf": ["weak_support_indicator"],
    "portfolio": []
  },
  "total_count": 3
}
```

**Normal Counts**:
- ✅ **0-5 dead features**: Normal, some features naturally less useful
- ⚠️ **6-15 dead features**: Review feature engineering
- ❌ **>15 dead features**: Significant feature quality issues

## Interpreting Results

### Training Health Assessment

**Excellent Attribution Health**:
```
🔍 Captum attribution analysis completed - feature importance updated
Attribution consensus: 0.823  ← High agreement
Attribution quality - Sparsity: 0.267, Concentration: 0.134  ← Balanced
Dead features: 2  ← Minimal unused features
Gradient flow: 0 vanishing, 0 exploding layers  ← Healthy training
```

**Concerning Attribution Health**:
```
⚠️ Attribution consensus: 0.234  ← Methods disagree strongly
⚠️ Attribution quality - Sparsity: 0.067, Concentration: 0.734  ← Poor distribution
⚠️ Dead features: 23  ← Many unused features
⚠️ Gradient flow: 5 vanishing, 2 exploding layers  ← Training problems
```

### Feature Importance Patterns

**Expected Branch Contributions**:

1. **Portfolio Branch** (typically 40-60% total importance):
   - Should show highest importance scores
   - Position P&L, cash ratio, hold time most critical
   - If low: Model not properly considering current state

2. **High-Frequency Branch** (15-35% during active trading):
   - Price velocity, volume ratios, spread dynamics
   - Higher during market open/close
   - If too high: Over-fitting to noise

3. **Medium-Frequency Branch** (20-40% importance):
   - Technical indicators: EMA, RSI, Bollinger Bands
   - Should be consistently moderate
   - If dominant: Over-reliance on lagging indicators

4. **Low-Frequency Branch** (10-25% importance):
   - Support/resistance, trend analysis
   - Should provide steady background signal
   - If too high: Model may be too slow to react

### Trading Decision Analysis

**Good Attribution for Buy Decisions**:
- Portfolio: Low cash ratio, no current position
- HF: Strong upward price velocity, high volume
- MF: Bullish EMA cross, RSI not overbought
- LF: Above support, in uptrend

**Good Attribution for Sell Decisions**:
- Portfolio: Large unrealized profit, long hold time
- HF: Downward price velocity, volume spike
- MF: Bearish divergence, overbought RSI
- LF: Near resistance, trend weakening

**Good Attribution for Hold Decisions**:
- Portfolio: Small position, recent entry
- HF: Low volatility, stable price
- MF: Sideways indicators, no clear signals
- LF: Between support/resistance

## Troubleshooting

### Low Attribution Consensus (<0.4)

**Possible Causes**:
- Model architecture too complex
- Inconsistent feature preprocessing
- Training instability
- Data quality issues

**Solutions**:
1. Simplify model architecture
2. Verify feature normalization
3. Reduce learning rate
4. Check for data leakage

### Many Dead Features (>15)

**Possible Causes**:
- Poor feature engineering
- Redundant features
- Wrong feature scaling
- Model capacity issues

**Solutions**:
1. Remove consistently unused features
2. Combine correlated features
3. Improve feature preprocessing
4. Increase model capacity

### Gradient Flow Problems

**Vanishing Gradients**:
- Reduce model depth
- Increase learning rate
- Add residual connections
- Check activation functions

**Exploding Gradients**:
- Reduce learning rate
- Add gradient clipping
- Check weight initialization
- Reduce model complexity

### Unstable Feature Rankings

**Possible Causes**:
- Training instability
- Insufficient data
- Model overfitting
- Poor episode termination

**Solutions**:
1. Increase training stability (lower LR)
2. More training episodes
3. Add regularization
4. Review episode reset logic

## Advanced Usage

### Manual Attribution Analysis

```python
# Access the analyzer directly
analyzer = trainer.metrics.model_internals_collector.attribution_analyzer

# Generate detailed report for specific states
states_to_analyze = [state1, state2, state3]
report = analyzer.generate_attribution_report(
    sample_states=states_to_analyze,
    save_path="custom_attribution_analysis.png"
)

# Compare two trading decisions
comparison = analyzer.compare_model_decisions(
    state_before_trade, 
    state_after_trade
)
print(f"Key differences: {comparison['key_discriminative_features']}")

# Check for dead features with custom threshold
dead_features = analyzer.get_dead_features(
    threshold=0.005,  # Lower threshold for stricter detection
    min_history=200   # Require more history
)

# Get gradient flow analysis
gradient_analysis = analyzer.analyze_gradient_flow()
if gradient_analysis['vanishing_gradients']:
    print(f"Warning: Vanishing gradients in {gradient_analysis['vanishing_gradients']}")
```

### Custom Attribution Configuration

```python
from feature.attribution import AttributionConfig, AttributionMethod

# Create custom configuration
config = AttributionConfig(
    primary_method=AttributionMethod.INTEGRATED_GRADIENTS,
    secondary_methods=[
        AttributionMethod.GRADIENT_SHAP,
        AttributionMethod.DEEP_LIFT,
        AttributionMethod.SALIENCY
    ],
    n_steps=100,  # More accurate but slower
    n_samples=50,  # More samples for stability
    use_noise_tunnel=True,
    noise_tunnel_samples=25,
    track_gradients=True,
    track_activations=True,
    baseline_type="gaussian"  # Different baseline strategy
)
```

### Debugging Attribution Issues

```python
# Check attribution analyzer status
if hasattr(trainer.metrics, 'model_internals_collector'):
    collector = trainer.metrics.model_internals_collector
    summary = collector.get_feature_attribution_summary()
    
    print(f"Attribution enabled: {summary['attribution_enabled']}")
    print(f"States buffered: {summary['states_buffered']}")
    print(f"Last analysis: {summary['last_attribution_analysis']}")
    print(f"Top features: {summary['top_features_by_branch']}")
    print(f"Dead features: {summary['dead_features']}")
```

### Integration with Other Analysis Tools

```python
# Export attribution data for external analysis
rankings = analyzer.get_feature_importance_ranking(n_top=20)

# Convert to pandas DataFrame for analysis
import pandas as pd
all_features = []
for branch, features in rankings.items():
    for name, importance in features:
        all_features.append({
            'branch': branch,
            'feature': name,
            'importance': importance
        })

df = pd.DataFrame(all_features)
print(df.groupby('branch')['importance'].describe())
```

## Best Practices

1. **Monitor Regularly**: Check attribution health every few training sessions
2. **Trust Consensus**: High consensus (>0.7) indicates reliable results
3. **Clean Dead Features**: Remove consistently unused features to improve efficiency
4. **Watch Gradient Flow**: Address gradient problems immediately
5. **Validate Against Trading Logic**: Attribution should make trading sense
6. **Use for Feature Engineering**: Let attribution guide new feature development

## Summary

The Captum feature attribution system provides unprecedented insight into your trading model's decision-making process. By monitoring the metrics described in this guide, you can:

- **Understand** what drives your model's trading decisions
- **Identify** problematic features or training issues early
- **Optimize** feature engineering based on actual importance
- **Debug** model behavior when performance degrades
- **Build confidence** in your model's reasoning process

The system runs automatically during training, providing continuous insights without additional overhead. Use the dashboard for real-time monitoring and this guide for interpreting the results to build better, more interpretable trading models.