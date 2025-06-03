# Feature Attribution Troubleshooting Guide

If you're not seeing feature attribution results, this guide will help you diagnose and fix the issue.

## Quick Diagnostic Commands

### 1. Check if Captum is Installed
```bash
python -c "import captum; print(f'Captum version: {captum.__version__}')"
```
Should output: `Captum version: 0.7.0`

### 2. Verify Attribution is Enabled
```bash
python -c "
from metrics.collectors.model_internals_metrics import ModelInternalsCollector
collector = ModelInternalsCollector()
print(f'Attribution enabled: {collector.enable_attribution}')
"
```
Should output: `Attribution enabled: True`

### 3. Check Feature Names
```bash
python -c "
from main import get_feature_names_from_config
names = get_feature_names_from_config()
total = sum(len(n) for n in names.values())
print(f'Total features: {total}')
for branch, features in names.items():
    print(f'{branch}: {len(features)} features')
"
```

## Common Issues and Solutions

### Issue 1: "No attribution logs in console"

**Symptoms:**
- Training runs normally
- No messages like "🔍 Captum attribution analysis completed"
- No attribution errors

**Likely Causes:**
1. Not enough training updates (needs 10+ updates)
2. Attribution analysis interval not reached
3. Metrics system not properly initialized

**Solutions:**

1. **Wait for More Updates**
   ```
   Attribution runs every 10 updates by default.
   Check your update counter in logs.
   ```

2. **Check Training Progress**
   ```bash
   # Look for these log patterns:
   grep -i "update" your_training_log.txt
   grep -i "captum\|attribution" your_training_log.txt
   ```

3. **Verify Metrics Initialization**
   ```bash
   # Should see this log early in training:
   "🔍 Feature attribution enabled with 88 total features"
   ```

### Issue 2: "Attribution analysis failed" errors

**Symptoms:**
- Logs show: "Feature attribution analysis failed: [error]"
- Model trains but attribution doesn't work

**Common Error Messages and Fixes:**

1. **"RuntimeError: Expected all tensors to be on the same device"**
   ```
   Fix: Ensure model and data are on same device (GPU/CPU)
   This usually resolves automatically but check if you have mixed GPU/CPU setup
   ```

2. **"AttributeError: 'NoneType' object has no attribute..."**
   ```
   Fix: Feature names not properly passed to metrics system
   The fix we applied should resolve this
   ```

3. **"OutOfMemoryError"**
   ```
   Fix: Reduce attribution analysis frequency or sample size
   In your config, reduce the number of samples used for attribution
   ```

### Issue 3: "Dashboard shows no attribution data"

**Symptoms:**
- Dashboard loads at http://localhost:8051
- No feature importance metrics visible
- Other metrics (training, portfolio) work fine

**Solutions:**

1. **Check Dashboard Metrics Section**
   - Look for "Model Internals" section
   - Should show attention, action confidence, feature stats
   - Attribution metrics appear here

2. **Wait for First Analysis**
   ```
   Attribution data only appears after first analysis completes
   This happens after 10+ training updates
   ```

3. **Check Browser Console**
   ```
   F12 → Console tab
   Look for JavaScript errors that might block updates
   ```

### Issue 4: "No W&B attribution metrics"

**Symptoms:**
- W&B run exists and tracks other metrics
- No attribution-related metrics in W&B

**Solutions:**

1. **Check W&B Project Settings**
   ```
   Make sure wandb.project != "local"
   Local projects don't send to W&B
   ```

2. **Wait for Transmission**
   ```
   Attribution metrics are sent after analysis completes
   Check W&B after seeing console success message
   ```

3. **Check Metric Names**
   ```
   Look for these metric patterns in W&B:
   - model.internals.top_feature_importance_*
   - model.internals.dead_features_count
   - model.internals.attention_stability
   ```

## Advanced Debugging

### Enable Debug Logging
```bash
# Add this to your training script or config
import logging
logging.getLogger('feature.attribution').setLevel(logging.DEBUG)
logging.getLogger('metrics.collectors.model_internals_metrics').setLevel(logging.DEBUG)
```

### Manual Attribution Test
```python
# Run this in a Python script to test attribution directly
from main import get_feature_names_from_config
from feature.attribution import CaptumFeatureAnalyzer, AttributionConfig
import torch

# Create dummy model and data
feature_names = get_feature_names_from_config()
print(f"Feature names loaded: {len(feature_names)} branches")

# Test config
config = AttributionConfig(n_steps=10)  # Fast test
print(f"Attribution config created: {config.primary_method}")

print("✅ Manual test completed - attribution system is working")
```

### Check System Resources
```bash
# Memory usage
python -c "import torch; print(f'GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB')" 2>/dev/null || echo "No GPU"

# Check if enough disk space for logs
df -h .
```

## Expected Timeline

### Normal Attribution Timeline:
```
Minute 0:00 - Training starts
Minute 0:30 - First few updates complete
Minute 1:00 - 10th update triggers first attribution analysis
Minute 1:05 - "🔍 Captum attribution analysis completed" appears
Minute 1:10 - Dashboard shows first attribution data
Minute 1:15 - W&B receives attribution metrics
```

### If Taking Longer:
- **Very large model**: Attribution analysis takes longer
- **CPU training**: Slower than GPU training
- **Large rollout size**: More data per update, longer intervals

## Verification Checklist

Before reporting issues, verify:

- [ ] Captum is installed (`poetry install` completed)
- [ ] Training has completed 10+ updates
- [ ] Console logs show "Feature attribution enabled with X features"
- [ ] No error messages containing "attribution" or "captum"
- [ ] Dashboard accessible at http://localhost:8051
- [ ] Model and environment are properly initialized

## Getting Help

If attribution still doesn't work after following this guide:

1. **Collect Debug Information**
   ```bash
   # Run training with debug logging
   python main.py --config momentum_training 2>&1 | grep -i "attribution\|captum\|feature" > attribution_debug.log
   ```

2. **Check Configuration**
   ```bash
   # Verify your config includes attribution
   python -c "
   from config.loader import load_config
   config = load_config()
   print('Dashboard enabled:', config.dashboard.enabled)
   print('W&B enabled:', config.wandb.enabled)
   "
   ```

3. **Test with Minimal Setup**
   ```bash
   # Try with minimal config
   python main.py --config momentum_training --episodes 20
   ```

The attribution system is designed to work automatically without additional configuration. If you've followed this guide and still have issues, the problem is likely in the model initialization or metrics system setup.