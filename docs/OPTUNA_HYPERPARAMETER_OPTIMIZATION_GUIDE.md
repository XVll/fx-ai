# Optuna Hyperparameter Optimization Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Understanding Configurations](#understanding-configurations)
4. [Running Optimization Studies](#running-optimization-studies)
5. [Monitoring Progress](#monitoring-progress)
6. [Analyzing Results](#analyzing-results)
7. [Interpreting Visualizations](#interpreting-visualizations)
8. [Best Practices](#best-practices)
9. [Advanced Usage](#advanced-usage)
10. [Troubleshooting](#troubleshooting)

## Introduction

Hyperparameter optimization is the process of finding the best configuration of parameters for your machine learning model. Instead of manually tuning parameters like learning rate, batch size, or model architecture, Optuna automates this process using intelligent search algorithms.

### Why Use Optuna?

- **Intelligent Search**: Uses advanced algorithms (TPE, CMA-ES) that learn from previous trials
- **Early Stopping**: Prunes unpromising trials to save computational resources
- **Parallel Execution**: Run multiple trials simultaneously on different GPUs
- **Visualization**: Rich plots to understand parameter relationships and optimization progress

### What Parameters Can Be Optimized?

In FxAI, you can optimize:
- **Training parameters**: learning rate, batch size, epochs, gamma, entropy coefficient
- **Model architecture**: layer count, model dimensions, dropout rates
- **Reward system**: coefficient values for different reward components
- **Environment settings**: episode lengths, thresholds, trading parameters

## Quick Start

### 1. Install Dependencies
```bash
poetry install  # Installs Optuna and related dependencies
```

### 2. Run 3-Phase Optimization
```bash
# Phase 1: Foundation optimization (2-4 hours)
poetry run poe optuna-foundation

# Transfer results and run Phase 2: Reward optimization (2-3 hours)  
poetry run poe optuna-transfer-1to2
poetry run poe optuna-reward

# Transfer results and run Phase 3: Fine-tuning (3-5 hours)
poetry run poe optuna-transfer-2to3
poetry run poe optuna-finetune
```

### 3. Monitor Progress
```bash
# Launch the Optuna dashboard in a separate terminal
poetry run poe optuna-dashboard
# Then open http://localhost:8052 in your browser

# Check status of all phases
poetry run poe optuna-status
```

### 4. View Results
```bash
# Show best parameters from each phase
poetry run poe optuna-best fx_ai_foundation
poetry run poe optuna-best fx_ai_reward  
poetry run poe optuna-best fx_ai_finetune

# Show comprehensive results
poetry run poe optuna-results
```

After optimization completes, check:
- Terminal output for best parameters found
- `optuna_results/` directory for detailed results and visualizations
- Optuna dashboard for interactive analysis

## Understanding Configurations

### Available Optimization Presets

#### 1. Quick Search (`quick_search.yaml`)
- **Purpose**: Fast optimization for beginners
- **Trials**: 50
- **Parameters**: 10 key parameters (learning rate, batch size, model size, key rewards)
- **Time**: ~2-4 hours
- **Use when**: First time optimizing, limited computational budget

#### 2. Comprehensive Search (`comprehensive_search.yaml`)
- **Purpose**: Full parameter space exploration
- **Trials**: 200
- **Parameters**: 25+ parameters covering all aspects
- **Time**: ~8-16 hours
- **Use when**: You have computational resources and want thorough optimization

#### 3. Reward Focused (`reward_focused.yaml`)
- **Purpose**: Optimize only reward system coefficients
- **Trials**: 100
- **Parameters**: 16 reward-related parameters
- **Time**: ~4-8 hours
- **Use when**: Your model architecture is good but rewards need tuning

#### 4. Parallel Search (`parallel_search.yaml`)
- **Purpose**: Multi-GPU parallel optimization
- **Trials**: 200 (8 parallel workers)
- **Parameters**: Balanced parameter set
- **Time**: ~2-4 hours with 8 GPUs
- **Use when**: You have multiple GPUs available

### Configuration Structure

Each optimization config contains:

```yaml
name: study_name
description: What this optimization does
studies:
  - study_name: unique_identifier
    direction: maximize  # or minimize
    metric_name: mean_reward  # what to optimize
    
    sampler:
      type: TPESampler  # optimization algorithm
      n_startup_trials: 10  # random trials before intelligent search
    
    pruner:
      type: MedianPruner  # early stopping strategy
      n_startup_trials: 5  # trials before pruning starts
    
    parameters:
      - name: training.learning_rate
        type: float_log  # logarithmic scale
        low: 0.00001
        high: 0.001
      
      - name: model.d_model
        type: categorical
        choices: [64, 128, 256]
    
    n_trials: 50
    training_config:
      # Base configuration for all trials
```

## Running Optimization Studies

### Basic Commands

```bash
# Run default optimization
poetry run poe optuna

# Run specific configuration
poetry run python optuna_optimization.py --spec config/optuna/quick_search.yaml

# Run with parallel workers (if you have multiple GPUs)
poetry run poe optuna-parallel 4

# Run specific study from a multi-study config
poetry run poe optuna-study fx_ai_comprehensive
```

### Command Line Options

```bash
# Show best configuration from completed study
poetry run poe optuna-best study_name

# Launch interactive dashboard
poetry run poe optuna-dashboard

# Run with custom number of parallel jobs
poetry run python optuna_optimization.py --n-jobs 8 --spec config/optuna/parallel_search.yaml
```

### What Happens During Optimization

1. **Study Creation**: Optuna creates a study with your specified configuration
2. **Trial Execution**: For each trial:
   - Optuna suggests parameter values based on previous results
   - A training run is executed with these parameters
   - Performance metrics are collected
   - Results are stored in the database
3. **Pruning**: Poor-performing trials are stopped early to save time
4. **Analysis**: Best parameters and visualizations are generated

## Monitoring Progress

### Terminal Output

During optimization, you'll see:

```
Trial 15
┌─ Trial Configuration ─┐
│ Parameters:            │
│   training.learning_rate: 0.0002341  │
│   training.batch_size: 128           │
│   model.d_model: 256                 │
│   env.reward.pnl_coefficient: 142.3  │
└────────────────────────┘

[Progress] ████████████████████░░░░ 75% | 38/50 trials | ETA: 1h 23m

Current best value: 15.42
```

Key indicators:
- **Trial number**: Current trial being executed
- **Parameters**: Values being tested in this trial
- **Progress bar**: Completion percentage and estimated time remaining
- **Current best**: Best metric value found so far

### Optuna Dashboard

Launch with `poetry run poe optuna-dashboard` and open http://localhost:8052

The dashboard shows:
- **Real-time progress**: Live updates of running trials
- **Optimization history**: How the best value improves over time
- **Parameter relationships**: Which parameters matter most
- **Trial details**: Individual trial results and logs

### Weights & Biases Integration

Each trial is automatically tracked in W&B:
- Project: `fx-ai-optuna`
- Tags: `[optuna, hyperparameter_search]`
- Metrics: All training metrics from each trial

## Analyzing Results

### Best Parameters

After optimization completes, you'll see:

```
Study Summary: fx_ai_quick
┌──────────────────┬──────────┐
│ Metric           │ Value    │
├──────────────────┼──────────┤
│ Best value       │ 18.45    │
│ Best trial       │ 42       │
│ Total trials     │ 50       │
│ Completed trials │ 47       │
│ Pruned trials    │ 3        │
│ Failed trials    │ 0        │
└──────────────────┴──────────┘

Best parameters:
  training.learning_rate: 0.0001523
  training.batch_size: 64
  training.n_epochs: 8
  model.d_model: 128
  env.reward.pnl_coefficient: 156.7
```

### Result Files

Results are saved in `optuna_results/[study_name]/`:

- **`best_params.json`**: Best parameters and their values
- **`[study_name]_results.json`**: All trial results with timestamps
- **`[study_name]_best_config.yaml`**: Complete configuration file for best trial
- **Visualization files**: HTML plots for analysis

### Key Metrics to Look For

1. **Best Value**: The highest (or lowest) metric achieved
2. **Convergence**: How quickly the optimization found good values
3. **Stability**: Whether similar parameter sets give consistent results
4. **Completion Rate**: Percentage of trials that completed vs. were pruned

### Success Indicators

- **Improvement over time**: Best value should generally increase with more trials
- **Parameter stability**: Best parameters shouldn't change drastically in final trials
- **Low pruning rate**: <30% pruned trials indicates good parameter ranges
- **Reasonable values**: Parameters should make intuitive sense

## Interpreting Visualizations

### 1. Optimization History (`optimization_history.html`)

**What it shows**: Best value found over time

**How to interpret**:
- **Steep initial climb**: Good - optimization is finding better parameters quickly
- **Plateauing**: Normal - optimization is converging to optimal values
- **Continued improvement**: More trials might yield better results
- **No improvement**: Parameter ranges might be too narrow or model is at optimal

**Example interpretation**:
```
If the line shows rapid improvement for first 20 trials, then plateaus around trial 30:
✓ Good: Quick convergence suggests well-chosen parameter ranges
✓ Good: Plateau suggests optimization found optimal region
? Consider: Running more trials to confirm convergence
```

### 2. Parameter Importances (`param_importances.html`)

**What it shows**: Which parameters most affect the objective

**How to interpret**:
- **High importance (>0.1)**: Critical parameters to focus on
- **Medium importance (0.05-0.1)**: Moderately important
- **Low importance (<0.05)**: May not need fine-tuning

**Example interpretation**:
```
learning_rate: 0.35    ← Most important parameter
batch_size: 0.28       ← Very important
pnl_coefficient: 0.15  ← Moderately important
dropout: 0.03          ← Less critical
```

**Action items**:
- Focus manual tuning on high-importance parameters
- Consider fixing low-importance parameters to reduce search space
- High learning_rate importance suggests sensitive optimization landscape

### 3. Parallel Coordinate Plot (`parallel_coordinate.html`)

**What it shows**: Relationship between all parameters and the objective

**How to interpret**:
- **Color coding**: Darker lines = better performance
- **Clustering**: Groups of similar parameter combinations
- **Trends**: How parameter values relate to performance

**Example interpretation**:
```
If you see:
- Dark lines (good trials) cluster around learning_rate=0.0002
- Dark lines spread across different batch_sizes
- Most dark lines have pnl_coefficient > 100

Conclusion:
✓ Learning rate is critical - keep around 0.0002
✓ Batch size is flexible - any reasonable value works
✓ PnL coefficient should be high
```

### 4. Slice Plot (`slice_plot.html`)

**What it shows**: Individual parameter effects on the objective

**How to interpret**:
- **Clear trends**: Parameter has strong effect
- **Scattered points**: Weak effect or interactions with other parameters
- **Optimal ranges**: Where most good trials cluster

**Example interpretation**:
```
learning_rate plot shows:
- Poor performance below 0.0001
- Best performance around 0.0002
- Declining performance above 0.0005

Action: Set learning_rate between 0.0001-0.0003 for future studies
```

### 5. Contour Plot (`contour_plot.html`)

**What it shows**: Interaction between two parameters

**How to interpret**:
- **Dark regions**: Good parameter combinations
- **Light regions**: Poor parameter combinations
- **Patterns**: How parameters interact

**Example interpretation**:
```
learning_rate vs batch_size contour shows:
- Dark region at (lr=0.0002, batch=64)
- Dark region at (lr=0.0001, batch=128)
- Light everywhere else

Conclusion: These parameters interact - low LR needs large batch, high LR needs small batch
```

## Best Practices

### 1. Start Small, Scale Up

```bash
# First run: Quick search to get baseline
poetry run python optuna_optimization.py --spec config/optuna/quick_search.yaml

# Then: Focus on important parameters found
# Edit reward_focused.yaml to include only high-importance parameters

# Finally: Comprehensive search with refined ranges
poetry run python optuna_optimization.py --spec config/optuna/comprehensive_search.yaml
```

### 2. Set Appropriate Parameter Ranges

**Too narrow**: Optimization might miss optimal values
```yaml
# BAD: Too narrow
learning_rate:
  low: 0.0001
  high: 0.0002
```

**Too wide**: Wastes trials on obviously bad values
```yaml
# BAD: Too wide
learning_rate:
  low: 0.000001
  high: 0.1
```

**Just right**: Covers reasonable range based on domain knowledge
```yaml
# GOOD: Reasonable range
learning_rate:
  low: 0.00005
  high: 0.001
```

### 3. Use Appropriate Distributions

- **Log scale for learning rates**: `type: float_log`
- **Categorical for discrete choices**: `type: categorical`
- **Linear for bounded continuous**: `type: float`

### 4. Monitor Resource Usage

```bash
# Check GPU memory usage during parallel runs
nvidia-smi

# Check disk space for result files
du -sh optuna_results/

# Monitor database size
ls -lh optuna_studies.db
```

### 5. Validate Results

After optimization:

1. **Re-run best configuration** to confirm results
2. **Compare to baseline** - ensure improvement over default parameters
3. **Test on different data** - verify generalization
4. **Check for overfitting** - ensure performance holds on validation set

## Advanced Usage

### Creating Custom Optimization Configs

1. **Copy existing config**:
```bash
cp config/optuna/quick_search.yaml config/optuna/my_custom_search.yaml
```

2. **Modify parameters**:
```yaml
parameters:
  # Add your custom parameters
  - name: env.max_episode_steps
    type: categorical
    choices: [256, 512, 1024]
  
  # Modify existing ranges
  - name: training.learning_rate
    type: float_log
    low: 0.0001  # Narrower range based on previous results
    high: 0.0005
```

3. **Run custom config**:
```bash
poetry run python optuna_optimization.py --spec config/optuna/my_custom_search.yaml
```

### Multi-Objective Optimization

For optimizing multiple metrics simultaneously:

```yaml
# In your config
direction: maximize  # This becomes the primary objective
# Additional objectives can be tracked in the trial results
```

### Resuming Interrupted Studies

Optuna automatically resumes from where it left off:

```bash
# If optimization was interrupted, just run again with same config
poetry run python optuna_optimization.py --spec config/optuna/quick_search.yaml
# It will continue from the last completed trial
```

### Comparing Multiple Studies

```python
# In the Optuna dashboard, you can compare different studies
# Or use the programmatic interface:

import optuna

study1 = optuna.load_study(study_name="fx_ai_quick", storage="sqlite:///optuna_studies.db")
study2 = optuna.load_study(study_name="fx_ai_comprehensive", storage="sqlite:///optuna_studies.db")

print(f"Quick search best: {study1.best_value}")
print(f"Comprehensive best: {study2.best_value}")
```

## Troubleshooting

### Common Issues and Solutions

#### 1. "Study not found" Error
```bash
Error: Study 'fx_ai_quick' not found
```
**Solution**: Check available studies in dashboard or create new one

#### 2. Training Fails in Trials
```bash
Trial 5 failed: CUDA out of memory
```
**Solutions**:
- Reduce batch size range in config
- Use fewer parallel workers
- Add memory cleanup between trials

#### 3. No Improvement in Optimization
```bash
Best value stuck at same level for many trials
```
**Solutions**:
- Check parameter ranges aren't too narrow
- Increase number of startup trials
- Switch to different sampler (CMA-ES instead of TPE)

#### 4. Slow Optimization
```bash
Each trial takes too long
```
**Solutions**:
- Reduce training episodes per trial
- Use more aggressive pruning
- Enable parallel execution

#### 5. Database Lock Errors
```bash
database is locked
```
**Solutions**:
- Make sure no other optimization is running
- Delete optuna_studies.db and restart
- Use different storage location

### Performance Optimization Tips

1. **Reduce trial duration**:
```yaml
training_config:
  training:
    total_updates: 50  # Reduce from default 100
    eval_frequency: 10  # Evaluate more often for better pruning
```

2. **Aggressive pruning**:
```yaml
pruner:
  type: SuccessiveHalvingPruner
  min_resource: 5
  reduction_factor: 2  # More aggressive than default 4
```

3. **Parallel execution**:
```bash
# Use all available GPUs
poetry run python optuna_optimization.py --n-jobs 8 --spec config/optuna/parallel_search.yaml
```

### Getting Help

If you encounter issues:

1. **Check the terminal output** for error messages
2. **Look at trial logs** in W&B for failed trials
3. **Examine the database** using Optuna dashboard
4. **Review configuration** for syntax errors
5. **Check resource usage** (GPU memory, disk space)

### Expected Results

After a successful optimization run, you should see:

- **10-30% improvement** in mean reward compared to default parameters
- **Stable best parameters** that don't change drastically in final trials
- **Clear parameter importance rankings** showing which factors matter most
- **Converged optimization** where additional trials don't improve results significantly

Remember: Hyperparameter optimization is an iterative process. Use the insights from each study to design better subsequent studies!