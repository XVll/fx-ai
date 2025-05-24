# Understanding Output Directories

This document explains the various output directories created by W&B, Hydra, and the FX-AI system.

## Directory Structure

### 1. **wandb/** (Weights & Biases)
- **Purpose**: Tracks experiments, metrics, and model performance
- **Keep in Git?**: ❌ NO - Add to .gitignore
- **What's inside**:
  - `wandb/run-*/` - Individual run data
  - `wandb/latest-run` - Symlink to most recent run
  - `wandb/debug*.log` - Debug logs
- **Important**: Your metrics are synced to the W&B cloud, so local files are just cache

### 2. **outputs/** (Hydra)
- **Purpose**: Stores run configurations, logs, and outputs for each execution
- **Keep in Git?**: ❌ NO - Add to .gitignore
- **What's inside**:
  - `outputs/YYYY-MM-DD/HH-MM-SS/` - Timestamped run directories
  - `config_used.yaml` - Exact configuration used for that run
  - `main.log` - Execution logs
  - `models/` - Model checkpoints (if saved here)
- **Use case**: Debugging specific runs, checking exact configs used

### 3. **best_models/** (Your Models)
- **Purpose**: Stores your best performing model checkpoints
- **Keep in Git?**: ✅ YES (selectively)
- **What's inside**:
  - `*.pt` - PyTorch model files
  - `*_meta.json` - Model metadata (performance, config)
- **Strategy**: Keep only significant milestone models, not every checkpoint

### 4. **logs/** (Application Logs)
- **Purpose**: Detailed application logs
- **Keep in Git?**: ❌ NO - Add to .gitignore
- **What's inside**: Various log files from different components

### 5. **multirun/** (Hydra Sweeps)
- **Purpose**: Output from hyperparameter sweeps
- **Keep in Git?**: ❌ NO - Add to .gitignore
- **What's inside**: Results from multiple runs with different parameters

## Best Practices

### What to Keep in Git:
1. **Configuration files** (`config/*.yaml`)
2. **Best model metadata** (`best_models/**/*.json`)
3. **Selected milestone models** (manually chosen)
4. **Documentation** about experiments

### What NOT to Keep in Git:
1. **W&B run data** (`wandb/`)
2. **Hydra outputs** (`outputs/`, `multirun/`)
3. **Log files** (`*.log`)
4. **Temporary model checkpoints**
5. **Cache files** (`__pycache__/`, `.cache/`)

### Why This Approach?

1. **W&B Cloud Storage**: Your experiment data is already stored in W&B cloud
2. **Reproducibility**: Hydra configs let you reproduce any run
3. **Storage Efficiency**: Model files are large; only keep important ones
4. **Clean Repository**: Keeps your Git history manageable

### Accessing Historical Data

- **W&B Dashboard**: View all metrics, charts, and artifacts at wandb.ai
- **Hydra Configs**: Each run's exact configuration is saved
- **Model Registry**: Use W&B artifacts or model registry for version control

### Local Development Tips

```bash
# Clean up old outputs (be careful!)
rm -rf outputs/2025-05-*/

# Archive important runs before cleaning
tar -czf important_runs.tar.gz outputs/2025-05-24/

# Check what's taking space
du -sh wandb/* outputs/* | sort -h
```

## Summary

- **wandb/** and **outputs/** are temporary/cache directories
- Your important data is in W&B cloud and **best_models/**
- The .gitignore file is configured to handle this automatically
- Focus on tracking configurations and significant model milestones in Git