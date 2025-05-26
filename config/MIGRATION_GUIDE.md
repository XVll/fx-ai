# Configuration System Migration Guide

## Overview

We've migrated from a dual YAML + dataclass system to a unified Pydantic-based configuration with minimal YAML overrides.

## Key Changes

### 1. **Single Source of Truth**
- All config structure and defaults are in `config/schemas.py`
- YAML files only contain overrides for experiments
- No more duplicate definitions

### 2. **Removed Redundant Configs**
- Removed unused `render_interval`, `log_reward_components`, etc.
- Consolidated reward system (removed v1, only v2 remains)
- Fixed action_dim inconsistency (now [3, 4] everywhere)

### 3. **Type Safety & Validation**
- Full type hints with IDE support
- Runtime validation with helpful error messages
- `extra="forbid"` prevents typos and unknown fields

### 4. **Simplified Structure**
```
config/
├── schemas.py          # All config definitions (THE source)
├── loader.py           # Config loading utilities
├── overrides/          # Minimal YAML overrides
│   ├── quick_test.yaml
│   ├── production.yaml
│   └── experiment_*.yaml
└── MIGRATION_GUIDE.md  # This file
```

## Usage Examples

### Basic Usage
```python
from config.loader import load_config

# Load with defaults
config = load_config()

# Load with overrides
config = load_config("quick_test")  # Loads config/overrides/quick_test.yaml
config = load_config("experiment_low_risk.yaml")
```

### Accessing Config Values
```python
# Direct access (type-safe)
hidden_dim = config.model.d_model
reward_scale = config.env.reward_v2.scale_factor

# With usage tracking (for detecting unused configs)
from config.loader import get_config_value
batch_size = get_config_value("training.batch_size")
```

### Creating Override Files
Only specify what you want to change:
```yaml
# config/overrides/my_experiment.yaml
model:
  d_model: 1024
  
training:
  learning_rate: 0.0001
```

## Benefits

1. **Type Safety**: Full IDE autocomplete and type checking
2. **Validation**: Catches errors at startup, not runtime
3. **Single Source**: No more searching multiple files
4. **Minimal Overrides**: Experiments only specify changes
5. **Usage Tracking**: Detect unused configs automatically

## Migration Steps

1. Replace imports:
   ```python
   # Old
   from config.config import Config
   from omegaconf import DictConfig
   
   # New
   from config.loader import load_config
   ```

2. Update main functions:
   ```python
   # Old
   @hydra.main(config_path="config", config_name="config")
   def main(cfg: DictConfig):
       ...
   
   # New
   def main():
       config = load_config(args.simulation_config)
       ...
   ```

3. Access config values:
   ```python
   # Old
   cfg.model.hidden_dim  # No type hints
   
   # New
   config.model.d_model  # Full type safety
   ```

## Common Patterns

### Experiment Configs
```yaml
# Only override what changes
training:
  total_updates: 50
  learning_rate: 0.001
```

### Sweep Configs
```python
# Define sweeps in Python with type safety
sweep_config = {
    "model.d_model": [256, 512, 1024],
    "training.learning_rate": [1e-4, 3e-4, 1e-3],
}
```

### Production vs Development
```yaml
# config/overrides/production.yaml
model:
  dropout: 0.2  # More regularization
wandb:
  tags: ["production"]
  
# config/overrides/development.yaml  
training:
  total_updates: 10  # Quick iterations
logging:
  level: "DEBUG"
```

## Debugging

### Check for Unused Configs
```python
from config.loader import check_unused_configs
# At end of training
check_unused_configs()  # Logs warnings about unused parameters
```

### Validate Config
```python
# Config validation happens automatically on load
try:
    config = load_config("my_override.yaml")
except ValidationError as e:
    print(f"Config error: {e}")
```

### Save Used Config
```python
# Automatically saved to outputs/configs/
# Filename: config_<experiment>_<timestamp>.yaml
config.save_used_config("path/to/save.yaml")  # Manual save
```