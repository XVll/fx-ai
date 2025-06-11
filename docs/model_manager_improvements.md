# Model Manager Improvements

## Overview
The ModelManager has been simplified and improved with better error handling and configuration-based directory management, while avoiding over-complicated typing.

## Key Improvements

### 1. Simple Configuration (`config/model/model_storage_config.py`)
- All directory paths are configurable (no hardcoded paths)
- Simple model retention policies
- Basic metadata storage options
- Clean and minimal configuration with just essential options

### 2. Simplified ModelManager (`core/model_manager.py`)
- **Better Error Handling**:
  - Custom exceptions: `ModelManagerError`, `ModelNotFoundError`
  - Proper error propagation with context
  - Graceful fallbacks for missing metadata

- **Improved Model Management**:
  - Automatic cleanup of old models (keeps top N)
  - Version management with configurable format
  - Support for both checkpoint and best model storage
  - Simple dictionary-based metadata tracking
  - No complex typing - uses simple dictionaries and basic types

### 3. Integration Updates
- Updated `main.py` to pass ModelStorageConfig to ModelManager
- Updated `training_manager.py` to use simple dictionary returns
- Updated `benchmark_runner.py` to accept ModelManager instance
- Added ModelStorageConfig to main configuration system

## Usage Example

```python
from config.model.model_storage_config import ModelStorageConfig
from core.model_manager import ModelManager

# Create simple configuration
config = ModelStorageConfig(
    checkpoint_dir="checkpoints",
    best_models_dir="best_models", 
    max_best_models=5,
    save_metadata=True
)

# Initialize manager with config and base directory
model_manager = ModelManager(
    config=config,
    base_dir=Path("outputs/2024-01-01")
)

# Save checkpoint
checkpoint_path = model_manager.save_checkpoint(
    model, optimizer, 
    global_step_counter=10000,
    global_episode_counter=500,
    global_update_counter=100,
    global_cycle_counter=2,
    metadata={"symbol": "AAPL", "day_quality": 0.8}
)

# Save as best model
best_path = model_manager.save_best_model(
    checkpoint_path,
    metrics={"sharpe": 1.5, "win_rate": 0.65},
    target_reward=0.95
)

# Load model
model, training_state = model_manager.load_model(model, optimizer)
print(f"Loaded model: step={training_state['global_step']}, reward={training_state['metadata'].get('reward', 0)}")
```

## Benefits

1. **Simplicity**: Uses simple dictionaries and basic types - no complex typing
2. **Error Resilience**: Better error handling prevents crashes and data corruption
3. **Configurability**: All paths and behaviors can be configured via YAML/Hydra
4. **Maintainability**: Clear and simple interfaces that are easy to understand
5. **Observability**: Metadata and logging for debugging and monitoring
6. **Backward Compatibility**: Maintains compatibility with existing code

## Testing
Comprehensive test suite added in `tests/core/test_model_manager.py` covering:
- Directory creation and management
- Model saving and loading
- Version management
- Simple metadata handling
- Error handling
- Model cleanup
- Backward compatibility