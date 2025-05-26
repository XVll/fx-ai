# Poe Tasks Reference

This project uses `poethepoet` (poe) for task automation. Install dependencies and poe with:

```bash
poetry install
```

## Available Tasks

### Training Tasks

```bash
# Initialize new model training from scratch
poe init

# Continue training from best model
poe train

# Quick test run (reduced steps for testing)
poe quick

# Train with specific date range
poe train-date  # trains Jan 1-31, 2025
# Edit pyproject.toml to change dates

# Train with specific number of days
poe train-days  # trains last 5 days
# Edit pyproject.toml to change days
```

### Backtesting Tasks

```bash
# Backtest single day
poe backtest       # Jan 15, 2025

# Backtest one week  
poe backtest-week  # Jan 13-17, 2025

# Backtest full month
poe backtest-month # Jan 1-31, 2025
```

### Hyperparameter Optimization

```bash
# Standard sweep (20 runs)
poe sweep

# Quick sweep (5 runs)
poe sweep-quick

# Extended sweep (50 runs)
poe sweep-long
```

### Testing & Utilities

```bash
# Run tests
poe test       # verbose mode
poe test-fast  # quick mode

# Setup
poe setup      # wandb login

# Cleanup
poe clean      # remove outputs, wandb logs, cache

# Dashboard info
poe dashboard  # shows dashboard URL
```

## Custom Usage

For custom parameters, use the scripts directly:

```bash
# Custom training
poetry run python scripts/run.py train --symbol AAPL --start-date 2024-12-01 --end-date 2024-12-31

# Custom backtest with specific model
poetry run python scripts/run.py backtest --symbol MLGO --model-path ./best_models/MLGO/model_v5.pt --start-date 2025-01-20 --end-date 2025-01-20

# Custom sweep
poetry run python scripts/sweep.py --config my_sweep.yaml --count 30 --project my-project
```

## Dashboard

The dashboard automatically launches during training at http://localhost:8050

To view from WSL, just open this URL in any Windows browser.

## Notes

- Default symbol is MLGO (configured in pyproject.toml)
- Training automatically continues from best model when using `poe train`
- Sweep uses Bayesian optimization by default
- All outputs go to `outputs/` directory
- Model checkpoints saved to `best_models/`