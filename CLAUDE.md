# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Development Commands
```bash
# Initial setup
poetry install
wandb login

# Quick model testing
poetry run poe quick

# Train new model from scratch
poetry run poe init

# Continue training from best model
poetry run poe train

# Run backtest
poetry run poe backtest

# Run with custom parameters
poetry run python scripts/run.py train --symbol MLGO --continue-training
poetry run python scripts/run.py backtest --symbol MLGO --start-date 2025-03-27 --end-date 2025-03-27

# Hyperparameter optimization
poetry run python scripts/sweep.py --config default.yaml --count 20
```

### Testing & Validation
```bash
# No test commands defined yet - tests directory exists but empty
```

## Architecture Overview

FxAIv2 is a reinforcement learning-based algorithmic trading system specializing in high-frequency momentum trading of low-float stocks (MLGO). The system uses:

### Core Components

1. **PPO Agent** (`agent/ppo_agent.py`)
   - Proximal Policy Optimization for trading decisions
   - Continuous training capability with model checkpointing
   - Custom callbacks for training lifecycle management

2. **Multi-Branch Transformer** (`ai/transformer.py`)
   - Processes features at different time scales:
     - High-frequency (1s) - price velocity, tape analysis, order flow
     - Medium-frequency (1m/5m) - technical indicators, candle patterns
     - Low-frequency (daily) - market context, support/resistance levels
   - ~100+ features extracted across timeframes

3. **Trading Environment** (`envs/trading_env.py`)
   - Gymnasium-compatible environment
   - Realistic market simulation with slippage and fees
   - Reward functions optimized for momentum trading

4. **Data Pipeline**
   - **DataManager** (`data/data_manager.py`) - Orchestrates data loading and preprocessing
   - **DatabentFileProvider** (`data/provider/data_bento/databento_file_provider.py`) - Handles Databento market data files
   - Supports tick-level data: trades, quotes (L1), order book (MBP-1)

5. **Feature Extraction** (`feature/feature_extractor.py`)
   - Aggregates 1s bars into 1m/5m timeframes
   - Calculates technical indicators and market microstructure features
   - Tracks session-level statistics (VWAP, cumulative volume, etc.)

6. **Market Simulation** (`simulators/`)
   - **MarketSimulator** - Manages market state and data windows
   - **ExecutionSimulator** - Simulates order execution with realistic slippage
   - **PortfolioSimulator** - Tracks positions, P&L, and risk metrics

7. **Metrics & Monitoring** (`metrics/`)
   - Comprehensive metric collection system
   - Weights & Biases integration for experiment tracking
   - Real-time performance monitoring during training

### Configuration System

Uses Hydra for hierarchical configuration management:
- `config/config.yaml` - Main configuration orchestrator
- `config/model/` - Model architecture settings
- `config/training/` - Training parameters (PPO, continuous)
- `config/data/` - Data source configurations
- `config/simulation/` - Trading simulation parameters
- `config/wandb/` - Experiment tracking settings

### Key Features

1. **Continuous Training**: Load best models and continue training with new data
2. **Multi-timeframe Analysis**: Processes data from tick-level to daily
3. **Realistic Simulation**: Includes bid-ask spreads, market impact, and trading fees
4. **Comprehensive Features**: Extracts 100+ features covering price action, volume, order flow, and market structure
5. **Experiment Tracking**: Full integration with Weights & Biases for monitoring

### Workflow

1. Data flows from Databento files → DataManager → MarketSimulator
2. MarketSimulator maintains rolling windows of market data
3. FeatureExtractor processes raw data into model features
4. PPO Agent receives features and outputs trading actions
5. ExecutionSimulator executes trades with realistic constraints
6. Metrics are collected and sent to W&B for analysis

## Important Notes

- The system focuses on momentum/squeeze trading strategies for low-float stocks
- Feature extraction is critical - the skeleton in `notes.md` shows planned v2 architecture
- Model checkpoints are saved in `best_models/MLGO/` with metadata
- Databento data files are stored in `dnb/Mlgo/` directory
- Logging is configured through `utils/logger.py` using Rich handler