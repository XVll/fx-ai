# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Development Commands
```bash
# Initial setup
poetry install
wandb login

# Quick model testing (reduced steps for architecture validation)
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

# Live dashboard (auto-launches with training)
# Access at http://localhost:8050
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
   - Continuous training capability with automatic model versioning (v1, v2, v3...)
   - Custom callbacks for training lifecycle management
   - Automatic learning rate annealing on continuation

2. **Multi-Branch Transformer** (`ai/transformer.py`)
   - Processes features at different time scales:
     - High-frequency (1s) - 60-second window: price velocity, tape analysis, order flow
     - Medium-frequency (1m/5m) - Technical indicators, patterns
     - Low-frequency (daily) - Market context, support/resistance levels
     - Static - Market cap, time encodings
     - Portfolio - Position, P&L, risk metrics
   - ~100+ features extracted across timeframes

3. **Trading Environment** (`envs/trading_env.py`)
   - Gymnasium-compatible environment
   - Discrete action space: (ActionType, PositionSize) tuples
   - Realistic market simulation with slippage and fees
   - RewardSystemV2 with modular components

4. **Data Pipeline**
   - **DataManager** (`data/data_manager.py`) - Orchestrates data loading with intelligent caching
   - **DatabentFileProvider** - Handles Databento market data files (.dbn.zst format)
   - **MarketSimulator** - Constructs uniform 1-second timelines (4 AM - 8 PM ET)
   - Supports tick-level data: trades, quotes (L1), order book (MBP-1), OHLCV

5. **Feature Extraction** (`feature/feature_extractor.py`)
   - Aggregates 1s bars into 1m/5m timeframes
   - Calculates technical indicators and market microstructure features
   - Tracks session-level statistics (VWAP, cumulative volume, etc.)
   - Handles previous day data for early session features

6. **Market Simulation** (`simulators/`)
   - **MarketSimulator** - Manages market state and rolling data windows
   - **ExecutionSimulator** - Simulates order execution with latency, slippage, market impact
   - **PortfolioSimulator** - Tracks positions, P&L, and risk metrics

7. **Reward System V2** (`rewards/`)
   - Modular design with 10+ specialized components
   - Anti-hacking measures: clipping, smoothing, exponential decay
   - Components: PnL, holding penalties, action efficiency, risk management
   - Individual component tracking for analysis

8. **Metrics & Monitoring** (`metrics/`)
   - Comprehensive metric collection system with factory pattern
   - Weights & Biases integration for experiment tracking
   - Live dashboard with real-time visualization
   - Collectors: training, trading, model, execution, reward metrics

### Configuration System

Uses Hydra for hierarchical configuration management:
- `config/config.yaml` - Main configuration with defaults composition
- `config/model/transformer.yaml` - Multi-branch transformer architecture
- `config/training/continuous.yaml` - Continuous training with model versioning
- `config/training/ppo.yaml` - PPO hyperparameters
- `config/env/trading.yaml` - Trading environment and reward system settings
- `config/data/databento.yaml` - Data source configurations
- `config/simulation/default.yaml` - Market simulation parameters
- `config/wandb/` - Experiment tracking settings

### Key Features

1. **Continuous Training**: Automatic model versioning, best model tracking, checkpoint syncing
2. **Multi-timeframe Analysis**: Processes data from tick-level to daily
3. **Realistic Simulation**: Bid-ask spreads, market impact, trading fees, latency
4. **Comprehensive Features**: 100+ features covering price action, volume, order flow, market structure
5. **Live Dashboard**: Real-time training visualization with charts, metrics, and component analysis
6. **Model Management**: Keeps top 5 models by reward with metadata tracking

### Data Flow

```
Databento Files → DataManager → MarketSimulator → FeatureExtractor → PPO Agent
                      ↓              ↓                                    ↓
                   Caching      Uniform Timeline                     Actions
                                     ↓                                    ↓
                              ExecutionSimulator ← PortfolioManager ← TradingEnv
                                                                          ↓
                                                                   RewardSystemV2
                                                                          ↓
                                                                  Metrics → W&B/Dashboard
```

### Important Implementation Details

1. **Action Space**: Discrete (ActionType × PositionSize) = 12 actions total
   - Prevents partial fills and simplifies execution
   - Position sizes: 25%, 50%, 75%, 100% of available capital

2. **Episode Management**:
   - Smart termination: bankruptcy, max loss, data end, invalid actions
   - Progress tracking with real-time updates
   - Episode history maintained for analysis

3. **Market Hours**: 4 AM - 8 PM ET with uniform 1-second intervals
   - Previous day data loaded for early session features
   - Efficient DataFrame-based caching with time range tracking

4. **Dashboard Integration**:
   - Auto-launches on port 8050 during training
   - Real-time updates via queue-based communication
   - Comprehensive visualization of all metrics and components

## Important Notes

- The system focuses on momentum/squeeze trading strategies for low-float stocks
- Feature extraction is critical - see README.md for planned v2 feature architecture
- Model checkpoints are saved in `best_models/MLGO/` with JSON metadata
- Databento data files are stored in `dnb/Mlgo/` directory structure
- Logging configured through `utils/logger.py` using Rich handler
- Invalid action handling is configurable with tracking and limits
- Design for modularity and configuration: Build systems that accept definitions as input and work independently. Use configurable, loosely-coupled components with clear interfaces so new functionality requires only new config definitions, not code changes.