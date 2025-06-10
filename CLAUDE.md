# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Development Commands
```bash
# Initial setup
poetry install
wandb login

# Momentum Training (RECOMMENDED - uses smart day selection and curriculum learning)
poetry run poe momentum                    # Start momentum training (symbols defined in curriculum stages)
poetry run poe momentum-continue           # Continue momentum training from best model
poetry run poe build-index                 # Build momentum day index (run once)

# Legacy Training (standard single-day training)
poetry run poe quick                       # Quick model testing
poetry run poe init                        # Train new model from scratch
poetry run poe train                       # Continue training from best model

# Override curriculum symbols via command line
poetry run python main.py --config curriculum_example --symbol AAPL
poetry run python main.py --config momentum_training --symbol TSLA --continue
poetry run poe momentum -- --symbol NVDA   # Pass symbol to poe task

# Create custom curriculum config with your symbols/stages
poetry run python main.py --config my_curriculum_config

# Run backtest
poetry run poe backtest                    # Run backtest (configure via curriculum or --symbol)
poetry run poe backtest-date               # Specify date: poetry run poe backtest-date 2025-04-15

# Scan for momentum days in data
poetry run poe scan                        # Min quality 0.5
poetry run poe scan-high                   # Min quality 0.7  
poetry run poe scan-all                    # All momentum days (0.0)

# Hyperparameter Optimization with Optuna
poetry run poe optuna-foundation           # Phase 1: Foundation optimization
poetry run poe optuna-reward               # Phase 2: Reward system optimization  
poetry run poe optuna-finetune             # Phase 3: Fine-tuning optimization
poetry run poe optuna-dashboard            # Launch Optuna dashboard

# Advanced usage (full features)
poetry run python optuna/optimization.py --spec optuna-1-foundation


```

### Testing & Validation
```bash
# Run all tests with verbose output
poetry run poe test

# Run tests with short traceback
poetry run poe test-fast

# Run specific test files
poetry run pytest tests/test_market_simulator.py -v
poetry run pytest tests/test_execution_simulator.py -v
poetry run pytest tests/test_portfolio_simulator.py -v
poetry run pytest tests/test_trading_environment.py -v
```

### Code Quality & Type Checking
When working on code, ensure to maintain code quality and type safety using the following commands:
```bash
# Type checking with pyright
poetry run pyright

# Linting and formatting with ruff
poetry run ruff check                      # Check for linting issues
poetry run ruff format                     # Auto-format code
poetry run ruff check --fix               # Auto-fix linting issues where possible

# Run both type checking and linting
poetry run pyright && poetry run ruff check
```

## Architecture Overview

FxAI is a reinforcement learning-based algorithmic trading system specializing in high-frequency momentum trading of low-float stocks. The system uses:

### Core Components

1. **PPO Agent** (`agent/ppo_agent.py`)
   - Proximal Policy Optimization for trading decisions
   - **Momentum-Based Training**: Automatically selects and switches between high-quality momentum days
   - **Curriculum Learning**: Progresses from easier to harder reset points based on performance
   - Continuous training capability with automatic model versioning (v1, v2, v3...)
   - Custom callbacks for training lifecycle management
   - Automatic learning rate annealing on continuation

2. **Multi-Branch Transformer** (`model/transformer.py`)
   - Processes features at different time scales:
     - High-frequency (1s) - 60-second window: price velocity, tape analysis, order flow
     - Medium-frequency (1m/5m) - Technical indicators, patterns
     - Low-frequency (daily) - Market context, support/resistance levels
     - Static - Market cap, time encodings
     - Portfolio - Position, P&L, risk metrics
   - ~100+ features extracted across timeframes

3. **Trading Environment** (`envs/trading_environment.py`)
   - Gymnasium-compatible environment
   - Discrete action space: (ActionType, PositionSize) tuples
   - Action enums: ActionTypeEnum (HOLD=0, BUY=1, SELL=2)
   - Position size enums: PositionSizeTypeEnum (SIZE_25=0, SIZE_50=1, SIZE_75=2, SIZE_100=3)
   - Realistic market simulation with slippage and fees
   - RewardSystemV2 with modular components

4. **Data Pipeline**
   - **DataManager** (`data/data_manager.py`) - Orchestrates data loading with intelligent caching
   - **MomentumScanner** (`data/scanner/momentum_scanner.py`) - Identifies high-quality momentum days and reset points
   - **DatabentFileProvider** - Handles Databento market data files (.dbn.zst format)
   - **MarketSimulator** - Constructs uniform 1-second timelines (4 AM - 8 PM ET)
   - Supports tick-level data: trades, quotes (L1), order book (MBP-1), OHLCV

5. **Feature Extraction** (`feature/`)
   - Organized by frequency: high-frequency (hf/), medium-frequency (mf/), low-frequency (lf/)
   - Specialized categories: pattern/, professional/, sequence_aware/, volume_analysis/
   - Feature attribution system in `feature/attribution/`
   - Simple feature manager coordinates feature extraction
   - Aggregates 1s bars into 1m/5m timeframes
   - Calculates technical indicators and market microstructure features
   - Tracks session-level statistics (VWAP, cumulative volume, etc.)
   - Handles previous day data for early session features

6. **Market Simulation** (`simulators/`)
   - **MarketSimulator** - Manages market state and rolling data windows
   - **ExecutionSimulator** - Simulates order execution with latency, slippage, market impact
   - **PortfolioSimulator** - Tracks positions, P&L, and risk metrics
   - Portfolio enums: OrderTypeEnum, OrderSideEnum, PositionSideEnum
   - Comprehensive test coverage for all simulators

7. **Reward System V2** (`rewards/`)
   - Modular design with 10+ specialized components
   - Anti-hacking measures: clipping, smoothing, exponential decay
   - Components: PnL, holding penalties, action efficiency, risk management
   - Individual component tracking for analysis

8. **Callback System** (`callbacks/`)
   - Core callbacks: base, checkpoint, context, factory, manager, metrics, wandb
   - Analysis callbacks: attribution analysis, performance monitoring
   - Optimization callbacks: early stopping, optuna integration
   - Comprehensive metric collection through callback system
   - Weights & Biases integration for experiment tracking
   - Old callback system maintained for backward compatibility

9. **Hyperparameter Optimization** (`optuna/`)
   - Advanced Optuna integration with multiple samplers (TPE, CMA-ES, etc.)
   - Configurable pruning strategies for efficient search
   - Parallel execution support for multi-GPU optimization
   - Study management with SQLite storage and result tracking
   - Automatic visualization generation (optimization history, parameter importance)
   - Predefined optimization configurations for different scenarios

### Configuration System

Uses Pydantic for configuration management with hierarchical organization:
- `config/config.py` - Main configuration with Pydantic models
- `config/model/model_config.py` - Multi-branch transformer architecture
- `config/training/training_config.py` - Continuous training with model versioning
- `config/environment/environment_config.py` - Trading environment settings
- `config/data/data_config.py` - Data source configurations
- `config/simulation/simulation_config.py` - Market simulation parameters
- `config/logging/logging_config.py` - Logging configuration
- `config/callbacks/callback_config.py` - Callback system configuration
- `config/rewards/reward_config.py` - Reward system settings
- `config/optuna/` - Hyperparameter optimization specifications:
  - `optuna_config.py` - Main Optuna configuration
  - `overrides/` - Phase-specific optimization configurations
- `config/overrides/` - Override configurations for different scenarios

### Key Features

1. **Momentum-Based Training**: Automatically selects and rotates through high-quality momentum days (quality scores 0.33-0.99)
2. **Curriculum Learning**: Progressive difficulty from easier to harder reset points based on performance
3. **Smart Day Selection**: Uses momentum indices with 21 available momentum days and 630 reset points
4. **Variable Reset Points**: Episodes start at different times (4 AM - 8 PM ET) based on momentum patterns, not fixed times
5. **Continuous Training**: Automatic model versioning, best model tracking, checkpoint syncing
6. **Multi-timeframe Analysis**: Processes data from tick-level to daily
7. **Realistic Simulation**: Bid-ask spreads, market impact, trading fees, latency
8. **Comprehensive Features**: 100+ features covering price action, volume, order flow, market structure
9. **Feature Monitoring**: Basic feature statistics tracking and gradient monitoring during training
10. **Live Dashboard**: Real-time training visualization with charts, metrics, and component analysis at http://localhost:8051
11. **Model Management**: Keeps top 5 models by reward with metadata tracking
12. **Test Suite**: Comprehensive testing for simulators and environment with pytest

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

## Software Design Principles
- Design for modularity and configuration: Build systems that accept definitions as input and work independently. Use configurable, loosely coupled components with clear interfaces, so the new functionality requires only new config definitions, not code changes.
 
## Important Notes

- The system focuses on momentum/squeeze trading strategies for low-float stocks
- **Feature Monitoring**: Basic feature statistics and gradient tracking for model performance monitoring
- Model checkpoints are saved in `cache/model/best/` with JSON metadata
- Databento data files are stored in `dnb/mlgo/` directory structure
- Logging configured through `core/logger.py` using Rich handler
- Invalid action handling is configurable with tracking and limits
- Use `TradingEnvironment` class from `envs.trading_environment` (not the old `TradingEnv`)
- Test suite available with pytest - run `poetry run poe test` for full test coverage
- Momentum day scanner available via `data/scanner/momentum_scanner.py`

## Memory

- We will use docs directory to keep track of features, metrics, rewards and keep implementation in sync with them