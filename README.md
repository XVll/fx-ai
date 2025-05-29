# FxAIv2 - Professional Momentum Trading System

*Reinforcement Learning-based algorithmic trading system for squeeze/breakout patterns*

## =Ú **Complete Documentation**

**=I [COMPLETE FEATURE & MODEL GUIDE](docs/COMPLETE_FEATURE_AND_MODEL_GUIDE.md)** - Start here for comprehensive system documentation

## =€ Quick Start

### Development Commands
```bash
# Setup
poetry install
wandb login

# Training
poetry run poe quick     # Quick validation
poetry run poe train     # Full training

# Testing  
poetry run poe test      # Run all tests
```

### Key Features
- **Professional Implementation**: pandas + ta library for robust calculations
- **4-Branch Transformer**: HF, MF, LF, Portfolio with attention fusion
- **74 High-Quality Features**: Optimized for momentum/squeeze trading
- **Sequence-Aware**: Features utilize full temporal windows efficiently

## =Ê Architecture Overview

```
Market Data ’ Feature Extraction ’ Multi-Branch Transformer ’ Trading Actions
             (74 features)        (4 branches + fusion)
```

### Feature Categories
- **HF (7)**: 1s aggregated microstructure
- **MF (43)**: 1m technical analysis (pandas/ta)  
- **LF (19)**: Session/daily context
- **Portfolio (5)**: Position/P&L tracking

## =' Recent Major Refactoring (May 2025)

 **Professional Tools**: Replaced manual calculations with pandas/ta library  
 **Sequence Efficiency**: Features now use full temporal windows  
 **Clean Architecture**: Eliminated confusing "static" branch  
 **Optimized Features**: 74 high-quality vs 80+ redundant features  

## =Ö Additional Documentation

- [Market Simulator Design](docs/market_simulator_design.md)
- [Metrics System Plan](docs/metrics_system_plan.md)  
- [Rewards System Plan](docs/rewards_system_plan.md)
- [POE Tasks](docs/POE_TASKS.md)

## <¯ Trading Focus

**Target**: Low-float momentum stocks (MLGO)  
**Strategy**: Squeeze/breakout patterns  
**Timeframe**: 3s-1m decisions, ~40 trades/day  
**Data**: Tick-level market microstructure

---

*For complete implementation details, feature specifications, and architectural decisions, see the [Complete Feature & Model Guide](docs/COMPLETE_FEATURE_AND_MODEL_GUIDE.md)*