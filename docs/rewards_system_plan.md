# Todo
* Add a penalty for halting trades.
# Reward System V2 Documentation

## Overview

The Reward System V2 provides a comprehensive, modular, and scalable reward framework for the FX-AI trading system. It implements advanced reward shaping with anti-hacking measures and detailed component tracking.

## Key Features

### 1. Modular Component Architecture
- **Base Classes**: `RewardComponent`, `RewardAggregator`, `RewardState`
- **Extensible Design**: Easy to add new reward components
- **Component Types**: Foundational, Shaping, Trade-specific, Terminal

### 2. Comprehensive Reward Components

#### Foundational Rewards
- **Realized PnL**: Direct profit/loss from closed trades (includes commissions)
- **Mark-to-Market**: Immediate feedback on unrealized P&L changes
- **Differential Sharpe**: Rewards improvements in risk-adjusted returns

#### Shaping Rewards (Penalties & Incentives)
- **Holding Time Penalty**: Discourages excessive position holding
- **Overtrading Penalty**: Prevents frequent unnecessary trades
- **Quick Profit Incentive**: Rewards capturing profits quickly
- **Drawdown Penalty**: Penalizes holding large losing positions

#### Trade-Specific Penalties
- **MAE Penalty**: Penalizes high risk during trades (Maximum Adverse Excursion)
- **MFE Penalty**: Penalizes giving back profits from peak (Maximum Favorable Excursion)

#### Terminal Penalties
- **Bankruptcy**: Large penalty for account depletion
- **Max Loss**: Penalty for hitting maximum loss threshold

### 3. Anti-Hacking Measures
- **Component Clipping**: Min/max bounds on individual rewards
- **Exponential Decay**: Reduces repeated behavior rewards over time
- **Reward Smoothing**: Moving average to prevent exploitation
- **Global Scaling**: Overall reward magnitude control

### 4. Comprehensive Metrics Integration

#### Per-Component Tracking
- Magnitude and frequency of triggers
- Correlation with agent behavior
- Impact on total reward
- Contribution percentage

#### W&B Integration
- Automatic logging of all component values
- Component statistics and correlations
- Episode-level summaries
- Training progression analysis

#### Dashboard Visualization
- Real-time component breakdown
- Top contributing components
- Reward vs penalty balance
- Historical trends

## Configuration

### Enable Reward V2
```yaml
env:
  use_reward_v2: true
  reward_v2:
    components:
      realized_pnl:
        enabled: true
        weight: 1.0
        clip_min: -10.0
        clip_max: 10.0
      # ... other components
    aggregator:
      global_scale: 0.01
      use_smoothing: true
      smoothing_window: 10
```

### Component Configuration
Each component supports:
- `enabled`: Toggle component on/off
- `weight`: Relative importance multiplier
- `clip_min/max`: Value bounds (anti-hacking)
- Component-specific parameters

## Implementation Details

### File Structure
```
rewards/
├── __init__.py          # Package exports
├── core.py              # Base classes and aggregator
├── components.py        # All reward component implementations
├── metrics.py           # Metrics tracking system
└── calculator.py        # Main reward system calculator
```

### Key Classes

#### RewardComponent (Abstract Base)
- `calculate()`: Compute reward value
- `apply_anti_hacking_measures()`: Apply clipping/decay
- Metadata for tracking and analysis

#### RewardAggregator
- Combines multiple components
- Applies global scaling and smoothing
- Tracks component statistics
- Prevents reward hacking

#### RewardSystemV2
- Main calculator interface
- Trade tracking for MAE/MFE
- Episode summaries
- Metrics integration

### Integration Points

#### Trading Environment
```python
# Automatically uses V2 when configured
if self.use_reward_v2:
    self.reward_calculator = RewardSystemV2(config, metrics_integrator)
```

#### Metrics System
- Automatic component registration
- Per-step value tracking
- Episode summaries
- W&B logging

#### Dashboard
- Real-time component display
- Sorted by impact
- Color-coded rewards/penalties

## Analysis Capabilities

### Component Analysis
- Which components dominate rewards?
- Are penalties too harsh/lenient?
- Is agent behavior being shaped correctly?

### Correlation Analysis
- Component values vs trading frequency
- Component values vs win rate
- Component interactions

### Tuning Insights
- Identify over/under-weighted components
- Detect reward hacking attempts
- Balance exploration vs exploitation

## Best Practices

### 1. Start Conservative
- Begin with lower weights
- Monitor component impacts
- Gradually tune based on behavior

### 2. Monitor for Hacking
- Watch for single component dominance
- Check for exploitative patterns
- Use anti-hacking measures

### 3. Balance Components
- Foundational rewards for core objectives
- Shaping rewards for behavior guidance
- Penalties to prevent bad habits

### 4. Use Metrics
- Regular analysis of component statistics
- Correlation with performance metrics
- Iterative refinement

## Example Usage

```python
# In training script
from hydra import compose, initialize

# Load config with reward v2
with initialize(config_path="../config"):
    cfg = compose(config_name="config.yaml", 
                  overrides=["env.use_reward_v2=true"])

# Environment automatically uses RewardSystemV2
env = TradingEnvironment(cfg, data_manager, metrics_integrator)

# Components tracked automatically in W&B and dashboard
```

## Future Enhancements

1. **Adaptive Weights**: Automatically adjust component weights based on training progress
2. **Meta-Learning**: Learn optimal reward configurations
3. **Multi-Objective**: Support for multiple competing objectives
4. **Curriculum Learning**: Progressive component activation