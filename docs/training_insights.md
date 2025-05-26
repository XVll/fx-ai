# Training Insights & Adjustments

This document tracks observations, interpretations, and adjustments made during the iterative training process.

## Training Interpretation Guide

### Key Metrics to Monitor

1. **Episode Metrics** (Primary Performance Indicators)
   - `episode_reward`: Total reward - should trend upward but expect high variance
   - `episode_pnl`: Actual profit/loss - the true measure of success
   - `win_rate`: Percentage of winning trades - aim for >50%
   - `sharpe_ratio`: Risk-adjusted returns - higher is better (>1.0 good, >2.0 excellent)
   - `max_drawdown`: Maximum loss from peak - lower is better (<10% ideal)

2. **Action Metrics** (Strategy Behavior)
   - `action_distribution`: Shows if model is exploring all actions
   - `hold_ratio`: Time spent holding vs trading - too high means overly cautious
   - `trade_frequency`: Number of trades per episode - balance is key
   - `position_utilization`: How much capital is being used

3. **Training Metrics** (Learning Progress)
   - `policy_loss`: Should stabilize or slowly decrease
   - `value_loss`: Should decrease over time
   - `entropy`: Measures exploration - should decrease gradually
   - `kl_divergence`: Keeps updates stable - spikes indicate instability
   - `learning_rate`: Will decay automatically in continuous training

4. **Reward Components** (What's Driving Behavior)
   - Watch individual reward components to understand what's influencing decisions
   - `pnl_reward`: Direct profit incentive
   - `holding_penalty`: Encourages action
   - `risk_penalty`: Discourages excessive risk
   - `action_consistency`: Rewards stable strategies

### Red Flags to Watch For

1. **Overtrading**: High trade frequency with negative PnL
2. **Analysis Paralysis**: Hold ratio >90%, very few trades
3. **Reward Hacking**: High reward but negative PnL
4. **Unstable Learning**: KL divergence spikes, erratic losses
5. **Poor Exploration**: Entropy drops too fast or action distribution too narrow

### Adjustment Guidelines

1. **If PnL is negative but rewards are high**:
   - Reduce non-PnL reward components
   - Increase PnL weight in reward system
   - Check for reward hacking behaviors

2. **If model holds too much**:
   - Reduce holding penalty decay rate
   - Increase base holding penalty
   - Check if spreads/fees are too high

3. **If model overtrades**:
   - Increase transaction costs
   - Add action rate penalty
   - Reduce momentum reward weight

4. **If learning is unstable**:
   - Reduce learning rate
   - Decrease PPO clip range
   - Increase batch size

5. **If no improvement after many episodes**:
   - Check feature quality (NaN, scaling issues)
   - Verify data variety (not stuck on same day)
   - Consider architecture changes

## Training Sessions

### Session Template
```
Date: YYYY-MM-DD
Model Version: vX
Starting Episode: X
Config Changes: None/List changes

Observations:
- Key metric values and trends
- Behavioral patterns
- Concerns or successes

Adjustments Made:
- Config changes with rationale
- Expected impact

Results:
- Immediate effects
- Longer-term outcomes
```

---

## Sessions Log

<!-- Training sessions will be logged below -->