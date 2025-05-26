# Trading Environment Redesign Guide

## Overview
This document outlines the redesign of the FxAIv2 trading environment to better support momentum-based, high-frequency trading strategies for low-float stocks. The goal is to create an environment that trains the model to execute quick, opportunistic trades rather than buy-and-hold strategies.

## Trading Style Requirements

### Core Trading Characteristics
- **Target Stocks**: Low float, high volume, high volatility (potentially high float)
- **Trade Duration**: Seconds to minutes (max few minutes)
- **Trading Hours**: Primarily pre-market (4-9:30 AM ET), some regular hours, minimal after-hours
- **Strategy**: Quick momentum scalping - enter/exit rapidly during momentum moves
- **Risk Management**: Minimal exposure time to reduce risk

### Trading Patterns
1. **Momentum Bursts**: Add positions incrementally as momentum builds, take profits quickly
2. **Squeeze Patterns**: Stocks may squeeze for 15 min, consolidate for 15 min, repeat
3. **Rapid Reversals**: Be prepared for 50% moves in 1 minute (both directions)
4. **Exit Speed**: When momentum fades, exit immediately and wait for next opportunity

## Current Environment Issues

1. **Random Episode Starting**: Episodes start at random points with no strategic logic
2. **Fixed Episode Length**: 2048 steps (34 min) doesn't align with momentum patterns
3. **No Position Closure**: Episodes terminate with open positions, losing PnL information
4. **Inadequate Reward System**: Doesn't incentivize quick trades or punish holding
5. **Poor Data Sampling**: Random sampling misses high-activity momentum periods

## Redesigned Environment Architecture

### 1. Episode Management Strategy

#### Episode Starting Logic
```python
class EpisodeStartStrategy:
    """Strategic episode starting based on market conditions"""
    
    def find_episode_start(self, day_data: pd.DataFrame) -> List[datetime]:
        """Find optimal starting points for training episodes"""
        
        potential_starts = []
        
        # 1. High Volume Periods (momentum opportunities)
        volume_threshold = day_data['volume'].quantile(0.8)
        high_volume_periods = day_data[day_data['volume'] > volume_threshold]
        
        # 2. Price Movement Periods (volatility spikes)
        price_changes = day_data['close'].pct_change().abs()
        volatility_threshold = price_changes.quantile(0.9)
        high_volatility_periods = day_data[price_changes > volatility_threshold]
        
        # 3. Market Open Periods (high activity)
        market_open_windows = [
            (time(4, 0), time(5, 0)),    # Pre-market open
            (time(9, 25), time(10, 0)),  # Regular market open
        ]
        
        # 4. Catalyst-Driven Periods (if available)
        # Future: integrate news/catalyst timing
        
        return potential_starts
```

#### Dynamic Episode Length
```python
class DynamicEpisodeManager:
    """Manage episodes with variable length based on market conditions"""
    
    def __init__(self):
        self.min_steps = 300  # 5 minutes minimum
        self.max_steps = 3600  # 60 minutes maximum
        self.momentum_threshold = 0.02  # 2% move indicates momentum
        
    def should_continue_episode(self, market_state, portfolio_state, steps) -> bool:
        """Determine if episode should continue"""
        
        # Always respect min/max bounds
        if steps < self.min_steps:
            return True
        if steps >= self.max_steps:
            return False
            
        # Continue if in active momentum
        if self.is_momentum_active(market_state):
            return True
            
        # Terminate if flat market for too long
        if self.consecutive_flat_steps > 180:  # 3 minutes
            return False
            
        # Continue if position is open (with limits)
        if portfolio_state.has_position and steps < 1800:  # 30 min max with position
            return True
            
        return True
```

#### Position Closure at Episode End
```python
class EpisodeTerminationHandler:
    """Handle graceful episode termination with position management"""
    
    def prepare_termination(self, env, reason: str) -> Dict:
        """Prepare for episode termination"""
        
        termination_info = {
            'reason': reason,
            'open_position': False,
            'forced_exit_pnl': 0.0
        }
        
        if env.portfolio_manager.has_position():
            # Force close position at market
            fill = env.execution_manager.force_close_position(
                symbol=env.primary_asset,
                reason='episode_termination'
            )
            
            if fill:
                termination_info['open_position'] = True
                termination_info['forced_exit_pnl'] = fill['realized_pnl']
                
                # Apply penalty for forced closure
                env.reward_calculator.apply_forced_closure_penalty(fill)
                
        return termination_info
```

### 2. Reward System for Momentum Trading

#### Core Reward Components
```python
class MomentumTradingRewardV3:
    """Reward system optimized for quick momentum trades"""
    
    def __init__(self, config):
        self.components = {
            # Primary rewards
            'realized_pnl': RealizedPnLReward(weight=2.0),  # Doubled weight
            'quick_profit': QuickProfitReward(weight=1.5),  # New component
            
            # Speed incentives
            'trade_duration': TradeDurationReward(
                optimal_duration=60,  # 1 minute optimal
                max_duration=300,     # 5 minute max
                weight=1.0
            ),
            'momentum_capture': MomentumCaptureReward(weight=1.2),
            
            # Risk penalties
            'holding_penalty': ExponentialHoldingPenalty(
                base_penalty=0.001,
                exponential_factor=1.1,  # Increases each step
                weight=1.0
            ),
            'drawdown_penalty': AggressiveDrawdownPenalty(
                threshold=0.02,  # 2% drawdown triggers
                weight=1.5
            ),
            
            # Efficiency rewards
            'trade_efficiency': TradeEfficiencyReward(
                min_profit_threshold=0.002,  # 0.2% minimum
                weight=0.8
            ),
            'exit_timing': ExitTimingReward(weight=1.0),
            
            # Anti-patterns
            'overholding_penalty': OverholdingPenalty(
                momentum_fade_threshold=0.5,  # 50% momentum reduction
                weight=2.0
            ),
            'missed_exit_penalty': MissedExitPenalty(weight=1.5)
        }
```

#### Specialized Reward Components

```python
class QuickProfitReward(RewardComponent):
    """Reward quick profitable trades"""
    
    def calculate(self, state: RewardState) -> float:
        if not state.trade_closed:
            return 0.0
            
        pnl = state.realized_pnl
        duration = state.trade_duration_seconds
        
        if pnl > 0:
            # Exponential bonus for quicker profits
            time_multiplier = np.exp(-duration / 60.0)  # Decay with 1-min half-life
            return pnl * time_multiplier * self.weight
        
        return 0.0

class MomentumCaptureReward(RewardComponent):
    """Reward entering during momentum and exiting before reversal"""
    
    def calculate(self, state: RewardState) -> float:
        if not state.in_position:
            return 0.0
            
        # Check if we're riding momentum
        price_velocity = state.get_price_velocity()
        position_direction = state.get_position_direction()
        
        if self.same_direction(price_velocity, position_direction):
            # Reward proportional to momentum strength
            return abs(price_velocity) * self.weight
        else:
            # Penalty for holding against momentum
            return -abs(price_velocity) * self.weight * 2.0

class ExitTimingReward(RewardComponent):
    """Reward well-timed exits"""
    
    def calculate(self, state: RewardState) -> float:
        if not state.trade_closed:
            return 0.0
            
        # Compare exit price to subsequent prices
        exit_price = state.exit_price
        future_prices = state.get_future_prices(window=60)  # Next 60 seconds
        
        if state.was_long:
            # For longs, reward if price went down after exit
            price_decline = (exit_price - min(future_prices)) / exit_price
            return min(price_decline * self.weight, 0.05)  # Cap at 5%
        else:
            # For shorts, reward if price went up after exit
            price_rise = (max(future_prices) - exit_price) / exit_price
            return min(price_rise * self.weight, 0.05)
```

### 3. Data Loading and Sampling Strategy

#### High-Activity Period Detection
```python
class HighActivityDataLoader:
    """Load data focusing on high-activity periods"""
    
    def identify_high_activity_periods(self, symbol: str, date: datetime) -> List[Tuple[datetime, datetime]]:
        """Identify periods of high trading activity"""
        
        day_data = self.load_full_day_data(symbol, date)
        
        periods = []
        
        # 1. Volume-based detection
        volume_ma = day_data['volume'].rolling(window=60).mean()
        volume_spikes = day_data[day_data['volume'] > 2 * volume_ma]
        
        # 2. Price movement detection
        price_changes = day_data['close'].pct_change(periods=60).abs()
        large_moves = day_data[price_changes > 0.05]  # 5% moves
        
        # 3. Volatility clustering
        volatility = day_data['close'].pct_change().rolling(window=60).std()
        high_volatility = day_data[volatility > volatility.quantile(0.8)]
        
        # Merge and create continuous periods
        activity_mask = (volume_spikes.index | large_moves.index | high_volatility.index)
        periods = self.merge_activity_periods(activity_mask, min_duration=300)
        
        return periods
    
    def load_training_data(self, symbol: str, dates: List[datetime]) -> Dict:
        """Load data optimized for momentum training"""
        
        training_data = {
            'high_activity_periods': [],
            'catalyst_days': [],
            'normal_periods': []  # Some normal periods for balance
        }
        
        for date in dates:
            # Check if catalyst day (future: integrate news data)
            if self.is_catalyst_day(symbol, date):
                training_data['catalyst_days'].append({
                    'date': date,
                    'full_day': True,
                    'weight': 2.0  # Higher sampling weight
                })
            
            # Find high activity periods
            periods = self.identify_high_activity_periods(symbol, date)
            for start, end in periods:
                training_data['high_activity_periods'].append({
                    'start': start,
                    'end': end,
                    'weight': 1.5
                })
                
        return training_data
```

### 4. Enhanced Episode Configuration

```yaml
# config/env/momentum_trading.yaml
env:
  episode_management:
    strategy: "high_activity"  # Options: random, high_activity, catalyst_driven
    
    # Episode length configuration
    min_episode_steps: 300      # 5 minutes
    max_episode_steps: 3600     # 60 minutes
    typical_episode_steps: 1200 # 20 minutes
    
    # Dynamic episode control
    momentum_continuation:
      enabled: true
      momentum_threshold: 0.02  # 2% price move
      max_extension: 1800       # 30 min max extension
    
    # Position management
    force_position_closure:
      enabled: true
      max_position_duration: 1800  # 30 minutes max
      closure_penalty: 0.1         # 10% of position value
    
    # Starting point selection
    start_selection:
      prefer_high_volume: true
      volume_percentile: 80
      prefer_volatility: true
      volatility_percentile: 75
      prefer_market_open: true
      market_open_windows:
        - ["04:00", "05:00"]  # Pre-market open
        - ["09:25", "10:00"]  # Regular open
      
  # Reward system configuration
  reward_v3:
    # Primary components
    realized_pnl:
      weight: 2.0
      scale: 0.001
      
    quick_profit:
      weight: 1.5
      optimal_duration: 60
      decay_rate: 0.02
      
    # Speed incentives
    trade_duration:
      weight: 1.0
      optimal_duration: 60
      max_duration: 300
      penalty_curve: "exponential"
      
    momentum_capture:
      weight: 1.2
      momentum_window: 10
      
    # Penalties
    holding_penalty:
      weight: 1.0
      base_penalty: 0.001
      exponential_factor: 1.1
      start_after_steps: 60
      
    overholding_penalty:
      weight: 2.0
      momentum_fade_threshold: 0.5
      max_penalty: 0.1
```

### 5. Implementation Priority

1. **Phase 1: Episode Management** (Highest Priority)
   - Implement dynamic episode length
   - Add position closure at termination
   - Create high-activity starting point selection

2. **Phase 2: Reward System** (High Priority)
   - Implement momentum-specific reward components
   - Add exponential holding penalties
   - Create quick profit incentives

3. **Phase 3: Data Loading** (Medium Priority)
   - Implement high-activity period detection
   - Create weighted sampling for training
   - Add catalyst day identification

4. **Phase 4: Monitoring & Validation**
   - Add momentum capture metrics
   - Create trade duration histograms
   - Implement reward component analysis

## Expected Outcomes

With this redesigned environment, we expect:

1. **Shorter Average Trade Duration**: From minutes/hours to seconds/minutes
2. **Higher Trade Frequency**: More trades per episode with quick turnover
3. **Better Momentum Capture**: Entry during momentum, exit before reversal
4. **Reduced Risk Exposure**: Minimal time in position = less drawdown risk
5. **Improved Training Efficiency**: Focus on high-value learning periods

## Success Metrics

- Average trade duration < 2 minutes
- 80%+ trades closed within 5 minutes
- Win rate > 55% on quick trades
- Maximum position duration < 30 minutes
- Episode PnL correlation with momentum periods > 0.7