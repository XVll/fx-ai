# Trading Environment Redesign - Final Implementation Plan

## Overview

This document consolidates the complete redesign plan for the trading environment, specifically optimized for **low-float momentum trading** with a focus on rapid entry/exit during squeeze events.

## Core Trading Strategy

- **Asset Type**: Low-float momentum stocks (e.g., MLGO)
- **Trading Style**: Sniper approach - quick in/out (3-5 seconds to 3-5 minutes per trade)
- **Episode Structure**: Single trading day (4 AM - 8 PM ET)
- **Position Management**: No overnight holds, multiple trades per squeeze
- **Decision Frequency**: 1-second bars for precise entry/exit

## Architecture Components

### 1. MomentumEpisodeScanner

**Purpose**: Scan daily market data to identify and categorize potential reset points for training episodes.

**Key Features**:
- Detects momentum/squeeze setups using volume, price velocity, and ATR expansion
- Categorizes market conditions (front side, back side, dead zones)
- Assigns quality scores based on multiple factors
- Creates metadata for intelligent episode selection

**Reset Point Categories**:

```python
class MomentumPhase(Enum):
    """Phases of momentum/squeeze moves"""
    FRONT_SIDE_BREAKOUT = "front_breakout"      # Initial squeeze breakout
    FRONT_SIDE_MOMENTUM = "front_momentum"       # Sustained upward momentum
    PARABOLIC = "parabolic"                      # Extreme momentum
    BACK_SIDE_FLUSH = "back_flush"              # Initial breakdown
    BACK_SIDE_FADE = "back_fade"                # Continued selling
    BOUNCE = "bounce"                            # Dead cat bounce
    CONSOLIDATION = "consolidation"              # Sideways after move
    ACCUMULATION = "accumulation"                # Building for next move
    DISTRIBUTION = "distribution"                # Top formation
    DEAD = "dead"                                # No activity
```

**Time-Based Quality Scoring**:

| Time Period | Hours | Quality Score | Reasoning |
|------------|-------|---------------|-----------|
| Early Premarket | 4-7 AM | 0.3 | Low volume, wide spreads |
| Active Premarket | 7-9:30 AM | 1.0 | Building volume, tightening spreads |
| Market Open | 9:30-10:30 AM | 1.0 | **Best** - High volume, momentum |
| Midday | 10:30 AM-2 PM | 0.4 | Often dead, choppy |
| Power Hour | 2-4 PM | 0.8 | Second best - EOD momentum |
| After Hours | 4-8 PM | 0.5 | Medium quality |

**Quality Calculation**:
```python
quality = (
    phase_score * 0.4 +        # Momentum phase (most important)
    volume_score * 0.3 +       # Volume vs average
    setup_quality * 0.2 +      # Pattern clarity
    spread_tightness * 0.1     # Execution quality
) * time_multiplier
```

### 2. Pattern Detection System

**Momentum Detection Criteria**:
- Volume surge: 2x+ 20-period average
- Price velocity: 0.2%+ move in 1 minute
- ATR expansion: 50%+ increase
- Pre-consolidation: 5+ minutes of low volatility

**Pattern Types**:
1. **Breakout**: Consolidation → Volume surge → Price expansion
2. **Flush**: Near highs → High volume → Sharp decline
3. **Bounce**: 3%+ decline → Volume → Recovery attempt
4. **Accumulation**: Low volatility near VWAP with building bids
5. **Distribution**: High of day with decreasing momentum

### 3. Episode Management

**Episode Structure**:
- Each episode operates within a single trading day
- Multiple reset points per day (momentum-based + fixed times)
- Episodes terminate at: market close, max loss (5%), or max duration (4 hours)
- No position carrying between days

**Fixed Reset Points**:
- 9:30 AM - Market open
- 10:30 AM - Post-open settlement
- 2:00 PM - Afternoon session
- 3:30 PM - Power hour

**Dead Zone Strategy**:
Include low-activity periods strategically to:
- Teach patience (avoid overtrading)
- Recognize pre-squeeze accumulation
- Learn risk management during choppy markets

### 4. Open Position Handling

**Episode Termination Handling**:
Since we're doing single-day episodes with intraday momentum trading, position handling is simpler but critical:

| Termination Reason | Position Action | Implementation |
|-------------------|-----------------|----------------|
| Market Close (8 PM) | Force close at bid | Real market constraint |
| Max Loss (10%) | Force close at bid | Risk management |
| Max Duration (4hr) | Continue normally | Position still valid |
| Data End | Force close at market | No other option |

**Position State Management**:
```python
class PositionHandler:
    def handle_episode_end(self, 
                          portfolio_state: PortfolioState,
                          termination_reason: str,
                          market_state: MarketState) -> Dict[str, Any]:
        """Handle open positions at episode end"""
        
        position = portfolio_state.get_position(self.symbol)
        
        if position.is_flat():
            return {'had_position': False}
            
        # Determine action based on termination
        if termination_reason in ["MARKET_CLOSE", "MAX_LOSS", "BANKRUPTCY"]:
            # Force close at bid (realistic exit)
            close_price = market_state.best_bid_price
            realized_pnl = self._calculate_exit_pnl(position, close_price)
            
            return {
                'had_position': True,
                'forced_exit': True,
                'exit_price': close_price,
                'realized_pnl': realized_pnl,
                'reason': termination_reason
            }
            
        elif termination_reason == "MAX_DURATION":
            # Episode ended but position might still be good
            # Let the next episode decide (within same day)
            return {
                'had_position': True,
                'forced_exit': False,
                'unrealized_pnl': position.unrealized_pnl,
                'hold_duration': position.duration
            }
```

**Intraday Continuity**:
- Within the same trading day, we track position performance across episode boundaries
- This teaches the model about position management through different market phases
- No overnight carries - all positions close by market close

### 5. Reward System for Sniper Trading

**Time-Based Multipliers**:
- 0-30 seconds: 2.0x multiplier (best)
- 30-120 seconds: 1.5x multiplier
- 120-300 seconds: 1.0x multiplier
- Over 5 minutes: 0.5x multiplier (penalty)

**Additional Components**:
- Quick profit bonus: Extra reward for hitting 3% target
- Momentum alignment: Bonus for trading with the squeeze
- Max position duration: 5 minutes
- Forced exit penalty: Additional penalty for positions closed due to market close

### 6. Progressive Curriculum

**Stage 1 - Prime Setups (0-1000 episodes)**:
- 80% high-quality momentum breakouts
- 20% risk scenarios (flushes, fakeouts)
- Focus on market open and power hour
- Learn basic entry/exit patterns

**Stage 2 - Expanded Conditions (1000-3000 episodes)**:
- 40% prime momentum
- 30% secondary setups
- 20% risk scenarios
- 10% educational patterns

**Stage 3 - Patience Training (3000-5000 episodes)**:
- Include 10% dead zones
- More diverse time periods
- Back side trading
- Pre-squeeze recognition

**Stage 4 - Full Market (5000+ episodes)**:
- Equal exposure to all conditions
- Complete market hours
- All pattern types
- Production-ready training

## Implementation Guide

### Phase 1: Core Scanner Implementation

```python
class EpisodeScanner:
    def scan_single_day(self, symbol: str, date: datetime) -> Dict[str, List[ResetPoint]]:
        """Scan one day and categorize all reset points"""
        
        # Load full day data
        day_data = self.data_manager.get_day_data(symbol, date)
        
        # Categorized storage
        reset_points = {
            'prime_momentum': [],
            'secondary_momentum': [],
            'risk_scenarios': [],
            'dead_zones': [],
            'time_based': []
        }
        
        # Scan every minute for patterns
        for timestamp in pd.date_range(start=day_data.index[0], 
                                     end=day_data.index[-1], 
                                     freq='1min'):
            
            # Detect momentum phase
            phase = self._detect_momentum_phase(day_data, timestamp)
            
            # Calculate quality metrics
            metrics = self._calculate_quality_metrics(day_data, timestamp)
            
            # Create categorized reset point
            reset_point = CategorizedResetPoint(
                timestamp=timestamp,
                momentum_phase=phase,
                quality_score=metrics['quality'],
                volume_ratio=metrics['volume_ratio'],
                pattern_type=metrics['pattern']
            )
            
            # Categorize
            if reset_point.quality_score > 0.8:
                reset_points['prime_momentum'].append(reset_point)
            elif reset_point.quality_score > 0.6:
                reset_points['secondary_momentum'].append(reset_point)
            # ... etc
            
        return reset_points
```

### Phase 2: Environment Modifications

```python
# Add to TradingEnvironment
def reset_for_momentum_training(self, reset_point: CategorizedResetPoint):
    """Reset environment at specific momentum point"""
    
    # Setup session at reset point
    self.setup_session(
        symbol=reset_point.symbol,
        start_time=reset_point.timestamp,
        end_time=min(
            reset_point.timestamp + timedelta(hours=4),
            reset_point.timestamp.replace(hour=20)  # Market close
        )
    )
    
    # Apply momentum-specific initialization
    self._apply_momentum_context(reset_point)
    
    return self.reset()

def get_episode_termination_info(self) -> Dict[str, Any]:
    """Get termination info without forcing position closure"""
    portfolio_state = self.portfolio_manager.get_portfolio_state(
        self.market_simulator.current_time
    )
    
    position_info = None
    if self.primary_asset:
        position = portfolio_state['positions'].get(self.primary_asset, {})
        if position.get('quantity', 0) != 0:
            position_info = {
                'has_position': True,
                'quantity': position['quantity'],
                'side': position['current_side'],
                'unrealized_pnl': position['unrealized_pnl'],
                'duration': self.current_step  # steps held
            }
    
    return {
        'portfolio_state': portfolio_state,
        'position_info': position_info,
        'termination_reason': self.current_termination_reason,
        'market_state': self.market_simulator.get_current_market_state()
    }
```

### Phase 3: Training Manager

```python
class TrainingManager:
    def __init__(self, config, data_manager, logger):
        self.position_handler = PositionHandler(config, logger)
        self.episode_scanner = MomentumEpisodeScanner(data_manager, config)
        self.curriculum_selector = CurriculumBasedSelector(config)
        
    def run_training(self, agent, num_episodes: int):
        """Main training loop with momentum focus"""
        
        # Pre-scan all available days
        all_reset_points = self._scan_all_training_days()
        
        for episode in range(num_episodes):
            # Select reset point based on curriculum
            reset_point = self._select_by_curriculum(
                all_reset_points, 
                episode, 
                self.performance_tracker
            )
            
            # Run episode
            obs = env.reset_for_momentum_training(reset_point)
            
            done = False
            while not done:
                action = agent.predict(obs)
                obs, reward, done, truncated, info = env.step(action)
                
            # Handle open positions at episode end
            termination_info = env.get_episode_termination_info()
            position_result = self.position_handler.handle_episode_end(
                termination_info['portfolio_state'],
                termination_info['termination_reason'],
                termination_info['market_state']
            )
            
            # Log position handling
            if position_result['had_position']:
                self.logger.info(
                    f"Episode {episode} ended with position: "
                    f"forced_exit={position_result.get('forced_exit', False)}, "
                    f"pnl={position_result.get('realized_pnl', 0):.4f}"
                )
                
            # Track performance by category
            self._update_performance_tracking(reset_point, info, position_result)
```

## Configuration

```yaml
momentum_trading:
  # Episode management
  episode:
    max_duration: 14400  # 4 hours
    stop_loss: 0.05      # 5%
    single_day_only: true
    
  # Reset point detection
  momentum_detection:
    volume_surge_threshold: 2.0      # 2x average
    price_velocity_threshold: 0.002  # 0.2% per minute
    atr_expansion_threshold: 1.5     # 50% expansion
    
  # Pattern recognition
  patterns:
    breakout:
      consolidation_minutes: 5
      volume_required: 3.0
    flush:
      from_high_threshold: 0.02
      volume_required: 2.0
      
  # Reward configuration
  rewards:
    time_multipliers:
      0_5_seconds: 2.0
      5_30_seconds: 1.5
      30_300_seconds: 1.0
      over_300_seconds: 0.5
    quick_profit_target: 0.003  # 0.3%
    stop_loss: 0.002           # 0.2%
    
  # Curriculum stages
  curriculum:
    stage_1:
      episodes: [0, 1000]
      prime_momentum_weight: 0.8
      risk_scenarios_weight: 0.2
    stage_2:
      episodes: [1000, 3000]
      prime_momentum_weight: 0.4
      secondary_momentum_weight: 0.3
      risk_scenarios_weight: 0.2
      educational_weight: 0.1
    stage_3:
      episodes: [3000, 5000]
      include_dead_zones: true
      dead_zone_weight: 0.1
    stage_4:
      episodes: [5000, null]
      all_categories_equal: true
```

## Expected Outcomes

1. **Training Efficiency**:
   - 50% faster convergence through intelligent episode selection
   - Reduced wasted compute on low-value market conditions

2. **Model Performance**:
   - Better entry timing on momentum breakouts
   - Improved exit timing on back side flushes
   - Reduced overtrading during dead periods
   - 20-30% improvement in Sharpe ratio

3. **Realistic Behavior**:
   - Sniper-like entries during high-probability setups
   - Quick profit-taking aligned with momentum trading
   - Proper risk management during adverse moves

## Implementation Timeline

**Week 1**: Scanner and pattern detection
**Week 2**: Environment modifications and reward system
**Week 3**: Training orchestration and curriculum
**Week 4**: Testing, validation, and optimization

## Monitoring Metrics

- Reset point quality distribution
- Performance by momentum phase
- Average holding time by pattern type
- Win rate during different market conditions
- Curriculum progression effectiveness

This design creates a training environment specifically optimized for low-float momentum trading, teaching the model to act like a sniper - waiting for high-probability setups and executing quick, profitable trades during squeeze events.