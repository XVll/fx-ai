# Trading Environment Redesign V2 - Complete Market Experience

## Core Philosophy
Train the model to experience and handle ALL market conditions - both the profitable momentum runs AND the devastating reversals. The model must learn to exit on its own, not rely on episode boundaries.

## Key Design Principles

### 1. No Artificial Position Management
- **No forced closures** at episode end - let positions carry over
- **No position-based episode termination** - model must learn to manage risk
- Episodes end based on time/steps only, NOT based on position status
- Model experiences the full consequences of poor exit decisions

### 2. Comprehensive Market Experience

#### Episode Structure: Full Market Cycles
```python
class MarketCycleEpisodeManager:
    """Manage episodes to cover complete market cycles"""
    
    def __init__(self):
        # Fixed episode length for consistent experience
        self.episode_length = 3600  # 60 minutes - enough for multiple cycles
        
        # Episode starting strategy
        self.start_strategies = {
            'pre_squeeze': 0.3,      # Start before big moves
            'mid_squeeze': 0.3,      # Start during momentum
            'post_squeeze': 0.2,     # Start after reversal
            'random': 0.2            # Random for variety
        }
        
    def select_episode_start(self, day_data: pd.DataFrame) -> datetime:
        """Select starting point to ensure varied market experiences"""
        
        strategy = self.np_random.choice(
            list(self.start_strategies.keys()),
            p=list(self.start_strategies.values())
        )
        
        if strategy == 'pre_squeeze':
            # Find calm before storm - low volatility before spike
            volatility = day_data['close'].pct_change().rolling(60).std()
            calm_periods = volatility < volatility.quantile(0.3)
            # Look ahead for volatility spike
            future_volatility = volatility.shift(-300)  # 5 min ahead
            pre_squeeze = calm_periods & (future_volatility > volatility.quantile(0.7))
            
        elif strategy == 'mid_squeeze':
            # Find active momentum periods
            momentum = day_data['close'].pct_change(60)
            high_momentum = abs(momentum) > momentum.quantile(0.8)
            
        elif strategy == 'post_squeeze':
            # Find reversal points - high followed by opposite move
            price_change = day_data['close'].pct_change(300)  # 5 min change
            reversals = (price_change > 0.1) & (price_change.shift(-300) < -0.05)
            
        else:  # random
            # Pure random selection
            
        return selected_timestamp
```

### 3. Reward System - Learning Complete Trade Lifecycle

#### Core Reward Philosophy
- Reward capturing momentum AND avoiding reversals
- Heavily penalize holding through reversals
- Incentivize recognizing momentum exhaustion

```python
class CompleteCycleRewardSystem:
    """Reward system that teaches full trade lifecycle"""
    
    def __init__(self):
        self.components = {
            # Base rewards
            'realized_pnl': RealizedPnLReward(
                weight=2.0,
                scale_by_duration=True  # Less reward for longer trades
            ),
            
            # Momentum rewards
            'momentum_alignment': MomentumAlignmentReward(
                weight=1.5,
                # Reward being positioned with momentum
                # Penalize being positioned against momentum
            ),
            
            'momentum_exhaustion_exit': MomentumExhaustionReward(
                weight=2.0,
                # Big reward for exiting near momentum peak
                # Penalty for holding through reversal
            ),
            
            # Risk management
            'reversal_avoidance': ReversalAvoidanceReward(
                weight=3.0,  # Heavy weight - critical skill
                reversal_threshold=0.03,  # 3% adverse move
            ),
            
            'drawdown_penalty': ProgressiveDrawdownPenalty(
                weight=2.0,
                thresholds=[0.02, 0.05, 0.10],  # 2%, 5%, 10%
                penalties=[0.1, 0.5, 2.0]        # Escalating
            ),
            
            # Timing rewards
            'entry_timing': EntryTimingReward(
                weight=1.0,
                # Reward entering at momentum start
                # Penalize chasing extended moves
            ),
            
            'hold_duration_penalty': AdaptiveHoldPenalty(
                weight=1.0,
                # Penalty increases with momentum decay
                # Not just time-based
            ),
            
            # Experience rewards
            'market_regime_adaptation': MarketRegimeReward(
                weight=0.5,
                # Small reward for correct behavior in different regimes
                # Helps learn context
            )
        }
```

#### Advanced Reward Components

```python
class MomentumExhaustionReward(RewardComponent):
    """Reward exiting when momentum shows exhaustion signs"""
    
    def calculate(self, state: RewardState) -> float:
        if not state.trade_closed:
            return 0.0
            
        # Get momentum at entry and exit
        entry_momentum = state.get_momentum_at_entry()
        exit_momentum = state.get_momentum_at_exit()
        peak_momentum = state.get_peak_momentum_during_trade()
        
        # Calculate momentum decay
        momentum_retention = exit_momentum / peak_momentum if peak_momentum > 0 else 0
        
        if state.profitable_trade:
            if momentum_retention > 0.7:
                # Exited while momentum still strong - good
                return 0.5 * self.weight
            elif momentum_retention > 0.3:
                # Exited as momentum fading - excellent
                return 1.0 * self.weight
            else:
                # Held too long, momentum reversed - bad
                return -0.5 * self.weight
        else:
            # Loss - check if cut losses quickly
            if state.trade_duration < 120:  # Under 2 min
                return 0.1 * self.weight  # Small reward for quick loss cut
            else:
                return -0.2 * self.weight  # Penalty for holding losers

class ReversalAvoidanceReward(RewardComponent):
    """Heavy penalty for holding through reversals"""
    
    def calculate(self, state: RewardState) -> float:
        if not state.in_position:
            return 0.0
            
        # Check if experiencing reversal
        position_pnl = state.unrealized_pnl_percent
        peak_pnl = state.peak_unrealized_pnl_percent
        
        # Reversal = significant pullback from peak
        pullback = peak_pnl - position_pnl
        
        if pullback > self.reversal_threshold:
            # In reversal - progressive penalty
            penalty = -pullback * 10 * self.weight  # 10x multiplier
            
            # Extra penalty if was profitable but now negative
            if peak_pnl > 0.01 and position_pnl < 0:
                penalty *= 2  # Double penalty for round trips
                
            return penalty
            
        return 0.0

class EntryTimingReward(RewardComponent):
    """Reward good entries, penalize chasing"""
    
    def calculate(self, state: RewardState) -> float:
        if not state.position_opened:
            return 0.0
            
        # Check momentum extension at entry
        price_run_before_entry = state.get_price_run_before_entry(window=300)
        
        if abs(price_run_before_entry) < 0.02:
            # Entered early in move - good
            return 0.2 * self.weight
        elif abs(price_run_before_entry) < 0.05:
            # Entered mid-move - neutral
            return 0.0
        else:
            # Chasing extended move - bad
            return -0.3 * self.weight
```

### 4. Episode Management Without Artificial Constraints

```python
class NaturalEpisodeManager:
    """Episode management that doesn't interfere with trading"""
    
    def __init__(self, config):
        self.episode_length = config.env.episode_length  # e.g., 3600 steps
        self.allow_overnight_positions = False  # For intraday only
        
    def should_terminate_episode(self, state: EnvState) -> Tuple[bool, str]:
        """Check natural termination conditions only"""
        
        # Time-based termination
        if state.steps >= self.episode_length:
            return True, "max_steps_reached"
            
        # Market close (for intraday)
        if state.current_time >= state.market_close and not self.allow_overnight_positions:
            return True, "market_closed"
            
        # Critical failures only
        if state.portfolio.bankrupt:
            return True, "bankruptcy"
            
        # Do NOT terminate for:
        # - Open positions
        # - Large losses (unless bankrupt)
        # - Any trading-related reason
        
        return False, ""
    
    def handle_episode_transition(self, env) -> Dict:
        """Smooth transition between episodes"""
        
        transition_info = {
            'carried_position': None,
            'carried_pnl': 0
        }
        
        if env.portfolio_manager.has_position():
            # Carry position to next episode
            position = env.portfolio_manager.get_position_details()
            transition_info['carried_position'] = position
            transition_info['carried_pnl'] = position['unrealized_pnl']
            
            # Do NOT close position
            # Do NOT apply penalties
            # Let it continue naturally
            
        return transition_info
```

### 5. Training Curriculum

To ensure the model learns both sides of momentum:

```python
class MomentumCurriculumScheduler:
    """Schedule training to cover all market conditions"""
    
    def __init__(self):
        self.curriculum_stages = [
            {
                'name': 'basic_momentum',
                'episodes': 1000,
                'focus': 'clean_trends',
                'difficulty': 0.3
            },
            {
                'name': 'reversals_introduction', 
                'episodes': 1000,
                'focus': 'trend_reversals',
                'difficulty': 0.5
            },
            {
                'name': 'whipsaws',
                'episodes': 1000,
                'focus': 'choppy_markets',
                'difficulty': 0.7
            },
            {
                'name': 'full_experience',
                'episodes': float('inf'),
                'focus': 'all_conditions',
                'difficulty': 1.0
            }
        ]
        
    def select_training_data(self, stage: Dict) -> List[str]:
        """Select appropriate training days for curriculum stage"""
        
        if stage['focus'] == 'clean_trends':
            # Days with clear directional moves
            return self.find_trending_days()
            
        elif stage['focus'] == 'trend_reversals':
            # Days with major reversals
            return self.find_reversal_days()
            
        elif stage['focus'] == 'choppy_markets':
            # High volatility, no clear direction
            return self.find_choppy_days()
            
        else:
            # Mix of everything
            return self.all_available_days
```

### 6. Implementation Configuration

```yaml
# config/env/momentum_complete.yaml
env:
  # Fixed episode length - no dynamic termination
  episode_length: 3600  # 60 minutes
  max_steps: 3600
  
  # Episode starting
  episode_start_strategy: "market_cycle"
  start_distribution:
    pre_squeeze: 0.3
    mid_squeeze: 0.3  
    post_squeeze: 0.2
    random: 0.2
  
  # No position-based termination
  terminate_on_position: false
  force_close_positions: false
  max_loss_termination: false  # Only bankruptcy
  
  # Natural boundaries only
  termination:
    bankruptcy_threshold: 0.1  # 90% loss
    time_limit_only: true
    
  # Reward configuration
  reward_v3:
    # Emphasize complete cycle learning
    momentum_exhaustion:
      weight: 2.0
      peak_decay_threshold: 0.3
      
    reversal_avoidance:
      weight: 3.0
      reversal_threshold: 0.03
      round_trip_multiplier: 2.0
      
    entry_timing:
      weight: 1.0
      chase_threshold: 0.05
      early_entry_bonus: 0.2
      
    # Adaptive penalties
    hold_penalty:
      base: 0.0  # No flat penalty
      momentum_based: true
      decay_factor: 0.1
```

## Expected Training Outcomes

1. **Natural Exit Learning**: Model learns to exit based on market conditions, not episode boundaries
2. **Reversal Recognition**: Heavy penalties teach reversal avoidance
3. **Complete Cycle Experience**: Model sees entry → momentum → exhaustion → reversal
4. **Risk Management**: Learns to cut losses quickly without artificial help
5. **Context Awareness**: Understands different market regimes

## Key Differences from V1

1. **No Forced Actions**: Episode boundaries don't affect positions
2. **Complete Experience**: Model experiences full consequences of decisions
3. **Natural Learning**: Rewards/penalties based on market reality, not artificial constraints
4. **Both Sides**: Explicitly trains on both momentum runs AND reversals
5. **Progressive Difficulty**: Curriculum introduces complexity gradually