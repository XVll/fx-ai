# Trading Environment Redesign V3 - Advanced Momentum Mastery

## Core Insight
Momentum trading is about **reading the tape, recognizing patterns, and having the discipline to cut losses fast while letting winners run**. The environment must teach these skills through experience, not rules.

## Revolutionary Design Elements

### 1. Multi-Phase Momentum Recognition

The model must learn that momentum has distinct phases, each requiring different actions:

```python
class MomentumPhaseTracker:
    """Track and reward actions based on momentum lifecycle"""
    
    PHASES = {
        'ACCUMULATION': {
            'characteristics': ['low_volume', 'tight_range', 'absorption'],
            'optimal_action': 'POSITION_BUILDING',
            'duration': (60, 300)  # 1-5 min typical
        },
        'BREAKOUT': {
            'characteristics': ['volume_spike', 'range_expansion', 'aggressive_buying'],
            'optimal_action': 'AGGRESSIVE_ADD',
            'duration': (30, 180)  # 30s-3 min
        },
        'MOMENTUM_RUN': {
            'characteristics': ['sustained_volume', 'trending', 'pullback_buying'],
            'optimal_action': 'HOLD_CORE_SCALE_OUT',
            'duration': (120, 900)  # 2-15 min
        },
        'EXHAUSTION': {
            'characteristics': ['volume_divergence', 'range_contraction', 'failed_highs'],
            'optimal_action': 'AGGRESSIVE_EXIT',
            'duration': (60, 300)  # 1-5 min
        },
        'DISTRIBUTION': {
            'characteristics': ['high_volume_selling', 'lower_highs', 'bid_weakness'],
            'optimal_action': 'FULL_EXIT_OR_SHORT',
            'duration': (30, 180)  # 30s-3 min
        },
        'REVERSAL': {
            'characteristics': ['panic_selling', 'cascade_stops', 'no_bid'],
            'optimal_action': 'FLAT_OR_SHORT',
            'duration': (60, 600)  # 1-10 min
        }
    }
    
    def identify_current_phase(self, market_state: MarketState) -> str:
        """Identify which momentum phase we're in"""
        
        features = self.extract_phase_features(market_state)
        
        # Use multiple indicators to determine phase
        # Volume profile, price action, tape speed, bid/ask behavior
        
        return detected_phase
    
    def calculate_phase_transition_probability(self, current_phase: str) -> Dict[str, float]:
        """Probability of transitioning to next phase"""
        
        # Based on time in phase, market conditions, volume patterns
        # This helps the model anticipate phase changes
        
        return transition_probabilities
```

### 2. Tape Reading Rewards

The model must learn to read order flow and tape speed:

```python
class TapeReadingReward(RewardComponent):
    """Reward correct interpretation of tape/order flow"""
    
    def calculate(self, state: RewardState) -> float:
        tape_features = state.get_tape_features()
        
        # Reward examples:
        # 1. Buying when large blocks hit ask (bullish tape)
        if state.action == 'BUY' and tape_features['large_ask_hits'] > tape_features['large_bid_hits']:
            reward += 0.1 * self.weight
            
        # 2. Selling when bid support disappears
        if state.action == 'SELL' and tape_features['bid_pull_rate'] > 0.8:
            reward += 0.2 * self.weight
            
        # 3. Holding during sustained aggressive buying
        if state.action == 'HOLD' and tape_features['buy_pressure'] > 0.7:
            reward += 0.05 * self.weight
            
        # 4. Recognizing absorption (big orders not moving price)
        if tape_features['absorption_detected'] and state.action == 'WAIT':
            reward += 0.15 * self.weight
            
        return reward

class MomentumQualityReward(RewardComponent):
    """Reward based on quality of momentum, not just direction"""
    
    def calculate(self, state: RewardState) -> float:
        quality_score = self.assess_momentum_quality(state)
        
        # High quality momentum characteristics:
        # - Volume expansion on moves up
        # - Tight pullbacks on low volume  
        # - Higher lows, higher highs
        # - Bid support following price
        
        if state.in_position and state.position_aligned_with_momentum:
            return quality_score * state.position_size * self.weight
        elif state.in_position and not state.position_aligned_with_momentum:
            return -quality_score * state.position_size * self.weight * 2
            
        return 0.0
```

### 3. Scaled Exit Mastery

Real traders scale out. The model must learn this:

```python
class ScaledExitReward(RewardComponent):
    """Teach optimal scaling out strategies"""
    
    def calculate(self, state: RewardState) -> float:
        if not state.is_partial_exit:
            return 0.0
            
        momentum_phase = state.get_momentum_phase()
        position_pnl = state.position_pnl_percent
        
        # Reward logic for different scenarios
        if momentum_phase == 'BREAKOUT' and position_pnl > 0.02:
            # Take some profit on initial pop
            if state.exit_percent <= 0.25:  # Taking 25% or less
                return 0.3 * self.weight  # Good - securing profits
        
        elif momentum_phase == 'MOMENTUM_RUN' and position_pnl > 0.05:
            # Scale out more aggressively
            if 0.25 <= state.exit_percent <= 0.5:
                return 0.5 * self.weight  # Excellent - riding trend
                
        elif momentum_phase == 'EXHAUSTION':
            # Should be exiting aggressively
            if state.exit_percent >= 0.75:
                return 1.0 * self.weight  # Perfect - recognizing exhaustion
            else:
                return -0.5 * self.weight  # Bad - holding too much
                
        return 0.0

class RunnersManagementReward(RewardComponent):
    """Reward for managing runners (last portion of position)"""
    
    def calculate(self, state: RewardState) -> float:
        if state.position_size_percent > 0.25:  # Not in runner management
            return 0.0
            
        # Rewards for runner management
        if state.trailing_stop_distance < state.atr * 2:
            # Tight stop on runners - good
            reward += 0.1 * self.weight
            
        if state.momentum_phase == 'MOMENTUM_RUN' and state.action == 'HOLD':
            # Letting runners run during momentum - excellent
            reward += 0.2 * self.weight
            
        if state.gave_back_more_than_50_percent_of_runner_profits:
            # Held runners too long - bad
            reward -= 0.5 * self.weight
            
        return reward
```

### 4. Psychological Realism

Trading is emotional. The reward system must reflect this:

```python
class PsychologicalRealismReward(RewardComponent):
    """Simulate psychological impact of trading decisions"""
    
    def __init__(self, config):
        super().__init__(config)
        self.regret_memory = deque(maxlen=100)
        self.confidence_score = 1.0
        
    def calculate(self, state: RewardState) -> float:
        psychological_impact = 0.0
        
        # 1. Regret from missed moves
        if not state.in_position and state.missed_move_percent > 0.1:
            regret = -state.missed_move_percent * 0.5 * self.weight
            self.regret_memory.append(regret)
            psychological_impact += regret
            
        # 2. Pain from round trips (profitable -> loss)
        if state.round_trip_detected:
            pain = -state.round_trip_magnitude * 3.0 * self.weight
            self.confidence_score *= 0.9  # Confidence hit
            psychological_impact += pain
            
        # 3. Euphoria danger (overconfidence after wins)
        if state.consecutive_wins > 3:
            # Penalty for increasing size after win streaks
            if state.position_size > state.average_position_size * 1.5:
                psychological_impact -= 0.2 * self.weight
                
        # 4. Revenge trading penalty
        if state.quick_reentry_after_loss and state.time_since_loss < 60:
            psychological_impact -= 0.5 * self.weight
            
        # 5. FOMO penalty
        if state.chasing_extended_move and state.move_extension > 0.1:
            psychological_impact -= state.move_extension * 2.0 * self.weight
            
        return psychological_impact
```

### 5. Context-Aware Position Sizing

Not all setups are equal:

```python
class SetupQualityPositionSizing(RewardComponent):
    """Reward appropriate position sizing based on setup quality"""
    
    def calculate(self, state: RewardState) -> float:
        setup_quality = self.evaluate_setup_quality(state)
        position_size_score = state.position_size / state.max_position_size
        
        # A+ Setup characteristics:
        # - Fresh breakout from consolidation
        # - Volume surge > 3x average
        # - Sector/market alignment
        # - Clear catalyst
        # - Early in day (more time to work)
        
        if setup_quality > 0.8:  # A+ setup
            if position_size_score > 0.75:
                return 0.5 * self.weight  # Good - sizing up on quality
            else:
                return -0.2 * self.weight  # Missing opportunity
                
        elif setup_quality < 0.4:  # C setup
            if position_size_score < 0.25:
                return 0.3 * self.weight  # Good - small on weak setup
            elif position_size_score > 0.5:
                return -0.5 * self.weight  # Bad - too big on weak setup
                
        return 0.0
    
    def evaluate_setup_quality(self, state: RewardState) -> float:
        """Score setup quality from 0-1"""
        
        score = 0.0
        
        # Volume characteristics
        if state.volume_surge_ratio > 3.0:
            score += 0.2
        
        # Price action quality
        if state.breakout_from_range and state.range_duration > 1800:  # 30min+ range
            score += 0.2
            
        # Catalyst presence
        if state.catalyst_detected:
            score += 0.2
            
        # Time of day
        if state.time_of_day < time(10, 30):  # Before 10:30 AM
            score += 0.1
            
        # Sector momentum
        if state.sector_momentum_aligned:
            score += 0.1
            
        # Failed breakout history
        if state.previous_failed_breakouts == 0:
            score += 0.2
            
        return min(score, 1.0)
```

### 6. Advanced Episode Design

Episodes that create realistic trading scenarios:

```python
class ScenarioBasedEpisodeManager:
    """Create specific trading scenarios for comprehensive learning"""
    
    def __init__(self):
        self.scenarios = {
            'clean_momentum': {
                'description': 'Clean breakout with sustained momentum',
                'weight': 0.15,
                'key_learning': 'Riding winners, scaling out'
            },
            'failed_breakout': {
                'description': 'Breakout attempt that fails and reverses',
                'weight': 0.20,
                'key_learning': 'Quick loss cutting, setup recognition'
            },
            'choppy_grind': {
                'description': 'Slow grind up with multiple shakeouts',
                'weight': 0.15,
                'key_learning': 'Patience, position management'
            },
            'parabolic_blowoff': {
                'description': 'Parabolic move followed by crash',
                'weight': 0.15,
                'key_learning': 'Exhaustion recognition, profit taking'
            },
            'opening_drive': {
                'description': 'Strong open that fades all day',
                'weight': 0.10,
                'key_learning': 'Time-based edge, morning vs afternoon'
            },
            'squeeze_continuation': {
                'description': 'Multiple legs up with consolidations',
                'weight': 0.15,
                'key_learning': 'Re-entry, continuation patterns'
            },
            'news_catalyst': {
                'description': 'News-driven move with specific dynamics',
                'weight': 0.10,
                'key_learning': 'Catalyst-based trading'
            }
        }
        
    def generate_episode_sequence(self, num_episodes: int) -> List[Dict]:
        """Generate a balanced sequence of scenarios"""
        
        # Ensure all scenarios are covered
        # But weight towards failure scenarios for robustness
        
        # Also create compound scenarios:
        # - Failed breakout followed by successful reversal
        # - Morning squeeze into afternoon fade
        # - Multiple attempts before real breakout
        
        return episode_sequence
```

### 7. Continuous Improvement Metrics

Track what really matters:

```python
class MomentumMasteryMetrics:
    """Track advanced metrics for momentum trading mastery"""
    
    def __init__(self):
        self.metrics = {
            # Entry metrics
            'entry_timing_score': [],  # How early in moves
            'setup_quality_correlation': [],  # Position size vs setup quality
            
            # Management metrics  
            'scale_out_efficiency': [],  # How well timed exits are
            'runner_performance': [],  # P&L from last 25% of position
            'hold_time_vs_momentum_phase': [],  # Holding appropriate to phase
            
            # Risk metrics
            'failed_breakout_loss_avg': [],  # Avg loss on failed patterns
            'reversal_avoidance_rate': [],  # % of reversals avoided
            'max_giveback_percent': [],  # Max profit given back
            
            # Advanced metrics
            'tape_reading_accuracy': [],  # Correlation of tape signals to actions
            'regime_adaptation_score': [],  # Performance in different regimes
            'psychological_discipline': [],  # Avoiding revenge/FOMO trades
        }
        
    def generate_mastery_report(self) -> Dict:
        """Generate comprehensive mastery assessment"""
        
        return {
            'overall_mastery_level': self.calculate_mastery_score(),
            'strengths': self.identify_strengths(),
            'weaknesses': self.identify_weaknesses(),
            'improvement_priorities': self.suggest_improvements()
        }
```

### 8. Final Configuration

```yaml
# config/env/momentum_mastery.yaml
env:
  # Episode configuration
  episode_length: 3600  # 60 min standard
  scenario_based_episodes: true
  scenario_weights:
    clean_momentum: 0.15
    failed_breakout: 0.20  # Higher weight on difficult scenarios
    choppy_grind: 0.15
    parabolic_blowoff: 0.15
    opening_drive: 0.10
    squeeze_continuation: 0.15
    news_catalyst: 0.10
    
  # Reward system v4 - Mastery focused
  reward_v4:
    # Core components
    realized_pnl:
      weight: 1.0  # Reduced - other components more important
      
    # Phase-based rewards
    momentum_phase_alignment:
      weight: 2.0
      phase_detection_window: 60
      
    tape_reading:
      weight: 1.5
      features: ['block_trades', 'bid_ask_pressure', 'tape_speed']
      
    # Exit mastery
    scaled_exit:
      weight: 2.0
      optimal_scaling_patterns:
        breakout: [0.25, 0.25, 0.25, 0.25]
        momentum_run: [0.20, 0.30, 0.30, 0.20]
        exhaustion: [0.50, 0.30, 0.20, 0.0]
        
    runner_management:
      weight: 1.5
      runner_threshold: 0.25  # Last 25% of position
      
    # Psychological realism
    psychological_factors:
      weight: 1.0
      track_confidence: true
      track_regret: true
      
    # Setup quality
    position_sizing_quality:
      weight: 1.5
      quality_factors: ['volume', 'range', 'catalyst', 'time', 'sector']
      
    # Advanced penalties
    momentum_quality:
      weight: 1.0
      quality_metrics: ['volume_profile', 'price_structure', 'bid_support']
      
  # Metrics tracking
  advanced_metrics:
    track_tape_reading: true
    track_phase_transitions: true
    track_setup_quality: true
    track_psychological_state: true
    generate_mastery_reports: true
    report_frequency: 100  # Every 100 episodes
```

## Expected Mastery Outcomes

1. **Entry Precision**: Model learns to enter at momentum start, not chase
2. **Exit Artistry**: Scales out optimally based on momentum phase
3. **Loss Discipline**: Cuts failed breakouts immediately
4. **Tape Reading**: Responds to order flow, not just price
5. **Psychological Stability**: Avoids revenge trading, FOMO, overconfidence
6. **Context Awareness**: Different behavior for different times/setups
7. **Risk Calibration**: Sizes positions based on setup quality

## The Ultimate Goal

Create a model that trades like an experienced momentum trader:
- Waits patiently for A+ setups
- Strikes aggressively when opportunity appears
- Scales out intelligently as momentum evolves
- Cuts losses instantly when wrong
- Never holds through major reversals
- Adapts to different market regimes
- Maintains psychological discipline

This is not just about profit - it's about developing complete momentum trading mastery through experience.