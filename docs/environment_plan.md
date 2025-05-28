# Todo :
* We need to teach about "HALT" status to model and include this in our environment.
* We also need a feature to handle "HALT" status in the environment.
* How do we teach the model about "HALT" status? We do not have LULD data, so we need to create a synthetic feature that simulates the "HALT" status.
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

### 1. MomentumEpisodeScanner (Runtime)

**Purpose**: Uses pre-built offline indices to efficiently select high-value training episodes.

**Key Features**:
- Loads pre-computed momentum indices from Parquet files
- No live scanning overhead - O(1) lookups from indexed data
- Categorizes market conditions (front side, back side, dead zones)
- Provides quality-based episode selection for curriculum training

**Offline Index Creation**:
- `MomentumScanner` runs as a separate offline process
- Scans entire symbol history to identify 10-30% movement days
- Creates indices with quality scores, reset points, and metadata
- Stores results in efficient Parquet format

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

### Phase 1: Scanner and Data Integration

**Offline Momentum Scanning (Implemented)**

The momentum scanner (`data/scanner/momentum_scanner.py`) runs as a separate offline process to build indices:

```python
class MomentumScanner:
    """Implemented offline scanner that builds momentum indices"""
    
    def scan_symbol(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Scan symbol history and identify momentum days"""
        
        momentum_days = []
        
        # Process databento files directly
        for file_path in self._get_symbol_files(symbol, start_date, end_date):
            day_data = self._process_file(file_path)
            
            if day_data is None:
                continue
                
            # Calculate intraday metrics
            day_metrics = self._calculate_day_metrics(day_data)
            
            # Only index high-movement days (10-30% moves)
            if 0.1 <= day_metrics['max_move'] <= 0.3:
                momentum_days.append({
                    'symbol': symbol,
                    'date': day_metrics['date'],
                    'quality_score': day_metrics['quality_score'],
                    'max_move': day_metrics['max_move'],
                    'volume_ratio': day_metrics['volume_ratio'],
                    'num_squeezes': day_metrics['num_squeezes'],
                    'best_period': day_metrics['best_period']
                })
                
        # Save to parquet indices
        return self._save_indices(pd.DataFrame(momentum_days))
```

**Runtime Episode Scanner (To Be Implemented)**

```python
class MomentumEpisodeScanner:
    """Runtime scanner that uses pre-built indices for O(1) lookups"""
    
    def __init__(self, index_path: str):
        # Load pre-computed indices from momentum scanner
        self.index_utils = MomentumIndexUtils(index_path)
        self.day_index = self.index_utils.load_index('MLGO')
        
    def get_training_days(self, symbol: str, stage: str = 'stage_1') -> List[Dict]:
        """Get curriculum-appropriate momentum days"""
        
        # Use curriculum selector for stage-based filtering
        selector = CurriculumBasedSelector(self.config)
        return selector.select_days(
            self.day_index,
            stage=stage,
            performance_metrics=self.performance_tracker.get_metrics()
        )
        
    def get_next_episode(self, symbol: str, current_episode: int) -> Dict:
        """Get next episode based on curriculum progression"""
        
        stage = self._get_curriculum_stage(current_episode)
        available_days = self.get_training_days(symbol, stage)
        
        # Select day based on quality and curriculum requirements
        return self._select_optimal_day(available_days, stage)
```

### Phase 2: Environment Modifications

```python
class TradingEnvironment:
    """Updated environment using MarketSimulator with pre-calculated features"""
    
    def __init__(self, config):
        self.config = config
        
        # Initialize DataManager with momentum index support
        self.data_manager = DataManager(
            provider=self._create_provider(),
            momentum_scanner=MomentumScanner(),
            preload_days=2
        )
        
        # MarketSimulator will be created per session
        self.current_market_sim = None
        self.next_market_sim = None
        
        # Episode management
        self.episode_scanner = MomentumEpisodeScanner(config.index_path)
        
    def setup_session(self, symbol: str, date: datetime):
        """Setup a new trading session with pre-calculated features"""
        
        # Create new MarketSimulator for this session
        self.current_market_sim = MarketSimulator(
            symbol=symbol,
            data_manager=self.data_manager,
            model_config=self.config.model,
            simulation_config=self.config.simulation
        )
        
        # Initialize day - this pre-calculates ALL features for entire day
        success = self.current_market_sim.initialize_day(date)
        
        if not success:
            raise ValueError(f"Failed to initialize {symbol} on {date}")
        
        # Load reset points from momentum index
        self.reset_points = self.data_manager.get_reset_points(symbol, date)
        self.current_reset_idx = 0
        
        # Log session info
        stats = self.current_market_sim.get_stats()
        self.logger.info(f"Session ready: {stats['total_seconds']} seconds, warmup: {stats['warmup_info']['has_warmup']}")
        
    def prepare_next_session(self, symbol: str, date: datetime):
        """Prepare next session in background for fast switching"""
        
        # Create MarketSimulator for next session
        self.next_market_sim = MarketSimulator(
            symbol=symbol,
            data_manager=self.data_manager,  # Shared data manager
            model_config=self.config.model,
            simulation_config=self.config.simulation
        )
        
        # Initialize in background thread for efficiency
        def background_init():
            success = self.next_market_sim.initialize_day(date)
            if success:
                self.logger.info(f"Next session ready: {symbol} {date}")
            else:
                self.logger.error(f"Failed to prepare next session: {symbol} {date}")
                
        # Could run in thread for true background processing
        background_init()
        
    def reset_at_point(self, reset_point: CategorizedResetPoint):
        """Reset to specific point within loaded day"""
        
        # No new data loading - use pre-loaded day
        self.current_time = reset_point.timestamp
        self.market_simulator.set_time(reset_point.timestamp)
        
        # Set episode boundaries
        self.episode_start = reset_point.timestamp
        self.episode_end = min(
            reset_point.timestamp + timedelta(hours=4),  # Max duration
            datetime.combine(reset_point.timestamp.date(), time(20, 0))  # Market close
        )
        
        # Apply momentum context
        self._apply_momentum_context(reset_point)
        
        # Get initial observation
        return self._get_observation()

    def step(self, action):
        """Execute action with awareness of reset boundaries"""
        
        # Normal step execution
        obs, reward, done, truncated, info = super().step(action)
        
        # Check if we should transition to next reset point
        if self._should_transition_reset():
            info['reset_transition'] = True
            done = True
            
        return obs, reward, done, truncated, info
        
    def _should_transition_reset(self):
        """Check if we should move to next reset point"""
        
        # Skip if no more reset points
        if self.current_reset_idx >= len(self.day_reset_points) - 1:
            return False
            
        next_reset = self.day_reset_points[self.current_reset_idx + 1]
        
        # Transition if we're close to next reset
        time_to_next = (next_reset.timestamp - self.current_time).total_seconds()
        return time_to_next <= 60  # Within 1 minute
```

### Phase 3: Training Manager

```python
class TrainingManager:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.data_manager = DataManager(config)
        self.episode_scanner = MomentumEpisodeScanner(config.index_path)
        self.position_handler = PositionHandler(config, logger)
        self.curriculum_selector = CurriculumBasedSelector(config)
        
    def run_training(self, agent, num_days: int):
        """Main training loop processing full days"""
        
        # Get available momentum days from index
        available_days = self.episode_scanner.get_training_days(
            self.config.symbol,
            quality_threshold=0.6
        )
        
        for day_idx in range(num_days):
            # Select day based on curriculum
            training_day = self._select_day_by_curriculum(
                available_days,
                day_idx,
                self.performance_tracker
            )
            
            # Load day once
            self.env.setup_day(training_day.symbol, training_day.date)
            
            # Train on all reset points in the day
            day_metrics = self._train_single_day(agent, training_day)
            
            # Update curriculum based on day performance
            self.curriculum_selector.update(day_metrics)
            
            # Pre-load next day in background
            if day_idx < num_days - 1:
                next_day = self._select_day_by_curriculum(
                    available_days, day_idx + 1, self.performance_tracker
                )
                self.data_manager.preload_day(next_day.symbol, next_day.date)
                
    def _train_single_day(self, agent, training_day):
        """Train on all reset points within a single day"""
        
        day_metrics = {
            'total_resets': len(self.env.day_reset_points),
            'completed_resets': 0,
            'total_pnl': 0,
            'positions_taken': 0
        }
        
        # Process each reset point
        for reset_idx, reset_point in enumerate(self.env.day_reset_points):
            # Reset at point
            obs = self.env.reset_at_point(reset_point)
            
            done = False
            episode_pnl = 0
            
            while not done:
                action = agent.predict(obs)
                obs, reward, done, truncated, info = self.env.step(action)
                episode_pnl += info.get('realized_pnl', 0)
                
                # Handle reset transitions
                if info.get('reset_transition', False):
                    self._handle_reset_transition(reset_idx)
                    break
                    
            day_metrics['completed_resets'] += 1
            day_metrics['total_pnl'] += episode_pnl
            
            # Log reset point completion
            self.logger.info(
                f"Reset {reset_idx}/{len(self.env.day_reset_points)} complete: "
                f"PnL={episode_pnl:.4f}, Phase={reset_point.momentum_phase}"
            )
            
        return day_metrics
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

## Implementation Status

**Completed**:
- ✅ Offline momentum scanner (`data/scanner/momentum_scanner.py`)
- ✅ Enhanced DataManager with 2-tier caching and momentum index (`data/data_manager.py`)
- ✅ **MarketSimulator with pre-calculated features** (`simulators/market_simulator.py`)
- ✅ **FeatureExtractor for pre-calculated features** (`feature/feature_extractor.py`)
- ✅ Unified data provider interface (`data/provider/data_provider.py`)
- ✅ Index utilities for momentum day queries (`data/utils/index_utils.py`)
- ✅ **Warmup data handling for early morning features**
- ✅ **Uniform timeline with forward-filling for missing data**

**Key Architecture Changes**:
- ✅ **Pre-calculated Features**: All features (HF/MF/LF/Static) calculated once during `initialize_day()`
- ✅ **O(1) Feature Access**: No feature calculation during training - just array lookups
- ✅ **Warmup Data Support**: Previous day data loaded automatically for early morning training
- ✅ **Uniform Timeline**: Every second has valid market state and features, even with sparse data
- ✅ **Smart Data Management**: Environment uses DataManager for momentum index, MarketSim for execution

**Next Steps**:
- 🔄 Environment integration with MarketSimulator
- 🔄 Training manager with momentum index-based day selection
- 🔄 Position handler for episode boundaries
- 🔄 Curriculum integration with quality scores

## Monitoring Metrics

- Reset point quality distribution
- Performance by momentum phase
- Average holding time by pattern type
- Win rate during different market conditions
- Curriculum progression effectiveness

This design creates a training environment specifically optimized for low-float momentum trading, teaching the model to act like a sniper - waiting for high-probability setups and executing quick, profitable trades during squeeze events.