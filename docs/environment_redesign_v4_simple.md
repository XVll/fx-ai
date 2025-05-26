# Trading Environment Redesign V4 - Simple & Smart

## Core Philosophy
Keep it simple. One class, one responsibility. Smart defaults over configuration complexity.

## Architecture Overview

```
EpisodeScanner → EpisodeController → EpisodeManager
                         ↓
                  OpenTradeHandler
```

## 1. EpisodeScanner - Finding Good Starting Points

Simple component that scans available data and identifies viable episode starting points.

```python
class EpisodeScanner:
    """Scans data to find good episode starting points"""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.min_warmup_seconds = 300  # 5 minutes minimum
        self.min_episode_length = 1800  # 30 minutes minimum
        
    def scan_date_range(self, start_date, end_date):
        """Scan date range and return viable episode starting points"""
        points = []
        
        current = start_date
        while current <= end_date:
            # Get market hours for this day
            market_open = current.replace(hour=9, minute=30)
            market_close = current.replace(hour=16, minute=0)
            
            # Scan every 5 minutes during market hours
            scan_time = market_open
            while scan_time < market_close:
                if self._is_viable_start(scan_time):
                    points.append({
                        'timestamp': scan_time,
                        'quality': self._assess_quality(scan_time)
                    })
                scan_time += timedelta(minutes=5)
                
            current += timedelta(days=1)
            
        return points
    
    def _is_viable_start(self, timestamp):
        """Check if this timestamp is a viable starting point"""
        # Check we have enough future data
        future_data = self.data_manager.get_data_length_after(timestamp)
        if future_data < self.min_episode_length:
            return False
            
        # Check we have warmup data
        past_data = self.data_manager.get_data_length_before(timestamp)
        if past_data < self.min_warmup_seconds:
            return False
            
        return True
    
    def _assess_quality(self, timestamp):
        """Simple quality score based on time of day"""
        hour = timestamp.hour
        
        # Best times: 9:45-11:30 AM and 2:00-3:30 PM
        if (9.75 <= hour <= 11.5) or (14 <= hour <= 15.5):
            return 1.0  # High quality
        # Good times: Rest of regular hours
        elif 9.5 <= hour <= 16:
            return 0.7  # Medium quality
        # Okay times: Extended hours
        else:
            return 0.3  # Low quality
```

## 2. EpisodeController - Unified Start/Reset Logic

Single class that handles both starting and resetting episodes intelligently.

```python
class EpisodeController:
    """Unified controller for episode lifecycle"""
    
    def __init__(self, scanner: EpisodeScanner):
        self.scanner = scanner
        self.episode_points = []
        self.current_index = 0
        self.training_progress = 0.0
        
    def initialize(self, date_range):
        """Initialize with available episode points"""
        self.episode_points = self.scanner.scan_date_range(
            date_range['start'], 
            date_range['end']
        )
        self._sort_by_difficulty()
        
    def get_next_episode_start(self, training_progress):
        """Get next episode starting point based on training progress"""
        self.training_progress = training_progress
        
        if training_progress < 0.3:
            # Early training: High quality points only
            candidates = [p for p in self.episode_points if p['quality'] >= 0.7]
        elif training_progress < 0.7:
            # Mid training: Mix of qualities
            candidates = [p for p in self.episode_points if p['quality'] >= 0.3]
        else:
            # Late training: All points (including challenging ones)
            candidates = self.episode_points
            
        # Simple selection with some randomness
        if not candidates:
            candidates = self.episode_points  # Fallback
            
        # 80% random, 20% sequential for coverage
        if random.random() < 0.8:
            point = random.choice(candidates)
        else:
            # Sequential to ensure coverage
            point = candidates[self.current_index % len(candidates)]
            self.current_index += 1
            
        return point['timestamp']
    
    def should_continue_episode(self, env_state):
        """Decide if episode should continue or reset"""
        # Simple rules for episode continuation
        if env_state['steps'] >= env_state['max_steps']:
            return False  # Time limit
        if env_state['capital'] <= 0:
            return False  # Bankruptcy
        if env_state['drawdown'] > 0.2:  # 20% drawdown
            return False  # Risk limit
        return True
        
    def _sort_by_difficulty(self):
        """Sort points by difficulty (quality inverse)"""
        self.episode_points.sort(key=lambda x: -x['quality'])
```

## 3. OpenTradeHandler - Managing Positions Across Episodes

Dedicated class for handling open positions when episodes end.

```python
class OpenTradeHandler:
    """Handles open positions across episode boundaries"""
    
    def __init__(self):
        self.carried_positions = {}  # symbol -> position data
        
    def handle_episode_end(self, symbol, position, reason):
        """Process open position at episode end"""
        if position['size'] == 0:
            return None
            
        if reason == 'bankruptcy' or reason == 'max_loss':
            # Force liquidation
            return self._force_liquidate(position)
            
        elif reason == 'time_limit':
            # Carry forward
            self._save_position(symbol, position)
            return {'action': 'carried', 'position': position}
            
        elif reason == 'data_end':
            # Mark to market
            return self._mark_to_market(position)
            
    def get_carried_position(self, symbol):
        """Retrieve carried position for symbol"""
        return self.carried_positions.pop(symbol, None)
        
    def _save_position(self, symbol, position):
        """Save position for next episode"""
        self.carried_positions[symbol] = {
            'size': position['size'],
            'avg_price': position['avg_price'],
            'unrealized_pnl': position['unrealized_pnl'],
            'carry_count': position.get('carry_count', 0) + 1
        }
        
    def _force_liquidate(self, position):
        """Simulate forced liquidation"""
        # Assume 2% slippage on liquidation
        slippage = 0.02
        if position['size'] > 0:
            exit_price = position['current_price'] * (1 - slippage)
        else:
            exit_price = position['current_price'] * (1 + slippage)
            
        pnl = (exit_price - position['avg_price']) * position['size']
        return {
            'action': 'liquidated',
            'pnl': pnl,
            'exit_price': exit_price
        }
        
    def _mark_to_market(self, position):
        """Calculate final value without closing"""
        return {
            'action': 'marked',
            'value': position['unrealized_pnl']
        }
```

## 4. EpisodeManager - Clean Orchestrator

Simple manager that ties everything together.

```python
class EpisodeManager:
    """Orchestrates episode lifecycle during training"""
    
    def __init__(self, env, config):
        self.env = env
        self.scanner = EpisodeScanner(env.data_manager)
        self.controller = EpisodeController(self.scanner)
        self.trade_handler = OpenTradeHandler()
        
        # Simple config
        self.max_steps_per_episode = config.get('max_steps', 3600)
        self.episodes_per_epoch = config.get('episodes_per_epoch', 10)
        
    def initialize_training(self, date_range):
        """Setup for training run"""
        self.controller.initialize(date_range)
        
    def run_epoch(self, epoch_num, total_epochs):
        """Run one training epoch"""
        training_progress = epoch_num / total_epochs
        epoch_stats = []
        
        for ep in range(self.episodes_per_epoch):
            # Get starting point
            start_time = self.controller.get_next_episode_start(training_progress)
            
            # Check for carried position
            carried = self.trade_handler.get_carried_position(self.env.symbol)
            
            # Reset environment
            obs, info = self.env.reset(
                start_time=start_time,
                carried_position=carried
            )
            
            # Run episode
            episode_stats = self._run_episode(obs)
            
            # Handle any open position
            if episode_stats['final_position'] != 0:
                trade_result = self.trade_handler.handle_episode_end(
                    self.env.symbol,
                    episode_stats['final_position'],
                    episode_stats['termination_reason']
                )
                episode_stats['position_handling'] = trade_result
                
            epoch_stats.append(episode_stats)
            
        return epoch_stats
        
    def _run_episode(self, initial_obs):
        """Run single episode"""
        obs = initial_obs
        total_reward = 0
        steps = 0
        
        while steps < self.max_steps_per_episode:
            # This is where agent would act
            # action = agent.act(obs)
            # For now, placeholder
            action = 0  
            
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            steps += 1
            
            # Check if should continue
            env_state = {
                'steps': steps,
                'max_steps': self.max_steps_per_episode,
                'capital': self.env.portfolio.capital,
                'drawdown': self.env.portfolio.current_drawdown
            }
            
            if not self.controller.should_continue_episode(env_state):
                truncated = True
                
            if terminated or truncated:
                return {
                    'total_reward': total_reward,
                    'steps': steps,
                    'termination_reason': info.get('reason', 'time_limit'),
                    'final_position': {
                        'size': self.env.position,
                        'avg_price': self.env.avg_entry_price,
                        'current_price': self.env.current_price,
                        'unrealized_pnl': self.env.unrealized_pnl
                    }
                }
                
            obs = next_obs
            
        # Episode ended by time limit
        return {
            'total_reward': total_reward,
            'steps': steps,
            'termination_reason': 'time_limit',
            'final_position': {
                'size': self.env.position,
                'avg_price': self.env.avg_entry_price,
                'current_price': self.env.current_price,
                'unrealized_pnl': self.env.unrealized_pnl
            }
        }
```

## Usage Example

```python
# In main.py or training script
def train_model(config):
    # Setup
    env = TradingEnvironment(config)
    agent = PPOAgent(config)
    episode_manager = EpisodeManager(env, config)
    
    # Initialize with date range
    episode_manager.initialize_training({
        'start': datetime(2025, 1, 1),
        'end': datetime(2025, 3, 31)
    })
    
    # Training loop
    for epoch in range(num_epochs):
        # Run epoch
        epoch_stats = episode_manager.run_epoch(epoch, num_epochs)
        
        # Update agent with collected experience
        agent.update()
        
        # Log statistics
        log_epoch_stats(epoch, epoch_stats)
```

## Key Improvements

1. **Unified Design**: EpisodeController handles both start selection and reset logic in one place
2. **Simple Scanner**: EpisodeScanner just identifies viable points with quality scores
3. **Dedicated Trade Handler**: OpenTradeHandler focuses solely on position management
4. **Clean Orchestration**: EpisodeManager provides simple, clear flow

## Configuration

```yaml
# Minimal configuration needed
episode:
  max_steps: 3600           # 1 hour episodes
  episodes_per_epoch: 10    # 10 episodes per training epoch
  
  # Scanner settings
  scanner:
    min_warmup: 300         # 5 minutes
    min_episode_length: 1800 # 30 minutes
    
  # Position handling
  position_handling:
    carry_forward: true     # Allow positions to carry
    max_carry_episodes: 3   # Maximum carries before force close
```

## Summary

This design is:
- **Simple**: Each component has one clear responsibility
- **Smart**: Intelligent defaults, minimal configuration
- **Practical**: Handles real trading scenarios without over-engineering
- **Flexible**: Easy to extend or modify individual components

The key insight is that we don't need complex strategies and multiple configuration options. Simple rules with smart defaults work better than over-engineered solutions.