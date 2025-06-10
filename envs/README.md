# V2 Trading Environment

## Overview

The V2 TradingEnvironment is a clean rewrite focusing on single episode execution with proper separation of concerns. It integrates seamlessly with the V2 architecture where:

- **TrainingManager** orchestrates the overall training process
- **EpisodeManager** handles episode selection, reset points, and day cycling
- **TradingEnvironment** focuses purely on single episode mechanics
- **PPOTrainer** handles the RL algorithm and model updates

## Key Improvements

### 1. Clean Separation of Concerns
- Episode management logic moved to EpisodeManager
- Environment focuses only on step execution and observation generation
- No more mixed responsibilities

### 2. Type-Safe Implementation
- Proper type hints throughout
- Clear data structures (EpisodeConfig)
- Type-safe action/observation handling

### 3. Improved Action Masking
- Clean ActionMask implementation
- Efficient validation logic
- Clear encoding/decoding methods

### 4. Simplified State Management
- Clear episode lifecycle
- Proper termination handling
- Clean observation generation

## Architecture

```
TrainingManager
    ├── EpisodeManager (handles episode selection)
    ├── PPOTrainer (handles RL algorithm)
    └── TradingEnvironment (handles single episode)
            ├── MarketSimulator (market data)
            ├── ExecutionSimulator (order execution)
            ├── PortfolioSimulator (portfolio state)
            ├── RewardSystem (reward calculation)
            └── ActionMask (action validation)
```

## Usage

### Basic Setup

```python
from envs import TradingEnvironment, EpisodeConfig
from data.data_manager import DataManager

# Create environment
env = TradingEnvironment(
    config=config,
    data_manager=data_manager,
    callback_manager=callback_manager
)

# Setup episode
episode_config = EpisodeConfig(
    symbol="AAPL",
    date=datetime(2024, 1, 15),
    start_time=datetime(2024, 1, 15, 14, 30),  # 9:30 AM ET in UTC
    end_time=datetime(2024, 1, 15, 16, 30),  # 11:30 AM ET in UTC
    max_steps=1000,
)

env.setup_episode(episode_config)

# Reset and run episode
obs, info = env.reset()

for _ in range(1000):
    action = agent.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break
```

### V2 Compatibility Mode

For backward compatibility with existing code:

```python
# Old style setup (adapter methods)
env.setup_session("AAPL", "2024-01-15")
obs, info = env.reset_at_point(reset_point_idx=0)

# Works the same as:
episode_config = EpisodeConfig(...)
env.setup_episode(episode_config)
obs, info = env.reset()
```

### Action Space

The environment uses a discrete action space with 12 possible actions:
- 3 action types (HOLD, BUY, SELL)
- 4 position sizes (25%, 50%, 75%, 100%)

Actions can be provided in multiple formats:
```python
# As tuple
action = (0, 1)  # HOLD with 50% size

# As array
action = np.array([0, 1])

# As linear index
action = 1  # Will be decoded to (0, 1)
```

### Action Masking

The environment provides dynamic action masking:

```python
# Get current valid actions
mask = env.get_action_mask()

# Use with agent
action_probs = agent.get_action_probs(obs)
masked_probs = env.action_mask.mask_action_probabilities(
    action_probs, portfolio_state, market_state
)
```

## Key Differences from V1

1. **No Episode Management**: The environment no longer manages episode selection or reset points. This is handled by EpisodeManager.

2. **Clean Interfaces**: Clear separation between setup (episode configuration) and execution (step/reset).

3. **Type Safety**: Extensive use of type hints and structured data (EpisodeConfig, etc).

4. **Simplified State**: The environment maintains minimal state, focusing on current episode execution.

5. **Better Error Handling**: Clear error messages and proper exception handling.

## Testing

Run the tests to verify integration:

```bash
# Unit tests
pytest v2/tests/environment/test_trading_environment.py -v

# Integration test
python v2/tests/test_integration.py
```

## Migration Guide

To migrate from the old TradingEnvironment:

1. **Episode Management**: Move any episode selection logic to use EpisodeManager
2. **Session Setup**: Use EpisodeConfig instead of separate session/reset calls
3. **State Access**: Use `get_current_state()` for PPOTrainer integration
4. **Action Masking**: Use the new ActionMask class methods

The compatibility methods (`setup_session`, `reset_at_point`) are provided for gradual migration but should be replaced with the new interfaces.