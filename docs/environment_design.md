# Trading Environment Redesign V4 - Simple & Smart

## Core Philosophy

Keep it simple. One class, one responsibility. Smart defaults over configuration complexity. Focus on momentum trading reality.

## Design Goals

1. **Simplicity**: Each component does one thing well
2. **Flexibility**: Easy to extend without modifying core
3. **Realism**: Mirrors actual momentum trading scenarios
4. **Testability**: Components can be tested in isolation

## Architecture Overview

The system consists of four main components that work together to manage the trading environment lifecycle:

### 1. EpisodeScanner - Finding Good Starting Points

The EpisodeScanner is responsible for scanning available market data and identifying viable episode starting points. It understands market structure and ensures
episodes start with sufficient historical data for feature calculation.

**Key Responsibilities:**

- Determine when episodes should terminate
- Balance exploration (random selection) with coverage (sequential selection)
- Select appropriate episode starting points based on training progress
- Scan date ranges for viable starting points
- Ensure minimum warmup data availability (Enough to calculate features for lookback periods)
- Ensure minimum forward data for meaningful episodes (At least about max episode step size)
- Assign quality scores based on time of day and market conditions

**Episode Termination Conditions:**

- Time limit reached (MAX_EPISODE_LENGTH)
- Risk limit exceeded (MAX_LOSS)
- Market closure (MARKET_CLOSE)
- Max carries reached (MAX_CARRIES)

**Reset/Starting Point Selection**
- Use a mix of random and sequential selection to cover diverse market conditions
- Ensure starting points have enough forward data for meaningful episodes
- Ensure starting points have enough historical data for feature calculation
- Ensure starting points are not too close to each other to avoid redundant training
- Ensure starting points are not too far apart to avoid missing important market transitions
- 
- Assign quality scores to starting points based on time of day, configured market conditions, and historical volatility




### 3. OpenTradeHandler - Managing Positions Across Episodes

The OpenTradeHandler is dedicated to managing open positions when episodes end. This is crucial for realistic training as real trading doesn't have artificial
episode boundaries.

**Key Responsibilities:**

- Handle open positions based on termination reason
- Maintain position continuity across episodes
- Track carry-forward statistics

**Position Handling Strategy:**

- **Max Loss**: Force liquidation at the bid
- **Time Limit**: Carry position forward to next episode
- **Market Close**: Force liquidation at the bid
- **Max Carries**: Force close after 3 episode carries to prevent stale positions

### 4. EpisodeManager - Clean Orchestrator

The EpisodeManager ties everything together, providing a clean interface for the training loop while coordinating all components.

**Key Responsibilities:**

- Initialize training with available date ranges
- Orchestrate episode execution
- Coordinate position handling between episodes
- Collect and report episode statistics
- Interface with the trading environment and agent

**Episode Flow:**

1. Get next starting point from controller
2. Check for carried positions from trade handler
3. Reset environment with appropriate state
4. Run episode collecting experience
5. Handle any open positions at episode end
6. Report statistics for training metrics


### Progressive Difficulty

The system implements natural curriculum learning by starting with high-quality, liquid market periods and progressively introducing more challenging scenarios
as training advances.

### Realistic Position Management

Open positions can carry across episodes, reflecting real trading where you can't simply reset when convenient. This teaches the model to manage positions
through various market conditions.


## Implementation Strategy

1. Start with EpisodeScanner to identify viable trading periods
3. Implement OpenTradeHandler for position continuity
4. Create EpisodeManager to orchestrate everything
5. Integrate with existing TradingEnvironment
6. Add comprehensive testing for each component