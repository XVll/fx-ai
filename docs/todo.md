
  Based on the configuration, existing code, and our design discussion, here are ALL the requirements:

  1. TrainingManager Requirements

  Core Training Loop

  - Single main training loop (no nested loops)
  - Global termination conditions:
    - termination_max_episodes - stop after X total episodes
    - termination_max_updates - stop after X total policy updates
    - termination_max_cycles - stop after X complete data cycles
  - Intelligent termination:
  -  # These type of terminations will be applied by callbacks we wont implement but will allow it
    - plateau_patience -  
    - degradation_threshold - 

  State Management

  - Track global counters: total_episodes, total_updates, total_cycles
  - Pass context parameters: device, output_path, run_id, start_timestamp
  - Counters should be tracked by Training manager, not trainer.

  Component Coordination

  - Delegate day/reset point management to EpisodeManager
  - Request episodes via episode_manager.get_next_episode()
  - Setup environment with episode context
  - Collect rollouts via trainer
  - Handle updates when buffer ready
  - Trigger callbacks at appropriate points

  2. EpisodeManager Requirements

  Day Management

  - Day selection modes: sequential, quality, random
  - Day filtering:
    - Filter by symbols list
    - Filter by date_range [start, end]
    - Filter by day_score_range [min, max]
  - Day cycling: Track used days, avoid repetition (except random mode)

  Reset Point Management

  - Reset point selection modes: sequential, quality, random
  - Reset point filtering:
    - Filter by reset_roc_range [min, max]
    - Filter by reset_activity_range [min, max]
    - Respect quality criteria from day filtering
  - Reset point cycling: Cycle through all points, then start over

  Daily Limits (when to switch days)

  - daily_max_episodes - max episodes per day before switching
  - daily_max_updates - max updates per day before switching
  - daily_max_cycles - max complete cycles through reset points before switching

  State and Interface

  - Maintain own loop/generator for days and reset points
  - Provide get_next_episode() → returns EpisodeContext or None
  - Handle on_episodes_completed(count, metrics) notifications
  - Handle on_update_completed(update_info) notifications
  - Track daily counters internally

  3. Environment Requirements

  Episode Management

  - Episode limits: episode_max_steps - max steps before episode reset
  - Setup episode with EpisodeContext (symbol, date, reset_point)
  - Reset at specific reset points via reset_at_point(index)
  - Handle episode termination (natural vs step limit)

  Integration

  - Work with existing TradingEnvironment class
  - Support setup_session(symbol, date) for day setup
  - Maintain compatibility with current portfolio/reward systems

  4. Trainer Requirements

  Rollout Collection

  - Target rollout steps: rollout_steps = 2048 (from config)
  - Collect steps across episode boundaries (don't stop mid-rollout for episode ends)
  - Handle partial episodes gracefully
  - Return RolloutResult with steps collected, episodes completed, buffer status

  Buffer Management

  - Accumulate data until buffer has enough for update (rollout_steps)
  - Trigger buffer_ready = True when ready for policy update
  - Clear/reset buffer after each update

  Policy Updates

  - Perform PPO updates when buffer ready
  - Return update metrics and information
  - Handle learning rate, clip epsilon, and other hyperparameters

  5. Context and Callback Requirements

  Context Creation

  - TrainingStartContext: config, trainer, environment, model, device, output_path, run_id, timestamp
  - EpisodeEndContext: episode info, trading metrics, portfolio state, trades, model info
  - UpdateEndContext: update info, losses, training metrics, model info
  - TrainingEndContext: final metrics, duration, termination reason

  Callback Integration

  - Trigger callbacks at appropriate lifecycle points
  - Handle callback errors gracefully (don't crash training)
  - Support all existing callback types (metrics, checkpoint, wandb, etc.)

  6. Data Management Requirements

  Integration with Existing Systems

  - Use existing DataManager and MomentumScanner
  - Load momentum days with filtering
  - Get reset points for selected days
  - Handle data caching and preloading

  Data Flow

  - EpisodeManager queries DataManager for available days
  - EpisodeManager gets reset points for selected days
  - Environment loads actual market data for trading sessions

  7. Configuration Requirements

  All Config Parameters Must Work

  - Training modes: training, optuna, benchmark
  - Termination: All termination_max_* and intelligent termination
  - Episode limits: episode_max_steps
  - Daily limits: All daily_max_* parameters
  - Selection modes: Both day and reset point selection modes
  - Filtering ranges: All score and quality ranges
  - Model management: Continue training, checkpointing, best model tracking

  8. State Persistence and Resumability

  State Saving/Loading

  - Save training state for resumability
  - Load previous model with training counters
  - Handle version management (v1, v2, v3...)
  - Track episode manager state across restarts

  9. Error Handling and Edge Cases

  Graceful Degradation

  - Handle data exhaustion (no more days/reset points)
  - Handle quality criteria not met
  - Handle preloading failures
  - Clean shutdown on errors or interruption

  Logging and Observability

  - Progress tracking at all levels
  - Clear state transitions logging
  - Performance metrics
  - Debug information for troubleshooting

  10. Performance Requirements

  Efficiency

  - No blocking operations in main loop
  - Efficient data loading and caching
  - Minimal memory overhead for state tracking
  - Fast episode transitions

  This covers ALL requirements from the configuration, existing code patterns, and our architectural decisions. Each requirement
  maps to specific config parameters and should be testable independently.

