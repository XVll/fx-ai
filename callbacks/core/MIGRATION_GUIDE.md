# Callback System Migration Guide

This guide helps you migrate from the old callback system to the new enhanced callback system.

## Overview of Changes

The new callback system provides:
- **30+ event hooks** (vs 4 in the old system)
- **Strongly-typed contexts** for each event
- **Priority-based execution**
- **Async event support**
- **Built-in state management**
- **Performance profiling**
- **Event filtering and routing**

## Quick Migration

### 1. Update Base Class

```python
# Old
from callbacks.core.base import BaseCallback

class MyCallback(BaseCallback):
    def __init__(self, config):
        super().__init__(config)

# New
from callbacks.core.base_v2 import BaseCallbackV2
from callbacks.core.events import EventPriority

class MyCallback(BaseCallbackV2):
    def __init__(self, config):
        super().__init__(
            name="MyCallback",
            priority=EventPriority.NORMAL,
            config=config
        )
```

### 2. Update Event Handlers

The new system uses specific context types for each event:

```python
# Old
def on_episode_end(self, context: EpisodeEndContext) -> None:
    episode_info = context.episode
    metrics = context.metrics

# New
def on_episode_end(self, context: EpisodeContext) -> None:
    # Direct access to all data
    reward = context.episode_reward
    pnl = context.total_pnl
    trades = context.trades
```

### 3. New Event Hooks

The new system provides many more hooks:

```python
# Step-level hooks
def on_step_start(self, context: StepContext) -> None:
    """Before environment step"""
    
def on_action_selected(self, context: StepContext) -> None:
    """After action selection"""
    
def on_reward_computed(self, context: StepContext) -> None:
    """After reward calculation"""

# Update hooks  
def on_gradient_computed(self, context: UpdateContext) -> None:
    """After gradients computed"""
    
def on_batch_start(self, context: BatchContext) -> None:
    """Before processing batch"""

# Data hooks
def on_day_switched(self, context: DataContext) -> None:
    """When switching trading day"""
```

## Feature Comparison

### State Management

```python
# Old - Manual state tracking
class OldCallback(BaseCallback):
    def __init__(self):
        self.episode_count = 0
        self.best_reward = float('-inf')
        
    def on_episode_end(self, context):
        self.episode_count += 1
        if context.episode.reward > self.best_reward:
            self.best_reward = context.episode.reward

# New - Built-in state management with persistence
class NewCallback(BaseCallbackV2):
    def __init__(self):
        super().__init__(
            state_dir=Path("outputs/callback_states")
        )
        
    def on_episode_end(self, context: EpisodeContext):
        # Automatic state tracking
        self.set_state('episode_count', 
                      self.get_state('episode_count', 0) + 1)
        
        # Or use StateManager for advanced features
        self.state_manager.increment('episode_count')
        self.state_manager.update_aggregate('reward', context.episode_reward)
```

### Performance Profiling

```python
# New - Automatic performance tracking
from callbacks.core.utils import profile_performance

class MyCallback(BaseCallbackV2):
    @profile_performance
    def on_episode_end(self, context: EpisodeContext):
        # Method execution time automatically tracked
        self.process_episode(context)
        
    def get_performance_report(self):
        return self.get_performance_stats()
```

### Event Filtering

```python
# New - Selective event listening
from callbacks.core.events import EventFilter, EventType

class MyCallback(BaseCallbackV2):
    def __init__(self):
        # Only listen to specific events
        event_filter = EventFilter(
            event_types={
                EventType.EPISODE_END,
                EventType.UPDATE_END
            }
        )
        super().__init__(event_filter=event_filter)
```

### Async Support

```python
# New - Async handlers for expensive operations
class MyCallback(BaseCallbackV2):
    def __init__(self):
        super().__init__(
            async_events={EventType.EPISODE_END}
        )
        
    async def on_episode_end_async(self, context: EpisodeContext):
        # Run expensive computation without blocking training
        results = await self.run_analysis(context)
        self.save_results(results)
```

## Migration Checklist

- [ ] Update imports from `base` to `base_v2`
- [ ] Add callback name and priority in `__init__`
- [ ] Update event handler signatures with new context types
- [ ] Replace manual state tracking with `StateManager`
- [ ] Add new event handlers for fine-grained control
- [ ] Use `@profile_performance` for performance-critical methods
- [ ] Add event filtering if not all events are needed
- [ ] Mark expensive operations as async
- [ ] Update callback manager usage to `CallbackManagerV2`

## Examples

### Complete Migration Example

```python
# Old callback
from callbacks.core.base import BaseCallback

class OldMetricsCallback(BaseCallback):
    def __init__(self, config):
        super().__init__(config)
        self.episode_rewards = []
        self.update_losses = []
        
    def on_episode_end(self, context):
        reward = context.episode.reward
        self.episode_rewards.append(reward)
        
        if len(self.episode_rewards) % 100 == 0:
            avg_reward = sum(self.episode_rewards[-100:]) / 100
            print(f"Avg reward: {avg_reward}")
            
    def on_update_end(self, context):
        self.update_losses.append(context.policy_loss)

# New callback with enhanced features
from callbacks.core.base_v2 import BaseCallbackV2
from callbacks.core.events import EventPriority, EventType
from callbacks.core.utils import MetricTracker, throttle
from callbacks.core.context_v2 import EpisodeContext, UpdateContext

class NewMetricsCallback(BaseCallbackV2):
    def __init__(self, config):
        super().__init__(
            name="EnhancedMetrics",
            priority=EventPriority.HIGH,
            state_dir=Path("outputs/metrics"),
            config=config
        )
        
        # Use built-in metric tracking
        self.metrics = MetricTracker(window_size=10000)
        
    def on_episode_end(self, context: EpisodeContext):
        # Rich context with all episode data
        self.metrics.record_episode('reward', context.episode_reward)
        self.metrics.record_episode('pnl', context.total_pnl)
        self.metrics.record_episode('trades', context.num_trades)
        
        # Automatic aggregation and statistics
        if context.global_episode % 100 == 0:
            stats = self.metrics.get_episode_stats('reward')
            self.logger.info(
                f"Episode {context.global_episode} - "
                f"Reward: {stats['mean']:.3f} ± {stats['std']:.3f} "
                f"(improving: {stats['improving']})"
            )
            
    def on_update_end(self, context: UpdateContext):
        # Track all losses automatically
        self.metrics.record('policy_loss', context.policy_loss)
        self.metrics.record('value_loss', context.value_loss)
        self.metrics.record('kl_divergence', context.kl_divergence)
        
    def on_gradient_computed(self, context: UpdateContext):
        # New hook for gradient analysis
        grad_info = context.get_gradient_info()
        for layer, stats in grad_info['layer_gradients'].items():
            self.metrics.record(f'grad_norm_{layer}', stats['norm'])
            
    @throttle(60.0)  # Once per minute
    def on_performance_log(self, context: BaseContext):
        # New hook for performance monitoring
        report = self.metrics.get_stats('policy_loss')
        self.logger.info(f"Loss trending: {report}")
```

## Best Practices

1. **Use specific event hooks** instead of doing everything in `on_episode_end`
2. **Leverage built-in utilities** like `MetricTracker` and `StateManager`
3. **Set appropriate priorities** for callback execution order
4. **Use async for expensive operations** to avoid blocking training
5. **Filter events** if you don't need all of them
6. **Profile performance** to identify bottlenecks
7. **Persist state** for fault tolerance and analysis

## Getting Help

- See `example_usage.py` for complete examples
- Check `hooks.py` for all available hooks and their documentation
- Use `HookRegistry.generate_callback_template()` to generate boilerplate
- Run `HookRegistry.validate_callback_implementation(YourCallback)` to check implementation