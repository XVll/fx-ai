"""
Example usage of the enhanced callback system.

This file demonstrates how to create advanced callbacks using the new
comprehensive event system.
"""

import numpy as np
from typing import Dict, Any, List
from pathlib import Path

from .base_v2 import BaseCallbackV2
from .events import EventType, EventPriority, EventFilter
from .context_v2 import (
    BaseContext, StepContext, EpisodeContext, UpdateContext,
    BatchContext, ModelContext, CustomContext
)
from .utils import (
    StateManager, MetricTracker, requires_components,
    profile_performance, throttle, event_handler
)


class AdvancedMetricsCallback(BaseCallbackV2):
    """
    Example callback showing advanced metric tracking and analysis.
    
    Demonstrates:
    - State management with persistence
    - Fine-grained event handling
    - Performance tracking
    - Metric aggregation
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            name="AdvancedMetrics",
            priority=EventPriority.HIGH,
            state_dir=Path(config.get('output_dir', 'outputs')) / 'callback_states',
            config=config
        )
        
        # State management
        self.state_manager = StateManager(self.name, self.state_dir)
        self.metric_tracker = MetricTracker(window_size=10000)
        
        # Configuration
        self.log_frequency = config.get('log_frequency', 100)
        self.track_gradients = config.get('track_gradients', True)
        self.track_features = config.get('track_features', False)
        
        # Initialize tracking
        self._step_times = []
        self._episode_data = []
    
    @profile_performance
    def on_training_start(self, context: BaseContext) -> None:
        """Initialize tracking at training start."""
        self.logger.info("Starting advanced metrics tracking")
        
        # Initialize state
        self.state_manager.set('training_start_time', context.timestamp)
        self.state_manager.set('total_episodes', 0)
        self.state_manager.set('total_updates', 0)
        
        # Analyze model architecture
        if context.trainer:
            model = context.get_model()
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            self.logger.info(f"Model: {total_params:,} total params, {trainable_params:,} trainable")
            self.state_manager.set('model_params', total_params)
    
    @requires_components('environment', 'portfolio_simulator')
    def on_step_end(self, context: StepContext) -> None:
        """Track step-level metrics."""
        # Track key metrics
        self.metric_tracker.record('reward', context.reward)
        self.metric_tracker.record('portfolio_value', context.portfolio_value)
        
        # Track action distribution
        action_key = f'action_{context.action}'
        self.state_manager.increment(action_key)
        
        # Track spread impact
        if context.spread is not None:
            self.metric_tracker.record('spread', context.spread)
            
            # Analyze spread impact on trades
            if context.action in [1, 2]:  # Buy or Sell
                spread_cost = context.spread * abs(context.position)
                self.state_manager.update_aggregate('spread_cost', spread_cost)
    
    @throttle(0.1)  # Max 10 times per second
    def on_action_selected(self, context: StepContext) -> None:
        """Analyze policy decisions."""
        if context.action_probs is not None:
            # Calculate policy entropy
            entropy = -np.sum(context.action_probs * np.log(context.action_probs + 1e-8))
            self.metric_tracker.record('policy_entropy', entropy)
            
            # Track confidence
            max_prob = np.max(context.action_probs)
            self.metric_tracker.record('action_confidence', max_prob)
    
    def on_episode_end(self, context: EpisodeContext) -> None:
        """Comprehensive episode analysis."""
        # Update counters
        total_episodes = self.state_manager.increment('total_episodes')
        
        # Track episode metrics
        self.metric_tracker.record_episode('reward', context.episode_reward)
        self.metric_tracker.record_episode('length', context.episode_length)
        self.metric_tracker.record_episode('pnl', context.total_pnl)
        self.metric_tracker.record_episode('trades', context.num_trades)
        
        # Analyze trading performance
        if context.num_trades > 0:
            self.metric_tracker.record_episode('win_rate', context.win_rate)
            self.metric_tracker.record_episode('avg_trade_pnl', context.avg_trade_pnl)
            
            # Track best episodes
            if context.episode_reward > self.state_manager.get('best_reward', float('-inf')):
                self.state_manager.set('best_reward', context.episode_reward)
                self.state_manager.set('best_episode', {
                    'num': context.episode_num,
                    'symbol': context.symbol,
                    'date': context.date.isoformat(),
                    'reward': context.episode_reward,
                    'pnl': context.total_pnl,
                    'trades': context.num_trades
                })
        
        # Periodic logging
        if total_episodes % self.log_frequency == 0:
            self._log_episode_stats()
    
    def on_update_end(self, context: UpdateContext) -> None:
        """Track training dynamics."""
        self.state_manager.increment('total_updates')
        
        # Track losses
        self.metric_tracker.record('policy_loss', context.policy_loss)
        self.metric_tracker.record('value_loss', context.value_loss)
        self.metric_tracker.record('total_loss', context.total_loss)
        
        # Track training stability metrics
        self.metric_tracker.record('kl_divergence', context.kl_divergence)
        self.metric_tracker.record('clip_fraction', context.clip_fraction)
        self.metric_tracker.record('explained_variance', context.explained_variance)
        
        # Gradient analysis
        if self.track_gradients and context.gradient_norm > 0:
            self.metric_tracker.record('gradient_norm', context.gradient_norm)
            
            # Check for gradient issues
            if context.gradient_norm > 100:
                self.logger.warning(f"High gradient norm: {context.gradient_norm:.2f}")
            elif context.gradient_norm < 0.0001:
                self.logger.warning(f"Very low gradient norm: {context.gradient_norm:.6f}")
        
        # Detailed gradient analysis
        if self.track_gradients and context.update_num % 50 == 0:
            grad_info = context.get_gradient_info()
            if grad_info:
                self._analyze_gradients(grad_info)
    
    def on_model_saved(self, context: ModelContext) -> None:
        """Track model checkpoints."""
        self.state_manager.set('last_checkpoint', {
            'version': context.model_version,
            'path': str(context.checkpoint_path),
            'reward': context.best_reward,
            'timestamp': context.timestamp.isoformat()
        })
        
        # Save callback state
        self.state_manager.save()
        self.logger.info(f"Model v{context.model_version} saved with reward {context.best_reward:.3f}")
    
    def on_training_end(self, context: BaseContext) -> None:
        """Generate final report."""
        # Calculate training duration
        start_time = self.state_manager.get('training_start_time')
        duration = context.timestamp - start_time
        
        # Get final stats
        total_episodes = self.state_manager.get('total_episodes', 0)
        total_updates = self.state_manager.get('total_updates', 0)
        best_reward = self.state_manager.get('best_reward', 0)
        
        # Generate comprehensive report
        report = {
            'duration': str(duration),
            'total_episodes': total_episodes,
            'total_updates': total_updates,
            'best_reward': best_reward,
            'episode_metrics': self._get_episode_summary(),
            'update_metrics': self._get_update_summary(),
            'action_distribution': self._get_action_distribution(),
            'performance': self.get_performance_stats()
        }
        
        # Save report
        report_path = self.state_dir / f"{self.name}_final_report.json"
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Training completed: {total_episodes} episodes, best reward: {best_reward:.3f}")
        self.logger.info(f"Report saved to {report_path}")
        
        # Save final state
        self.state_manager.save()
    
    def _log_episode_stats(self) -> None:
        """Log periodic episode statistics."""
        episode_stats = self.metric_tracker.get_episode_stats('reward')
        
        if episode_stats:
            self.logger.info(
                f"Episode Stats - "
                f"Mean Reward: {episode_stats['mean']:.3f} ± {episode_stats['std']:.3f}, "
                f"Max: {episode_stats['max']:.3f}, "
                f"Improving: {episode_stats.get('improving', False)}"
            )
        
        # Log action distribution
        action_dist = self._get_action_distribution()
        if action_dist:
            self.logger.info(f"Action Distribution: {action_dist}")
    
    def _analyze_gradients(self, grad_info: Dict[str, Any]) -> None:
        """Analyze gradient flow through network."""
        layer_grads = grad_info.get('layer_gradients', {})
        
        # Check for vanishing/exploding gradients
        for layer_name, stats in layer_grads.items():
            self.state_manager.append_time_series(
                f'grad_norm_{layer_name}',
                stats['norm']
            )
            
            # Track gradient flow issues
            if stats['norm'] < 1e-6:
                self.logger.warning(f"Vanishing gradient in {layer_name}: {stats['norm']:.2e}")
            elif stats['norm'] > 1000:
                self.logger.warning(f"Exploding gradient in {layer_name}: {stats['norm']:.2e}")
    
    def _get_episode_summary(self) -> Dict[str, Any]:
        """Generate episode metrics summary."""
        metrics = {}
        
        for metric_name in ['reward', 'pnl', 'length', 'trades', 'win_rate']:
            stats = self.metric_tracker.get_episode_stats(metric_name)
            if stats:
                metrics[metric_name] = stats
        
        return metrics
    
    def _get_update_summary(self) -> Dict[str, Any]:
        """Generate update metrics summary."""
        metrics = {}
        
        for metric_name in ['policy_loss', 'value_loss', 'kl_divergence', 'gradient_norm']:
            stats = self.metric_tracker.get_stats(metric_name)
            if stats:
                metrics[metric_name] = stats
        
        return metrics
    
    def _get_action_distribution(self) -> Dict[str, float]:
        """Calculate action distribution percentages."""
        total_actions = 0
        action_counts = {}
        
        # Collect action counts
        for i in range(12):  # Assuming 12 actions
            count = self.state_manager.get(f'action_{i}', 0)
            if count > 0:
                action_counts[f'action_{i}'] = count
                total_actions += count
        
        # Convert to percentages
        if total_actions > 0:
            return {k: (v / total_actions * 100) for k, v in action_counts.items()}
        
        return {}


class AttributionAnalysisCallback(BaseCallbackV2):
    """
    Example callback for feature attribution analysis.
    
    Demonstrates:
    - Async event handling for expensive computations
    - Custom event integration
    - Selective event filtering
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Only listen to specific events
        event_filter = EventFilter(
            event_types={
                EventType.EPISODE_END,
                EventType.UPDATE_END,
                EventType.CUSTOM
            }
        )
        
        # Mark expensive operations as async
        async_events = {EventType.EPISODE_END}
        
        super().__init__(
            name="AttributionAnalysis",
            priority=EventPriority.LOW,  # Run after other callbacks
            event_filter=event_filter,
            async_events=async_events,
            config=config
        )
        
        self.analysis_frequency = config.get('analysis_frequency', 100)
        self.methods = config.get('methods', ['integrated_gradients'])
    
    async def on_episode_end_async(self, context: EpisodeContext) -> None:
        """Perform async attribution analysis."""
        if context.global_episode % self.analysis_frequency != 0:
            return
        
        self.logger.info(f"Starting attribution analysis for episode {context.episode_num}")
        
        # Simulate expensive computation
        import asyncio
        await asyncio.sleep(0.1)  # Replace with actual attribution
        
        # Generate results
        attribution_results = {
            'episode': context.episode_num,
            'top_features': ['feature1', 'feature2', 'feature3'],
            'method': self.methods[0]
        }
        
        # Trigger custom event with results
        custom_context = CustomContext(
            event_metadata=context.event_metadata,
            event_name='attribution_complete',
            event_data={'results': attribution_results}
        )
        
        # Note: In real implementation, would trigger through manager
        self.logger.info(f"Attribution analysis complete for episode {context.episode_num}")
    
    def on_custom_event(self, context: CustomContext) -> None:
        """Handle attribution results from other callbacks."""
        if context.event_name == 'attribution_request':
            # Schedule attribution analysis
            self.logger.info(f"Received attribution request: {context.event_data}")


# Example of creating a callback with decorators
class DecoratedCallback(BaseCallbackV2):
    """Example using utility decorators."""
    
    def __init__(self):
        super().__init__(name="DecoratedExample")
        self._performance_stats = {}
    
    @event_handler(EventType.EPISODE_END, EventType.UPDATE_END)
    @profile_performance
    @requires_components('trainer', 'environment')
    def handle_completion_events(self, context: Union[EpisodeContext, UpdateContext]) -> None:
        """Handle both episode and update completion with profiling."""
        if isinstance(context, EpisodeContext):
            self.logger.info(f"Episode {context.episode_num} completed")
        elif isinstance(context, UpdateContext):
            self.logger.info(f"Update {context.update_num} completed")
    
    @throttle(1.0)  # Max once per second
    def on_step_end(self, context: StepContext) -> None:
        """Throttled step processing."""
        self.logger.debug("Processing step (throttled)")


# Example usage
def create_advanced_callbacks(config: Dict[str, Any]) -> List[BaseCallbackV2]:
    """Create a set of advanced callbacks."""
    callbacks = [
        AdvancedMetricsCallback(config),
        AttributionAnalysisCallback(config),
        DecoratedCallback()
    ]
    
    return callbacks