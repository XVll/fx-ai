"""
Hook registry and documentation for the callback system.

Provides a central registry of all available hooks, their signatures,
and when they are triggered in the training process.
"""

from typing import Dict, List, Optional, Set, Any, Type
from dataclasses import dataclass
from enum import Enum

from .events import EventType, EventRegistry
from .context_v2 import (
    BaseContext, StepContext, EpisodeContext, RolloutContext,
    UpdateContext, BatchContext, ModelContext, EvaluationContext,
    DataContext, ErrorContext, CustomContext,
    get_context_class
)


@dataclass
class HookInfo:
    """Information about a callback hook."""
    event_type: EventType
    method_name: str
    context_class: Type[BaseContext]
    description: str
    trigger_location: str
    frequency: str
    performance_impact: str
    example_usage: Optional[str] = None
    related_hooks: Optional[List[str]] = None
    available_data: Optional[List[str]] = None


class PerformanceImpact(Enum):
    """Performance impact levels for hooks."""
    MINIMAL = "minimal"      # < 1ms typical
    LOW = "low"              # 1-10ms typical
    MEDIUM = "medium"        # 10-100ms typical
    HIGH = "high"            # > 100ms typical
    VARIABLE = "variable"    # Depends on implementation


class HookRegistry:
    """
    Central registry of all callback hooks.
    
    Provides comprehensive documentation and metadata for each hook,
    including when it's triggered, what data is available, and performance
    considerations.
    """
    
    # Complete hook definitions
    HOOKS = {
        # Training lifecycle hooks
        EventType.TRAINING_START: HookInfo(
            event_type=EventType.TRAINING_START,
            method_name="on_training_start",
            context_class=BaseContext,
            description="Called once when training begins, after all components are initialized",
            trigger_location="TrainingManager.start() after component initialization",
            frequency="Once per training run",
            performance_impact=PerformanceImpact.MINIMAL.value,
            available_data=[
                "trainer", "environment", "data_manager", "episode_manager",
                "training_state", "model_manager"
            ],
            example_usage="""
def on_training_start(self, context: BaseContext):
    # Initialize tracking, connect to external services
    self.start_time = context.timestamp
    self.episodes_completed = 0
    
    # Access model for architecture analysis
    model = context.get_model()
    self.log_model_architecture(model)
"""
        ),
        
        EventType.TRAINING_END: HookInfo(
            event_type=EventType.TRAINING_END,
            method_name="on_training_end",
            context_class=BaseContext,
            description="Called once when training completes or is terminated",
            trigger_location="TrainingManager._finalize_training()",
            frequency="Once per training run",
            performance_impact=PerformanceImpact.MINIMAL.value,
            available_data=["final_stats", "termination_reason"],
            related_hooks=["on_training_start"],
            example_usage="""
def on_training_end(self, context: BaseContext):
    # Save final metrics, cleanup resources
    duration = context.timestamp - self.start_time
    self.log_final_metrics(duration, self.episodes_completed)
    self.cleanup_resources()
"""
        ),
        
        # Episode hooks
        EventType.EPISODE_START: HookInfo(
            event_type=EventType.EPISODE_START,
            method_name="on_episode_start",
            context_class=EpisodeContext,
            description="Called at the beginning of each episode after environment reset",
            trigger_location="TrainingManager.start() after environment.reset()",
            frequency="Every episode",
            performance_impact=PerformanceImpact.LOW.value,
            available_data=[
                "episode_num", "symbol", "date", "reset_point_idx",
                "starting_cash", "starting_portfolio_value"
            ],
            example_usage="""
def on_episode_start(self, context: EpisodeContext):
    # Track episode configuration
    self.current_episode = {
        'symbol': context.symbol,
        'date': context.date,
        'start_value': context.starting_portfolio_value
    }
"""
        ),
        
        EventType.EPISODE_END: HookInfo(
            event_type=EventType.EPISODE_END,
            method_name="on_episode_end",
            context_class=EpisodeContext,
            description="Called at the end of each episode with complete episode statistics",
            trigger_location="TrainingManager after rollout with completed episodes",
            frequency="Every episode",
            performance_impact=PerformanceImpact.MEDIUM.value,
            available_data=[
                "episode_reward", "episode_length", "trades", "total_pnl",
                "win_rate", "max_drawdown", "termination_reason"
            ],
            example_usage="""
def on_episode_end(self, context: EpisodeContext):
    # Analyze episode performance
    self.metric_tracker.record_episode('reward', context.episode_reward)
    self.metric_tracker.record_episode('pnl', context.total_pnl)
    
    # Track best episodes
    if context.episode_reward > self.best_reward:
        self.best_reward = context.episode_reward
        self.save_episode_data(context)
"""
        ),
        
        # Step hooks
        EventType.STEP_START: HookInfo(
            event_type=EventType.STEP_START,
            method_name="on_step_start",
            context_class=StepContext,
            description="Called before each environment step",
            trigger_location="PPOTrainer.collect_rollout() before env.step()",
            frequency="Every environment step",
            performance_impact=PerformanceImpact.MINIMAL.value,
            available_data=["observation", "step_num", "episode_step"],
            example_usage="""
def on_step_start(self, context: StepContext):
    # Pre-step analysis or intervention
    if self.should_override_action(context.observation):
        # Can't modify action here, but can prepare
        self.prepare_override(context)
"""
        ),
        
        EventType.STEP_END: HookInfo(
            event_type=EventType.STEP_END,
            method_name="on_step_end",
            context_class=StepContext,
            description="Called after each environment step with complete step information",
            trigger_location="PPOTrainer.collect_rollout() after env.step()",
            frequency="Every environment step",
            performance_impact=PerformanceImpact.LOW.value,
            available_data=[
                "observation", "action", "reward", "next_observation",
                "terminated", "truncated", "info", "current_price",
                "position", "portfolio_value"
            ],
            example_usage="""
def on_step_end(self, context: StepContext):
    # Track step-level metrics
    self.steps_taken += 1
    
    # Analyze action impact
    if context.action in [1, 2]:  # Buy or Sell
        self.analyze_trade_decision(context)
    
    # Check for anomalies
    if abs(context.reward) > self.reward_threshold:
        self.log_anomaly(context)
"""
        ),
        
        EventType.ACTION_SELECTED: HookInfo(
            event_type=EventType.ACTION_SELECTED,
            method_name="on_action_selected",
            context_class=StepContext,
            description="Called immediately after action is selected by the policy",
            trigger_location="PPOTrainer.collect_rollout() after model inference",
            frequency="Every environment step",
            performance_impact=PerformanceImpact.MINIMAL.value,
            available_data=[
                "action", "action_probs", "action_logprob", "value_estimate"
            ],
            example_usage="""
def on_action_selected(self, context: StepContext):
    # Analyze policy decisions
    entropy = -np.sum(context.action_probs * np.log(context.action_probs + 1e-8))
    self.track_policy_entropy(entropy)
    
    # Track action distribution
    self.action_counts[context.action] += 1
"""
        ),
        
        # Update hooks
        EventType.UPDATE_START: HookInfo(
            event_type=EventType.UPDATE_START,
            method_name="on_update_start",
            context_class=UpdateContext,
            description="Called before policy update begins",
            trigger_location="TrainingManager before trainer.update_policy()",
            frequency="Every N steps (based on rollout_steps)",
            performance_impact=PerformanceImpact.MINIMAL.value,
            available_data=[
                "update_num", "batch_size", "learning_rate", "clip_epsilon"
            ],
            example_usage="""
def on_update_start(self, context: UpdateContext):
    # Prepare for update monitoring
    self.update_start_time = time.time()
    
    # Capture model state before update
    if self.track_weight_changes:
        self.model_weights_before = self.get_model_weights(context.get_model())
"""
        ),
        
        EventType.UPDATE_END: HookInfo(
            event_type=EventType.UPDATE_END,
            method_name="on_update_end",
            context_class=UpdateContext,
            description="Called after policy update completes with all metrics",
            trigger_location="TrainingManager after trainer.update_policy()",
            frequency="Every N steps (based on rollout_steps)",
            performance_impact=PerformanceImpact.MEDIUM.value,
            available_data=[
                "policy_loss", "value_loss", "entropy_loss", "total_loss",
                "kl_divergence", "clip_fraction", "gradient_norm",
                "explained_variance"
            ],
            example_usage="""
def on_update_end(self, context: UpdateContext):
    # Track training metrics
    self.metric_tracker.record('policy_loss', context.policy_loss)
    self.metric_tracker.record('value_loss', context.value_loss)
    self.metric_tracker.record('kl_divergence', context.kl_divergence)
    
    # Check for training issues
    if context.kl_divergence > self.kl_threshold:
        self.log_warning(f"High KL divergence: {context.kl_divergence}")
    
    # Compute weight changes
    if self.track_weight_changes:
        weight_change = self.compute_weight_change(
            self.model_weights_before,
            self.get_model_weights(context.get_model())
        )
        self.track_weight_stability(weight_change)
"""
        ),
        
        EventType.GRADIENT_COMPUTED: HookInfo(
            event_type=EventType.GRADIENT_COMPUTED,
            method_name="on_gradient_computed",
            context_class=UpdateContext,
            description="Called after gradients are computed but before optimizer step",
            trigger_location="PPOTrainer.update_policy() after loss.backward()",
            frequency="Every batch in every update",
            performance_impact=PerformanceImpact.HIGH.value,
            available_data=["gradient_info", "gradient_norm", "gradient_max"],
            example_usage="""
def on_gradient_computed(self, context: UpdateContext):
    # Analyze gradients for training stability
    grad_info = context.get_gradient_info()
    
    for layer_name, stats in grad_info['layer_gradients'].items():
        if stats['norm'] > self.gradient_threshold:
            self.log_gradient_explosion(layer_name, stats)
        
        # Track gradient flow
        self.gradient_norms[layer_name].append(stats['norm'])
"""
        ),
        
        # Batch hooks
        EventType.BATCH_START: HookInfo(
            event_type=EventType.BATCH_START,
            method_name="on_batch_start",
            context_class=BatchContext,
            description="Called before processing each training batch",
            trigger_location="PPOTrainer.update_policy() inner loop",
            frequency="Every batch in every epoch of update",
            performance_impact=PerformanceImpact.MINIMAL.value,
            available_data=[
                "batch_idx", "batch_size", "observations", "actions",
                "returns", "advantages"
            ],
            example_usage="""
def on_batch_start(self, context: BatchContext):
    # Analyze batch statistics
    self.track_advantage_distribution(context.advantages)
    
    # Data quality checks
    if torch.isnan(context.advantages).any():
        self.log_error("NaN in advantages!")
"""
        ),
        
        # Model hooks
        EventType.MODEL_SAVED: HookInfo(
            event_type=EventType.MODEL_SAVED,
            method_name="on_model_saved",
            context_class=ModelContext,
            description="Called when model checkpoint is saved",
            trigger_location="CheckpointCallback or ModelManager.save_model()",
            frequency="Based on checkpoint frequency",
            performance_impact=PerformanceImpact.LOW.value,
            available_data=[
                "checkpoint_path", "model_version", "best_reward",
                "saved_at_step", "model_metadata"
            ],
            example_usage="""
def on_model_saved(self, context: ModelContext):
    # Track model versions
    self.model_history.append({
        'version': context.model_version,
        'reward': context.best_reward,
        'path': context.checkpoint_path,
        'timestamp': context.timestamp
    })
    
    # Upload to cloud storage
    if self.cloud_backup_enabled:
        self.upload_checkpoint(context.checkpoint_path)
"""
        ),
        
        # Data hooks
        EventType.DAY_SWITCHED: HookInfo(
            event_type=EventType.DAY_SWITCHED,
            method_name="on_day_switched",
            context_class=DataContext,
            description="Called when training switches to a new trading day",
            trigger_location="EpisodeManager when selecting new day",
            frequency="When day selection changes",
            performance_impact=PerformanceImpact.LOW.value,
            available_data=[
                "symbol", "date", "momentum_score", "data_points",
                "time_range_start", "time_range_end"
            ],
            example_usage="""
def on_day_switched(self, context: DataContext):
    # Track day progression
    self.days_trained.append(context.date)
    
    # Log day characteristics
    self.logger.info(
        f"Switched to {context.symbol} {context.date} "
        f"(momentum: {context.momentum_score:.3f})"
    )
    
    # Prepare day-specific features
    self.load_day_specific_data(context.date)
"""
        ),
        
        # Custom hooks
        EventType.CUSTOM: HookInfo(
            event_type=EventType.CUSTOM,
            method_name="on_custom_event",
            context_class=CustomContext,
            description="Called for user-defined custom events",
            trigger_location="Anywhere via trigger_event(EventType.CUSTOM, ...)",
            frequency="As triggered by user code",
            performance_impact=PerformanceImpact.VARIABLE.value,
            available_data=["event_name", "event_data"],
            example_usage="""
def on_custom_event(self, context: CustomContext):
    # Handle custom events by name
    if context.event_name == "attribution_complete":
        attribution_results = context.event_data['results']
        self.process_attribution(attribution_results)
    
    elif context.event_name == "evaluation_requested":
        self.schedule_evaluation(context.event_data)
"""
        )
    }
    
    @classmethod
    def get_hook_info(cls, event_type: EventType) -> Optional[HookInfo]:
        """Get detailed information about a hook."""
        return cls.HOOKS.get(event_type)
    
    @classmethod
    def get_hooks_by_frequency(cls, frequency: str) -> List[HookInfo]:
        """Get all hooks with a specific frequency pattern."""
        return [
            hook for hook in cls.HOOKS.values()
            if frequency.lower() in hook.frequency.lower()
        ]
    
    @classmethod
    def get_hooks_by_impact(cls, impact: str) -> List[HookInfo]:
        """Get all hooks with a specific performance impact."""
        return [
            hook for hook in cls.HOOKS.values()
            if hook.performance_impact == impact
        ]
    
    @classmethod
    def get_hook_locations(cls) -> Dict[str, List[EventType]]:
        """Get all hooks grouped by trigger location."""
        locations = {}
        for event_type, hook in cls.HOOKS.items():
            location = hook.trigger_location.split('.')[0]  # Get class name
            if location not in locations:
                locations[location] = []
            locations[location].append(event_type)
        return locations
    
    @classmethod
    def validate_callback_implementation(cls, callback_class: type) -> Dict[str, Any]:
        """
        Validate that a callback class implements hooks correctly.
        
        Returns dict with:
        - implemented_hooks: List of correctly implemented hooks
        - missing_hooks: List of hooks that could be implemented
        - incorrect_signatures: List of hooks with wrong signatures
        """
        implemented = []
        missing = []
        incorrect = []
        
        for event_type, hook_info in cls.HOOKS.items():
            method_name = hook_info.method_name
            
            if hasattr(callback_class, method_name):
                method = getattr(callback_class, method_name)
                # Basic check - could be enhanced with signature validation
                implemented.append(method_name)
            else:
                missing.append(method_name)
        
        return {
            'implemented_hooks': implemented,
            'missing_hooks': missing,
            'incorrect_signatures': incorrect,
            'coverage': len(implemented) / len(cls.HOOKS)
        }
    
    @classmethod
    def generate_callback_template(cls, event_types: Optional[Set[EventType]] = None) -> str:
        """Generate a callback class template with selected hooks."""
        if event_types is None:
            event_types = set(EventType)
        
        template = '''"""
Custom callback implementation.
"""

from callbacks.core.base_v2 import BaseCallbackV2
from callbacks.core.context_v2 import *
from callbacks.core.events import EventType, EventPriority


class CustomCallback(BaseCallbackV2):
    """Custom callback for ..."""
    
    def __init__(self, config=None):
        super().__init__(
            name="CustomCallback",
            priority=EventPriority.NORMAL,
            config=config
        )
        
        # Initialize your state here
        self.episodes_seen = 0
        self.best_reward = float('-inf')
'''
        
        for event_type in event_types:
            hook_info = cls.HOOKS.get(event_type)
            if hook_info:
                template += f'''
    
    def {hook_info.method_name}(self, context: {hook_info.context_class.__name__}) -> None:
        """{hook_info.description}"""
        # TODO: Implement your logic here
        pass
'''
        
        return template