"""
Clean evaluation callback focused solely on model evaluation.

This callback runs evaluations and shares results with other callbacks
through the event system. Other callbacks handle best model tracking,
adaptive training, etc.
"""

import logging
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from datetime import datetime

from callbacks.core.base import BaseCallback

if TYPE_CHECKING:
    from core.evaluation.evaluator import Evaluator
    from core.evaluation import EvaluationResult


class EvaluationCallback(BaseCallback):
    """
    Pure evaluation callback that focuses only on running evaluations.
    
    Responsibilities:
    - Run periodic evaluations at configured intervals
    - Share evaluation results via event system
    - Track evaluation history for analysis
    
    Does NOT handle:
    - Best model tracking (handled by ModelManagerCallback)
    - Adaptive frequency (handled by AdaptiveTrainingCallback)
    - Metrics logging (handled by WandbCallback/MetricsCallback)
    - Optuna reporting (handled by OptunaCallback)
    """
    
    def __init__(
        self, 
        evaluator: "Evaluator",
        update_frequency: int = 50,
        episode_frequency: Optional[int] = None,
        time_frequency_minutes: Optional[float] = None,
        enabled: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize evaluation callback.
        
        Args:
            evaluator: Evaluator instance for model assessment
            update_frequency: Number of updates between evaluations
            episode_frequency: Number of episodes between evaluations (optional)
            time_frequency_minutes: Minutes between evaluations (optional)
            enabled: Whether callback is active
            config: Additional configuration
        """
        super().__init__(name="EvaluationCallback", enabled=enabled, config=config)
        
        self.evaluator = evaluator
        self.update_frequency = update_frequency
        self.episode_frequency = episode_frequency
        self.time_frequency_minutes = time_frequency_minutes
        
        # Simple state tracking
        self.last_evaluation_update = 0
        self.last_evaluation_episode = 0
        self.last_evaluation_time = datetime.now()
        self.evaluation_history: List["EvaluationResult"] = []
        
        self.logger.info(
            f"🔍 EvaluationCallback initialized (update_freq={update_frequency})"
        )
    
    def on_training_start(self, context: Dict[str, Any]) -> None:
        """Reset evaluation state when training starts."""
        self.logger.info("🔍 EvaluationCallback ready for training")
        self.last_evaluation_update = 0
        self.last_evaluation_episode = 0
        self.last_evaluation_time = datetime.now()
        self.evaluation_history.clear()
    
    def on_update_end(self, context: Dict[str, Any]) -> None:
        """Check if evaluation should be triggered after update."""
        if self._should_evaluate_by_updates(context):
            self._run_evaluation(context, trigger_type="update")
    
    def on_episode_end(self, context: Dict[str, Any]) -> None:
        """Check if evaluation should be triggered after episode."""
        if self.episode_frequency and self._should_evaluate_by_episodes(context):
            self._run_evaluation(context, trigger_type="episode")
    
    def on_training_end(self, context: Dict[str, Any]) -> None:
        """Run final evaluation when training ends."""
        self.logger.info("🔍 Running final evaluation")
        self._run_evaluation(context, trigger_type="final")
        
        if self.evaluation_history:
            self.logger.info(
                f"🏁 Completed {len(self.evaluation_history)} evaluations during training"
            )
    
    def _should_evaluate_by_updates(self, context: Dict[str, Any]) -> bool:
        """Check if evaluation should be triggered based on updates."""
        current_updates = context.get('global_updates', 0)
        
        if current_updates <= 0:
            return False
        
        updates_since_last = current_updates - self.last_evaluation_update
        return updates_since_last >= self.update_frequency
    
    def _should_evaluate_by_episodes(self, context: Dict[str, Any]) -> bool:
        """Check if evaluation should be triggered based on episodes."""
        current_episodes = context.get('total_episodes', self.episodes_seen)
        episodes_since_last = current_episodes - self.last_evaluation_episode
        return episodes_since_last >= self.episode_frequency
    
    def _should_evaluate_by_time(self) -> bool:
        """Check if evaluation should be triggered based on time."""
        if not self.time_frequency_minutes:
            return False
        
        time_since_last = (datetime.now() - self.last_evaluation_time).total_seconds() / 60
        return time_since_last >= self.time_frequency_minutes
    
    def _run_evaluation(self, context: Dict[str, Any], trigger_type: str = "periodic") -> None:
        """
        Execute model evaluation and share results.
        
        Args:
            context: Current training context
            trigger_type: What triggered this evaluation (update, episode, time, final)
        """
        if not self.trainer or not self.environment or not self.data_manager:
            self.logger.warning("Missing required components for evaluation")
            return
        
        self.logger.info(f"🔍 Running {trigger_type} evaluation")
        
        try:
            # Run evaluation
            result = self.evaluator.evaluate_model(
                trainer=self.trainer,
                environment=self.environment,
                data_manager=self.data_manager,
                episode_manager=context.get('episode_manager')
            )
            
            if not result:
                self.logger.warning("Evaluation failed")
                return
            
            # Store result and update state
            self.evaluation_history.append(result)
            self._update_evaluation_state(context)
            
            # Log basic result
            self.logger.info(
                f"📊 {trigger_type.capitalize()} Evaluation: "
                f"mean={result.mean_reward:.4f}±{result.std_reward:.4f}, "
                f"episodes={result.total_episodes}"
            )
            
            # Create evaluation context for other callbacks
            eval_context = self._create_evaluation_context(result, trigger_type, context)
            
            # Share results with other callbacks via event system
            if hasattr(self, 'callback_manager') and self.callback_manager:
                self.callback_manager.trigger_evaluation_complete(eval_context)
            else:
                # Fallback: add to current context
                context.update(eval_context)
                
        except Exception as e:
            self.logger.error(f"Evaluation failed with error: {e}", exc_info=True)
    
    def _update_evaluation_state(self, context: Dict[str, Any]) -> None:
        """Update evaluation tracking state."""
        self.last_evaluation_update = context.get('global_updates', 0)
        self.last_evaluation_episode = context.get('total_episodes', self.episodes_seen)
        self.last_evaluation_time = datetime.now()
    
    def _create_evaluation_context(
        self, 
        result: "EvaluationResult", 
        trigger_type: str,
        original_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create context dict for sharing evaluation results with other callbacks."""
        return {
            # Core evaluation data
            'evaluation_result': result,
            'evaluation_trigger': trigger_type,
            'is_final_evaluation': trigger_type == 'final',
            'evaluation_timestamp': datetime.now(),
            'evaluation_count': len(self.evaluation_history),
            
            # Training context (pass through for other callbacks)
            'global_updates': original_context.get('global_updates', 0),
            'total_episodes': original_context.get('total_episodes', self.episodes_seen),
            'trainer': self.trainer,
            'environment': self.environment,
            'data_manager': self.data_manager,
            'episode_manager': original_context.get('episode_manager'),
            
            # Basic metrics (for easy access by other callbacks)
            'evaluation_metrics': {
                'mean_reward': result.mean_reward,
                'std_reward': result.std_reward,
                'min_reward': result.min_reward,
                'max_reward': result.max_reward,
                'total_episodes': result.total_episodes,
                'timestamp': result.timestamp.isoformat() if hasattr(result, 'timestamp') else None,
            },
            
            # Additional metrics if available
            'extended_metrics': self._extract_extended_metrics(result)
        }
    
    def _extract_extended_metrics(self, result: "EvaluationResult") -> Dict[str, Any]:
        """Extract any additional metrics from evaluation result."""
        extended = {}
        
        # Common optional metrics
        optional_attrs = [
            'success_rate', 'sharpe_ratio', 'max_drawdown', 'total_return',
            'win_rate', 'profit_factor', 'sortino_ratio', 'calmar_ratio',
            'volatility', 'skewness', 'kurtosis'
        ]
        
        for attr in optional_attrs:
            if hasattr(result, attr):
                extended[attr] = getattr(result, attr)
        
        return extended
    
    # Public API for other callbacks and external access
    
    def get_latest_result(self) -> Optional["EvaluationResult"]:
        """Get the most recent evaluation result."""
        return self.evaluation_history[-1] if self.evaluation_history else None
    
    def get_evaluation_history(self) -> List["EvaluationResult"]:
        """Get all evaluation results."""
        return self.evaluation_history.copy()
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get basic evaluation summary statistics."""
        if not self.evaluation_history:
            return {"total_evaluations": 0}
        
        rewards = [r.mean_reward for r in self.evaluation_history]
        return {
            "total_evaluations": len(self.evaluation_history),
            "latest_mean_reward": rewards[-1],
            "best_mean_reward": max(rewards),
            "worst_mean_reward": min(rewards),
            "mean_reward_trend": rewards[-1] - rewards[0] if len(rewards) > 1 else 0.0,
            "last_evaluation_update": self.last_evaluation_update,
            "last_evaluation_time": self.last_evaluation_time.isoformat()
        }
    
    def force_evaluation(self, context: Dict[str, Any]) -> Optional["EvaluationResult"]:
        """Force an immediate evaluation (useful for testing/debugging)."""
        self.logger.info("🔍 Forcing immediate evaluation")
        self._run_evaluation(context, trigger_type="forced")
        return self.get_latest_result()
    
    def update_frequency(self, new_frequency: int) -> None:
        """Update evaluation frequency (useful for adaptive callbacks)."""
        if new_frequency > 0:
            self.update_frequency = new_frequency
            self.logger.debug(f"Updated evaluation frequency to {new_frequency}")