"""
Evaluation callback for integrating Evaluator with training loop.

This callback bridges the gap between the training loop and the evaluation system,
triggering evaluations at appropriate intervals and handling results.
"""

import logging
from typing import Optional, Dict, Any

from callbacks.core.base import BaseCallback
from callbacks.core.context import ComponentState
from core.evaluation.evaluator import Evaluator
from core.evaluation import EvaluationResult
from config.evaluation.evaluation_config import EvaluationConfig


class EvaluationCallback(BaseCallback):
    """
    Callback that integrates Evaluator with the training loop.
    
    Responsibilities:
    - Trigger evaluation at specified frequencies
    - Handle evaluation results (logging, metrics, storage)
    - Coordinate with other callbacks (e.g., wandb, checkpointing)
    """
    
    def __init__(self, evaluator: Evaluator, frequency: int = 50):
        """
        Initialize evaluation callback.
        
        Args:
            evaluator: Evaluator instance to use for model evaluation
            frequency: Number of updates between evaluations
        """
        super().__init__()
        self.evaluator = evaluator
        self.frequency = frequency
        self.logger = logging.getLogger(f"{__name__}.EvaluationCallback")
        
        # State tracking
        self.last_evaluation_update: int = 0
        self.evaluation_history: list[EvaluationResult] = []
        
        self.logger.info(f"🔍 EvaluationCallback initialized (frequency={frequency})")
    
    def on_update_end(self, state: ComponentState) -> None:
        """
        Called after each policy update - check if evaluation should be triggered.
        
        Args:
            state: Current component state with trainer, environment, etc.
        """
        if not self._should_evaluate(state):
            return
        
        self.logger.info(f"🔍 Triggering evaluation at update {state.training_state.global_updates}")
        
        # Run evaluation
        result = self.evaluator.evaluate_model(
            trainer=state.trainer,
            environment=state.environment,
            data_manager=state.data_manager,
            episode_manager=state.episode_manager
        )
        
        if result:
            # Store result
            self.evaluation_history.append(result)
            self.last_evaluation_update = state.training_state.global_updates
            
            # Handle evaluation result
            self._handle_evaluation_result(state, result)
        else:
            self.logger.warning("Evaluation failed")
    
    def on_training_start(self, state: ComponentState) -> None:
        """Called when training starts."""
        self.logger.info("🔍 EvaluationCallback ready for training")
        self.last_evaluation_update = 0
        self.evaluation_history.clear()
    
    def on_training_end(self, state: ComponentState) -> None:
        """Called when training ends - run final evaluation."""
        if state.training_state.global_updates > self.last_evaluation_update:
            self.logger.info("🔍 Running final evaluation")
            
            result = self.evaluator.evaluate_model(
                trainer=state.trainer,
                environment=state.environment,
                data_manager=state.data_manager,
                episode_manager=state.episode_manager
            )
            
            if result:
                self.evaluation_history.append(result)
                self._handle_evaluation_result(state, result, is_final=True)
        
        # Log evaluation summary
        if self.evaluation_history:
            best_result = max(self.evaluation_history, key=lambda r: r.mean_reward)
            self.logger.info(
                f"🏆 Best evaluation: {best_result.mean_reward:.4f} "
                f"(from {len(self.evaluation_history)} evaluations)"
            )
    
    def _should_evaluate(self, state: ComponentState) -> bool:
        """Check if evaluation should be triggered."""
        current_updates = state.training_state.global_updates
        
        # Skip if no updates yet
        if current_updates <= 0:
            return False
        
        # Check frequency
        updates_since_last = current_updates - self.last_evaluation_update
        return updates_since_last >= self.frequency
    
    def _handle_evaluation_result(
        self, 
        state: ComponentState, 
        result: EvaluationResult,
        is_final: bool = False
    ) -> None:
        """
        Handle evaluation result - logging, metrics, notifications.
        
        Args:
            state: Current component state
            result: Evaluation result to handle
            is_final: Whether this is the final evaluation
        """
        # Log result
        eval_type = "Final" if is_final else "Periodic"
        self.logger.info(
            f"📊 {eval_type} Evaluation Result: "
            f"mean={result.mean_reward:.4f}, "
            f"std={result.std_reward:.4f}, "
            f"episodes={result.total_episodes}"
        )
        
        # Add evaluation data to event_data for other callbacks
        eval_data = {
            "evaluation_result": result,
            "evaluation_type": eval_type.lower(),
            "is_final_evaluation": is_final
        }
        
        # Update state's event_data
        if state.event_data is None:
            state.event_data = {}
        state.event_data.update(eval_data)
        
        # Store metrics for potential use by other systems
        self._store_evaluation_metrics(result, state)
    
    def _store_evaluation_metrics(self, result: EvaluationResult, state: ComponentState) -> None:
        """Store evaluation metrics for tracking and analysis."""
        # This could be extended to store in database, files, etc.
        metrics = {
            "eval/mean_reward": result.mean_reward,
            "eval/std_reward": result.std_reward,
            "eval/min_reward": result.min_reward,
            "eval/max_reward": result.max_reward,
            "eval/total_episodes": result.total_episodes,
            "eval/timestamp": result.timestamp.isoformat()
        }
        
        # Add to event_data for wandb callback or other metric collectors
        if "metrics" not in state.event_data:
            state.event_data["metrics"] = {}
        state.event_data["metrics"].update(metrics)
    
    def get_latest_result(self) -> Optional[EvaluationResult]:
        """Get the most recent evaluation result."""
        return self.evaluation_history[-1] if self.evaluation_history else None
    
    def get_best_result(self) -> Optional[EvaluationResult]:
        """Get the best evaluation result by mean reward."""
        if not self.evaluation_history:
            return None
        return max(self.evaluation_history, key=lambda r: r.mean_reward)
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations."""
        if not self.evaluation_history:
            return {"total_evaluations": 0}
        
        rewards = [r.mean_reward for r in self.evaluation_history]
        return {
            "total_evaluations": len(self.evaluation_history),
            "best_mean_reward": max(rewards),
            "worst_mean_reward": min(rewards),
            "latest_mean_reward": rewards[-1],
            "improvement_trend": rewards[-1] - rewards[0] if len(rewards) > 1 else 0.0
        }