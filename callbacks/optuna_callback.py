"""Optuna integration callback for hyperparameter optimization."""

import logging
from typing import Optional, Dict, Any

import optuna

from callbacks.core.base import BaseCallback
from callbacks.core.types import CallbackContext
from core.data_structures import EvaluationResult

logger = logging.getLogger(__name__)


class OptunaCallback(BaseCallback):
    """Callback for Optuna hyperparameter optimization integration.
    
    This callback integrates with Optuna trials to:
    - Report evaluation metrics to Optuna
    - Handle pruning decisions based on intermediate values
    - Track optimization progress
    
    The callback listens for evaluation complete events and reports the results
    to the Optuna trial. If the trial should be pruned, it raises TrialPruned.
    """
    
    def __init__(
        self,
        trial: Optional[optuna.Trial] = None,
        metric_name: str = "mean_reward",
        report_interval: int = 1,
        use_best_value: bool = False,
    ):
        """Initialize Optuna callback.
        
        Args:
            trial: Optuna trial object
            metric_name: Name of the metric to optimize (e.g., "mean_reward", "sharpe_ratio")
            report_interval: Report to Optuna every N evaluations
            use_best_value: Whether to report best value seen so far instead of current
        """
        super().__init__(name="OptunaCallback")
        self.trial = trial
        self.metric_name = metric_name
        self.report_interval = report_interval
        self.use_best_value = use_best_value
        
        self.evaluation_count = 0
        self.best_value = float('-inf')
        self.current_step = 0
        
    def on_evaluation_complete(self, context: CallbackContext) -> None:
        """Report evaluation results to Optuna trial.
        
        This method is called after each evaluation completes. It extracts
        the specified metric from the evaluation result and reports it to
        Optuna for hyperparameter optimization.
        
        Args:
            context: Callback context containing evaluation results
        """
        if not self.trial:
            return
            
        # Get evaluation result from context
        eval_result = context.shared_data.get('evaluation_result')
        if not isinstance(eval_result, EvaluationResult):
            logger.warning("No evaluation result found in context")
            return
            
        self.evaluation_count += 1
        
        # Only report at specified intervals
        if self.evaluation_count % self.report_interval != 0:
            return
            
        # Extract metric value
        value = self._extract_metric_value(eval_result)
        if value is None:
            logger.warning(f"Could not extract metric '{self.metric_name}' from evaluation result")
            return
            
        # Update best value if needed
        if value > self.best_value:
            self.best_value = value
            
        # Report value to Optuna
        report_value = self.best_value if self.use_best_value else value
        self.current_step = context.global_updates
        
        logger.info(
            f"Reporting to Optuna - Step: {self.current_step}, "
            f"Metric: {self.metric_name}, Value: {report_value:.4f}"
        )
        
        self.trial.report(report_value, self.current_step)
        
        # Check if trial should be pruned
        if self.trial.should_prune():
            logger.info(f"Trial pruned at step {self.current_step}")
            raise optuna.TrialPruned()
            
    def _extract_metric_value(self, eval_result: EvaluationResult) -> Optional[float]:
        """Extract metric value from evaluation result.
        
        Args:
            eval_result: Evaluation result object
            
        Returns:
            Metric value or None if not found
        """
        # Direct attribute access
        if hasattr(eval_result, self.metric_name):
            return float(getattr(eval_result, self.metric_name))
            
        # Try common metric mappings
        metric_mappings = {
            "mean_reward": "mean_reward",
            "total_reward": "total_reward",
            "sharpe_ratio": "sharpe_ratio",
            "max_drawdown": "max_drawdown",
            "win_rate": "win_rate",
            "profit_factor": "profit_factor",
            "mean_episode_length": "mean_episode_length",
        }
        
        if self.metric_name in metric_mappings:
            attr_name = metric_mappings[self.metric_name]
            if hasattr(eval_result, attr_name):
                return float(getattr(eval_result, attr_name))
                
        # Try to find in metrics dict if available
        if hasattr(eval_result, 'metrics') and isinstance(eval_result.metrics, dict):
            if self.metric_name in eval_result.metrics:
                return float(eval_result.metrics[self.metric_name])
                
        return None
        
    def on_training_end(self, context: CallbackContext) -> None:
        """Handle training end - report final value if available."""
        if not self.trial:
            return
            
        # Try to get final evaluation result
        eval_result = context.shared_data.get('evaluation_result')
        if isinstance(eval_result, EvaluationResult):
            value = self._extract_metric_value(eval_result)
            if value is not None:
                # Report final value
                self.trial.report(value, self.current_step)
                logger.info(f"Reported final value to Optuna: {value:.4f}")
                
    def get_optimization_value(self) -> float:
        """Get the current optimization value for this trial.
        
        Returns:
            Current best value if use_best_value is True, otherwise last reported value
        """
        return self.best_value