"""
Early stopping callback.

Provides early stopping functionality based on training metrics
to prevent overfitting and save computational resources.
"""

from typing import Dict, Any, Optional
from ..core.base import BaseCallback


class EarlyStoppingCallback(BaseCallback):
    """
    Early stopping callback.
    
    Monitors training progress and stops training early if
    no improvement is observed for a specified number of episodes.
    
    Placeholder implementation - will be expanded when early stopping
    is needed in v2 system.
    """
    
    def __init__(
        self,
        enabled: bool = True,
        patience: int = 100,
        min_delta: float = 0.001,
        metric: str = "episode_reward",
        trainer: Optional[Any] = None,
        name: Optional[str] = None
    ):
        """
        Initialize early stopping callback.
        
        Args:
            enabled: Whether callback is active
            patience: Number of episodes to wait for improvement
            min_delta: Minimum improvement to consider as progress
            metric: Metric to monitor for early stopping
            trainer: PPO trainer instance
            name: Optional custom name
        """
        super().__init__(enabled, name)
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.trainer = trainer
        
        # Early stopping state
        self.best_value = float('-inf')
        self.wait_count = 0
        self.stopped_early = False
        self.stop_episode = None
    
    def on_training_start(self, context: Dict[str, Any]) -> None:
        """Initialize early stopping tracking."""
        super().on_training_start(context)
        self.logger.info(f"⏰ Early stopping callback initialized")
        self.logger.info(f"  Patience: {self.patience} episodes")
        self.logger.info(f"  Min delta: {self.min_delta}")
        self.logger.info(f"  Monitoring: {self.metric}")
    
    def on_episode_end(self, context: Dict[str, Any]) -> None:
        """Check early stopping criteria."""
        super().on_episode_end(context)
        
        if self.stopped_early:
            return
        
        # Get metric value to monitor
        current_value = self._get_metric_value(context)
        if current_value is None:
            return
        
        episode_info = context.get("episode", {})
        episode_num = episode_info.get("num", self.episode_count)
        
        # Check for improvement
        if current_value > self.best_value + self.min_delta:
            self.best_value = current_value
            self.wait_count = 0
            self.logger.debug(f"⏰ New best {self.metric}: {current_value:.3f} at episode {episode_num}")
        else:
            self.wait_count += 1
            
        # Check if patience exceeded
        if self.wait_count >= self.patience:
            self.stopped_early = True
            self.stop_episode = episode_num
            
            self.logger.info(f"⏰ Early stopping triggered at episode {episode_num}")
            self.logger.info(f"   Best {self.metric}: {self.best_value:.3f}")
            self.logger.info(f"   No improvement for {self.patience} episodes")
            
            # TODO: Signal trainer to stop training
            # if self.trainer and hasattr(self.trainer, 'request_early_stop'):
            #     self.trainer.request_early_stop()
    
    def _get_metric_value(self, context: Dict[str, Any]) -> Optional[float]:
        """Extract metric value from context."""
        if self.metric == "episode_reward":
            return context.get("episode", {}).get("reward")
        elif self.metric == "portfolio_value":
            return context.get("metrics", {}).get("portfolio_value")
        elif self.metric == "total_profit":
            return context.get("metrics", {}).get("total_profit")
        else:
            self.logger.warning(f"Unknown metric: {self.metric}")
            return None
    
    def on_training_end(self, context: Dict[str, Any]) -> None:
        """Log early stopping results."""
        super().on_training_end(context)
        
        if self.stopped_early:
            self.logger.info(f"⏰ Training stopped early at episode {self.stop_episode}")
            self.logger.info(f"   Best {self.metric}: {self.best_value:.3f}")
        else:
            self.logger.info(f"⏰ Training completed without early stopping")
            self.logger.info(f"   Final best {self.metric}: {self.best_value:.3f}")
    
    @property
    def should_stop(self) -> bool:
        """Whether early stopping has been triggered."""
        return self.stopped_early