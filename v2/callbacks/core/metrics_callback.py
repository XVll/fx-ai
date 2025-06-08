"""
Basic metrics logging callback.

Provides fundamental metrics tracking and console logging
for training progress monitoring with strongly typed contexts.
"""

from typing import Optional
import logging
from .base import BaseCallback
from .context import TrainingStartContext, EpisodeEndContext, UpdateEndContext, TrainingEndContext


class MetricsCallback(BaseCallback):
    """
    Basic metrics logging callback.
    
    Tracks and logs essential training metrics like episode rewards,
    training losses, and portfolio performance to console and logs.
    """
    
    def __init__(
        self,
        enabled: bool = True,
        log_freq: int = 10,
        console_output: bool = True,
        name: Optional[str] = None
    ):
        """
        Initialize metrics callback.
        
        Args:
            enabled: Whether callback is active
            log_freq: Frequency for logging metrics (episodes)
            console_output: Whether to output to console
            name: Optional custom name
        """
        super().__init__(enabled, name)
        self.log_freq = log_freq
        self.console_output = console_output
        
        # Metrics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.total_trades = 0
        self.total_profit = 0.0
        
        # Training metrics
        self.policy_losses = []
        self.value_losses = []
        
    def on_training_start(self, context: TrainingStartContext) -> None:
        """Log training start."""
        super().on_training_start(context)
        
        self.logger.info("🚀 Training started with MetricsCallback")
        self.logger.info(f"  Log frequency: {self.log_freq} episodes")
        self.logger.info(f"  Console output: {self.console_output}")
        self.logger.info(f"  Run ID: {context.run_id}")
        self.logger.info(f"  Output path: {context.output_path}")
        
        if self.console_output:
            print("\n" + "="*80)
            print("🚀 FxAI V2 Training Started")
            print(f"Run ID: {context.run_id}")
            print("="*80)
    
    def on_episode_end(self, context: EpisodeEndContext) -> None:
        """Track episode metrics."""
        super().on_episode_end(context)
        
        # Extract typed data from context
        episode_num = context.episode.num
        reward = context.episode.reward
        length = context.episode.length
        
        # Track metrics
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        
        # Track trading metrics
        trades_count = context.metrics.trades_count
        portfolio_value = context.metrics.portfolio_value
        profit = context.metrics.total_profit
        
        self.total_trades += trades_count
        self.total_profit = profit
        
        # Log at specified frequency
        if episode_num % self.log_freq == 0:
            self._log_episode_summary(episode_num, reward, length, context)
    
    def on_update_end(self, context: Dict[str, Any]) -> None:
        """Track training update metrics."""
        super().on_update_end(context)
        
        losses = context.get("losses", {})
        
        # Track losses
        policy_loss = losses.get("policy_loss", 0.0)
        value_loss = losses.get("value_loss", 0.0)
        
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        
        # Log training metrics periodically
        update_num = context.get("update", {}).get("num", self.update_count)
        if update_num % (self.log_freq * 5) == 0:  # Less frequent for updates
            self._log_training_summary(update_num, losses)
    
    def on_training_end(self, context: Dict[str, Any]) -> None:
        """Log training completion summary."""
        super().on_training_end(context)
        
        final_metrics = context.get("final_metrics", {})
        total_episodes = context.get("total_episodes", len(self.episode_rewards))
        duration = context.get("duration", "unknown")
        
        self._log_final_summary(total_episodes, duration, final_metrics)
        
        if self.console_output:
            print("\n" + "="*80)
            print("✅ Training Completed")
            print("="*80)
    
    def _log_episode_summary(
        self, 
        episode_num: int, 
        reward: float, 
        length: int, 
        context: EpisodeEndContext
    ) -> None:
        """Log episode summary."""
        # Calculate running averages
        recent_rewards = self.episode_rewards[-self.log_freq:]
        avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0.0
        
        recent_lengths = self.episode_lengths[-self.log_freq:]
        avg_length = sum(recent_lengths) / len(recent_lengths) if recent_lengths else 0.0
        
        # Log to logger
        self.logger.info(f"Episode {episode_num:6d} | "
                        f"Reward: {reward:8.3f} | "
                        f"Avg Reward: {avg_reward:8.3f} | "
                        f"Length: {length:4d} | "
                        f"Avg Length: {avg_length:6.1f}")
        
        # Console output
        if self.console_output:
            portfolio_value = context.metrics.portfolio_value
            trades_count = context.metrics.trades_count
            win_rate = context.metrics.win_rate
            
            print(f"Episode {episode_num:6d} | "
                  f"Reward: {reward:8.3f} | "
                  f"Portfolio: ${portfolio_value:10,.2f} | "
                  f"Trades: {trades_count:3d} | "
                  f"Win Rate: {win_rate:5.1%}")
    
    def _log_training_summary(self, update_num: int, losses: Dict[str, Any]) -> None:
        """Log training update summary."""
        # Calculate recent loss averages
        recent_policy = self.policy_losses[-10:] if self.policy_losses else [0.0]
        recent_value = self.value_losses[-10:] if self.value_losses else [0.0]
        
        avg_policy_loss = sum(recent_policy) / len(recent_policy)
        avg_value_loss = sum(recent_value) / len(recent_value)
        
        self.logger.info(f"Update {update_num:6d} | "
                        f"Policy Loss: {avg_policy_loss:.6f} | "
                        f"Value Loss: {avg_value_loss:.6f} | "
                        f"Total Loss: {avg_policy_loss + avg_value_loss:.6f}")
    
    def _log_final_summary(
        self, 
        total_episodes: int, 
        duration: str, 
        final_metrics: Dict[str, Any]
    ) -> None:
        """Log final training summary."""
        if not self.episode_rewards:
            self.logger.info("No episodes completed")
            return
        
        # Calculate final statistics
        total_reward = sum(self.episode_rewards)
        avg_reward = total_reward / len(self.episode_rewards)
        max_reward = max(self.episode_rewards)
        min_reward = min(self.episode_rewards)
        
        avg_length = sum(self.episode_lengths) / len(self.episode_lengths) if self.episode_lengths else 0
        
        # Log summary
        self.logger.info("=" * 80)
        self.logger.info("TRAINING COMPLETED - FINAL METRICS")
        self.logger.info("=" * 80)
        self.logger.info(f"Duration: {duration}")
        self.logger.info(f"Total Episodes: {total_episodes}")
        self.logger.info(f"Total Updates: {self.update_count}")
        self.logger.info(f"Average Episode Reward: {avg_reward:.3f}")
        self.logger.info(f"Max Episode Reward: {max_reward:.3f}")
        self.logger.info(f"Min Episode Reward: {min_reward:.3f}")
        self.logger.info(f"Average Episode Length: {avg_length:.1f}")
        self.logger.info(f"Total Trades: {self.total_trades}")
        self.logger.info(f"Total Profit: ${self.total_profit:.2f}")
        self.logger.info("=" * 80)
        
        # Console summary
        if self.console_output:
            print(f"\n📊 Final Results:")
            print(f"  Episodes: {total_episodes}")
            print(f"  Avg Reward: {avg_reward:.3f}")
            print(f"  Max Reward: {max_reward:.3f}")
            print(f"  Total Profit: ${self.total_profit:.2f}")
            print(f"  Total Trades: {self.total_trades}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get current metrics summary."""
        if not self.episode_rewards:
            return {}
        
        return {
            "episodes_completed": len(self.episode_rewards),
            "updates_completed": self.update_count,
            "average_reward": sum(self.episode_rewards) / len(self.episode_rewards),
            "max_reward": max(self.episode_rewards),
            "min_reward": min(self.episode_rewards),
            "total_profit": self.total_profit,
            "total_trades": self.total_trades,
            "average_length": sum(self.episode_lengths) / len(self.episode_lengths) if self.episode_lengths else 0
        }