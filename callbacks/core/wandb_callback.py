"""
Weights & Biases logging callback.

Provides comprehensive experiment tracking with WandB integration
for metrics, model parameters, and training artifacts.
"""

from typing import Dict, Any, Optional, List
import logging
from .base import BaseCallback

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class WandBCallback(BaseCallback):
    """
    Weights & Biases logging callback.
    
    Logs comprehensive training metrics, model parameters, and artifacts
    to WandB for experiment tracking and analysis.
    """
    
    def __init__(
        self,
        enabled: bool = True,
        project: str = "fxai-v2",
        entity: Optional[str] = None,
        tags: Optional[List[str]] = None,
        log_freq: int = 10,
        log_gradients: bool = False,
        log_parameters: bool = True,
        name: Optional[str] = None
    ):
        """
        Initialize WandB callback.
        
        Args:
            enabled: Whether callback is active
            project: WandB project name
            entity: WandB entity/team name
            tags: List of tags for the run
            log_freq: Frequency for logging metrics (episodes)
            log_gradients: Whether to log gradient norms
            log_parameters: Whether to log model parameters
            name: Optional custom name
        """
        super().__init__(enabled, name)
        
        if not WANDB_AVAILABLE:
            self.logger.warning("WandB not available, disabling WandBCallback")
            self.enabled = False
            return
        
        self.project = project
        self.entity = entity
        self.tags = tags or []
        self.log_freq = log_freq
        self.log_gradients = log_gradients
        self.log_parameters = log_parameters
        
        # WandB run
        self.run = None
        self.run_initialized = False
        
        # Metrics tracking
        self.episode_metrics = []
        self.update_metrics = []
    
    def on_training_start(self, context: Dict[str, Any]) -> None:
        """Initialize WandB run."""
        super().on_training_start(context)
        
        if not self.enabled:
            return
        
        try:
            config = context.get("config", {})
            trainer = context.get("trainer")
            
            # Initialize WandB run
            self.run = wandb.init(
                project=self.project,
                entity=self.entity,
                tags=self.tags,
                config=config,
                reinit=True
            )
            
            self.run_initialized = True
            
            self.logger.info(f"🔗 WandB run initialized: {self.run.name}")
            self.logger.info(f"   Project: {self.project}")
            self.logger.info(f"   URL: {self.run.url}")
            
            # Log model architecture if available
            if trainer and hasattr(trainer, 'network') and self.log_parameters:
                try:
                    wandb.watch(trainer.network, log='all', log_freq=self.log_freq * 10)
                    self.logger.info("   Model watching enabled")
                except Exception as e:
                    self.logger.warning(f"Could not watch model: {e}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize WandB: {e}")
            self.enabled = False
    
    def on_episode_end(self, context: Dict[str, Any]) -> None:
        """Log episode metrics to WandB."""
        super().on_episode_end(context)
        
        if not self.enabled or not self.run_initialized:
            return
        
        try:
            episode_info = context.get("episode", {})
            metrics = context.get("metrics", {})
            portfolio = context.get("portfolio", {})
            
            episode_num = episode_info.get("num", self.episode_count)
            
            # Prepare episode metrics
            wandb_metrics = {
                "episode/number": episode_num,
                "episode/reward": episode_info.get("reward", 0.0),
                "episode/length": episode_info.get("length", 0),
                "episode/terminated": episode_info.get("terminated", False),
            }
            
            # Add trading metrics
            if metrics:
                wandb_metrics.update({
                    "trading/portfolio_value": metrics.get("portfolio_value", 0.0),
                    "trading/total_profit": metrics.get("total_profit", 0.0),
                    "trading/trades_count": metrics.get("trades_count", 0),
                    "trading/win_rate": metrics.get("win_rate", 0.0),
                    "trading/avg_trade_profit": metrics.get("avg_trade_profit", 0.0),
                    "trading/max_drawdown": metrics.get("max_drawdown", 0.0),
                })
            
            # Add portfolio metrics
            if portfolio:
                wandb_metrics.update({
                    "portfolio/cash": portfolio.get("cash", 0.0),
                    "portfolio/position_value": portfolio.get("position_value", 0.0),
                    "portfolio/total_value": portfolio.get("total_value", 0.0),
                    "portfolio/position_count": portfolio.get("position_count", 0),
                })
            
            # Log at specified frequency
            if episode_num % self.log_freq == 0:
                self.run.log(wandb_metrics, step=episode_num)
                
                # Log running averages
                self._log_running_averages(episode_num)
            
            # Store metrics for averaging
            self.episode_metrics.append(wandb_metrics)
            
        except Exception as e:
            self.logger.error(f"Failed to log episode metrics to WandB: {e}")
    
    def on_update_end(self, context: Dict[str, Any]) -> None:
        """Log training update metrics to WandB."""
        super().on_update_end(context)
        
        if not self.enabled or not self.run_initialized:
            return
        
        try:
            update_info = context.get("update", {})
            losses = context.get("losses", {})
            metrics = context.get("metrics", {})
            
            update_num = update_info.get("num", self.update_count)
            
            # Prepare training metrics
            wandb_metrics = {
                "training/update_number": update_num,
                "training/learning_rate": update_info.get("learning_rate", 0.0),
                "training/clip_epsilon": update_info.get("clip_epsilon", 0.0),
            }
            
            # Add loss metrics
            if losses:
                wandb_metrics.update({
                    "losses/policy_loss": losses.get("policy_loss", 0.0),
                    "losses/value_loss": losses.get("value_loss", 0.0),
                    "losses/entropy_loss": losses.get("entropy_loss", 0.0),
                    "losses/total_loss": losses.get("total_loss", 0.0),
                })
            
            # Add training metrics
            if metrics:
                wandb_metrics.update({
                    "training/kl_divergence": metrics.get("kl_divergence", 0.0),
                    "training/clip_fraction": metrics.get("clip_fraction", 0.0),
                    "training/gradient_norm": metrics.get("gradient_norm", 0.0),
                    "training/explained_variance": metrics.get("explained_variance", 0.0),
                })
            
            # Log training metrics (less frequent)
            if update_num % (self.log_freq * 2) == 0:
                self.run.log(wandb_metrics, step=update_num)
            
            # Store metrics
            self.update_metrics.append(wandb_metrics)
            
        except Exception as e:
            self.logger.error(f"Failed to log training metrics to WandB: {e}")
    
    def on_training_end(self, context: Dict[str, Any]) -> None:
        """Log final summary and close WandB run."""
        super().on_training_end(context)
        
        if not self.enabled or not self.run_initialized:
            return
        
        try:
            final_metrics = context.get("final_metrics", {})
            total_episodes = context.get("total_episodes", len(self.episode_metrics))
            duration = context.get("duration", "unknown")
            
            # Log final summary
            final_summary = {
                "summary/total_episodes": total_episodes,
                "summary/total_updates": self.update_count,
                "summary/duration": str(duration),
                "summary/best_reward": final_metrics.get("best_reward", 0.0),
                "summary/average_reward": final_metrics.get("average_reward", 0.0),
                "summary/total_profit": final_metrics.get("total_profit", 0.0),
            }
            
            self.run.log(final_summary)
            
            # Create summary table if we have episode data
            if self.episode_metrics:
                self._create_summary_table()
            
            self.logger.info(f"🔗 WandB logging completed")
            self.logger.info(f"   Total episodes logged: {len(self.episode_metrics)}")
            self.logger.info(f"   Total updates logged: {len(self.update_metrics)}")
            
        except Exception as e:
            self.logger.error(f"Failed to log final summary to WandB: {e}")
        
        finally:
            if self.run:
                try:
                    self.run.finish()
                    self.logger.info("   WandB run finished")
                except Exception as e:
                    self.logger.error(f"Failed to finish WandB run: {e}")
    
    def _log_running_averages(self, episode_num: int) -> None:
        """Log running averages of key metrics."""
        if len(self.episode_metrics) < self.log_freq:
            return
        
        try:
            # Get recent metrics
            recent_metrics = self.episode_metrics[-self.log_freq:]
            
            # Calculate averages
            avg_reward = sum(m.get("episode/reward", 0) for m in recent_metrics) / len(recent_metrics)
            avg_length = sum(m.get("episode/length", 0) for m in recent_metrics) / len(recent_metrics)
            avg_profit = sum(m.get("trading/total_profit", 0) for m in recent_metrics) / len(recent_metrics)
            
            running_averages = {
                f"running_avg/reward_{self.log_freq}ep": avg_reward,
                f"running_avg/length_{self.log_freq}ep": avg_length,
                f"running_avg/profit_{self.log_freq}ep": avg_profit,
            }
            
            self.run.log(running_averages, step=episode_num)
            
        except Exception as e:
            self.logger.warning(f"Failed to log running averages: {e}")
    
    def _create_summary_table(self) -> None:
        """Create summary table of key metrics."""
        try:
            import pandas as pd
            
            # Extract key metrics for table
            table_data = []
            for i, metrics in enumerate(self.episode_metrics[-100:]):  # Last 100 episodes
                table_data.append({
                    "Episode": metrics.get("episode/number", i),
                    "Reward": metrics.get("episode/reward", 0.0),
                    "Length": metrics.get("episode/length", 0),
                    "Profit": metrics.get("trading/total_profit", 0.0),
                    "Trades": metrics.get("trading/trades_count", 0),
                    "Win Rate": metrics.get("trading/win_rate", 0.0),
                })
            
            if table_data:
                df = pd.DataFrame(table_data)
                table = wandb.Table(dataframe=df)
                self.run.log({"summary/episode_table": table})
                self.logger.info("   Summary table created")
                
        except Exception as e:
            self.logger.warning(f"Failed to create summary table: {e}")
    
    def log_custom_metric(self, name: str, value: Any, step: Optional[int] = None) -> None:
        """
        Log custom metric to WandB.
        
        Args:
            name: Metric name
            value: Metric value
            step: Optional step for x-axis
        """
        if not self.enabled or not self.run_initialized:
            return
        
        try:
            self.run.log({name: value}, step=step)
        except Exception as e:
            self.logger.error(f"Failed to log custom metric {name}: {e}")
    
    def log_artifact(self, file_path: str, name: str, artifact_type: str = "model") -> None:
        """
        Log artifact to WandB.
        
        Args:
            file_path: Path to file to upload
            name: Artifact name
            artifact_type: Type of artifact
        """
        if not self.enabled or not self.run_initialized:
            return
        
        try:
            artifact = wandb.Artifact(name, type=artifact_type)
            artifact.add_file(file_path)
            self.run.log_artifact(artifact)
            self.logger.info(f"Logged artifact: {name}")
        except Exception as e:
            self.logger.error(f"Failed to log artifact {name}: {e}")
    
    def shutdown(self) -> None:
        """Shutdown WandB callback."""
        super().shutdown()
        
        if self.run and self.run_initialized:
            try:
                self.run.finish()
                self.logger.info("WandB run finished during shutdown")
            except Exception as e:
                self.logger.error(f"Error finishing WandB run during shutdown: {e}")