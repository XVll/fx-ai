"""Weights & Biases callback for experiment tracking and visualization."""

from typing import Dict, Any
import numpy as np
from collections import deque
from datetime import datetime

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from agent.callbacks import V1BaseCallback


class WandBCallbackV1(V1BaseCallback):
    """Callback for Weights & Biases experiment tracking.

    This callback:
    - Logs all training metrics to W&B
    - Tracks portfolio performance and trades
    - Records model internals and gradients
    - Handles feature attribution logging
    - Only performs calculations when enabled
    """

    def __init__(self, config: Dict[str, Any], enabled: bool = True):
        """Initialize W&B callback.

        Args:
            config: W&B configuration including project, tags, etc.
            enabled: Whether this callback is active
        """
        super().__init__(enabled)

        if not WANDB_AVAILABLE and enabled:
            self.logger.warning("wandb not installed. WandBCallback disabled.")
            self.enabled = False
            return

        self.config = config
        self.run = None

        # Tracking buffers
        self.episode_buffer = deque(maxlen=100)
        self.trade_history = []
        self.position_history = []

        # Aggregated stats
        self.total_episodes = 0
        self.total_steps = 0
        self.total_updates = 0

        # Portfolio tracking
        self.portfolio_stats = {
            "total_trades": 0,
            "winning_trades": 0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
            "peak_equity": 10000.0,  # Default starting capital
        }

        # Performance tracking
        self.step_timer = None
        self.update_timer = None
        self.rollout_timer = None

    def on_training_start(self, config: Dict[str, Any]) -> None:
        """Initialize W&B run."""
        if not self.enabled:
            return

        # Check if a run is already active
        if self.run is not None or wandb.run is not None:
            self.logger.info("W&B run already active, skipping initialization")
            self.run = wandb.run
            return

        # Use full_config if available, otherwise use the passed config
        wandb_config = config.get("full_config", config)

        # Initialize W&B run
        self.run = wandb.init(
            project=self.config.get("project", "fx-ai"),
            name=self.config.get("name", config.get("experiment_name", "training")),
            tags=self.config.get("tags", []),
            notes=self.config.get("notes", ""),
            config=wandb_config,
            resume=self.config.get("resume", "allow"),
            mode=self.config.get("mode", "online"),
            reinit=False,  # Prevent reinitialization if run exists
        )
        
        # Register with graceful shutdown manager
        try:
            from utils.graceful_shutdown import register_wandb_for_shutdown
            register_wandb_for_shutdown(self.run)
        except ImportError:
            # Graceful shutdown not available, continue without it
            pass

        # Define custom metrics only if this is a new run
        if self.run:
            wandb.define_metric("episode")
            wandb.define_metric("update")
            wandb.define_metric("global_step")

            # Set step metrics
            wandb.define_metric("episode/*", step_metric="episode")
            wandb.define_metric("update/*", step_metric="update")
            wandb.define_metric("step/*", step_metric="global_step")

            self.logger.info(f"W&B run initialized: {self.run.name}")
        else:
            self.logger.warning("Failed to initialize W&B run")

    def on_episode_start(self, episode_num: int, reset_info: Dict[str, Any]) -> None:
        """Track episode start."""
        if not self.enabled or not wandb.run:
            return

        self.total_episodes = episode_num

        # Log reset info
        date_str = ""
        if "date" in reset_info and reset_info["date"] is not None:
            try:
                date_str = reset_info["date"].strftime("%Y-%m-%d")
            except (AttributeError, TypeError):
                date_str = str(reset_info["date"])

        wandb.log(
            {
                "episode/symbol": reset_info.get("symbol", "UNKNOWN"),
                "episode/date": date_str,
                "episode/reset_time": reset_info.get("reset_time", ""),
                "episode": episode_num,
            }
        )

    def on_episode_end(self, episode_num: int, episode_data: Dict[str, Any]) -> None:
        """Log episode metrics."""
        if not self.enabled or not wandb.run:
            return

        # Extract episode metrics
        episode_reward = episode_data.get("episode_reward", 0.0)
        episode_length = episode_data.get("episode_length", 0)

        # Add to buffer for averaging
        self.episode_buffer.append(
            {
                "reward": episode_reward,
                "length": episode_length,
                "final_equity": episode_data.get("final_equity", 10000.0),
                "num_trades": episode_data.get("num_trades", 0),
            }
        )

        # Calculate running averages
        recent_episodes = list(self.episode_buffer)
        mean_reward = np.mean([ep["reward"] for ep in recent_episodes])
        mean_length = np.mean([ep["length"] for ep in recent_episodes])

        # Log episode metrics
        metrics = {
            "episode": episode_num,
            "episode/reward": episode_reward,
            "episode/length": episode_length,
            "episode/mean_reward": mean_reward,
            "episode/mean_length": mean_length,
            "episode/final_equity": episode_data.get("final_equity", 10000.0),
            "episode/num_trades": episode_data.get("num_trades", 0),
            "episode/final_position": episode_data.get("final_position", 0),
        }

        # Add termination reason
        if "termination_reason" in episode_data:
            metrics["episode/termination_reason"] = episode_data["termination_reason"]

        wandb.log(metrics)

    def on_episode_step(self, step_data: Dict[str, Any]) -> None:
        """Track step-level metrics."""
        if not self.enabled or not wandb.run:
            return

        self.total_steps += 1

        # Log every N steps to avoid overwhelming W&B
        if self.total_steps % 100 == 0:
            metrics = {
                "global_step": self.total_steps,
                "step/reward": step_data.get("reward", 0.0),
                "step/position": step_data.get("info", {}).get("position", 0),
                "step/equity": step_data.get("info", {}).get("total_equity", 10000.0),
            }

            # Add price if available
            if "current_price" in step_data.get("info", {}):
                metrics["step/price"] = step_data["info"]["current_price"]

            wandb.log(metrics)

    def on_rollout_start(self) -> None:
        """Track rollout timing."""
        if not self.enabled:
            return

        import time

        self.rollout_timer = time.time()

    def on_rollout_end(self, rollout_data: Dict[str, Any]) -> None:
        """Log rollout statistics."""
        if not self.enabled:
            return

        import time

        rollout_time = time.time() - self.rollout_timer if self.rollout_timer else 0

        metrics = {
            "rollout/time": rollout_time,
            "rollout/episodes": rollout_data.get("num_episodes", 0),
            "rollout/steps": rollout_data.get("num_steps", 0),
            "rollout/mean_reward": rollout_data.get("mean_reward", 0.0),
            "rollout/mean_length": rollout_data.get("mean_length", 0.0),
            "update": self.total_updates,
        }

        wandb.log(metrics)

    def on_update_start(self, update_num: int) -> None:
        """Track update timing."""
        if not self.enabled:
            return

        import time

        self.update_timer = time.time()
        self.total_updates = update_num

    def on_update_end(self, update_num: int, update_metrics: Dict[str, Any]) -> None:
        """Log PPO update metrics."""
        if not self.enabled or not wandb.run:
            return

        import time

        update_time = time.time() - self.update_timer if self.update_timer else 0

        # Log all update metrics
        metrics = {
            "update": update_num,
            "update/time": update_time,
            "update/policy_loss": update_metrics.get("policy_loss", 0.0),
            "update/value_loss": update_metrics.get("value_loss", 0.0),
            "update/entropy": update_metrics.get("entropy", 0.0),
            "update/total_loss": update_metrics.get("total_loss", 0.0),
            "update/learning_rate": update_metrics.get("learning_rate", 0.0),
            "update/clip_fraction": update_metrics.get("clip_fraction", 0.0),
            "update/kl_divergence": update_metrics.get("kl_divergence", 0.0),
            "update/explained_variance": update_metrics.get("explained_variance", 0.0),
        }

        # Add gradient norms if available
        if "grad_norm" in update_metrics:
            metrics["update/grad_norm"] = update_metrics["grad_norm"]
        if "grad_norm_clipped" in update_metrics:
            metrics["update/grad_norm_clipped"] = update_metrics["grad_norm_clipped"]

        wandb.log(metrics)

    def on_evaluation_end(self, eval_results: Dict[str, Any]) -> None:
        """Log evaluation results."""
        if not self.enabled or not wandb.run:
            return

        metrics = {
            "eval/mean_reward": eval_results.get("mean_reward", 0.0),
            "eval/mean_length": eval_results.get("mean_length", 0.0),
            "eval/win_rate": eval_results.get("win_rate", 0.0),
            "eval/profit_factor": eval_results.get("profit_factor", 0.0),
            "eval/sharpe_ratio": eval_results.get("sharpe_ratio", 0.0),
            "eval/max_drawdown": eval_results.get("max_drawdown", 0.0),
            "update": self.total_updates,
        }

        wandb.log(metrics)

    def on_order_filled(self, fill_data: Dict[str, Any]) -> None:
        """Track order execution metrics."""
        if not self.enabled or not wandb.run:
            return

        # Log execution quality
        metrics = {
            "execution/slippage": fill_data.get("slippage", 0.0),
            "execution/commission": fill_data.get("commission", 0.0),
            "execution/latency_ms": fill_data.get("latency_ms", 0.0),
            "global_step": self.total_steps,
        }

        wandb.log(metrics)

    def on_position_closed(self, trade_result: Dict[str, Any]) -> None:
        """Track completed trades."""
        if not self.enabled:
            return

        # Update portfolio stats
        pnl = trade_result.get("pnl", 0.0)
        self.portfolio_stats["total_trades"] += 1
        self.portfolio_stats["total_pnl"] += pnl
        if pnl > 0:
            self.portfolio_stats["winning_trades"] += 1

        # Calculate win rate
        win_rate = self.portfolio_stats["winning_trades"] / max(
            1, self.portfolio_stats["total_trades"]
        )

        # Log trade metrics
        metrics = {
            "trade/pnl": pnl,
            "trade/return_pct": trade_result.get("return_pct", 0.0),
            "trade/duration": trade_result.get("duration", 0),
            "trade/side": 1 if trade_result.get("side") == "long" else -1,
            "portfolio/total_trades": self.portfolio_stats["total_trades"],
            "portfolio/win_rate": win_rate,
            "portfolio/total_pnl": self.portfolio_stats["total_pnl"],
            "global_step": self.total_steps,
        }

        # Add to trade history for table logging
        self.trade_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "symbol": trade_result.get("symbol", "UNKNOWN"),
                "side": trade_result.get("side", "unknown"),
                "entry_price": trade_result.get("entry_price", 0.0),
                "exit_price": trade_result.get("exit_price", 0.0),
                "quantity": trade_result.get("quantity", 0),
                "pnl": pnl,
                "return_pct": trade_result.get("return_pct", 0.0),
                "duration": trade_result.get("duration", 0),
            }
        )

        wandb.log(metrics)

        # Periodically log trade table
        if len(self.trade_history) % 10 == 0:
            wandb.log(
                {
                    "trades_table": wandb.Table(
                        columns=list(self.trade_history[0].keys()),
                        data=[
                            list(t.values()) for t in self.trade_history[-50:]
                        ],  # Last 50 trades
                    )
                }
            )

    def on_portfolio_update(self, portfolio_state: Dict[str, Any]) -> None:
        """Track portfolio state."""
        if not self.enabled or not wandb.run:
            return

        equity = portfolio_state.get("total_equity", 10000.0)

        # Update drawdown
        if equity > self.portfolio_stats["peak_equity"]:
            self.portfolio_stats["peak_equity"] = equity

        drawdown = (
            self.portfolio_stats["peak_equity"] - equity
        ) / self.portfolio_stats["peak_equity"]
        self.portfolio_stats["max_drawdown"] = max(
            self.portfolio_stats["max_drawdown"], drawdown
        )

        # Log portfolio metrics
        metrics = {
            "portfolio/equity": equity,
            "portfolio/cash": portfolio_state.get("cash", 10000.0),
            "portfolio/position_value": portfolio_state.get("position_value", 0.0),
            "portfolio/unrealized_pnl": portfolio_state.get("unrealized_pnl", 0.0),
            "portfolio/drawdown": drawdown,
            "portfolio/max_drawdown": self.portfolio_stats["max_drawdown"],
            "global_step": self.total_steps,
        }

        wandb.log(metrics)

    def on_model_forward(self, forward_data: Dict[str, Any]) -> None:
        """Track model internals."""
        if not self.enabled:
            return

        # Log action probabilities
        if "action_probs" in forward_data:
            action_probs = forward_data["action_probs"]
            metrics = {
                f"model/action_prob_{i}": prob for i, prob in enumerate(action_probs)
            }
            metrics["global_step"] = self.total_steps
            wandb.log(metrics)

        # Log attention weights periodically
        if self.total_steps % 1000 == 0 and "attention_weights" in forward_data:
            # Log as histogram
            wandb.log(
                {
                    "model/attention_weights": wandb.Histogram(
                        forward_data["attention_weights"].flatten()
                    ),
                    "global_step": self.total_steps,
                }
            )

    def on_gradient_update(self, gradient_data: Dict[str, Any]) -> None:
        """Track gradient statistics."""
        if not self.enabled:
            return

        # Log gradient norms by layer
        if "layer_grad_norms" in gradient_data:
            metrics = {
                f"gradients/{name}": norm
                for name, norm in gradient_data["layer_grad_norms"].items()
            }
            metrics["update"] = self.total_updates
            wandb.log(metrics)

    def on_attribution_computed(self, attribution_data: Dict[str, Any]) -> None:
        """Log feature attribution results."""
        if not self.enabled:
            return

        # Log top feature importances
        if "feature_importances" in attribution_data:
            importances = attribution_data["feature_importances"]
            top_features = sorted(
                importances.items(), key=lambda x: abs(x[1]), reverse=True
            )[:20]

            metrics = {f"attribution/{name}": value for name, value in top_features}
            metrics["update"] = self.total_updates
            wandb.log(metrics)

        # Log dead feature count
        if "dead_features" in attribution_data:
            wandb.log(
                {
                    "attribution/dead_feature_count": len(
                        attribution_data["dead_features"]
                    ),
                    "attribution/dead_feature_ratio": attribution_data.get(
                        "dead_feature_ratio", 0.0
                    ),
                    "update": self.total_updates,
                }
            )

    def on_momentum_day_change(self, day_info: Dict[str, Any]) -> None:
        """Track momentum day changes."""
        if not self.enabled or not self.run:
            return

        # Extract day_info from event data structure (handle both formats)
        if "day_info" in day_info:
            actual_day_info = day_info["day_info"]
        else:
            actual_day_info = day_info

        # Safe date handling
        date_str = ""
        if "date" in actual_day_info and actual_day_info["date"] is not None:
            try:
                date_str = actual_day_info["date"].strftime("%Y-%m-%d")
            except (AttributeError, TypeError):
                date_str = str(actual_day_info["date"])

        wandb.log(
            {
                "momentum/day": date_str,
                "momentum/quality_score": actual_day_info.get("quality_score", 0.0),
                "momentum/activity_score": actual_day_info.get("activity_score", 0.0),
                "momentum/stage": actual_day_info.get("curriculum_stage", 0),
                "episode": self.total_episodes,
            }
        )

    def on_curriculum_stage_change(self, stage_info: Dict[str, Any]) -> None:
        """Track curriculum progression."""
        if not self.enabled:
            return

        wandb.log(
            {
                "curriculum/stage": stage_info.get("stage", 0),
                "curriculum/progress": stage_info.get("progress", 0.0),
                "curriculum/performance_threshold": stage_info.get(
                    "performance_threshold", 0.0
                ),
                "episode": self.total_episodes,
            }
        )

    def on_training_end(self, final_stats: Dict[str, Any]) -> None:
        """Log final summary and close W&B run."""
        if not self.enabled or not self.run:
            return

        try:
            # Check if this is an interruption - if so, finish immediately
            is_interrupted = final_stats.get("interrupted", False)

            if is_interrupted:
                # For interruptions, just mark as interrupted and finish quickly
                wandb.run.summary["interrupted"] = True
                wandb.run.summary["final_status"] = "interrupted"
                wandb.finish(quiet=True)
                self.logger.info("W&B run interrupted and closed")
                return

            # For normal completion, do full logging
            # Log final summary
            summary = {
                "total_episodes": self.total_episodes,
                "total_steps": self.total_steps,
                "total_updates": self.total_updates,
                "total_trades": self.portfolio_stats["total_trades"],
                "final_win_rate": (
                    self.portfolio_stats["winning_trades"]
                    / max(1, self.portfolio_stats["total_trades"])
                ),
                "total_pnl": self.portfolio_stats["total_pnl"],
                "max_drawdown": self.portfolio_stats["max_drawdown"],
                "best_episode_reward": max(
                    [ep["reward"] for ep in self.episode_buffer], default=0.0
                ),
                "final_mean_reward": np.mean(
                    [ep["reward"] for ep in self.episode_buffer]
                )
                if self.episode_buffer
                else 0.0,
                "final_status": "completed",
            }

            # Add any additional final stats
            summary.update(final_stats)

            # Update W&B summary
            for key, value in summary.items():
                wandb.run.summary[key] = value

            # Log final trade table
            if self.trade_history:
                wandb.log(
                    {
                        "final_trades_table": wandb.Table(
                            columns=list(self.trade_history[0].keys()),
                            data=[list(t.values()) for t in self.trade_history],
                        )
                    }
                )

            # Finish the run
            wandb.finish()
            self.logger.info("W&B run completed and closed")

        except Exception as e:
            # If anything fails during cleanup, force finish
            self.logger.warning(f"Error during W&B cleanup: {e}")
            try:
                wandb.finish(quiet=True)
            except:
                pass
