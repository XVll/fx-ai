# agent/wandb_callback.py
import wandb
import numpy as np
import pandas as pd
from typing import Dict, Any
import matplotlib.pyplot as plt
import io
from PIL import Image
import torch

from agent.callbacks import TrainingCallback


class WandbCallback(TrainingCallback):
    """Callback for tracking experiments with Weights & Biases."""

    def __init__(self,
                 project_name="ai-trading",
                 entity=None,
                 log_freq=1,
                 config=None,
                 log_model=True,
                 log_code=True):
        """
        Initialize W&B tracking.

        Args:
            project_name: W&B project name
            entity: W&B entity name (username or team)
            log_freq: Logging frequency in steps
            config: Configuration dict to track
            log_model: Whether to log model checkpoints
            log_code: Whether to track code changes
        """
        self.project_name = project_name
        self.entity = entity
        self.log_freq = log_freq
        self.config = config or {}
        self.log_model = log_model
        self.log_code = log_code
        self.step_count = 0
        self.run = None
        self.best_reward = -float('inf')

        # Trade analytics tracking
        self.trade_history = []
        self.profit_factor = 0
        self.accuracy = 0
        self.avg_win = 0
        self.avg_loss = 0

        # Training analytics
        self.reward_history = []
        self.action_history = []
        self.price_history = []
        self.position_history = []

        # Market state snapshots
        self.marked_snapshots = []

    def _create_custom_charts(self, trainer):
        """Create custom charts for W&B."""
        # Trade analysis chart
        if self.trade_history:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

            # Chart 1: Cumulative PnL
            pnl_df = pd.DataFrame(self.trade_history)
            if 'realized_pnl' in pnl_df.columns:
                cumulative_pnl = pnl_df['realized_pnl'].cumsum()
                ax1.plot(cumulative_pnl.values)
                ax1.set_title('Cumulative PnL')
                ax1.set_xlabel('Trade #')
                ax1.set_ylabel('Cumulative PnL ($)')
                ax1.grid(True)

            # Chart 2: Win/Loss Distribution
            if 'realized_pnl' in pnl_df.columns:
                ax2.hist(pnl_df['realized_pnl'].values, bins=20)
                ax2.set_title('PnL Distribution')
                ax2.set_xlabel('PnL per Trade ($)')
                ax2.set_ylabel('Frequency')
                ax2.grid(True)

            plt.tight_layout()

            # Convert plot to image
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img = Image.open(buf)

            # Log to W&B
            wandb.log({"trade_analysis": wandb.Image(img)})
            plt.close(fig)

        # Market analysis
        if len(self.price_history) > 0 and len(self.position_history) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))

            # Price chart with position overlay
            ax.plot(self.price_history, label='Price')
            ax.set_title('Price with Position Sizing')
            ax.set_xlabel('Step')
            ax.set_ylabel('Price ($)')

            # Create a twin axis for position
            ax2 = ax.twinx()
            ax2.plot(self.position_history, 'r-', alpha=0.5, label='Position')
            ax2.set_ylabel('Position Size')

            # Add legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

            plt.grid(True)
            plt.tight_layout()

            # Convert to image
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img = Image.open(buf)

            # Log to W&B
            wandb.log({"market_position_analysis": wandb.Image(img)})
            plt.close(fig)

    def _log_model_gradients(self, trainer):
        """Log model parameter gradients to W&B."""
        gradients = {}
        for name, param in trainer.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                gradients[f"gradients/{name}"] = wandb.Histogram(param.grad.detach().cpu().numpy())
        wandb.log(gradients)

    def on_training_start(self, trainer):
        """Called when training starts."""
        # Initialize W&B run
        self.run = wandb.init(
            project=self.project_name,
            entity=self.entity,
            config=self.config,
            save_code=self.log_code,
            job_type="training"
        )

        # Log model architecture as a summary
        wandb.run.summary["model_summary"] = str(trainer.model)

        # Create a table for tracking detailed trade information
        self.trade_table = wandb.Table(columns=[
            "trade_id", "entry_time", "exit_time", "entry_price",
            "exit_price", "position_size", "realized_pnl",
            "trade_duration", "entry_signal", "exit_signal"
        ])

        # Log model graph (architecture)
        try:
            # Sample input for forward pass
            sample_batch = {
                'hf_features': torch.zeros((1, trainer.model.hf_seq_len, trainer.model.hf_feat_dim)).to(trainer.device),
                'mf_features': torch.zeros((1, trainer.model.mf_seq_len, trainer.model.mf_feat_dim)).to(trainer.device),
                'lf_features': torch.zeros((1, trainer.model.lf_seq_len, trainer.model.lf_feat_dim)).to(trainer.device),
                'static_features': torch.zeros((1, trainer.model.static_feat_dim)).to(trainer.device)
            }
            wandb.watch(trainer.model, log="all", log_freq=100)
        except Exception as e:
            trainer.logger.warning(f"Failed to log model graph: {str(e)}")

    def on_training_end(self, trainer, stats):
        """Called when training ends."""
        # Log final performance metrics
        wandb.log({
            "final/episodes": stats["total_episodes"],
            "final/steps": stats["total_steps"],
            "final/best_reward": stats["best_mean_reward"],
            "final/training_time": stats["elapsed_time"]
        })

        # Create and log final custom visualizations
        self._create_custom_charts(trainer)

        # Log the best model if requested
        if self.log_model and stats.get("best_model_path"):
            wandb.save(stats["best_model_path"])

        # Compute and log final trade statistics
        if self.trade_history:
            trade_df = pd.DataFrame(self.trade_history)
            if 'realized_pnl' in trade_df.columns:
                wins = trade_df[trade_df['realized_pnl'] > 0]
                losses = trade_df[trade_df['realized_pnl'] <= 0]

                accuracy = len(wins) / len(trade_df) if len(trade_df) > 0 else 0
                avg_win = wins['realized_pnl'].mean() if len(wins) > 0 else 0
                avg_loss = losses['realized_pnl'].mean() if len(losses) > 0 else 0
                profit_factor = abs(wins['realized_pnl'].sum() / losses['realized_pnl'].sum()) if len(losses) > 0 and \
                                                                                                  losses[
                                                                                                      'realized_pnl'].sum() != 0 else 0

                wandb.run.summary.update({
                    "final_accuracy": accuracy,
                    "final_profit_factor": profit_factor,
                    "final_avg_win": avg_win,
                    "final_avg_loss": avg_loss,
                    "total_trades": len(trade_df),
                    "total_pnl": trade_df['realized_pnl'].sum()
                })

        # Close the W&B run
        wandb.finish()

    def on_rollout_start(self, trainer):
        """Called before collecting rollouts."""
        wandb.log({"rollout/start_time": wandb.run.start_time})

    def on_rollout_end(self, trainer):
        """Called after collecting rollouts."""
        pass

    def on_step(self, trainer, state, action, reward, next_state, info):
        """Called after each environment step."""
        self.step_count += 1

        # Record data for later visualization
        if hasattr(trainer.env.simulator, 'market_simulator'):
            market_state = trainer.env.simulator.market_simulator.get_current_market_state()
            if market_state:
                self.price_history.append(market_state.get('price', 0))

        # Record position for visualization
        position = 0
        if hasattr(trainer.env.simulator, 'portfolio_simulator'):
            portfolio_state = trainer.env.simulator.portfolio_simulator.get_portfolio_state()
            if portfolio_state and 'positions' in portfolio_state:
                for symbol, pos in portfolio_state['positions'].items():
                    position = pos.get('quantity', 0)
        self.position_history.append(position)

        # Record action
        if isinstance(action, torch.Tensor):
            action_value = action.item() if action.numel() == 1 else action.cpu().numpy()
        else:
            action_value = action
        self.action_history.append(action_value)

        # Log metrics at specified frequency
        if self.step_count % self.log_freq == 0:
            metrics = {
                "step/reward": reward,
                "step/action": action_value,
                "step/total_steps": trainer.total_steps
            }

            # Add info metrics if available
            if isinstance(info, dict):
                for k, v in info.items():
                    if isinstance(v, (int, float)):
                        metrics[f"step/info_{k}"] = v

            wandb.log(metrics, step=trainer.total_steps)

            # Log gradients periodically
            if self.step_count % (self.log_freq * 10) == 0:
                self._log_model_gradients(trainer)

    def on_episode_end(self, trainer, episode_reward, episode_length, info):
        """Called at the end of an episode."""
        # Record reward for later analysis
        self.reward_history.append(episode_reward)

        # Log episode metrics
        metrics = {
            "episode/reward": episode_reward,
            "episode/length": episode_length,
            "episode/total_episodes": trainer.total_episodes
        }

        # Log trade information if available
        if info and "episode" in info:
            episode_info = info["episode"]
            for k, v in episode_info.items():
                if isinstance(v, (int, float)):
                    metrics[f"episode/{k}"] = v

        # Track best performance
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            metrics["episode/best_reward"] = self.best_reward

        # Process trade data if available
        if hasattr(trainer.env.simulator, 'portfolio_simulator'):
            trades = trainer.env.simulator.portfolio_simulator.get_trade_history()
            if trades:
                # Add new trades to history
                for trade in trades:
                    if trade not in self.trade_history:
                        self.trade_history.append(trade)

                        # Add to W&B table
                        self.trade_table.add_data(
                            len(self.trade_history),
                            str(trade.get('open_time', '')),
                            str(trade.get('close_time', '')),
                            trade.get('entry_price', 0),
                            trade.get('exit_price', 0),
                            trade.get('quantity', 0),
                            trade.get('realized_pnl', 0),
                            str(trade.get('close_time', '') - trade.get('open_time', '')) if trade.get(
                                'close_time') and trade.get('open_time') else '',
                            trade.get('entry_signal', 'unknown'),
                            trade.get('exit_signal', 'unknown')
                        )

                # Update trade metrics
                win_trades = [t for t in trades if t.get('realized_pnl', 0) > 0]
                loss_trades = [t for t in trades if t.get('realized_pnl', 0) <= 0]

                self.accuracy = len(win_trades) / len(trades) if len(trades) > 0 else 0
                self.avg_win = sum(t.get('realized_pnl', 0) for t in win_trades) / len(win_trades) if len(
                    win_trades) > 0 else 0
                self.avg_loss = sum(t.get('realized_pnl', 0) for t in loss_trades) / len(loss_trades) if len(
                    loss_trades) > 0 else 0

                win_sum = sum(t.get('realized_pnl', 0) for t in win_trades)
                loss_sum = abs(sum(t.get('realized_pnl', 0) for t in loss_trades))
                self.profit_factor = win_sum / loss_sum if loss_sum > 0 else float('inf')

                metrics.update({
                    "trades/accuracy": self.accuracy,
                    "trades/profit_factor": self.profit_factor,
                    "trades/avg_win": self.avg_win,
                    "trades/avg_loss": self.avg_loss,
                    "trades/total": len(trades)
                })

                # Log trade table periodically
                if len(self.trade_history) % 10 == 0:
                    wandb.log({"trades_table": self.trade_table})

        wandb.log(metrics)

        # Create custom charts every 5 episodes
        if trainer.total_episodes % 5 == 0:
            self._create_custom_charts(trainer)

    def on_update_start(self, trainer):
        """Called before policy update."""
        pass

    def on_update_end(self, trainer, metrics):
        """Called after policy update."""
        # Log update metrics
        update_metrics = {f"update/{k}": v for k, v in metrics.items()}
        update_metrics["update/updates"] = trainer.updates
        wandb.log(update_metrics)

    def on_update_iteration_end(self, trainer, update_iter, update_metrics, rollout_stats):
        """Called at the end of each update iteration (rollout + update)."""
        combined_metrics = {}
        combined_metrics.update({f"update_iter/{k}": v for k, v in update_metrics.items()})
        combined_metrics.update({f"rollout_iter/{k}": v for k, v in rollout_stats.items()
                                 if k not in ["episode_rewards", "episode_lengths"]})
        combined_metrics["iteration"] = update_iter

        wandb.log(combined_metrics)