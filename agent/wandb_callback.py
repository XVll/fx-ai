# agent/wandb_callback.py
import os
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Any, Optional
from datetime import datetime
import io
from PIL import Image

from agent.callbacks import TrainingCallback


class WandbCallback(TrainingCallback):
    """
    Callback for tracking training with Weights & Biases.

    This callback handles all WandB logging, visualization creation,
    and model saving throughout the training process.
    """

    def __init__(
            self,
            project_name: str = "ai-trading",
            entity: Optional[str] = None,
            log_freq: int = 10,
            config: Optional[Dict[str, Any]] = None,
            log_model: bool = True,
            log_code: bool = True,
    ):
        """
        Initialize WandB callback.

        Args:
            project_name: WandB project name
            entity: WandB team or username
            log_freq: Frequency for logging step metrics
            config: Configuration dict to log to WandB
            log_model: Whether to save model checkpoints to WandB
            log_code: Whether to track code with WandB
        """
        # Basic settings
        self.project_name = project_name
        self.entity = entity
        self.log_freq = log_freq
        self.config = config or {}
        self.log_model = log_model
        self.log_code = log_code

        # Initialize WandB run
        self.run = None

        # Step tracking - single source of truth for step counts
        self.global_step = 0
        self.step_count = 0
        self.last_logged_step = 0

        # Metric tracking
        self.best_reward = -float('inf')
        self.best_model_path = None

        # Data for visualizations
        self.trade_history = []
        self.price_history = []
        self.position_history = []
        self.action_history = []
        self.reward_history = []

        # Logger
        self.logger = None

    def on_training_start(self, trainer):
        """Called when training starts."""
        # Create WandB directory in the output directory
        wandb_dir = os.path.join(trainer.output_dir, "wandb")
        os.makedirs(wandb_dir, exist_ok=True)

        # Set logger
        self.logger = trainer.logger

        # Initialize the WandB run
        self.run = wandb.init(
            dir=wandb_dir,  # Use the created directory for WandB files
            project=self.project_name,
            entity=self.entity,
            config=self.config,
            save_code=self.log_code,
            job_type="training",
            reinit=True  # Ensure we create a new run
        )
        # CRITICAL: Initialize step tracking to match trainer
        if hasattr(trainer, 'total_steps'):
            self.global_step = trainer.total_steps
        else:
            self.global_step = 0

        self._last_logged_step = self.global_step - 1  # Ensure next log will use current step
        # Log model architecture
        if hasattr(trainer, 'model'):
            wandb.run.summary["model_architecture"] = str(trainer.model)

            # Set up model watching if available
            if hasattr(wandb, 'watch') and self.config.get('watch_model', True):
                wandb.watch(
                    trainer.model,
                    log="all",
                    log_freq=max(100, self.log_freq * 10)
                )

        # Create tables for tracking
        self.trade_table = wandb.Table(columns=[
            "trade_id", "entry_time", "exit_time", "entry_price",
            "exit_price", "position_size", "realized_pnl",
            "duration_seconds", "trade_type"
        ])

        # Log initial message
        print(f"WandB initialized: {wandb.run.name} ({wandb.run.id})")
    def on_training_end(self, trainer, stats):
        """Called when training ends."""
        # Log final statistics
        final_metrics = {
            "final/total_episodes": stats.get("total_episodes", 0),
            "final/total_steps": stats.get("total_steps", 0),
            "final/best_reward": stats.get("best_mean_reward", 0),
            "final/training_time_seconds": stats.get("elapsed_time", 0),
        }

        # Add trade statistics if available
        if hasattr(trainer.env, 'portfolio_simulator'):
            trade_stats = trainer.env.portfolio_simulator.get_statistics()
            if trade_stats:
                final_metrics.update({
                    "final/trade_count": trade_stats.get("total_trades", 0),
                    "final/win_rate": trade_stats.get("win_rate", 0),
                    "final/total_pnl": trade_stats.get("total_pnl", 0),
                })

        # Log final metrics
        self._safe_wandb_log(final_metrics)

        # Log trade table if we have trades
        if len(self.trade_table.data) > 0:
            wandb.log({"final_trades": self.trade_table})

        # Create final visualizations
        self._create_final_visualizations()

        # Upload best model if we have one and log_model is enabled
        if self.log_model and self.best_model_path and os.path.exists(self.best_model_path):
            wandb.save(self.best_model_path, base_path=trainer.output_dir)
            wandb.run.summary["best_model_path"] = self.best_model_path

        # Finish the run
        wandb.finish()
    def on_rollout_start(self, trainer):
        """Called before collecting rollouts."""
        # We don't need to do anything here
        pass
    def on_rollout_end(self, trainer):
        """Called after collecting rollouts."""
        # We don't need to do anything here
        pass
    def on_step(self, trainer, state, action, reward, next_state, info):
        """Called after each environment step."""
        # Update our step counter
        self.step_count += 1

        # Track data for visualizations
        self.reward_history.append(reward)

        # Track action
        if isinstance(action, torch.Tensor):
            action_value = action.item() if action.numel() == 1 else action.cpu().numpy().tolist()
        else:
            action_value = action
        self.action_history.append(action_value)

        # Track market data if available
        if hasattr(trainer.env, 'market_simulator'):
            market_state = trainer.env.market_simulator.get_current_market_state()
            if market_state and 'price' in market_state:
                self.price_history.append(market_state['price'])
            elif market_state and 'current_1s_bar' in market_state and market_state['current_1s_bar']:
                # Try to get price from 1s bar
                self.price_history.append(market_state['current_1s_bar'].get('close', 0))

        # Track position if available
        if hasattr(trainer.env, 'portfolio_simulator'):
            portfolio_state = trainer.env.portfolio_simulator.get_portfolio_state()
            if portfolio_state:
                position = portfolio_state.get('position', 0)
                self.position_history.append(position)

        # Log metrics at specified frequency
        if self.step_count % self.log_freq == 0:
            # Increment global step to ensure monotonicity
            self.global_step += 1

            # Create metrics dict
            metrics = {
                "step/reward": reward,
                "step/action": action_value,
                "step/step_count": self.step_count,
                "step/global_step": self.global_step,
            }

            # Add market data if available
            if hasattr(trainer.env, 'market_simulator'):
                market_state = trainer.env.market_simulator.get_current_market_state()
                if market_state:
                    if 'current_1s_bar' in market_state and market_state['current_1s_bar']:
                        bar = market_state['current_1s_bar']
                        metrics["market/price"] = bar.get('close', 0)
                        metrics["market/volume"] = bar.get('volume', 0)

            # Add position data if available
            if hasattr(trainer.env, 'portfolio_simulator'):
                portfolio_state = trainer.env.portfolio_simulator.get_portfolio_state()
                if portfolio_state:
                    metrics["portfolio/cash"] = portfolio_state.get('cash', 0)
                    metrics["portfolio/position"] = portfolio_state.get('position', 0)
                    metrics["portfolio/total_value"] = portfolio_state.get('total_value', 0)

            # Add info metrics
            if isinstance(info, dict):
                for k, v in info.items():
                    if isinstance(v, (int, float)) and k not in ['timestamp', 'action_result']:
                        metrics[f"info/{k}"] = v

            # Log metrics
            self._safe_wandb_log(metrics)
    def on_episode_end(self, trainer, episode_reward, episode_length, info):
        """Called at the end of an episode."""
        # Increment global step
        self.global_step += 1

        # Create base metrics
        metrics = {
            "episode/reward": episode_reward,
            "episode/length": episode_length,
            "episode/number": trainer.total_episodes,
        }

        # Add episode info if available
        if isinstance(info, dict) and 'episode' in info:
            episode_info = info['episode']
            for k, v in episode_info.items():
                if isinstance(v, (int, float)):
                    metrics[f"episode/{k}"] = v

        # Process trades
        if hasattr(trainer.env, 'portfolio_simulator'):
            # Get trade history
            trades = trainer.env.portfolio_simulator.get_trade_history()

            # Add new trades to our trade records
            for trade in trades:
                # Check if we've already recorded this trade
                if trade not in self.trade_history:
                    self.trade_history.append(trade)

                    # Add to WandB table
                    trade_row = [
                        len(self.trade_history),  # trade_id
                        str(trade.get('open_time', '')),  # entry_time
                        str(trade.get('close_time', '')),  # exit_time
                        trade.get('entry_price', 0),  # entry_price
                        trade.get('exit_price', 0),  # exit_price
                        trade.get('quantity', 0),  # position_size
                        trade.get('realized_pnl', 0),  # realized_pnl
                        (trade.get('close_time', 0) - trade.get('open_time', 0)).total_seconds()
                        if trade.get('close_time') and trade.get('open_time') else 0,  # duration
                        'buy' if trade.get('quantity', 0) > 0 else 'sell'  # trade_type
                    ]
                    self.trade_table.add_data(*trade_row)

            # Calculate trade statistics
            if self.trade_history:
                # Split into wins and losses
                win_trades = [t for t in self.trade_history if t.get('realized_pnl', 0) > 0]
                loss_trades = [t for t in self.trade_history if t.get('realized_pnl', 0) <= 0]

                # Calculate metrics
                win_rate = len(win_trades) / len(self.trade_history) if self.trade_history else 0
                avg_win = sum(t.get('realized_pnl', 0) for t in win_trades) / len(win_trades) if win_trades else 0
                avg_loss = sum(t.get('realized_pnl', 0) for t in loss_trades) / len(loss_trades) if loss_trades else 0

                # Calculate profit factor (avoid division by zero)
                win_sum = sum(t.get('realized_pnl', 0) for t in win_trades)
                loss_sum = abs(sum(t.get('realized_pnl', 0) for t in loss_trades))
                profit_factor = win_sum / loss_sum if loss_sum > 0 else float('inf')

                # Add to metrics
                metrics.update({
                    "trades/count": len(self.trade_history),
                    "trades/win_rate": win_rate,
                    "trades/avg_win": avg_win,
                    "trades/avg_loss": avg_loss,
                    "trades/profit_factor": profit_factor,
                    "trades/total_pnl": sum(t.get('realized_pnl', 0) for t in self.trade_history)
                })

                # Log trade table periodically
                if len(self.trade_history) % 10 == 0:
                    self._safe_wandb_log({"trades_table": self.trade_table})

        # Log metrics
        self._safe_wandb_log(metrics)

        # Create visualizations periodically
        if trainer.total_episodes % 5 == 0:
            self._create_episode_visualizations()
    def on_update_start(self, trainer):
        """Called before policy update."""
        # We don't need to do anything here
        pass
    def on_update_end(self, trainer, metrics):
        """Called after policy update."""
        # Increment global step
        self.global_step += 1

        # Log update metrics
        update_metrics = {f"update/{k}": v for k, v in metrics.items()}
        update_metrics["update/count"] = trainer.updates

        # Log with our global step
        self._safe_wandb_log(update_metrics)

        # Log parameter histograms periodically
        if trainer.updates % 10 == 0:
            self._log_model_gradients(trainer)
    def _log_model_gradients(self, trainer):
        """
        Log model parameter gradients to WandB for visualization.

        Args:
            trainer: The PPO trainer instance containing the model
        """
        if not hasattr(trainer, 'model'):
            return

        try:
            gradient_dict = {}

            # Log parameter histograms
            for name, param in trainer.model.named_parameters():
                if param.requires_grad:
                    # Log the parameter values
                    gradient_dict[f"params/{name}"] = wandb.Histogram(param.detach().cpu().numpy())

                    # Log the gradients if they exist
                    if param.grad is not None:
                        gradient_dict[f"grads/{name}"] = wandb.Histogram(param.grad.detach().cpu().numpy())

            # Log everything at once
            self._safe_wandb_log(gradient_dict)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error logging gradients: {e}", exc_info=True)
            else:
                print(f"Error logging gradients: {e}")
    def on_update_iteration_end(self, trainer, update_iter, update_metrics, rollout_stats):
        """Called at the end of each update iteration (rollout + update)."""
        # Increment global step
        self.global_step += 1

        # Combine metrics
        combined_metrics = {
            "iteration/number": update_iter,
            "iteration/mean_reward": rollout_stats.get("mean_reward", 0),
            "iteration/episodes": rollout_stats.get("episodes", 0),
            "iteration/steps": rollout_stats.get("total_steps", 0),
        }

        # Add update metrics
        for k, v in update_metrics.items():
            combined_metrics[f"iteration/update_{k}"] = v

        # Log the metrics
        self._safe_wandb_log(combined_metrics)

        # Check if this is the best model so far
        mean_reward = rollout_stats.get("mean_reward", -float('inf'))
        if mean_reward > self.best_reward:
            self.best_reward = mean_reward

            # If we have a new best model, mark it
            if hasattr(trainer, 'model_dir'):
                self.best_model_path = os.path.join(
                    trainer.model_dir,
                    f"best_model_reward{mean_reward:.2f}_iter{update_iter}.pt"
                )

                # Log a message
                self._safe_wandb_log({
                    "best_model/reward": mean_reward,
                    "best_model/iteration": update_iter
                })
    def _safe_wandb_log(self, metrics_dict, step=None):
        """
        Safely log to WandB with guaranteed monotonically increasing steps.
        Fixed to prevent step resets and warnings.
        """
        # Make sure we're initialized
        if not wandb.run:
            return

        # Use provided step or global_step
        current_step = step if step is not None else self.global_step

        # Use a class variable to track the last logged step
        if not hasattr(self, '_last_logged_step'):
            self._last_logged_step = 0

        # Ensure step is monotonically increasing
        if current_step is None or current_step <= self._last_logged_step:
            # Use last step + 1 to maintain monotonicity
            current_step = self._last_logged_step + 1

            # Only log at debug level to reduce noise
            if self.logger:
                self.logger.debug(f"Adjusted step from {step} to {current_step} to maintain monotonicity")

        try:
            # Log the metrics
            wandb.log(metrics_dict, step=current_step)
            self._last_logged_step = current_step

            # Update global step to at least match the logged step
            self.global_step = max(self.global_step, current_step + 1)
        except Exception as e:
            # Log error but don't crash
            if self.logger:
                self.logger.error(f"WandB logging failed: {e}")
            else:
                print(f"WandB logging failed: {e}")
    def _create_episode_visualizations(self):
        """Create visualizations after episodes."""
        try:
            # Skip if we don't have enough data
            if len(self.price_history) < 10:
                return

            # 1. Create price & position chart
            if len(self.price_history) > 0 and len(self.position_history) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))

                # Plot price
                ax.plot(range(len(self.price_history)), self.price_history,
                        color='blue', label='Price', linewidth=1.5)
                ax.set_ylabel('Price', color='blue')
                ax.set_title('Price and Position Over Time')

                # Create twin axis for position
                ax2 = ax.twinx()
                ax2.plot(range(len(self.position_history)), self.position_history,
                         color='red', label='Position', alpha=0.7)
                ax2.set_ylabel('Position', color='red')

                # Create legend
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

                # Save as image and log to WandB
                buf = io.BytesIO()
                plt.tight_layout()
                plt.savefig(buf, format='png', dpi=100)
                buf.seek(0)

                # Convert BytesIO to PIL Image before passing to wandb.Image
                img = Image.open(buf)

                # Log to WandB
                self._safe_wandb_log({"chart/price_position": wandb.Image(img)})

                # Close both the figure and the buffer
                plt.close(fig)
                buf.close()

            # 2. Create reward chart
            if len(self.reward_history) > 0:
                fig, ax = plt.subplots(figsize=(10, 4))

                # Plot rewards
                ax.plot(range(len(self.reward_history)), self.reward_history, color='green')
                ax.set_title('Reward History')
                ax.set_ylabel('Reward')
                ax.set_xlabel('Step')

                # Add moving average
                window = min(100, len(self.reward_history))
                if window > 10:
                    rewards_ma = pd.Series(self.reward_history).rolling(window).mean().iloc[window - 1:].values
                    ax.plot(range(window - 1, len(self.reward_history)),
                            rewards_ma, color='red', linewidth=2, label=f'MA-{window}')
                    ax.legend()

                # Save and log
                buf = io.BytesIO()
                plt.tight_layout()
                plt.savefig(buf, format='png', dpi=100)
                buf.seek(0)

                # Convert BytesIO to PIL Image
                img = Image.open(buf)

                # Log to WandB
                self._safe_wandb_log({"chart/reward": wandb.Image(img)})

                # Close figure and buffer
                plt.close(fig)
                buf.close()

        except Exception as e:
            print(f"Error creating episode visualizations: {e}")
    def _create_final_visualizations(self):
        """Create final visualizations at the end of training."""
        try:
            # Only create if we have trade data
            if not self.trade_history:
                return

            # Create trade analysis visualization
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))

            # Convert to DataFrame for easier analysis
            trades_df = pd.DataFrame(self.trade_history)

            # 1. Win/Loss pie chart
            if 'realized_pnl' in trades_df.columns:
                win_count = (trades_df['realized_pnl'] > 0).sum()
                loss_count = len(trades_df) - win_count

                axs[0, 0].pie([win_count, loss_count],
                              labels=[f'Wins ({win_count})', f'Losses ({loss_count})'],
                              autopct='%1.1f%%',
                              colors=['green', 'red'],
                              startangle=90)

                axs[0, 0].set_title(f'Win/Loss Ratio: {win_count / max(1, len(trades_df)):.1%}')

            # 2. PnL Distribution
            if 'realized_pnl' in trades_df.columns:
                axs[0, 1].hist(trades_df['realized_pnl'], bins=20, color='skyblue')
                axs[0, 1].axvline(x=0, color='black', linestyle='--')
                axs[0, 1].set_title('P&L Distribution')
                axs[0, 1].set_xlabel('P&L')

                # Add mean line
                mean_pnl = trades_df['realized_pnl'].mean()
                axs[0, 1].axvline(x=mean_pnl, color='red', linestyle='-',
                                  label=f'Mean: ${mean_pnl:.2f}')
                axs[0, 1].legend()

            # 3. Trade Duration
            if 'duration_seconds' in trades_df.columns:
                axs[1, 0].hist(trades_df['duration_seconds'], bins=20, color='skyblue')
                axs[1, 0].set_title('Trade Duration')
                axs[1, 0].set_xlabel('Seconds')

                # Add mean line
                mean_duration = trades_df['duration_seconds'].mean()
                axs[1, 0].axvline(x=mean_duration, color='red', linestyle='-',
                                  label=f'Mean: {mean_duration:.1f}s')
                axs[1, 0].legend()

            # 4. Cumulative P&L
            if 'realized_pnl' in trades_df.columns:
                cumulative_pnl = trades_df['realized_pnl'].cumsum()
                axs[1, 1].plot(range(len(cumulative_pnl)), cumulative_pnl, color='blue')
                axs[1, 1].set_title('Cumulative P&L')
                axs[1, 1].set_xlabel('Trade Number')
                axs[1, 1].set_ylabel('Cumulative P&L ($)')

            # Save and log
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=120)
            buf.seek(0)

            # Convert BytesIO to PIL Image
            img = Image.open(buf)

            # Log to WandB
            self._safe_wandb_log({"final/trade_analysis": wandb.Image(img)})

            # Close both figure and buffer
            plt.close(fig)
            buf.close()

        except Exception as e:
            print(f"Error creating final visualizations: {e}")