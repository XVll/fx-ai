# agent/wandb_callback.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import wandb
from typing import Dict,  Any, Optional
import io
from PIL import Image
import time
from collections import defaultdict

from agent.callbacks import TrainingCallback
from simulators.portfolio_simulator import PositionSideEnum


class WandbCallback(TrainingCallback):
    """
    Enhanced W&B callback for tracking training metrics, visualizations,
    and model checkpoints with the updated portfolio and trading environment.

    Features:
    - Robust monotonic step tracking to prevent W&B warnings
    - Comprehensive trade tracking and visualization
    - Detailed portfolio metrics logging
    - Model gradient and parameter tracking
    - Automatic chart generation
    """

    def __init__(
            self,
            project_name: str = "ai-trading",
            entity: Optional[str] = None,
            log_freq: int = 10,
            config: Optional[Dict[str, Any]] = None,
            log_model: bool = True,
            log_code: bool = True,
            log_batch_metrics: bool = False,
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
            log_batch_metrics: Whether to log detailed batch metrics (can be noisy)
        """
        # Basic settings
        self.project_name = project_name
        self.entity = entity
        self.log_freq = log_freq
        self.config = config or {}
        self.log_model = log_model
        self.log_code = log_code
        self.log_batch_metrics = log_batch_metrics

        # Initialize WandB run (will be set in on_training_start)
        self.run = None

        # Step counting for monotonic logging
        self._monotonic_step = 0
        self._last_logged_step = 0
        self.global_step = 0
        self.episode_count = 0
        self.update_count = 0

        # Performance tracking
        self.training_start_time = None
        self.best_reward = -float('inf')
        self.best_model_path = None

        # Data tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.update_metrics_history = []

        # Price and portfolio tracking
        self.price_history = []
        self.equity_history = []
        self.cash_history = []
        self.position_history = []
        self.realized_pnl_history = []
        self.unrealized_pnl_history = []

        # Action tracking
        self.action_history = []
        self.action_type_counts = defaultdict(int)
        self.action_size_counts = defaultdict(int)

        # Trade tracking
        self.trades_table = None
        self.trade_stats = defaultdict(list)
        self.completed_trades = []

        # Visualization settings
        self.chart_update_freq = 5  # Every N episodes

        # Logger
        self.logger = None

    def on_training_start(self, trainer):
        """Initialize WandB run and prepare for tracking when training starts."""
        # Set logger
        self.logger = trainer.logger if hasattr(trainer, 'logger') else None

        # Store start time
        self.training_start_time = time.time()

        # Initialize WandB if not already done
        if self.run is None:
            self.run = wandb.init(
                project=self.project_name,
                entity=self.entity,
                config=self.config,
                save_code=self.log_code,
                job_type="training",
                reinit=True
            )

        # Initialize trades table
        self.trades_table = wandb.Table(columns=[
            "trade_id", "symbol", "side", "entry_time", "exit_time",
            "entry_price", "exit_price", "quantity", "realized_pnl",
            "commission", "fees", "slippage", "duration_seconds",
            "max_favorable_excursion", "max_adverse_excursion", "exit_reason"
        ])

        # Log model architecture to W&B
        if hasattr(trainer, 'model'):
            wandb.run.summary["model_architecture"] = str(trainer.model)

            # Set up model watching if enabled
            if self.config.get('watch_model', True):
                wandb.watch(
                    trainer.model,
                    log="all",
                    log_freq=max(100, self.log_freq * 10)
                )

        # Initialize step counters
        # If this is a resumed run, we need to ensure our step counters
        # are at least as large as the wandb run's current step
        if wandb.run and hasattr(wandb.run, 'step'):
            current_wandb_step = wandb.run.step
            self._monotonic_step = max(current_wandb_step + 1, self._monotonic_step)
            self._last_logged_step = max(current_wandb_step, self._last_logged_step)
            self.global_step = max(current_wandb_step, self.global_step)
        else:
            # For new runs, we can start from 0
            self._monotonic_step = 0
            self._last_logged_step = 0
            self.global_step = 0

        # These counters can always start from 0 as they're just for tracking
        self.episode_count = 0
        self.update_count = 0

        if self.logger:
            self.logger.info(f"WandB initialized: {wandb.run.name} ({wandb.run.id})")
        else:
            print(f"WandB initialized: {wandb.run.name} ({wandb.run.id})")

    def on_training_end(self, trainer, stats):
        """Log final statistics and visualizations when training ends."""
        # Calculate training time
        training_duration = time.time() - self.training_start_time

        # Log final summary statistics
        final_metrics = {
            "final/total_episodes": stats.get("total_episodes", self.episode_count),
            "final/total_steps": stats.get("total_steps", self.global_step),
            "final/total_updates": stats.get("total_updates", self.update_count),
            "final/best_reward": stats.get("best_mean_reward", self.best_reward),
            "final/training_time_seconds": training_duration,
            "final/mean_episode_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0,
            "final/mean_episode_length": np.mean(self.episode_lengths) if self.episode_lengths else 0,
        }

        # Add portfolio metrics if available
        if hasattr(trainer.env, 'portfolio_manager'):
            portfolio_metrics = trainer.env.portfolio_manager.get_trader_vue_metrics()
            for k, v in portfolio_metrics.items():
                final_metrics[f"final/portfolio_{k}"] = v

        # Log all the final metrics
        self._safe_wandb_log(final_metrics)

        # Log trades table
        if self.trades_table and len(self.trades_table.data) > 0:
            self._safe_wandb_log({"final_trades": self.trades_table})

        # Create and log final visualizations
        self._create_final_visualizations()

        # Upload best model if we have one and log_model is enabled
        if self.log_model and self.best_model_path and os.path.exists(self.best_model_path):
            base_path = trainer.output_dir if hasattr(trainer, 'output_dir') else "."
            wandb.save(self.best_model_path, base_path=base_path)
            wandb.run.summary["best_model_path"] = self.best_model_path

        # Finish W&B run
        wandb.finish()

        if self.logger:
            self.logger.info("W&B logging finished")

    def on_rollout_start(self, trainer):
        """Called before collecting rollouts."""
        # Not much to do here, but increment the step for strict monotonicity
        self._monotonic_step += 1

    def on_rollout_end(self, trainer):
        """Called after collecting rollouts."""
        # Not much to do here, but increment the step for strict monotonicity
        self._monotonic_step += 1

    def on_step(self, trainer, state, action, reward, next_state, info):
        """
        Record metrics after each environment step.

        This is called frequently, so we only log to W&B periodically.
        """
        # Update global step counter
        self.global_step += 1

        # Track basic metrics
        self.episode_rewards.append(reward)

        # Track action
        if isinstance(action, torch.Tensor):
            action_value = action.cpu().numpy().tolist() if action.dim() > 0 else action.item()
        else:
            action_value = action
        self.action_history.append(action_value)

        # Track prices and portfolio state if available in info
        if isinstance(info, dict):
            if 'portfolio_equity' in info:
                self.equity_history.append(info['portfolio_equity'])
            if 'portfolio_cash' in info:
                self.cash_history.append(info['portfolio_cash'])
            if 'portfolio_unrealized_pnl' in info:
                self.unrealized_pnl_history.append(info['portfolio_unrealized_pnl'])
            if 'portfolio_realized_pnl_session_net' in info:
                self.realized_pnl_history.append(info['portfolio_realized_pnl_session_net'])

            # Track position for primary asset
            for key in info:
                if key.startswith('position_') and key.endswith('_qty'):
                    self.position_history.append(info[key])
                    break

            # Check for market price
            if hasattr(trainer.env, 'market_simulator'):
                market_state = trainer.env.market_simulator.get_current_market_state()
                if market_state:
                    price = market_state.get('current_price')
                    if price is not None:
                        self.price_history.append(price)

            # Track action metrics
            if 'action_decoded' in info:
                action_info = info['action_decoded']
                if action_info:
                    action_type = action_info.get('type')
                    action_size = action_info.get('size_enum')
                    if action_type:
                        self.action_type_counts[action_type.name] += 1
                    if action_size:
                        self.action_size_counts[action_size.name] += 1

            # Check for trade fills
            if 'fills_step' in info and info['fills_step']:
                for fill in info['fills_step']:
                    if self.logger:
                        self.logger.debug(f"Trade fill logged: {fill}")

        # Only log to W&B at specified intervals to reduce overhead
        if self.global_step % self.log_freq == 0:
            metrics = {
                "step/reward": reward,
                "step/global_step": self.global_step,
                "step/action": action_value,
            }

            # Add info metrics
            if isinstance(info, dict):
                for k, v in info.items():
                    if isinstance(v, (int, float)) and not k.startswith('position_') and k not in ['timestamp_iso', 'step']:
                        metrics[f"step/{k}"] = v

                # Add position metrics
                for k, v in info.items():
                    if k.startswith('position_') and isinstance(v, (int, float, str)):
                        metrics[f"position/{k.replace('position_', '')}"] = v

            # Log metrics to W&B
            self._safe_wandb_log(metrics)

    def on_episode_end(self, trainer, episode_reward, episode_length, info):
        """
        Process metrics and create visualizations at the end of each episode.
        """
        # Increment episode counter
        self.episode_count += 1
        self._monotonic_step += 1

        # Store episode data
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)

        # Prepare metrics dictionary
        metrics = {
            "episode/reward": episode_reward,
            "episode/length": episode_length,
            "episode/number": self.episode_count,
        }

        # Add episode info if available
        if isinstance(info, dict):
            # Add episode summary metrics
            if 'episode_summary' in info:
                summary = info['episode_summary']
                for k, v in summary.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        metrics[f"episode/{k}"] = v

            # Process trades if available
            if hasattr(trainer.env, 'portfolio_manager'):
                # Get completed trades from portfolio manager
                portfolio_manager = trainer.env.portfolio_manager
                trade_log = getattr(portfolio_manager, 'trade_log', [])

                # Process new trades
                for trade in trade_log:
                    # Check if we've already recorded this trade
                    trade_id = trade.get('trade_id')
                    if trade_id and trade_id not in [t.get('trade_id') for t in self.completed_trades]:
                        self.completed_trades.append(trade)

                        # Add to W&B table
                        trade_row = [
                            trade.get('trade_id', ''),
                            trade.get('asset_id', ''),
                            trade.get('side', PositionSideEnum.FLAT).value,
                            trade.get('entry_timestamp', ''),
                            trade.get('exit_timestamp', ''),
                            trade.get('avg_entry_price', 0.0),
                            trade.get('avg_exit_price', 0.0),
                            trade.get('entry_quantity_total', 0.0),
                            trade.get('realized_pnl', 0.0),
                            trade.get('commission_total', 0.0),
                            trade.get('fees_total', 0.0),
                            trade.get('slippage_total_trade_usd', 0.0),
                            trade.get('holding_period_seconds', 0.0),
                            trade.get('max_favorable_excursion_usd', 0.0),
                            trade.get('max_adverse_excursion_usd', 0.0),
                            trade.get('reason_for_exit', '')
                        ]
                        self.trades_table.add_data(*trade_row)

                        # Track in trade stats
                        self.trade_stats['pnl'].append(trade.get('realized_pnl', 0.0))
                        self.trade_stats['duration'].append(trade.get('holding_period_seconds', 0.0))
                        self.trade_stats['entry_price'].append(trade.get('avg_entry_price', 0.0))
                        self.trade_stats['exit_price'].append(trade.get('avg_exit_price', 0.0))
                        self.trade_stats['quantity'].append(trade.get('entry_quantity_total', 0.0))
                        self.trade_stats['mfe'].append(trade.get('max_favorable_excursion_usd', 0.0))
                        self.trade_stats['mae'].append(trade.get('max_adverse_excursion_usd', 0.0))

                # Calculate trade statistics
                if self.completed_trades:
                    num_trades = len(self.completed_trades)
                    win_trades = [t for t in self.completed_trades if t.get('realized_pnl', 0) > 0]
                    loss_trades = [t for t in self.completed_trades if t.get('realized_pnl', 0) <= 0]

                    win_rate = len(win_trades) / num_trades if num_trades > 0 else 0

                    metrics.update({
                        "trades/count": num_trades,
                        "trades/win_rate": win_rate * 100,  # as percentage
                        "trades/avg_pnl": np.mean(self.trade_stats['pnl']) if self.trade_stats['pnl'] else 0,
                        "trades/avg_duration": np.mean(self.trade_stats['duration']) if self.trade_stats['duration'] else 0,
                    })

                    # Log trade table occasionally
                    if num_trades % 10 == 0 or num_trades <= 10:
                        self._safe_wandb_log({"trades_table": self.trades_table})

        # Log the metrics
        self._safe_wandb_log(metrics)

        # Create visualizations periodically
        if self.episode_count % self.chart_update_freq == 0:
            self._create_episode_visualizations()

    def on_update_start(self, trainer):
        """Called before policy update."""
        # Not much to do here, but increment the step for strict monotonicity
        self._monotonic_step += 1

    def on_update_end(self, trainer, metrics):
        """Log metrics after policy update."""
        # Increment update counter
        self.update_count += 1
        self._monotonic_step += 1

        # Prepare metrics dictionary
        update_metrics = {f"update/{k}": v for k, v in metrics.items()}
        update_metrics["update/count"] = self.update_count

        # Store for history
        self.update_metrics_history.append(metrics)

        # Log the metrics
        self._safe_wandb_log(update_metrics)

        # Log parameter histograms periodically
        if self.update_count % 10 == 0:
            self._log_model_parameters(trainer)

    def _log_model_parameters(self, trainer):
        """Log model parameter histograms and gradients."""
        if not hasattr(trainer, 'model'):
            return

        try:
            histogram_dict = {}

            # Log parameter histograms
            for name, param in trainer.model.named_parameters():
                if param.requires_grad:
                    # Log the parameter values
                    histogram_dict[f"parameters/{name}"] = wandb.Histogram(param.detach().cpu().numpy())

                    # Log the gradients if they exist
                    if param.grad is not None:
                        histogram_dict[f"gradients/{name}"] = wandb.Histogram(param.grad.detach().cpu().numpy())
                        # Also log gradient norms
                        histogram_dict[f"gradient_norms/{name}"] = torch.norm(param.grad).item()

            # Log everything at once
            self._safe_wandb_log(histogram_dict)

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error logging model parameters: {e}")
            else:
                print(f"Error logging model parameters: {e}")

    def on_update_iteration_end(self, trainer, update_iter, update_metrics, rollout_stats):
        """Log metrics at the end of each update iteration (rollout + update)."""
        # Increment step counter
        self._monotonic_step += 1

        # Combine metrics
        combined_metrics = {
            "iteration/number": update_iter,
            "iteration/mean_reward": rollout_stats.get("mean_reward", 0),
            "iteration/mean_episode_length": rollout_stats.get("mean_episode_length", 0),
            "iteration/episodes_completed": rollout_stats.get("num_episodes", 0),
            "iteration/steps_collected": rollout_stats.get("collected_steps", 0),
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

            # If we have a path to a new best model, mark it
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

    def _safe_wandb_log(self, metrics_dict: Dict[str, Any], step: Optional[int] = None):
        """
        Safely log to W&B with guaranteed monotonically increasing steps.
        Fixes issues with step resets and warnings.
        """
        # Make sure W&B is initialized
        if not wandb.run:
            if self.logger:
                self.logger.warning("W&B not initialized. Skipping logging.")
            return

        # Always use the monotonic step counter to ensure unique, increasing steps
        # Ignore any provided step parameter to avoid conflicts
        step = self._monotonic_step
        self._monotonic_step += 1

        # Extra safety check to ensure we're always ahead of the last logged step
        if step <= self._last_logged_step:
            step = self._last_logged_step + 1
            self._monotonic_step = step + 1  # Update monotonic step to stay ahead

        try:
            # Clean metrics dict to ensure all values are loggable
            clean_metrics = {}
            for k, v in metrics_dict.items():
                if isinstance(v, (int, float, str, bool, np.ndarray, wandb.Histogram, wandb.Image, wandb.Table)):
                    clean_metrics[k] = v
                elif isinstance(v, torch.Tensor):
                    try:
                        if v.numel() == 1:
                            clean_metrics[k] = v.item()
                        else:
                            clean_metrics[k] = wandb.Histogram(v.detach().cpu().numpy())
                    except Exception:
                        # Skip tensors that can't be converted
                        pass
                elif v is None:
                    # Skip None values
                    pass
                else:
                    # Try to convert to string for other types
                    try:
                        clean_metrics[k] = str(v)
                    except Exception:
                        # Skip values that can't be converted
                        pass

            # Log the metrics
            wandb.log(clean_metrics, step=step)
            self._last_logged_step = step

        except Exception as e:
            # Log error but don't crash
            if self.logger:
                self.logger.error(f"W&B logging failed: {e}")
            else:
                print(f"W&B logging failed: {e}")

    def _create_episode_visualizations(self):
        """
        Create and log visualizations for current episode.
        """
        try:
            # Only create visualizations if we have enough data
            if len(self.price_history) < 10 or len(self.equity_history) < 10:
                return

            # 1. Portfolio Performance Chart
            fig, axes = plt.subplots(3, 1, figsize=(12, 14), gridspec_kw={'height_ratios': [2, 1, 1]})

            # Plot portfolio equity
            ax_equity = axes[0]
            x_range = list(range(len(self.equity_history)))
            ax_equity.plot(x_range, self.equity_history, color='blue', linewidth=2, label='Portfolio Value')
            ax_equity.set_ylabel('Portfolio Value ($)', color='blue')
            ax_equity.set_title('Portfolio Performance')
            ax_equity.grid(True, alpha=0.3)

            # Plot realized PnL on twin axis
            if self.realized_pnl_history:
                ax_pnl = ax_equity.twinx()
                ax_pnl.plot(x_range[-len(self.realized_pnl_history):],
                            self.realized_pnl_history,
                            color='green', linestyle='--', label='Realized PnL')
                ax_pnl.set_ylabel('Realized PnL ($)', color='green')

                # Get lines and labels for the legend
                lines_equity, labels_equity = ax_equity.get_legend_handles_labels()
                lines_pnl, labels_pnl = ax_pnl.get_legend_handles_labels()
                ax_equity.legend(lines_equity + lines_pnl, labels_equity + labels_pnl, loc='upper left')
            else:
                ax_equity.legend(loc='upper left')

            # Plot position size
            ax_pos = axes[1]
            if self.position_history:
                ax_pos.plot(x_range[-len(self.position_history):],
                            self.position_history,
                            color='purple', label='Position Size')
                ax_pos.set_ylabel('Position Size', color='purple')
                ax_pos.set_title('Position Size Over Time')
                ax_pos.grid(True, alpha=0.3)
                ax_pos.legend(loc='upper left')

            # Plot price
            ax_price = axes[2]
            if self.price_history:
                ax_price.plot(x_range[-len(self.price_history):],
                              self.price_history,
                              color='orange', label='Price')
                ax_price.set_ylabel('Price ($)', color='orange')
                ax_price.set_title('Asset Price')
                ax_price.grid(True, alpha=0.3)
                ax_price.set_xlabel('Step')
                ax_price.legend(loc='upper left')

            # Adjust layout and convert to image
            plt.tight_layout()
            portfolio_img = self._fig_to_image(fig)
            plt.close(fig)

            # 2. Action Distribution Chart
            if self.action_type_counts:
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))

                # Plot action type distribution
                ax_type = axes[0]
                action_types = list(self.action_type_counts.keys())
                action_type_values = list(self.action_type_counts.values())
                ax_type.bar(action_types, action_type_values, color='skyblue')
                ax_type.set_title('Action Type Distribution')
                ax_type.set_xlabel('Action Type')
                ax_type.set_ylabel('Count')

                # Plot action size distribution
                if self.action_size_counts:
                    ax_size = axes[1]
                    action_sizes = list(self.action_size_counts.keys())
                    action_size_values = list(self.action_size_counts.values())
                    ax_size.bar(action_sizes, action_size_values, color='lightgreen')
                    ax_size.set_title('Position Size Distribution')
                    ax_size.set_xlabel('Position Size')
                    ax_size.set_ylabel('Count')

                plt.tight_layout()
                action_img = self._fig_to_image(fig)
                plt.close(fig)

                # Log both images
                self._safe_wandb_log({
                    "chart/portfolio_performance": wandb.Image(portfolio_img),
                    "chart/action_distribution": wandb.Image(action_img)
                })
            else:
                # Log just the portfolio image
                self._safe_wandb_log({
                    "chart/portfolio_performance": wandb.Image(portfolio_img)
                })

            # 3. Trade Analysis chart if we have trades
            if len(self.completed_trades) > 5:
                self._create_trade_analysis_chart()

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating episode visualizations: {e}")
            else:
                print(f"Error creating episode visualizations: {e}")

    def _create_trade_analysis_chart(self):
        """Create and log trade analysis visualizations."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # 1. PnL Distribution
            ax_pnl = axes[0, 0]
            if self.trade_stats['pnl']:
                ax_pnl.hist(self.trade_stats['pnl'], bins=20, color='skyblue', alpha=0.7)
                ax_pnl.axvline(x=0, color='red', linestyle='--')
                ax_pnl.set_title('Trade PnL Distribution')
                ax_pnl.set_xlabel('PnL ($)')
                ax_pnl.set_ylabel('Frequency')

                # Add mean line
                mean_pnl = np.mean(self.trade_stats['pnl'])
                ax_pnl.axvline(x=mean_pnl, color='green', linestyle='-',
                               label=f'Mean: ${mean_pnl:.2f}')
                ax_pnl.legend()

            # 2. Trade Duration
            ax_dur = axes[0, 1]
            if self.trade_stats['duration']:
                ax_dur.hist(self.trade_stats['duration'], bins=20, color='lightgreen', alpha=0.7)
                ax_dur.set_title('Trade Duration Distribution')
                ax_dur.set_xlabel('Duration (seconds)')
                ax_dur.set_ylabel('Frequency')

                # Add mean line
                mean_dur = np.mean(self.trade_stats['duration'])
                ax_dur.axvline(x=mean_dur, color='blue', linestyle='-',
                               label=f'Mean: {mean_dur:.1f}s')
                ax_dur.legend()

            # 3. Win vs. Loss Pie Chart
            ax_win = axes[1, 0]
            if self.trade_stats['pnl']:
                wins = sum(1 for pnl in self.trade_stats['pnl'] if pnl > 0)
                losses = sum(1 for pnl in self.trade_stats['pnl'] if pnl <= 0)

                if wins + losses > 0:
                    ax_win.pie([wins, losses],
                               labels=[f'Wins ({wins})', f'Losses ({losses})'],
                               autopct='%1.1f%%',
                               colors=['green', 'red'],
                               startangle=90)
                    ax_win.set_title(f'Win/Loss Ratio: {wins / (wins + losses):.1%}')

            # 4. Cumulative PnL
            ax_cum = axes[1, 1]
            if self.trade_stats['pnl']:
                cumulative_pnl = np.cumsum(self.trade_stats['pnl'])
                ax_cum.plot(cumulative_pnl, color='blue')
                ax_cum.set_title('Cumulative PnL')
                ax_cum.set_xlabel('Trade Number')
                ax_cum.set_ylabel('Cumulative PnL ($)')
                ax_cum.grid(True, alpha=0.3)

            plt.tight_layout()
            trade_img = self._fig_to_image(fig)
            plt.close(fig)

            # Log the image
            self._safe_wandb_log({
                "chart/trade_analysis": wandb.Image(trade_img)
            })

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating trade analysis chart: {e}")
            else:
                print(f"Error creating trade analysis chart: {e}")

    def _create_final_visualizations(self):
        """Create comprehensive final visualizations at the end of training."""
        try:
            # Create training summary visualization
            if self.episode_rewards and self.episode_lengths:
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))

                # 1. Episode Rewards
                ax_rew = axes[0, 0]
                ep_range = list(range(1, len(self.episode_rewards) + 1))
                ax_rew.plot(ep_range, self.episode_rewards, color='blue')

                # Safe moving average - ensure x and y have same length
                if len(self.episode_rewards) > 2:
                    window_size = min(10, len(self.episode_rewards))
                    rewards_series = pd.Series(self.episode_rewards)
                    rewards_ma = rewards_series.rolling(window_size, min_periods=1).mean().values
                    # Ensure ep_range and rewards_ma have the same length
                    if len(ep_range) == len(rewards_ma):
                        ax_rew.plot(ep_range, rewards_ma, color='red', linewidth=2, label=f'MA-{window_size}')
                    else:
                        if self.logger:
                            self.logger.warning(f"Skipping MA plot: x and y have different lengths ({len(ep_range)} vs {len(rewards_ma)})")
                        else:
                            print(f"Skipping MA plot: x and y have different lengths ({len(ep_range)} vs {len(rewards_ma)})")

                # We don't need this second moving average calculation as it's redundant and could cause issues
                # The first calculation above with min_periods=1 already provides a moving average for all points

                ax_rew.set_title('Episode Rewards')
                ax_rew.set_xlabel('Episode')
                ax_rew.set_ylabel('Reward')
                ax_rew.grid(True, alpha=0.3)
                ax_rew.legend()

                # 2. Episode Lengths
                ax_len = axes[0, 1]
                # Always create a new ep_range that matches the length of episode_lengths
                # This ensures we always have matching lengths
                ep_range_lengths = list(range(1, len(self.episode_lengths) + 1))
                ax_len.plot(ep_range_lengths, self.episode_lengths, color='green')

                # Log the lengths for debugging
                if self.logger and len(ep_range) != len(self.episode_lengths):
                    self.logger.debug(f"Note: episode_rewards length ({len(ep_range)}) differs from episode_lengths ({len(self.episode_lengths)})")
                    self.logger.debug(f"Using separate range for episode lengths plot to ensure correct visualization")
                ax_len.set_title('Episode Lengths')
                ax_len.set_xlabel('Episode')
                ax_len.set_ylabel('Steps')
                ax_len.grid(True, alpha=0.3)

                # 3. Update Metrics
                ax_upd = axes[1, 0]
                if self.update_metrics_history:
                    update_range = list(range(1, len(self.update_metrics_history) + 1))

                    # Extract common metrics
                    if 'actor_loss' in self.update_metrics_history[0]:
                        actor_losses = [m.get('actor_loss', 0) for m in self.update_metrics_history]
                        ax_upd.plot(update_range, actor_losses, label='Actor Loss', color='blue')

                    if 'critic_loss' in self.update_metrics_history[0]:
                        critic_losses = [m.get('critic_loss', 0) for m in self.update_metrics_history]
                        ax_upd.plot(update_range, critic_losses, label='Critic Loss', color='red')

                    if 'entropy' in self.update_metrics_history[0]:
                        entropies = [m.get('entropy', 0) for m in self.update_metrics_history]
                        ax_upd.plot(update_range, entropies, label='Entropy', color='green')

                    ax_upd.set_title('Training Metrics')
                    ax_upd.set_xlabel('Update')
                    ax_upd.set_ylabel('Value')
                    ax_upd.grid(True, alpha=0.3)
                    ax_upd.legend()

                # 4. Action Distribution Pie Chart
                ax_act = axes[1, 1]
                if self.action_type_counts:
                    labels = list(self.action_type_counts.keys())
                    sizes = list(self.action_type_counts.values())

                    ax_act.pie(sizes, labels=labels, autopct='%1.1f%%',
                               startangle=90, colors=plt.cm.tab10.colors[:len(labels)])
                    ax_act.set_title('Action Type Distribution')

                plt.tight_layout()
                summary_img = self._fig_to_image(fig)
                plt.close(fig)

                # Log the image
                self._safe_wandb_log({
                    "chart/training_summary": wandb.Image(summary_img)
                })

            # Create detailed trade analysis if we have trades
            if len(self.completed_trades) > 0:
                # Already handled by _create_trade_analysis_chart
                self._create_trade_analysis_chart()

                # Add additional specific trade insights
                # TODO: Add specialized visualizations for your trading strategy
                # For example, visualize entries/exits relative to price patterns

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating final visualizations: {e}")
            else:
                print(f"Error creating final visualizations: {e}")

    def _fig_to_image(self, fig):
        """Convert matplotlib figure to PIL Image."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img = Image.open(buf)
        return img
