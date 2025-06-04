import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import io
from PIL import Image
import wandb
import logging

logger = logging.getLogger(__name__)


class EpisodeVisualizer:
    """Creates trading episode visualizations for W&B tracking."""

    def __init__(self, figsize: Tuple[int, int] = (16, 12)):
        self.figsize = figsize
        self.episode_data = []
        self.trades = []
        self.current_episode_info = {}

    def start_episode(self, episode_info: Dict[str, Any]):
        """Initialize data collection for a new episode."""
        self.episode_data = []
        self.trades = []
        self.current_episode_info = episode_info

    def add_step(self, step_data: Dict[str, Any]):
        """Add a single step's data to the episode."""
        self.episode_data.append(step_data)

    def add_trade(self, trade_info: Dict[str, Any]):
        """Record a trade (buy or sell)."""
        self.trades.append(trade_info)

    def create_episode_chart(self) -> Optional[wandb.Image]:
        """Create a comprehensive trading chart for the episode."""
        if not self.episode_data:
            return None

        try:
            # Create figure with subplots
            fig = plt.figure(figsize=self.figsize)
            gs = GridSpec(5, 2, height_ratios=[3, 1, 1, 1, 1], width_ratios=[3, 1])

            # Main price chart
            ax_price = fig.add_subplot(gs[0, 0])
            ax_stats = fig.add_subplot(gs[0, 1])
            ax_volume = fig.add_subplot(gs[1, 0], sharex=ax_price)
            ax_position = fig.add_subplot(gs[2, 0], sharex=ax_price)
            ax_reward = fig.add_subplot(gs[3, 0], sharex=ax_price)
            ax_indicators = fig.add_subplot(gs[4, 0], sharex=ax_price)

            # Extract data
            df = pd.DataFrame(self.episode_data)

            # Plot price with candles
            self._plot_price_chart(ax_price, df)

            # Plot volume
            self._plot_volume(ax_volume, df)

            # Plot position
            self._plot_position(ax_position, df)

            # Plot rewards
            self._plot_rewards(ax_reward, df)

            # Plot key indicators
            self._plot_indicators(ax_indicators, df)

            # Add statistics panel
            self._add_statistics(ax_stats, df)

            # Style adjustments
            plt.tight_layout()

            # Convert to image for W&B
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            img = Image.open(buf)
            plt.close(fig)

            return wandb.Image(
                img,
                caption=f"Episode {self.current_episode_info.get('episode_num', 'N/A')}",
            )

        except Exception as e:
            logger.error(f"Error creating episode chart: {e}")
            return None

    def _plot_price_chart(self, ax, df):
        """Plot price chart with buy/sell markers and trade outcomes."""
        # Plot price line
        ax.plot(df.index, df["price"], "k-", linewidth=1, alpha=0.7)

        # Add moving averages if available
        if "sma_fast" in df.columns:
            ax.plot(
                df.index, df["sma_fast"], "b-", linewidth=1, alpha=0.5, label="Fast MA"
            )
        if "sma_slow" in df.columns:
            ax.plot(
                df.index, df["sma_slow"], "r-", linewidth=1, alpha=0.5, label="Slow MA"
            )

        # Plot trades
        for trade in self.trades:
            idx = trade["step"]
            price = trade["price"]

            if trade["action"] == "buy":
                ax.scatter(
                    idx,
                    price,
                    color="green",
                    s=100,
                    marker="^",
                    edgecolor="darkgreen",
                    linewidth=2,
                    zorder=5,
                )
                ax.annotate(
                    f"BUY\n${price:.4f}",
                    (idx, price),
                    xytext=(5, 10),
                    textcoords="offset points",
                    fontsize=8,
                    color="green",
                    weight="bold",
                )
            elif trade["action"] == "sell":
                color = "red" if trade.get("pnl", 0) < 0 else "lime"
                ax.scatter(
                    idx,
                    price,
                    color=color,
                    s=100,
                    marker="v",
                    edgecolor="darkred" if color == "red" else "darkgreen",
                    linewidth=2,
                    zorder=5,
                )
                pnl_text = f"${trade.get('pnl', 0):.2f}" if "pnl" in trade else ""
                ax.annotate(
                    f"SELL\n${price:.4f}\n{pnl_text}",
                    (idx, price),
                    xytext=(5, -25),
                    textcoords="offset points",
                    fontsize=8,
                    color=color,
                    weight="bold",
                )

        # Add VWAP if available
        if "vwap" in df.columns:
            ax.plot(df.index, df["vwap"], "m--", linewidth=1, alpha=0.5, label="VWAP")

        ax.set_ylabel("Price ($)", fontsize=10)
        ax.set_title(
            f"Trading Episode - {self.current_episode_info.get('symbol', 'N/A')} "
            f"({self.current_episode_info.get('date', 'N/A')})",
            fontsize=12,
            weight="bold",
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)

    def _plot_volume(self, ax, df):
        """Plot volume bars with color coding."""
        if "volume" not in df.columns:
            return

        # Color based on price movement
        colors = [
            "g" if i == 0 or df["price"].iloc[i] >= df["price"].iloc[i - 1] else "r"
            for i in range(len(df))
        ]

        ax.bar(df.index, df["volume"], color=colors, alpha=0.5)
        ax.set_ylabel("Volume", fontsize=10)
        ax.grid(True, alpha=0.3)

    def _plot_position(self, ax, df):
        """Plot position over time."""
        if "position" not in df.columns:
            return

        ax.fill_between(
            df.index,
            0,
            df["position"],
            where=(df["position"] > 0),
            color="green",
            alpha=0.3,
            label="Long",
        )
        ax.fill_between(
            df.index,
            0,
            df["position"],
            where=(df["position"] < 0),
            color="red",
            alpha=0.3,
            label="Short",
        )
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.set_ylabel("Position", fontsize=10)
        ax.set_ylim(-1.5, 1.5)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)

    def _plot_rewards(self, ax, df):
        """Plot cumulative rewards."""
        if "reward" not in df.columns:
            return

        cumulative_reward = df["reward"].cumsum()
        ax.plot(df.index, cumulative_reward, "b-", linewidth=2)
        ax.fill_between(
            df.index,
            0,
            cumulative_reward,
            where=(cumulative_reward >= 0),
            color="green",
            alpha=0.3,
        )
        ax.fill_between(
            df.index,
            0,
            cumulative_reward,
            where=(cumulative_reward < 0),
            color="red",
            alpha=0.3,
        )
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.set_ylabel("Cumulative Reward", fontsize=10)
        ax.grid(True, alpha=0.3)

    def _plot_indicators(self, ax, df):
        """Plot key technical indicators."""
        # RSI
        if "rsi" in df.columns:
            ax2 = ax.twinx()
            ax2.plot(df.index, df["rsi"], "purple", linewidth=1, alpha=0.7, label="RSI")
            ax2.axhline(y=70, color="purple", linestyle="--", alpha=0.5)
            ax2.axhline(y=30, color="purple", linestyle="--", alpha=0.5)
            ax2.set_ylabel("RSI", fontsize=10, color="purple")
            ax2.set_ylim(0, 100)
            ax2.tick_params(axis="y", labelcolor="purple")

        # Volatility or other indicators
        if "volatility" in df.columns:
            ax.plot(
                df.index,
                df["volatility"],
                "orange",
                linewidth=1,
                alpha=0.7,
                label="Volatility",
            )
            ax.set_ylabel("Volatility", fontsize=10)

        ax.set_xlabel("Step", fontsize=10)
        ax.grid(True, alpha=0.3)
        if ax.get_lines():
            ax.legend(loc="upper left", fontsize=8)

    def _add_statistics(self, ax, df):
        """Add episode statistics panel."""
        ax.axis("off")

        # Calculate statistics
        total_trades = len([t for t in self.trades if t["action"] in ["buy", "sell"]])
        winning_trades = len([t for t in self.trades if t.get("pnl", 0) > 0])
        losing_trades = len([t for t in self.trades if t.get("pnl", 0) < 0])

        total_pnl = sum(t.get("pnl", 0) for t in self.trades)
        max_profit = max([t.get("pnl", 0) for t in self.trades], default=0)
        max_loss = min([t.get("pnl", 0) for t in self.trades], default=0)

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # Episode info
        stats_text = "Episode Statistics\n"
        stats_text += "=" * 25 + "\n\n"
        stats_text += f"Symbol: {self.current_episode_info.get('symbol', 'N/A')}\n"
        stats_text += f"Date: {self.current_episode_info.get('date', 'N/A')}\n"
        stats_text += (
            f"Episode: {self.current_episode_info.get('episode_num', 'N/A')}\n\n"
        )

        # Trading stats
        stats_text += f"Total Trades: {total_trades}\n"
        stats_text += f"Winning: {winning_trades}\n"
        stats_text += f"Losing: {losing_trades}\n"
        stats_text += f"Win Rate: {win_rate:.1f}%\n\n"

        # P&L stats
        stats_text += f"Total P&L: ${total_pnl:.2f}\n"
        stats_text += f"Max Profit: ${max_profit:.2f}\n"
        stats_text += f"Max Loss: ${max_loss:.2f}\n\n"

        # Market stats
        if len(df) > 0:
            stats_text += (
                f"Price Range: ${df['price'].min():.4f} - ${df['price'].max():.4f}\n"
            )
            stats_text += f"Total Volume: {df['volume'].sum():,.0f}\n"

        # Reward stats
        if "reward" in df.columns:
            stats_text += f"\nTotal Reward: {df['reward'].sum():.4f}\n"
            stats_text += f"Avg Reward: {df['reward'].mean():.4f}\n"

        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
        )

    def create_debug_charts(self) -> List[wandb.Image]:
        """Create additional debug charts for detailed analysis."""
        charts = []

        if not self.episode_data:
            return charts

        df = pd.DataFrame(self.episode_data)

        # Feature importance chart
        feature_chart = self._create_feature_chart(df)
        if feature_chart:
            charts.append(feature_chart)

        # Action distribution chart
        action_chart = self._create_action_distribution_chart(df)
        if action_chart:
            charts.append(action_chart)

        # Reward components chart
        reward_chart = self._create_reward_components_chart(df)
        if reward_chart:
            charts.append(reward_chart)

        return charts

    def _create_feature_chart(self, df) -> Optional[wandb.Image]:
        """Create feature visualization chart."""
        try:
            # Select key features to visualize
            feature_cols = [
                col
                for col in df.columns
                if col.startswith(("momentum", "volume_", "rsi", "volatility"))
            ]

            if not feature_cols:
                return None

            fig, axes = plt.subplots(
                len(feature_cols), 1, figsize=(12, 2 * len(feature_cols)), sharex=True
            )
            if len(feature_cols) == 1:
                axes = [axes]

            for ax, col in zip(axes, feature_cols):
                ax.plot(df.index, df[col], linewidth=1)
                ax.set_ylabel(col, fontsize=8)
                ax.grid(True, alpha=0.3)

            axes[-1].set_xlabel("Step", fontsize=10)
            plt.suptitle("Feature Evolution", fontsize=12)
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            img = Image.open(buf)
            plt.close(fig)

            return wandb.Image(img, caption="Feature Evolution")

        except Exception as e:
            logger.error(f"Error creating feature chart: {e}")
            return None

    def _create_action_distribution_chart(self, df) -> Optional[wandb.Image]:
        """Create action distribution visualization."""
        try:
            if "action" not in df.columns:
                return None

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

            # Action counts
            action_counts = df["action"].value_counts()
            ax1.bar(action_counts.index, action_counts.values)
            ax1.set_xlabel("Action")
            ax1.set_ylabel("Count")
            ax1.set_title("Action Distribution")

            # Action over time
            action_numeric = (
                df["action"].map({"hold": 0, "buy": 1, "sell": -1}).fillna(0)
            )
            ax2.plot(df.index, action_numeric, linewidth=1)
            ax2.set_xlabel("Step")
            ax2.set_ylabel("Action")
            ax2.set_title("Actions Over Time")
            ax2.set_yticks([-1, 0, 1])
            ax2.set_yticklabels(["Sell", "Hold", "Buy"])
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            img = Image.open(buf)
            plt.close(fig)

            return wandb.Image(img, caption="Action Analysis")

        except Exception as e:
            logger.error(f"Error creating action chart: {e}")
            return None

    def _create_reward_components_chart(self, df) -> Optional[wandb.Image]:
        """Create reward components breakdown."""
        try:
            reward_cols = [
                col for col in df.columns if "reward" in col.lower() and col != "reward"
            ]

            if not reward_cols:
                return None

            fig, ax = plt.subplots(figsize=(10, 6))

            # Stack plot of reward components
            bottom = np.zeros(len(df))
            for col in reward_cols:
                ax.fill_between(
                    df.index, bottom, bottom + df[col], label=col, alpha=0.7
                )
                bottom += df[col]

            ax.plot(df.index, df["reward"], "k-", linewidth=2, label="Total Reward")
            ax.set_xlabel("Step")
            ax.set_ylabel("Reward")
            ax.set_title("Reward Components Breakdown")
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            img = Image.open(buf)
            plt.close(fig)

            return wandb.Image(img, caption="Reward Components")

        except Exception as e:
            logger.error(f"Error creating reward chart: {e}")
            return None
