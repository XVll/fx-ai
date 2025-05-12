# visualization/trade_visualizer.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict, List, Union, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
import os
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

# Set Seaborn style for better visuals
sns.set_style("whitegrid")


class TradeVisualizer:
    """
    Visualization tool for trading data and performance metrics.

    Features:
    - Chart price data with trades overlaid
    - Portfolio equity curve
    - Performance metrics visualization
    - Trade analysis (win/loss, durations, etc.)
    - Multi-timeframe visualization
    """

    def __init__(self, save_path: str = "charts", figsize: Tuple[int, int] = (12, 8), logger: logging.Logger = None):
        """
        Initialize the visualizer.

        Args:
            save_path: Directory to save charts
            figsize: Default figure size
            logger: Optional logger
        """
        self.save_path = save_path
        self.figsize = figsize
        self.logger = logger or logging.getLogger(__name__)

        # Create save directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')

        # Color scheme
        self.colors = {
            'buy': '#22c55e',  # Green
            'sell': '#ef4444',  # Red
            'price': '#4b5563',  # Gray
            'vwap': '#8b5cf6',  # Purple
            'ema_9': '#3b82f6',  # Blue
            'ema_20': '#0ea5e9',  # Light blue
            'ema_50': '#f59e0b',  # Orange
            'ema_200': '#7c3aed',  # Violet
            'volume': '#cbd5e1',  # Light gray
            'equity': '#0ea5e9',  # Light blue
            'win': '#22c55e',  # Green
            'loss': '#ef4444',  # Red
            'background': '#f8fafc',  # Light background
        }

    def _log(self, message: str, level: int = logging.INFO):
        """Helper method for logging."""
        if self.logger:
            self.logger.log(level, message)

    def plot_price_chart_with_trades(self,
                                     price_data: pd.DataFrame,
                                     trades: List[Dict[str, Any]],
                                     title: str = "Price Chart with Trades",
                                     save_filename: Optional[str] = None,
                                     show_indicators: bool = True,
                                     show_volume: bool = True) -> plt.Figure:
        """
        Plot price chart with trades overlaid.

        Args:
            price_data: DataFrame with price data (must have timestamp index and OHLCV columns)
            trades: List of trade dictionaries
            title: Chart title
            save_filename: Filename to save chart (if None, not saved)
            show_indicators: Whether to show technical indicators
            show_volume: Whether to show volume

        Returns:
            Matplotlib figure
        """
        # Ensure DataFrame has required columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in price_data.columns for col in required_cols):
            self._log(f"Price data missing required columns. Required: {required_cols}", logging.ERROR)
            return None

        # Create subplots - price and optionally volume
        n_rows = 2 if show_volume else 1
        fig, axs = plt.subplots(n_rows, 1, figsize=self.figsize, sharex=True,
                                gridspec_kw={'height_ratios': [3, 1] if show_volume else [1]})

        # Get the price axis (first subplot or only subplot if no volume)
        ax_price = axs[0] if show_volume else axs

        # Plot price data (using fill_between for better clarity)
        ax_price.plot(price_data.index, price_data['close'], color=self.colors['price'],
                      label='Price', linewidth=1.5)

        # Plot technical indicators if requested
        if show_indicators:
            # Calculate and plot EMA lines if not already in the DataFrame
            if 'ema_9' not in price_data.columns:
                price_data['ema_9'] = price_data['close'].ewm(span=9, adjust=False).mean()
            if 'ema_20' not in price_data.columns:
                price_data['ema_20'] = price_data['close'].ewm(span=20, adjust=False).mean()
            if 'vwap' not in price_data.columns and 'volume' in price_data.columns:
                # Calculate VWAP if not available
                typical_price = (price_data['high'] + price_data['low'] + price_data['close']) / 3
                price_data['vwap'] = (typical_price * price_data['volume']).cumsum() / price_data['volume'].cumsum()

            # Plot EMAs
            ax_price.plot(price_data.index, price_data['ema_9'],
                          color=self.colors['ema_9'], label='EMA 9', linewidth=1, alpha=0.8)
            ax_price.plot(price_data.index, price_data['ema_20'],
                          color=self.colors['ema_20'], label='EMA 20', linewidth=1, alpha=0.8)

            # Plot VWAP if available
            if 'vwap' in price_data.columns:
                ax_price.plot(price_data.index, price_data['vwap'],
                              color=self.colors['vwap'], label='VWAP', linewidth=1, alpha=0.8)

        # Plot volume if requested
        if show_volume and 'volume' in price_data.columns:
            ax_vol = axs[1]
            ax_vol.bar(price_data.index, price_data['volume'], color=self.colors['volume'], alpha=0.7)
            ax_vol.set_ylabel('Volume')
            ax_vol.yaxis.set_major_formatter(
                FuncFormatter(lambda x, _: f'{x / 1000:.0f}K' if x < 1e6 else f'{x / 1e6:.1f}M'))
            ax_vol.tick_params(axis='y', labelsize=8)

        # Plot trades
        for trade in trades:
            # Extract trade info
            entry_time = trade.get('open_time')
            exit_time = trade.get('close_time')
            entry_price = trade.get('entry_price', 0.0)
            exit_price = trade.get('exit_price', 0.0)
            quantity = trade.get('quantity', 0.0)
            pnl = trade.get('realized_pnl', 0.0)

            if not all([entry_time, exit_time, entry_price, exit_price]):
                continue

            # Determine if winning or losing trade
            is_win = pnl > 0
            color = self.colors['win'] if is_win else self.colors['loss']

            # Plot entry and exit markers
            ax_price.scatter(entry_time, entry_price, color=color, marker='^', s=100, zorder=5)
            ax_price.scatter(exit_time, exit_price, color=color, marker='v', s=100, zorder=5)

            # Connect entry and exit with a line
            ax_price.plot([entry_time, exit_time], [entry_price, exit_price],
                          color=color, linestyle='--', linewidth=1.5, alpha=0.7)

            # Annotate trade with P&L
            mid_time = entry_time + (exit_time - entry_time) / 2
            mid_price = (entry_price + exit_price) / 2
            ax_price.annotate(f"${pnl:.2f}",
                              xy=(mid_time, mid_price),
                              xytext=(0, 10 if is_win else -20),
                              textcoords='offset points',
                              color=color,
                              fontweight='bold',
                              ha='center')

        # Set up axis labels and title
        ax_price.set_title(title, fontsize=16, pad=10)
        ax_price.set_ylabel('Price ($)', fontsize=12)

        # Format x-axis with date
        ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        fig.autofmt_xdate(rotation=45)

        # Add grid
        ax_price.grid(True, linestyle='--', alpha=0.7)

        # Add legend
        ax_price.legend(loc='best', frameon=True, framealpha=0.8)

        # Finish plot
        plt.tight_layout()

        # Save if filename provided
        if save_filename:
            save_path = os.path.join(self.save_path, save_filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self._log(f"Chart saved to {save_path}")

        return fig

    def plot_portfolio_performance(self,
                                   portfolio_history: pd.Series,
                                   trades: List[Dict[str, Any]],
                                   title: str = "Portfolio Performance",
                                   save_filename: Optional[str] = None) -> plt.Figure:
        """
        Plot portfolio equity curve with trades marked.

        Args:
            portfolio_history: Series with portfolio values (timestamp index)
            trades: List of trade dictionaries
            title: Chart title
            save_filename: Filename to save chart (if None, not saved)

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot equity curve
        ax.plot(portfolio_history.index, portfolio_history.values,
                color=self.colors['equity'], linewidth=2)

        # Calculate drawdown
        rolling_max = portfolio_history.cummax()
        drawdown = (portfolio_history - rolling_max) / rolling_max * 100

        # Create a twin axis for drawdown
        ax2 = ax.twinx()
        ax2.fill_between(drawdown.index, drawdown.values, 0,
                         color=self.colors['loss'], alpha=0.2, label='Drawdown')
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.set_ylim(bottom=min(drawdown.min() * 1.5, -1), top=1)  # Set some extra space at bottom

        # Mark trades on the equity curve
        for trade in trades:
            exit_time = trade.get('close_time')
            pnl = trade.get('realized_pnl', 0.0)

            if not exit_time:
                continue

            # Find portfolio value at trade exit
            port_value = portfolio_history.loc[portfolio_history.index >= exit_time].iloc[0] if any(
                portfolio_history.index >= exit_time) else None

            if port_value is not None:
                # Determine marker properties based on P&L
                is_win = pnl > 0
                color = self.colors['win'] if is_win else self.colors['loss']
                marker = '^' if is_win else 'v'

                # Plot marker
                ax.scatter(exit_time, port_value, color=color, marker=marker, s=80, zorder=5)

        # Set up axis labels and title
        ax.set_title(title, fontsize=16, pad=10)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)

        # Format x-axis with date
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        fig.autofmt_xdate(rotation=45)

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)

        # Format y-axis with dollar sign
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))

        # Create custom legend
        custom_lines = [
            Line2D([0], [0], color=self.colors['equity'], lw=2),
            Line2D([0], [0], marker='^', color=self.colors['win'], linestyle='None', markersize=8),
            Line2D([0], [0], marker='v', color=self.colors['loss'], linestyle='None', markersize=8),
            mpatches.Patch(color=self.colors['loss'], alpha=0.2)
        ]
        ax.legend(custom_lines, ['Equity', 'Winning Trade', 'Losing Trade', 'Drawdown'],
                  loc='best', frameon=True, framealpha=0.8)

        # Finish plot
        plt.tight_layout()

        # Save if filename provided
        if save_filename:
            save_path = os.path.join(self.save_path, save_filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self._log(f"Chart saved to {save_path}")

        return fig

    def plot_trade_analysis(self,
                            trades: List[Dict[str, Any]],
                            title: str = "Trade Analysis",
                            save_filename: Optional[str] = None) -> plt.Figure:
        """
        Plot trade analysis charts (win/loss, durations, P&L distribution).

        Args:
            trades: List of trade dictionaries
            title: Chart title
            save_filename: Filename to save chart (if None, not saved)

        Returns:
            Matplotlib figure
        """
        # Create DataFrame from trades list
        if not trades:
            self._log("No trades to analyze", logging.WARNING)
            return None

        trade_df = pd.DataFrame(trades)

        # Convert times to datetime if they're strings
        for col in ['open_time', 'close_time']:
            if col in trade_df.columns and trade_df[col].dtype == 'object':
                trade_df[col] = pd.to_datetime(trade_df[col])

        # Calculate trade durations in seconds
        if 'duration' not in trade_df.columns and 'open_time' in trade_df.columns and 'close_time' in trade_df.columns:
            trade_df['duration'] = (trade_df['close_time'] - trade_df['open_time']).dt.total_seconds()

        # Classify trades as win or loss
        trade_df['win'] = trade_df['realized_pnl'] > 0

        # Create subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        plt.suptitle(title, fontsize=16, y=0.95)

        # 1. Win/Loss Pie Chart
        win_count = trade_df['win'].sum()
        loss_count = len(trade_df) - win_count
        win_rate = win_count / len(trade_df) if len(trade_df) > 0 else 0

        axs[0, 0].pie([win_count, loss_count],
                      labels=[f'Wins ({win_count})', f'Losses ({loss_count})'],
                      autopct='%1.1f%%',
                      colors=[self.colors['win'], self.colors['loss']],
                      startangle=90,
                      wedgeprops={'alpha': 0.8})
        axs[0, 0].set_title(f'Win/Loss Ratio: {win_rate:.1%}', fontsize=14)

        # 2. P&L Distribution
        axs[0, 1].hist(trade_df['realized_pnl'], bins=20, color=self.colors['equity'], alpha=0.7)
        axs[0, 1].axvline(x=0, color='black', linestyle='--', alpha=0.7)
        axs[0, 1].set_title('P&L Distribution', fontsize=14)
        axs[0, 1].set_xlabel('P&L ($)')
        axs[0, 1].set_ylabel('Frequency')

        # Add mean and median P&L lines
        mean_pnl = trade_df['realized_pnl'].mean()
        median_pnl = trade_df['realized_pnl'].median()
        axs[0, 1].axvline(x=mean_pnl, color='red', linestyle='-', alpha=0.7, label=f'Mean: ${mean_pnl:.2f}')
        axs[0, 1].axvline(x=median_pnl, color='green', linestyle='-', alpha=0.7, label=f'Median: ${median_pnl:.2f}')
        axs[0, 1].legend()

        # 3. Trade Duration Distribution
        if 'duration' in trade_df.columns:
            # Convert seconds to more readable units
            trade_df['duration_display'] = trade_df['duration']
            trade_df.loc[trade_df['duration'] >= 60, 'duration_display'] = trade_df.loc[trade_df[
                                                                                            'duration'] >= 60, 'duration'] / 60

            # Set appropriate label based on duration range
            if trade_df['duration'].max() < 60:
                duration_label = 'Duration (seconds)'
                duration_bins = np.linspace(0, max(60, trade_df['duration'].max()), 20)
            else:
                duration_label = 'Duration (minutes)'
                duration_bins = np.linspace(0, max(5, trade_df['duration'].max() / 60), 20)

            axs[1, 0].hist(trade_df['duration_display'], bins=duration_bins, color=self.colors['equity'], alpha=0.7)
            axs[1, 0].set_title('Trade Duration Distribution', fontsize=14)
            axs[1, 0].set_xlabel(duration_label)
            axs[1, 0].set_ylabel('Frequency')
        else:
            axs[1, 0].text(0.5, 0.5, 'Duration data not available',
                           horizontalalignment='center',
                           verticalalignment='center',
                           transform=axs[1, 0].transAxes)

        # 4. Cumulative P&L
        trade_df['cumulative_pnl'] = trade_df['realized_pnl'].cumsum()
        axs[1, 1].plot(range(len(trade_df)), trade_df['cumulative_pnl'],
                       color=self.colors['equity'], marker='o', markersize=4)
        axs[1, 1].set_title('Cumulative P&L', fontsize=14)
        axs[1, 1].set_xlabel('Trade Number')
        axs[1, 1].set_ylabel('Cumulative P&L ($)')
        axs[1, 1].grid(True, linestyle='--', alpha=0.7)

        # Format y-axis with dollar sign
        axs[1, 1].yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))

        # Finish plot
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle

        # Save if filename provided
        if save_filename:
            save_path = os.path.join(self.save_path, save_filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self._log(f"Chart saved to {save_path}")

        return fig

    def plot_trade_metrics(self,
                           trades: List[Dict[str, Any]],
                           title: str = "Trade Metrics Summary",
                           save_filename: Optional[str] = None) -> plt.Figure:
        """
        Plot trade metrics summary (win rate, avg win/loss, profit factor, etc.)

        Args:
            trades: List of trade dictionaries
            title: Chart title
            save_filename: Filename to save chart (if None, not saved)

        Returns:
            Matplotlib figure
        """
        if not trades:
            self._log("No trades to analyze", logging.WARNING)
            return None

        # Calculate metrics
        win_trades = [t for t in trades if t.get('realized_pnl', 0) > 0]
        loss_trades = [t for t in trades if t.get('realized_pnl', 0) <= 0]

        total_trades = len(trades)
        win_count = len(win_trades)
        loss_count = len(loss_trades)

        win_rate = win_count / total_trades if total_trades > 0 else 0

        avg_win = np.mean([t.get('realized_pnl', 0) for t in win_trades]) if win_trades else 0
        avg_loss = np.mean([t.get('realized_pnl', 0) for t in loss_trades]) if loss_trades else 0

        total_profit = sum(t.get('realized_pnl', 0) for t in win_trades)
        total_loss = abs(sum(t.get('realized_pnl', 0) for t in loss_trades))

        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

        avg_trade = np.mean([t.get('realized_pnl', 0) for t in trades])

        trade_durations = [t.get('duration', 0) for t in trades if t.get('duration') is not None]
        avg_duration = np.mean(trade_durations) if trade_durations else 0

        # Create figure for metrics
        fig, ax = plt.subplots(figsize=(10, 6))

        # No plotting on this axis - using it as a placeholder
        ax.axis('off')

        # Create the metrics table as text
        metrics_text = f"""
        Trade Metrics Summary
        ---------------------
        Total Trades: {total_trades}
        Win Rate: {win_rate:.1%}
        Win Count: {win_count}
        Loss Count: {loss_count}

        Average Winning Trade: ${avg_win:.2f}
        Average Losing Trade: ${avg_loss:.2f}
        Win/Loss Ratio: {win_loss_ratio:.2f}

        Total Profit: ${total_profit:.2f}
        Total Loss: ${total_loss:.2f}
        Net P&L: ${total_profit - total_loss:.2f}

        Profit Factor: {profit_factor:.2f}
        Average Trade P&L: ${avg_trade:.2f}

        Average Trade Duration: {avg_duration:.1f} seconds
        """

        ax.text(0.5, 0.5, metrics_text, ha='center', va='center',
                fontfamily='monospace', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Set title
        ax.set_title(title, fontsize=16, pad=20)

        # Save if filename provided
        if save_filename:
            save_path = os.path.join(self.save_path, save_filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self._log(f"Chart saved to {save_path}")

        return fig

    def plot_multi_timeframe_view(self,
                                  data_dict: Dict[str, pd.DataFrame],
                                  timeframes: List[str],
                                  trades: List[Dict[str, Any]],
                                  timestamp: Optional[datetime] = None,
                                  window_size: Dict[str, int] = None,
                                  title: str = "Multi-Timeframe Analysis",
                                  save_filename: Optional[str] = None) -> plt.Figure:
        """
        Plot multiple timeframes side by side for a specific timestamp.

        Args:
            data_dict: Dictionary mapping timeframes to DataFrames
            timeframes: List of timeframes to display
            trades: List of trade dictionaries
            timestamp: Specific timestamp to center on (if None, uses latest)
            window_size: Dictionary mapping timeframes to window sizes
            title: Chart title
            save_filename: Filename to save chart (if None, not saved)

        Returns:
            Matplotlib figure
        """
        if not timeframes or not data_dict:
            self._log("No data or timeframes provided", logging.WARNING)
            return None

        # Default window sizes
        default_window = {'1s': 120, '10s': 60, '1m': 30, '5m': 24, '1d': 30}
        window_size = window_size or default_window

        # Create subplots
        n_rows = min(3, len(timeframes))
        n_cols = (len(timeframes) + n_rows - 1) // n_rows  # Ceiling division

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), squeeze=False)
        plt.suptitle(title, fontsize=16, y=0.95)

        # Flatten axs for easy indexing
        axs_flat = axs.flatten()

        # Plot each timeframe
        for i, tf in enumerate(timeframes):
            if i >= len(axs_flat):
                break

            ax = axs_flat[i]

            if tf not in data_dict or data_dict[tf].empty:
                ax.text(0.5, 0.5, f'No data for {tf}',
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=ax.transAxes)
                continue

            df = data_dict[tf]

            # Determine window to display
            if timestamp is None:
                # Use latest timestamp
                end_idx = len(df) - 1
            else:
                # Find nearest timestamp
                end_idx = df.index.get_indexer([timestamp], method='nearest')[0]

            start_idx = max(0, end_idx - window_size.get(tf, default_window.get(tf, 30)))

            # Get visible data
            visible_df = df.iloc[start_idx:end_idx + 1]

            # Plot price data
            ax.plot(visible_df.index, visible_df['close'], color=self.colors['price'], linewidth=1.5)

            # Add EMA lines if available
            for ema, color in [('ema_9', self.colors['ema_9']),
                               ('ema_20', self.colors['ema_20'])]:
                if ema in visible_df.columns:
                    ax.plot(visible_df.index, visible_df[ema], color=color, linewidth=1, alpha=0.8, label=ema.upper())

            # Add VWAP if available
            if 'vwap' in visible_df.columns:
                ax.plot(visible_df.index, visible_df['vwap'], color=self.colors['vwap'],
                        linewidth=1, alpha=0.8, label='VWAP')

            # Find trades visible in this window
            visible_trades = []
            for trade in trades:
                entry_time = trade.get('open_time')
                exit_time = trade.get('close_time')

                if entry_time is None or exit_time is None:
                    continue

                # Check if trade overlaps with visible window
                if (entry_time >= visible_df.index[0] and entry_time <= visible_df.index[-1]) or \
                        (exit_time >= visible_df.index[0] and exit_time <= visible_df.index[-1]):
                    visible_trades.append(trade)

            # Plot visible trades
            for trade in visible_trades:
                entry_time = trade.get('open_time')
                exit_time = trade.get('close_time')
                entry_price = trade.get('entry_price', 0.0)
                exit_price = trade.get('exit_price', 0.0)
                pnl = trade.get('realized_pnl', 0.0)

                # Determine color based on P&L
                color = self.colors['win'] if pnl > 0 else self.colors['loss']

                # Plot entry point if visible
                if entry_time >= visible_df.index[0] and entry_time <= visible_df.index[-1]:
                    ax.scatter(entry_time, entry_price, color=color, marker='^', s=80, zorder=5)

                # Plot exit point if visible
                if exit_time >= visible_df.index[0] and exit_time <= visible_df.index[-1]:
                    ax.scatter(exit_time, exit_price, color=color, marker='v', s=80, zorder=5)

                # Connect entry and exit with line if both visible or partially visible
                if (entry_time <= visible_df.index[-1] and exit_time >= visible_df.index[0]):
                    # Calculate visible start and end points
                    vis_start_time = max(entry_time, visible_df.index[0])
                    vis_end_time = min(exit_time, visible_df.index[-1])

                    # Interpolate prices if needed
                    if vis_start_time > entry_time:
                        # Start point is after entry - interpolate price
                        time_ratio = (vis_start_time - entry_time) / (exit_time - entry_time)
                        vis_start_price = entry_price + (exit_price - entry_price) * time_ratio
                    else:
                        vis_start_price = entry_price

                    if vis_end_time < exit_time:
                        # End point is before exit - interpolate price
                        time_ratio = (vis_end_time - entry_time) / (exit_time - entry_time)
                        vis_end_price = entry_price + (exit_price - entry_price) * time_ratio
                    else:
                        vis_end_price = exit_price

                    # Draw the line
                    ax.plot([vis_start_time, vis_end_time], [vis_start_price, vis_end_price],
                            color=color, linestyle='--', linewidth=1.5, alpha=0.7)

            # Set up axis labels
            ax.set_title(f"{tf} Timeframe", fontsize=12)
            ax.set_ylabel('Price ($)', fontsize=10)

            # Format x-axis with appropriate date format
            if tf in ['1d']:
                date_format = '%m/%d'
            elif tf in ['1h', '4h']:
                date_format = '%m/%d %H:%M'
            else:
                date_format = '%H:%M:%S'

            ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))

            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)

            # Add legend
            ax.legend(loc='upper left', frameon=True, framealpha=0.8, fontsize=8)

        # Remove empty subplots
        for i in range(len(timeframes), len(axs_flat)):
            fig.delaxes(axs_flat[i])

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle

        # Save if filename provided
        if save_filename:
            save_path = os.path.join(self.save_path, save_filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self._log(f"Chart saved to {save_path}")

        return fig

    def plot_tape_analysis(self,
                           trades_df: pd.DataFrame,
                           price_data: pd.DataFrame,
                           trade_events: List[Dict[str, Any]],
                           window_size: int = 120,
                           title: str = "Tape Analysis",
                           save_filename: Optional[str] = None) -> plt.Figure:
        """
        Plot tape analysis visualization with trade volume, imbalance, and prints.

        Args:
            trades_df: DataFrame with trade data (tape)
            price_data: DataFrame with price data
            trade_events: List of trade execution events
            window_size: Number of seconds to display
            title: Chart title
            save_filename: Filename to save chart (if None, not saved)

        Returns:
            Matplotlib figure
        """
        if trades_df.empty or price_data.empty:
            self._log("No data provided for tape analysis", logging.WARNING)
            return None

        # Create subplots
        fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True,
                                gridspec_kw={'height_ratios': [3, 1, 1]})
        plt.suptitle(title, fontsize=16, y=0.95)

        # Get the timeframe for display
        end_time = price_data.index[-1]
        start_time = end_time - pd.Timedelta(seconds=window_size)

        # Filter data to the window
        price_window = price_data[(price_data.index >= start_time) & (price_data.index <= end_time)]
        trades_window = trades_df[(trades_df.index >= start_time) & (trades_df.index <= end_time)]

        # 1. Top panel: Price chart with trades
        ax_price = axs[0]

        # Plot price
        ax_price.plot(price_window.index, price_window['close'], color=self.colors['price'],
                      label='Price', linewidth=1.5)

        # Plot EMAs and VWAP if available
        for indicator, color, label in [
            ('ema_9', self.colors['ema_9'], 'EMA 9'),
            ('ema_20', self.colors['ema_20'], 'EMA 20'),
            ('vwap', self.colors['vwap'], 'VWAP')
        ]:
            if indicator in price_window.columns:
                ax_price.plot(price_window.index, price_window[indicator],
                              color=color, label=label, linewidth=1, alpha=0.8)

        # Plot trading events
        for event in trade_events:
            timestamp = event.get('timestamp')
            action = event.get('action', '')
            price = event.get('fill_price', 0.0)

            if timestamp < start_time or timestamp > end_time:
                continue

            # Set marker properties based on action
            if action.lower() == 'buy':
                marker = '^'
                color = self.colors['buy']
            elif action.lower() == 'sell':
                marker = 'v'
                color = self.colors['sell']
            else:
                continue

            # Plot marker
            ax_price.scatter(timestamp, price, color=color, marker=marker, s=100, zorder=5)

            # Annotate with action
            ax_price.annotate(action.upper(),
                              xy=(timestamp, price),
                              xytext=(0, 10 if action.lower() == 'buy' else -20),
                              textcoords='offset points',
                              color=color,
                              fontweight='bold',
                              ha='center')

        ax_price.set_title('Price Chart with Trading Actions', fontsize=14)
        ax_price.set_ylabel('Price ($)', fontsize=12)
        ax_price.legend(loc='best')
        ax_price.grid(True, linestyle='--', alpha=0.7)

        # 2. Middle panel: Trade volume
        ax_vol = axs[1]

        # Aggregate trades by second and calculate volume
        if not trades_window.empty and 'size' in trades_window.columns:
            # Resample to second and sum volume
            vol_by_second = trades_window.resample('1S')['size'].sum().fillna(0)

            # Get tick colors (buy/sell) if available
            if 'side' in trades_window.columns:
                # Resample buy and sell volume separately
                buy_mask = trades_window['side'] == 'B'
                sell_mask = trades_window['side'] == 'A'

                buy_vol = trades_window[buy_mask].resample('1S')['size'].sum().fillna(0)
                sell_vol = trades_window[sell_mask].resample('1S')['size'].sum().fillna(0)

                # Plot stacked bars for buy and sell volume
                ax_vol.bar(buy_vol.index, buy_vol, color=self.colors['buy'], label='Buy Volume', alpha=0.7)
                ax_vol.bar(sell_vol.index, -sell_vol, color=self.colors['sell'], label='Sell Volume', alpha=0.7)

                # Set y-limits to be symmetric
                max_vol = max(buy_vol.max(), sell_vol.max()) * 1.1
                ax_vol.set_ylim(-max_vol, max_vol)

                # Add zero line
                ax_vol.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            else:
                # Just plot total volume
                ax_vol.bar(vol_by_second.index, vol_by_second, color=self.colors['volume'], alpha=0.7)

        ax_vol.set_title('Trade Volume by Second', fontsize=14)
        ax_vol.set_ylabel('Volume', fontsize=12)
        ax_vol.legend(loc='best')
        ax_vol.grid(True, linestyle='--', alpha=0.7)

        # 3. Bottom panel: Tape imbalance
        ax_imb = axs[2]

        # Calculate and plot tape imbalance if side information is available
        if not trades_window.empty and 'side' in trades_window.columns:
            # Resample by second and calculate imbalance
            def calc_imbalance(x):
                buy_vol = x[x['side'] == 'B']['size'].sum()
                sell_vol = x[x['side'] == 'A']['size'].sum()
                total_vol = buy_vol + sell_vol
                return (buy_vol - sell_vol) / total_vol if total_vol > 0 else 0

            imbalance = trades_window.resample('1S').apply(calc_imbalance)

            # Plot imbalance
            ax_imb.bar(imbalance.index, imbalance,
                       color=[self.colors['buy'] if x > 0 else self.colors['sell'] for x in imbalance],
                       alpha=0.7)

            # Add zero line
            ax_imb.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

            # Set y-limits
            ax_imb.set_ylim(-1.1, 1.1)

            ax_imb.set_title('Tape Imbalance (Buy - Sell) / Total', fontsize=14)
            ax_imb.set_ylabel('Imbalance', fontsize=12)
        else:
            ax_imb.text(0.5, 0.5, 'No trade side data available for imbalance calculation',
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=ax_imb.transAxes)

        ax_imb.grid(True, linestyle='--', alpha=0.7)

        # Format x-axis with time
        ax_imb.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        fig.autofmt_xdate(rotation=45)

        # Set common x-axis label
        ax_imb.set_xlabel('Time', fontsize=12)

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle

        # Save if filename provided
        if save_filename:
            save_path = os.path.join(self.save_path, save_filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self._log(f"Chart saved to {save_path}")

        return fig