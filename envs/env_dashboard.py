# envs/dashboard.py
import logging
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Deque
from dataclasses import dataclass, field
import threading
import time
import sys
from io import StringIO

from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align
from rich import box
from rich.columns import Columns
from rich.logging import RichHandler

from simulators.portfolio_simulator import PositionSideEnum, OrderSideEnum


@dataclass
class DashboardState:
    """Holds all the state information for the dashboard"""
    # Environment state
    step: int = 0
    timestamp: str = "N/A"
    symbol: str = "N/A"
    episode_reward: float = 0.0
    total_reward: float = 0.0

    # Market data
    current_price: Optional[float] = None
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None
    market_session: str = "UNKNOWN"

    # Position data
    position_qty: float = 0.0
    position_side: str = "FLAT"
    position_avg_entry: float = 0.0
    position_market_value: float = 0.0
    position_unrealized_pnl: float = 0.0

    # Portfolio metrics
    total_equity: float = 0.0
    cash: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    # Session totals
    total_commissions: float = 0.0
    total_fees: float = 0.0
    total_slippage: float = 0.0
    total_volume: float = 0.0
    total_turnover: float = 0.0

    # Action info
    last_action_type: str = "N/A"
    last_action_size: str = "N/A"
    last_action_invalid: bool = False
    last_action_reason: str = ""
    invalid_actions_count: int = 0

    # Recent history
    recent_actions: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=10))
    recent_fills: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=10))
    recent_logs: Deque[str] = field(default_factory=lambda: deque(maxlen=20))

    # Status
    is_terminated: bool = False
    is_truncated: bool = False
    termination_reason: str = ""

    # Performance metrics
    initial_capital: float = 25000.0
    session_pnl: float = 0.0
    session_pnl_pct: float = 0.0

    # Update tracking
    last_update_time: float = 0.0
    update_count: int = 0


class LogCapture(logging.Handler):
    """Custom log handler to capture logs for dashboard display"""

    def __init__(self, dashboard_state: DashboardState):
        super().__init__()
        self.dashboard_state = dashboard_state
        self.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    def emit(self, record):
        try:
            msg = self.format(record)
            # Truncate very long messages
            if len(msg) > 120:
                msg = msg[:117] + "..."
            self.dashboard_state.recent_logs.append(msg)
        except Exception:
            pass  # Don't let logging errors break the dashboard


class TradingDashboard:
    """
    A live dashboard for monitoring trading environment using Rich Live display.
    Now with improved updates, log display, and error handling.
    """

    def __init__(self, console: Optional[Console] = None, show_logs: bool = True):
        """
        Initialize the trading dashboard.

        Args:
            console: Rich console instance (creates new if None)
            show_logs: Whether to show logs in the dashboard
        """
        self.console = console or Console()
        self.show_logs = show_logs
        self.state = DashboardState()
        self.live: Optional[Live] = None
        self.layout: Optional[Layout] = None
        self.logger = logging.getLogger(__name__)

        # Log capture
        self.log_handler: Optional[LogCapture] = None
        if self.show_logs:
            self.log_handler = LogCapture(self.state)
            self.log_handler.setLevel(logging.INFO)
            # Add to root logger to capture all logs
            logging.getLogger().addHandler(self.log_handler)

        # State management
        self._running = False
        self._last_update = 0
        self._force_update = False

        self._setup_layout()

    def _setup_layout(self):
        """Create the dashboard layout structure"""
        self.layout = Layout()

        if self.show_logs:
            # Split into header, main content, logs, and footer
            self.layout.split_column(
                Layout(name="header", size=3),
                Layout(name="main", ratio=2),
                Layout(name="logs", size=8),
                Layout(name="footer", size=3)
            )
        else:
            # Split into header, main content, and footer
            self.layout.split_column(
                Layout(name="header", size=3),
                Layout(name="main", ratio=1),
                Layout(name="footer", size=3)
            )

        # Split main area into left and right sections
        self.layout["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=3)
        )

        # Split left into market data and position
        self.layout["left"].split_column(
            Layout(name="market", size=8),
            Layout(name="position", size=8),
            Layout(name="costs", size=6)
        )

        # Split right into portfolio, actions, and fills
        self.layout["right"].split_column(
            Layout(name="portfolio", size=8),
            Layout(name="actions", ratio=1),
            Layout(name="fills", ratio=1)
        )

    def start(self):
        """Start the live dashboard"""
        if self._running:
            return

        self._running = True

        try:
            self.live = Live(
                self.layout,
                console=self.console,
                refresh_per_second=4,  # Refresh 4 times per second
                screen=True,
                auto_refresh=True
            )

            # Start the live display
            self.live.start()
            self._update_display()  # Initial update

            self.logger.info("Trading dashboard started")

        except Exception as e:
            self.logger.error(f"Failed to start dashboard: {e}")
            self._running = False

    def stop(self):
        """Stop the live dashboard"""
        self._running = False

        if self.live:
            try:
                self.live.stop()
            except Exception as e:
                self.logger.error(f"Error stopping dashboard: {e}")

        # Remove log handler
        if self.log_handler:
            logging.getLogger().removeHandler(self.log_handler)

        self.logger.info("Trading dashboard stopped")

    def force_update(self):
        """Force an immediate update of the dashboard"""
        self._force_update = True
        if self._running:
            self._update_display()

    def _update_display(self):
        """Update all dashboard components"""
        if not self.layout or not self._running:
            return

        try:
            current_time = time.time()

            # Update state tracking
            self.state.last_update_time = current_time
            self.state.update_count += 1

            # Update each section
            self.layout["header"].update(self._create_header())
            self.layout["market"].update(self._create_market_panel())
            self.layout["position"].update(self._create_position_panel())
            self.layout["costs"].update(self._create_costs_panel())
            self.layout["portfolio"].update(self._create_portfolio_panel())
            self.layout["actions"].update(self._create_actions_panel())
            self.layout["fills"].update(self._create_fills_panel())

            if self.show_logs:
                self.layout["logs"].update(self._create_logs_panel())

            self.layout["footer"].update(self._create_footer())

            self._force_update = False

        except Exception as e:
            self.logger.error(f"Error updating display components: {e}")

    def update_state(self, info_dict: Dict[str, Any], market_state: Optional[Dict[str, Any]] = None):
        """
        Update dashboard state with new information from the environment.
        Now updates immediately every time it's called.
        """
        try:
            # Basic environment info
            self.state.step = info_dict.get('step', 0)
            self.state.timestamp = info_dict.get('timestamp_iso', 'N/A')
            self.state.episode_reward = info_dict.get('reward_step', 0.0)
            self.state.total_reward = info_dict.get('episode_cumulative_reward', 0.0)

            # Portfolio metrics
            self.state.total_equity = info_dict.get('portfolio_equity', 0.0)
            self.state.cash = info_dict.get('portfolio_cash', 0.0)
            self.state.unrealized_pnl = info_dict.get('portfolio_unrealized_pnl', 0.0)
            self.state.realized_pnl = info_dict.get('portfolio_realized_pnl_session_net', 0.0)

            # Calculate session PnL
            self.state.session_pnl = self.state.total_equity - self.state.initial_capital
            if self.state.initial_capital > 0:
                self.state.session_pnl_pct = (self.state.session_pnl / self.state.initial_capital) * 100

            # Position data
            symbol = self.state.symbol
            if symbol != "N/A":
                self.state.position_qty = info_dict.get(f'position_{symbol}_qty', 0.0)
                self.state.position_side = info_dict.get(f'position_{symbol}_side', 'FLAT')
                self.state.position_avg_entry = info_dict.get(f'position_{symbol}_avg_entry', 0.0)

            # Action info
            action_decoded = info_dict.get('action_decoded', {})
            if action_decoded:
                action_type = action_decoded.get('type')
                self.state.last_action_type = action_type.name if hasattr(action_type, 'name') else str(action_type)

                size_enum = action_decoded.get('size_enum')
                self.state.last_action_size = size_enum.name if hasattr(size_enum, 'name') else str(size_enum)

                self.state.last_action_invalid = bool(action_decoded.get('invalid_reason'))
                self.state.last_action_reason = str(action_decoded.get('invalid_reason', ''))

                # Add to recent actions
                self.state.recent_actions.append({
                    'step': self.state.step,
                    'timestamp': self.state.timestamp,
                    'type': self.state.last_action_type,
                    'size': self.state.last_action_size,
                    'invalid': self.state.last_action_invalid,
                    'reason': self.state.last_action_reason
                })

            self.state.invalid_actions_count = info_dict.get('invalid_actions_total_episode', 0)

            # Market data
            if market_state:
                self.state.current_price = market_state.get('current_price')
                self.state.bid_price = market_state.get('best_bid_price')
                self.state.ask_price = market_state.get('best_ask_price')
                self.state.bid_size = market_state.get('best_bid_size', 0)
                self.state.ask_size = market_state.get('best_ask_size', 0)
                self.state.market_session = market_state.get('market_session', 'UNKNOWN')

            # Fills
            fills_step = info_dict.get('fills_step', [])
            if fills_step:
                for fill in fills_step:
                    self.state.recent_fills.append({
                        'step': self.state.step,
                        'timestamp': self.state.timestamp,
                        'side': fill.get('order_side', 'N/A'),
                        'quantity': fill.get('executed_quantity', 0.0),
                        'price': fill.get('executed_price', 0.0),
                        'commission': fill.get('commission', 0.0),
                        'fees': fill.get('fees', 0.0),
                        'slippage': fill.get('slippage_cost_total', 0.0)
                    })

            # Episode summary if available
            episode_summary = info_dict.get('episode_summary', {})
            if episode_summary:
                self.state.total_commissions = episode_summary.get('session_total_commissions', 0.0)
                self.state.total_fees = episode_summary.get('session_total_fees', 0.0)
                self.state.total_slippage = episode_summary.get('session_total_slippage_cost', 0.0)
                self.state.termination_reason = episode_summary.get('termination_reason', '')

            # Status
            self.state.is_terminated = info_dict.get('termination_reason') is not None
            self.state.is_truncated = info_dict.get('TimeLimit.truncated', False)

            # Force immediate update
            self._update_display()

        except Exception as e:
            self.logger.error(f"Error updating dashboard state: {e}")
            # Still try to update display to show error
            self._update_display()

    def set_symbol(self, symbol: str):
        """Set the trading symbol"""
        self.state.symbol = symbol
        self._update_display()

    def set_initial_capital(self, capital: float):
        """Set the initial capital for PnL calculations"""
        self.state.initial_capital = capital
        self._update_display()

    def _create_header(self) -> Panel:
        """Create the header panel with basic info"""
        # Parse timestamp for display
        time_str = "N/A"
        if self.state.timestamp != "N/A":
            try:
                time_str = self.state.timestamp.split('T')[1].split('.')[0]
            except:
                time_str = "N/A"

        # Status indicator
        status = "🟢 ACTIVE"
        if self.state.is_terminated:
            status = "🔴 TERMINATED"
        elif self.state.is_truncated:
            status = "🟡 TRUNCATED"

        # Update info
        update_info = f"Updates: {self.state.update_count}"

        header_table = Table.grid(padding=1)
        header_table.add_column(justify="left")
        header_table.add_column(justify="center")
        header_table.add_column(justify="right")

        header_table.add_row(
            f"[bold cyan]STEP {self.state.step}[/bold cyan] | [cyan]{time_str}[/cyan]",
            f"[bold white]{self.state.symbol}[/bold white] | {status}",
            f"[bold green]REWARD: {self.state.total_reward:.4f}[/bold green] | {update_info}"
        )

        return Panel(header_table, style="bright_blue", box=box.HEAVY)

    def _create_market_panel(self) -> Panel:
        """Create market data panel"""
        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))

        # Current price
        price_text = f"${self.state.current_price:.2f}" if self.state.current_price else "N/A"
        table.add_row("Price", Text(price_text, style="bold yellow"))

        # Bid/Ask
        bid_text = f"${self.state.bid_price:.2f}" if self.state.bid_price else "N/A"
        ask_text = f"${self.state.ask_price:.2f}" if self.state.ask_price else "N/A"
        table.add_row("Bid", Text(bid_text, style="red"))
        table.add_row("Ask", Text(ask_text, style="green"))

        # Spread
        if self.state.bid_price and self.state.ask_price:
            spread = self.state.ask_price - self.state.bid_price
            spread_bps = (spread / self.state.ask_price) * 10000 if self.state.ask_price > 0 else 0
            table.add_row("Spread", Text(f"${spread:.4f} ({spread_bps:.1f}bps)", style="white"))
        else:
            table.add_row("Spread", "N/A")

        # Session
        table.add_row("Session", Text(self.state.market_session, style="cyan"))

        return Panel(table, title="[bold]Market Data", border_style="yellow")

    def _create_position_panel(self) -> Panel:
        """Create position information panel"""
        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))

        # Position side with color coding
        side_style = {
            "LONG": "green",
            "SHORT": "red",
            "FLAT": "white"
        }.get(self.state.position_side, "white")

        table.add_row("Side", Text(self.state.position_side, style=f"bold {side_style}"))
        table.add_row("Quantity", Text(f"{self.state.position_qty:.2f}", style="white"))

        # Average entry price
        entry_text = f"${self.state.position_avg_entry:.2f}" if self.state.position_avg_entry > 0 else "N/A"
        table.add_row("Avg Entry", Text(entry_text, style="white"))

        # Current P&L vs entry
        if (self.state.current_price and self.state.position_avg_entry > 0 and
                self.state.position_qty != 0):
            price_diff = self.state.current_price - self.state.position_avg_entry
            if self.state.position_side == "SHORT":
                price_diff = -price_diff
            price_diff_pct = (price_diff / self.state.position_avg_entry) * 100

            diff_style = "green" if price_diff > 0 else "red" if price_diff < 0 else "white"
            diff_text = f"${price_diff:.2f} ({price_diff_pct:+.2f}%)"
            table.add_row("P&L vs Entry", Text(diff_text, style=diff_style))
        else:
            table.add_row("P&L vs Entry", "N/A")

        # Market value
        if self.state.position_qty != 0 and self.state.current_price:
            market_value = abs(self.state.position_qty * self.state.current_price)
            table.add_row("Market Value", Text(f"${market_value:.2f}", style="white"))
        else:
            table.add_row("Market Value", "N/A")

        return Panel(table, title="[bold]Position", border_style="blue")

    def _create_costs_panel(self) -> Panel:
        """Create costs breakdown panel"""
        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))

        table.add_row("Commissions", Text(f"${self.state.total_commissions:.2f}", style="red"))
        table.add_row("Fees", Text(f"${self.state.total_fees:.2f}", style="red"))
        table.add_row("Slippage", Text(f"${self.state.total_slippage:.2f}", style="red"))

        # Total costs
        total_costs = self.state.total_commissions + self.state.total_fees + self.state.total_slippage
        table.add_row("Total Costs", Text(f"${total_costs:.2f}", style="bold red"))

        return Panel(table, title="[bold]Costs", border_style="red")

    def _create_portfolio_panel(self) -> Panel:
        """Create portfolio metrics panel"""
        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))

        # Total equity with color coding
        equity_style = "bold green" if self.state.total_equity >= self.state.initial_capital else "bold red"
        table.add_row("Total Equity", Text(f"${self.state.total_equity:.2f}", style=equity_style))

        # Cash
        table.add_row("Cash", Text(f"${self.state.cash:.2f}", style="white"))

        # Unrealized PnL
        unreal_style = "green" if self.state.unrealized_pnl > 0 else "red" if self.state.unrealized_pnl < 0 else "white"
        table.add_row("Unrealized P&L", Text(f"${self.state.unrealized_pnl:.2f}", style=unreal_style))

        # Realized PnL
        real_style = "green" if self.state.realized_pnl > 0 else "red" if self.state.realized_pnl < 0 else "white"
        table.add_row("Realized P&L", Text(f"${self.state.realized_pnl:.2f}", style=real_style))

        # Session PnL
        session_style = "bold green" if self.state.session_pnl > 0 else "bold red" if self.state.session_pnl < 0 else "bold white"
        table.add_row("Session P&L",
                      Text(f"${self.state.session_pnl:.2f} ({self.state.session_pnl_pct:+.2f}%)", style=session_style))

        # Step reward
        reward_style = "green" if self.state.episode_reward > 0 else "red" if self.state.episode_reward < 0 else "white"
        table.add_row("Step Reward", Text(f"{self.state.episode_reward:.4f}", style=reward_style))

        return Panel(table, title="[bold]Portfolio", border_style="green")

    def _create_actions_panel(self) -> Panel:
        """Create recent actions panel"""
        table = Table(box=box.SIMPLE, show_edge=False, padding=(0, 1))
        table.add_column("Step", width=6)
        table.add_column("Action", width=8)
        table.add_column("Size", width=8)
        table.add_column("Status", width=8)

        # Add recent actions (most recent first)
        for action in list(self.state.recent_actions)[-5:]:  # Last 5 actions
            step = str(action['step'])
            action_type = action['type']
            size = action['size']

            # Status with color
            if action['invalid']:
                status = Text("INVALID", style="bold red")
            else:
                status_text = "OK"
                if action_type == "BUY":
                    status = Text(status_text, style="green")
                elif action_type == "SELL":
                    status = Text(status_text, style="red")
                else:
                    status = Text(status_text, style="white")

            table.add_row(step, action_type, size, status)

        # If no actions, show placeholder
        if not self.state.recent_actions:
            table.add_row("", "No actions yet", "", "")

        return Panel(table, title="[bold]Recent Actions", border_style="magenta")

    def _create_fills_panel(self) -> Panel:
        """Create recent fills panel"""
        table = Table(box=box.SIMPLE, show_edge=False, padding=(0, 1))
        table.add_column("Step", width=6)
        table.add_column("Side", width=6)
        table.add_column("Qty", width=8)
        table.add_column("Price", width=10)
        table.add_column("Costs", width=8)

        # Add recent fills (most recent first)
        for fill in list(self.state.recent_fills)[-5:]:  # Last 5 fills
            step = str(fill['step'])

            # Side with color
            side_obj = fill['side']
            if hasattr(side_obj, 'value'):
                side_text = side_obj.value
            else:
                side_text = str(side_obj)

            side_style = "green" if side_text == "BUY" else "red" if side_text == "SELL" else "white"
            side = Text(side_text, style=side_style)

            qty = f"{fill['quantity']:.1f}"
            price = f"${fill['price']:.2f}"

            # Total costs
            total_cost = fill['commission'] + fill['fees'] + fill['slippage']
            costs = f"${total_cost:.2f}"

            table.add_row(step, side, qty, price, costs)

        # If no fills, show placeholder
        if not self.state.recent_fills:
            table.add_row("", "No fills yet", "", "", "")

        return Panel(table, title="[bold]Recent Fills", border_style="yellow")

    def _create_logs_panel(self) -> Panel:
        """Create logs panel"""
        if not self.show_logs:
            return Panel("Logs disabled", title="[bold]Logs", border_style="dim")

        table = Table(box=box.SIMPLE, show_header=False, show_edge=False, padding=(0, 1))
        table.add_column("Log Message", ratio=1)

        # Add recent logs
        recent_logs = list(self.state.recent_logs)[-15:]  # Last 15 log messages

        for log_msg in recent_logs:
            # Color code by log level
            style = "white"
            if " ERROR " in log_msg:
                style = "red"
            elif " WARNING " in log_msg:
                style = "yellow"
            elif " INFO " in log_msg:
                style = "cyan"
            elif " DEBUG " in log_msg:
                style = "dim"

            table.add_row(Text(log_msg, style=style))

        if not recent_logs:
            table.add_row(Text("No logs yet...", style="dim"))

        return Panel(table, title="[bold]Recent Logs", border_style="cyan")

    def _create_footer(self) -> Panel:
        """Create footer with status and alerts"""
        footer_text = ""

        # Termination info
        if self.state.is_terminated and self.state.termination_reason:
            footer_text = f"[bold red]TERMINATED: {self.state.termination_reason}[/bold red]"
        elif self.state.is_truncated:
            footer_text = "[bold yellow]TRUNCATED: Max steps reached[/bold yellow]"
        elif self.state.invalid_actions_count > 0:
            footer_text = f"[yellow]Invalid actions this episode: {self.state.invalid_actions_count}[/yellow]"
        else:
            footer_text = "[green]Environment running normally[/green]"

        # Add refresh info
        refresh_info = f" | Last update: {time.strftime('%H:%M:%S')}"
        footer_text += refresh_info

        return Panel(
            Align.center(Text.from_markup(footer_text)),
            style="bright_white"
        )


# Convenience function for easy integration
def create_dashboard(console: Optional[Console] = None, show_logs: bool = True) -> TradingDashboard:
    """Create and return a new trading dashboard instance"""
    return TradingDashboard(console=console, show_logs=show_logs)