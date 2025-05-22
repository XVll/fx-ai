# envs/env_dashboard.py - Enhanced with centralized logging and 2-column layout
import logging
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Deque
from dataclasses import dataclass, field
import time

from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align
from rich import box
from rich.console import Group
from rich.rule import Rule

from simulators.portfolio_simulator import PositionSideEnum, OrderSideEnum
from utils.logger import CentralizedLogger, get_logger


@dataclass
class DashboardState:
    """Holds all the state information for the dashboard"""
    # Environment state
    step: int = 0
    timestamp: str = "N/A"
    symbol: str = "N/A"
    episode_reward: float = 0.0
    step_reward: float = 0.0
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

    # Recent history for dashboard display only
    recent_actions: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=5))
    recent_fills: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=5))

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

    # Training info
    episode_number: int = 0
    total_episodes: int = 0
    total_steps: int = 0
    update_count_training: int = 0
    is_training: bool = True
    is_evaluating: bool = False


class EnhancedTradingDashboard:
    """
    Enhanced trading dashboard with centralized logging and 2-column layout.
    Left column: Live logs
    Right column: Trading dashboard
    """

    def __init__(self, logger_manager: Optional[CentralizedLogger] = None):
        """
        Initialize the enhanced trading dashboard.

        Args:
            logger_manager: Centralized logger instance
        """
        self.logger_manager = logger_manager or get_logger()
        self.console = self.logger_manager.console
        self.state = DashboardState()
        self.live: Optional[Live] = None
        self.layout: Layout = self._create_layout()

        # State management
        self._running = False
        self._last_log_count = 0

    def _create_layout(self) -> Layout:
        """Create the 2-column dashboard layout structure"""
        layout = Layout()

        # Main structure: 2 columns
        layout.split_row(
            Layout(name="logs_column", ratio=1),  # Left: Logs
            Layout(name="dashboard_column", ratio=1)  # Right: Dashboard
        )

        # Left column: Logs section
        layout["logs_column"].split_column(
            Layout(name="logs_header", size=3),
            Layout(name="logs_content", ratio=1)
        )

        # Right column: Dashboard sections
        layout["dashboard_column"].split_column(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=3)
        )

        # Dashboard body: 2 columns
        layout["body"].split_row(
            Layout(name="left_info", ratio=1),
            Layout(name="right_info", ratio=1)
        )

        # Left info: Market data, Position
        layout["left_info"].split_column(
            Layout(name="market", size=9),
            Layout(name="position", size=9),
            Layout(name="training", size=6)
        )

        # Right info: Portfolio, Actions, Fills
        layout["right_info"].split_column(
            Layout(name="portfolio", size=9),
            Layout(name="actions", size=8),
            Layout(name="fills", size=7)
        )

        # Initialize with empty content
        self._update_layout(layout)
        return layout

    def start(self):
        """Start the enhanced dashboard with centralized logging"""
        if self._running:
            return

        try:
            # Create Live display
            self.live = Live(
                self.layout,
                console=self.console,
                refresh_per_second=4,  # Higher refresh rate for logs
                screen=False,
                auto_refresh=True,
                transient=False,
                redirect_stdout=False,  # Don't redirect - we handle logging differently
                redirect_stderr=False,
                vertical_overflow="ellipsis"
            )

            # Start the live display
            self.live.start()
            self._running = True

            # Log startup message using centralized logger
            self.logger_manager.info("🚀 Enhanced Trading Dashboard Started", "dashboard")
            self.logger_manager.info("📊 Logs displayed in left column, dashboard in right column", "dashboard")

        except Exception as e:
            self.logger_manager.error(f"Failed to start enhanced dashboard: {e}", "dashboard")
            self._running = False

    def stop(self):
        """Stop the enhanced dashboard"""
        if not self._running:
            return

        self._running = False

        if self.live:
            try:
                self.logger_manager.info("🛑 Stopping enhanced trading dashboard...", "dashboard")
                self.live.stop()
            except Exception as e:
                print(f"Error stopping enhanced dashboard: {e}")

    def update_state(self, info_dict: Dict[str, Any], market_state: Optional[Dict[str, Any]] = None):
        """
        Update dashboard state and refresh display.
        """
        if not self._running or not self.live:
            return

        try:
            # Update state data
            self._update_state_data(info_dict, market_state)

            # Update the layout with new data
            self._update_layout(self.layout)

            # Refresh the display
            self.live.update(self.layout)

        except Exception as e:
            self.logger_manager.error(f"Error updating enhanced dashboard: {e}", "dashboard")

    def _update_state_data(self, info_dict: Dict[str, Any], market_state: Optional[Dict[str, Any]] = None):
        """Update internal state data"""
        # Basic environment info
        self.state.step = info_dict.get('step', self.state.step)
        self.state.timestamp = info_dict.get('timestamp_iso', 'N/A')
        self.state.episode_reward = info_dict.get('episode_cumulative_reward', 0.0)
        self.state.step_reward = info_dict.get('reward_step', 0.0)
        self.state.total_reward = self.state.episode_reward

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
                'type': self.state.last_action_type,
                'size': self.state.last_action_size,
                'invalid': self.state.last_action_invalid
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
                    'side': fill.get('order_side', 'N/A'),
                    'quantity': fill.get('executed_quantity', 0.0),
                    'price': fill.get('executed_price', 0.0)
                })

        # Episode summary
        episode_summary = info_dict.get('episode_summary', {})
        if episode_summary:
            self.state.total_commissions = episode_summary.get('session_total_commissions', 0.0)
            self.state.total_fees = episode_summary.get('session_total_fees', 0.0)
            self.state.total_slippage = episode_summary.get('session_total_slippage_cost', 0.0)
            self.state.termination_reason = episode_summary.get('termination_reason', '')

        # Status
        self.state.is_terminated = info_dict.get('termination_reason') is not None
        self.state.is_truncated = info_dict.get('TimeLimit.truncated', False)

        # Update tracking
        self.state.last_update_time = time.time()
        self.state.update_count += 1

    def _update_layout(self, layout: Layout):
        """Update all layout components with current state data"""
        # Update logs section
        layout["logs_header"].update(self._create_logs_header())
        layout["logs_content"].update(self._create_logs_content())

        # Update dashboard sections
        layout["header"].update(self._create_header())
        layout["market"].update(self._create_market_panel())
        layout["position"].update(self._create_position_panel())
        layout["training"].update(self._create_training_panel())
        layout["portfolio"].update(self._create_portfolio_panel())
        layout["actions"].update(self._create_actions_panel())
        layout["fills"].update(self._create_fills_panel())
        layout["footer"].update(self._create_footer())

    def _create_logs_header(self) -> Panel:
        """Create the logs section header"""
        log_count = len(self.logger_manager.get_recent_logs())
        header_text = f"[bold cyan]📋 Live Logs[/bold cyan] ([yellow]{log_count}[/yellow] entries)"

        if self.state.is_training:
            status = "[green]🏃 TRAINING[/green]"
        elif self.state.is_evaluating:
            status = "[blue]🔍 EVALUATING[/blue]"
        else:
            status = "[yellow]⏸️ IDLE[/yellow]"

        header_table = Table.grid(padding=1)
        header_table.add_column(justify="left")
        header_table.add_column(justify="right")

        header_table.add_row(header_text, status)

        return Panel(header_table, style="bright_blue", box=box.HEAVY)

    def _create_logs_content(self) -> Panel:
        """Create the live logs content panel"""
        # Get recent logs from centralized logger
        formatted_logs = self.logger_manager.get_formatted_logs_for_display(count=30)

        if not formatted_logs:
            content = Text("No logs yet...", style="dim white")
        else:
            # Show most recent logs at the bottom
            content = Group(*formatted_logs[-30:])  # Show last 30 logs

        return Panel(
            content,
            title="",
            border_style="white",
            box=box.SIMPLE,
            padding=(0, 1)
        )

    def _create_training_panel(self) -> Panel:
        """Create training information panel"""
        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))

        table.add_row("Episode", Text(f"{self.state.episode_number}", style="bold cyan"))
        table.add_row("Total Steps", Text(f"{self.state.total_steps}", style="white"))
        table.add_row("Updates", Text(f"{self.state.update_count_training}", style="white"))

        if self.state.is_training:
            status = Text("TRAINING", style="bold green")
        elif self.state.is_evaluating:
            status = Text("EVALUATING", style="bold blue")
        else:
            status = Text("IDLE", style="yellow")

        table.add_row("Status", status)

        return Panel(table, title="[bold]Training", border_style="cyan")

    # [Rest of the panel creation methods remain the same as before...]
    def _create_header(self) -> Panel:
        """Create the header panel with basic info"""
        time_str = "N/A"
        if self.state.timestamp != "N/A":
            try:
                time_str = self.state.timestamp.split('T')[1].split('.')[0]
            except:
                time_str = "N/A"

        status = "🟢 ACTIVE"
        if self.state.is_terminated:
            status = "🔴 TERMINATED"
        elif self.state.is_truncated:
            status = "🟡 TRUNCATED"

        header_table = Table.grid(padding=1)
        header_table.add_column(justify="left")
        header_table.add_column(justify="center")
        header_table.add_column(justify="right")

        reward_text = f"[bold green]EPISODE: {self.state.episode_reward:.4f}[/bold green] | [yellow]STEP: {self.state.step_reward:.4f}[/yellow]"

        header_table.add_row(
            f"[bold cyan]STEP {self.state.step}[/bold cyan] | [cyan]{time_str}[/cyan]",
            f"[bold white]{self.state.symbol}[/bold white] | {status}",
            reward_text
        )

        return Panel(header_table, style="bright_blue", box=box.HEAVY)

    def _create_market_panel(self) -> Panel:
        """Create market data panel"""
        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))

        price_text = f"${self.state.current_price:.4f}" if self.state.current_price else "N/A"
        table.add_row("Price", Text(price_text, style="bold yellow"))

        bid_text = f"${self.state.bid_price:.4f}" if self.state.bid_price else "N/A"
        ask_text = f"${self.state.ask_price:.4f}" if self.state.ask_price else "N/A"
        table.add_row("Bid", Text(bid_text, style="red"))
        table.add_row("Ask", Text(ask_text, style="green"))

        if self.state.bid_price and self.state.ask_price:
            spread = self.state.ask_price - self.state.bid_price
            spread_bps = (spread / self.state.ask_price) * 10000 if self.state.ask_price > 0 else 0
            table.add_row("Spread", Text(f"${spread:.4f} ({spread_bps:.1f}bps)", style="white"))
        else:
            table.add_row("Spread", "N/A")

        table.add_row("Session", Text(self.state.market_session, style="cyan"))

        return Panel(table, title="[bold]Market Data", border_style="yellow")

    def _create_position_panel(self) -> Panel:
        """Create position information panel"""
        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))

        side_style = {
            "LONG": "green",
            "SHORT": "red",
            "FLAT": "white"
        }.get(self.state.position_side, "white")

        table.add_row("Side", Text(self.state.position_side, style=f"bold {side_style}"))
        table.add_row("Quantity", Text(f"{self.state.position_qty:.2f}", style="white"))

        entry_text = f"${self.state.position_avg_entry:.4f}" if self.state.position_avg_entry > 0 else "N/A"
        table.add_row("Avg Entry", Text(entry_text, style="white"))

        if (self.state.current_price and self.state.position_avg_entry > 0 and self.state.position_qty != 0):
            price_diff = self.state.current_price - self.state.position_avg_entry
            if self.state.position_side == "SHORT":
                price_diff = -price_diff
            price_diff_pct = (price_diff / self.state.position_avg_entry) * 100

            diff_style = "green" if price_diff > 0 else "red" if price_diff < 0 else "white"
            diff_text = f"${price_diff:.4f} ({price_diff_pct:+.2f}%)"
            table.add_row("P&L vs Entry", Text(diff_text, style=diff_style))
        else:
            table.add_row("P&L vs Entry", "N/A")

        if self.state.position_qty != 0 and self.state.current_price:
            market_value = abs(self.state.position_qty * self.state.current_price)
            table.add_row("Market Value", Text(f"${market_value:.2f}", style="white"))
        else:
            table.add_row("Market Value", "N/A")

        return Panel(table, title="[bold]Position", border_style="blue")

    def _create_portfolio_panel(self) -> Panel:
        """Create portfolio metrics panel"""
        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))

        equity_style = "bold green" if self.state.total_equity >= self.state.initial_capital else "bold red"
        table.add_row("Total Equity", Text(f"${self.state.total_equity:.2f}", style=equity_style))

        table.add_row("Cash", Text(f"${self.state.cash:.2f}", style="white"))

        unreal_style = "green" if self.state.unrealized_pnl > 0 else "red" if self.state.unrealized_pnl < 0 else "white"
        table.add_row("Unrealized P&L", Text(f"${self.state.unrealized_pnl:.4f}", style=unreal_style))

        real_style = "green" if self.state.realized_pnl > 0 else "red" if self.state.realized_pnl < 0 else "white"
        table.add_row("Realized P&L", Text(f"${self.state.realized_pnl:.4f}", style=real_style))

        session_style = "bold green" if self.state.session_pnl > 0 else "bold red" if self.state.session_pnl < 0 else "bold white"
        table.add_row("Session P&L",
                      Text(f"${self.state.session_pnl:.4f} ({self.state.session_pnl_pct:+.2f}%)", style=session_style))

        return Panel(table, title="[bold]Portfolio", border_style="green")

    def _create_actions_panel(self) -> Panel:
        """Create recent actions panel"""
        table = Table(box=box.SIMPLE, show_edge=False, padding=(0, 1))
        table.add_column("Step", width=6)
        table.add_column("Action", width=8)
        table.add_column("Size", width=8)
        table.add_column("Status", width=8)

        recent_actions = list(self.state.recent_actions)[-5:]
        for action in recent_actions:
            step = str(action['step'])
            action_type = action['type']
            size = action['size']

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

        recent_fills = list(self.state.recent_fills)[-5:]
        for fill in recent_fills:
            step = str(fill['step'])

            side_obj = fill['side']
            if hasattr(side_obj, 'value'):
                side_text = side_obj.value
            else:
                side_text = str(side_obj)

            side_style = "green" if side_text == "BUY" else "red" if side_text == "SELL" else "white"
            side = Text(side_text, style=side_style)

            qty = f"{fill['quantity']:.2f}"
            price = f"${fill['price']:.4f}"

            table.add_row(step, side, qty, price)

        if not self.state.recent_fills:
            table.add_row("", "No fills yet", "", "")

        return Panel(table, title="[bold]Recent Fills", border_style="yellow")

    def _create_footer(self) -> Panel:
        """Create footer with status and alerts"""
        footer_text = ""

        if self.state.is_terminated and self.state.termination_reason:
            footer_text = f"[bold red]TERMINATED: {self.state.termination_reason}[/bold red]"
        elif self.state.is_truncated:
            footer_text = "[bold yellow]TRUNCATED: Max steps reached[/bold yellow]"
        elif self.state.invalid_actions_count > 0:
            footer_text = f"[yellow]Invalid actions this episode: {self.state.invalid_actions_count}[/yellow]"
        else:
            footer_text = "[green]Environment running normally[/green]"

        footer_text += f" | Step: {self.state.step} | Dashboard updates: {self.state.update_count}"

        return Panel(
            Align.center(Text.from_markup(footer_text)),
            style="bright_white"
        )

    def set_symbol(self, symbol: str):
        """Set the trading symbol"""
        self.state.symbol = symbol

    def set_initial_capital(self, capital: float):
        """Set the initial capital for PnL calculations"""
        self.state.initial_capital = capital

    def set_training_info(self, episode_num: int = 0, total_episodes: int = 0,
                          total_steps: int = 0, update_count: int = 0,
                          buffer_size: int = 0, is_training: bool = True,
                          is_evaluating: bool = False, learning_rate: float = 0.0):
        """Set training information for dashboard display"""
        self.state.episode_number = episode_num
        self.state.total_episodes = total_episodes
        self.state.total_steps = total_steps
        self.state.update_count_training = update_count
        self.state.is_training = is_training
        self.state.is_evaluating = is_evaluating


# Convenience function for easy integration
def create_enhanced_dashboard(logger_manager: Optional[CentralizedLogger] = None) -> EnhancedTradingDashboard:
    """Create and return a new enhanced trading dashboard instance"""
    return EnhancedTradingDashboard(logger_manager=logger_manager)