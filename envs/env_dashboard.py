# envs/env_dashboard.py - FIXED: Enhanced dashboard with improved training metrics and reward system

import logging
import threading
import time
from collections import deque
from typing import Any, Dict, Optional, Deque, List
from dataclasses import dataclass, field
from enum import Enum
import io
import sys
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone

from rich.console import Console, Group
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align
from rich import box
from rich.logging import RichHandler
from rich.progress import Progress, BarColumn, TextColumn


class TrainingStage(Enum):
    """Training stages for dashboard display"""
    INITIALIZING = "Initializing"
    LOADING_DATA = "Loading Data"
    SETTING_UP_ENV = "Setting Up Environment"
    LOADING_MODEL = "Loading Model"
    COLLECTING_ROLLOUT = "Collecting Rollout"
    UPDATING_POLICY = "Updating Policy"
    EVALUATING = "Evaluating"
    SAVING_MODEL = "Saving Model"
    COMPLETED = "Training Completed"
    ERROR = "Error Occurred"


@dataclass
class DashboardState:
    """Enhanced dashboard state with improved training and reward tracking"""
    # Environment state
    step: int = 0
    episode_step: int = 0
    timestamp: str = "N/A"
    market_time: str = "N/A"
    symbol: str = "N/A"
    episode_reward: float = 0.0
    step_reward: float = 0.0
    total_reward: float = 0.0

    # FIXED: Simplified reward tracking - focus on reward purpose and impact
    reward_breakdown: Dict[str, float] = field(default_factory=dict)
    reward_impact_tracking: Dict[str, float] = field(default_factory=lambda: {
        "profit_driving": 0.0,  # Rewards that drive profit
        "cost_penalties": 0.0,  # Transaction and fee penalties
        "risk_penalties": 0.0,  # Risk management penalties
        "behavior_shaping": 0.0  # Rewards that shape behavior
    })

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

    # Action info
    last_action_type: str = "N/A"
    last_action_size: str = "N/A"
    last_action_invalid: bool = False
    invalid_actions_count: int = 0

    # FIXED: Increased recent history from 3 to 5
    recent_actions: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=5))
    recent_fills: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=5))

    # Training state
    episode_number: int = 0
    total_episodes: int = 0
    total_steps: int = 0
    update_count: int = 0
    current_stage: TrainingStage = TrainingStage.INITIALIZING
    stage_progress: float = 0.0
    substage_progress: float = 0.0
    stage_details: str = ""

    # FIXED: Comprehensive training metrics (will span 2 columns)
    learning_rate: float = 0.0
    batch_size: int = 0
    buffer_size: int = 0
    rollout_steps: int = 0
    collected_steps: int = 0
    mean_episode_reward: float = 0.0
    mean_episode_length: float = 0.0
    actor_loss: float = 0.0
    critic_loss: float = 0.0
    entropy: float = 0.0
    gradient_norm: float = 0.0
    value_loss: float = 0.0
    policy_entropy: float = 0.0

    # FIXED: Additional training metrics for expanded panel
    episodes_per_update: int = 0
    steps_per_second: float = 0.0
    time_per_update: float = 0.0
    total_training_time: float = 0.0
    estimated_time_remaining: float = 0.0
    policy_kl_divergence: float = 0.0
    value_function_explained_variance: float = 0.0
    clipfrac: float = 0.0  # Fraction of training data that was clipped
    approx_kl: float = 0.0

    # Status tracking
    is_training: bool = False
    is_evaluating: bool = False
    is_terminated: bool = False
    is_truncated: bool = False
    current_termination_reason: str = ""

    # Performance metrics
    initial_capital: float = 25000.0
    session_pnl: float = 0.0
    session_pnl_pct: float = 0.0

    # Update tracking
    last_update_time: float = 0.0
    update_count_dashboard: int = 0
    state_version: int = 0


class TradingDashboard:
    """
    FIXED: Enhanced trading dashboard with improved layout and comprehensive training metrics.
    """

    def __init__(self, log_height: int = 20):
        """
        Initialize the enhanced trading dashboard.

        Args:
            log_height: Height of the log section at bottom (in lines)
        """
        self.log_height = log_height
        self.state = DashboardState()

        # FIXED: Optimized thread safety and update throttling
        self._state_lock = threading.RLock()
        self._update_throttle = 0.3  # Update every 0.3 seconds
        self._last_update_time = 0.0
        self._update_in_progress = False

        # State management for stability
        self._stable_state = DashboardState()
        self._last_valid_update = 0.0

        # Create main console for dashboard
        self.console = Console()

        # Create separate console for logs with Rich features
        self.log_console = Console(
            width=120,
            height=log_height - 2,
            force_terminal=True,
            color_system="auto"
        )

        # Create layout
        self.layout = self._create_layout()

        # Live display
        self.live: Optional[Live] = None
        self._running = False

        # Enhanced log capture that preserves Rich formatting
        self.log_buffer = deque(maxlen=100)
        self._setup_dashboard_logging()

    def _setup_dashboard_logging(self):
        """Setup logging to capture Rich-formatted output"""

        class DashboardRichHandler(RichHandler):
            def __init__(self, dashboard_instance):
                super().__init__(
                    console=dashboard_instance.log_console,
                    show_time=True,
                    show_path=False,
                    rich_tracebacks=True,
                    tracebacks_show_locals=False,
                    markup=True
                )
                self.dashboard = dashboard_instance

            def emit(self, record):
                try:
                    with io.StringIO() as buffer:
                        temp_console = Console(file=buffer, width=120, force_terminal=False)
                        temp_handler = RichHandler(
                            console=temp_console,
                            show_time=True,
                            show_path=False,
                            markup=True
                        )
                        temp_handler.setFormatter(self.formatter)
                        temp_handler.emit(record)

                        rich_text = buffer.getvalue().strip()

                        if rich_text:
                            self.dashboard.log_buffer.append({
                                'text': rich_text,
                                'timestamp': time.time(),
                                'level': record.levelname
                            })

                except Exception:
                    pass

        self.dashboard_handler = DashboardRichHandler(self)
        self.dashboard_handler.setLevel(logging.INFO)

    def _create_layout(self) -> Layout:
        """FIXED: Create improved 3-column layout with expanded training metrics"""
        layout = Layout()

        # Split main layout: Dashboard (top) and Logs (bottom)
        layout.split_column(
            Layout(name="dashboard", ratio=3),
            Layout(name="logs", size=self.log_height)
        )

        # Compact header and footer
        dashboard_layout = layout["dashboard"]
        dashboard_layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=2)
        )

        # FIXED: 3-column layout with expanded training metrics
        dashboard_layout["body"].split_row(
            Layout(name="left_column", ratio=1),  # Market & Position & Portfolio
            Layout(name="training_column", ratio=2),  # EXPANDED: Training metrics (2x width)
            Layout(name="right_column", ratio=1)  # Reward & Actions & Status
        )

        # Left column (Market + Position + Portfolio)
        dashboard_layout["left_column"].split_column(
            Layout(name="market", ratio=1),
            Layout(name="position", ratio=1),
            Layout(name="portfolio", ratio=1)
        )

        # FIXED: Expanded training column with comprehensive metrics
        dashboard_layout["training_column"].split_column(
            Layout(name="training_stage", ratio=1),
            Layout(name="training_metrics_main", ratio=1),  # Main training metrics
            Layout(name="training_metrics_advanced", ratio=1)  # Advanced training metrics
        )

        # Right column (Reward + Actions + Status)
        dashboard_layout["right_column"].split_column(
            Layout(name="reward_system", ratio=1),  # FIXED: Focused reward system
            Layout(name="actions", ratio=1),
            Layout(name="episode_status", ratio=1)  # FIXED: Episode status including termination reasons
        )

        return layout

    def start(self):
        """Start the full-screen dashboard"""
        if self._running:
            return

        try:
            root_logger = logging.getLogger()
            self.original_handlers = root_logger.handlers.copy()
            root_logger.handlers.clear()
            root_logger.addHandler(self.dashboard_handler)

            self._update_layout()

            self.live = Live(
                self.layout,
                console=self.console,
                refresh_per_second=3,
                screen=False,
                auto_refresh=True,
                transient=False,
                redirect_stdout=False,
                redirect_stderr=False
            )

            self.live.start()
            self._running = True

            logging.info("Enhanced Trading Dashboard Started")

        except Exception as e:
            logging.error(f"Failed to start dashboard: {e}")
            self._running = False

    def stop(self):
        """Stop the dashboard"""
        if not self._running:
            return

        self._running = False

        try:
            root_logger = logging.getLogger()
            root_logger.handlers.clear()

            if hasattr(self, 'original_handlers'):
                for handler in self.original_handlers:
                    root_logger.addHandler(handler)

            if self.live:
                logging.info("Stopping enhanced trading dashboard...")
                self.live.stop()

        except Exception as e:
            print(f"Error stopping dashboard: {e}")

    def set_training_stage(self, stage: TrainingStage, overall_progress: float = None, details: str = "",
                           substage_progress: float = None):
        """Set the current training stage with proper progress tracking"""
        with self._state_lock:
            self.state.current_stage = stage
            if overall_progress is not None:
                self.state.stage_progress = max(0.0, min(1.0, overall_progress))
            if substage_progress is not None:
                self.state.substage_progress = max(0.0, min(1.0, substage_progress))
            if details:
                self.state.stage_details = str(details)
            self.state.state_version += 1

        if self._running:
            self._safe_throttled_update()

    def update_training_metrics(self, metrics: Dict[str, Any]):
        """Update training metrics with enhanced tracking"""
        if not metrics:
            return

        with self._state_lock:
            for key, value in metrics.items():
                if hasattr(self.state, key) and value is not None:
                    if isinstance(value, (int, float)) and not (isinstance(value, float) and (value != value)):
                        setattr(self.state, key, value)

            # FIXED: Process reward breakdown for better visualization
            if 'reward_components' in metrics and isinstance(metrics['reward_components'], dict):
                self._process_reward_breakdown(metrics['reward_components'])

            self.state.state_version += 1

        if self._running:
            self._safe_throttled_update()

    def _process_reward_breakdown(self, reward_components: Dict[str, float]):
        """FIXED: Process reward components into meaningful categories"""
        self.state.reward_breakdown = reward_components.copy()

        # Categorize rewards by their purpose and impact
        profit_driving = 0.0
        cost_penalties = 0.0
        risk_penalties = 0.0
        behavior_shaping = 0.0

        for component, value in reward_components.items():
            component_lower = component.lower()

            if any(x in component_lower for x in ['equity', 'pnl', 'profit', 'realized']):
                profit_driving += value
            elif any(x in component_lower for x in ['commission', 'fee', 'transaction', 'slippage']):
                cost_penalties += value
            elif any(x in component_lower for x in ['drawdown', 'risk', 'leverage', 'bankruptcy']):
                risk_penalties += value
            else:
                behavior_shaping += value

        self.state.reward_impact_tracking = {
            "profit_driving": profit_driving,
            "cost_penalties": cost_penalties,
            "risk_penalties": risk_penalties,
            "behavior_shaping": behavior_shaping
        }

    def update_state(self, info_dict: Dict[str, Any], market_state: Optional[Dict[str, Any]] = None):
        """Enhanced state update with improved tracking"""
        if not self._running or not self.live:
            return

        if not info_dict and not market_state:
            return

        try:
            with self._state_lock:
                if self._update_in_progress:
                    return

                self._update_in_progress = True
                self._update_state_data_enhanced(info_dict, market_state)
                self._update_in_progress = False

            self._safe_throttled_update()

        except Exception as e:
            self._update_in_progress = False
            logging.error(f"Error updating dashboard state: {e}")

    def _update_state_data_enhanced(self, info_dict: Dict[str, Any], market_state: Optional[Dict[str, Any]] = None):
        """Enhanced state update with comprehensive tracking"""

        # FIXED: Properly update global step counter
        if 'global_step_counter' in info_dict and isinstance(info_dict['global_step_counter'], (int, float)):
            global_step = int(info_dict.get('global_step_counter'))
            if global_step >= 0:
                self.state.step = global_step
                self.state.total_steps = global_step  # FIXED: Ensure both are updated

        # Update episode step separately
        if 'step' in info_dict and isinstance(info_dict['step'], (int, float)):
            new_episode_step = int(info_dict['step'])
            if new_episode_step >= 0:
                self.state.episode_step = new_episode_step

        # Update episode number properly
        if 'episode_number' in info_dict and isinstance(info_dict['episode_number'], (int, float)):
            episode_num = int(info_dict['episode_number'])
            if episode_num > 0:
                self.state.episode_number = episode_num

        # FIXED: Update reward tracking
        if 'reward_step' in info_dict and isinstance(info_dict['reward_step'], (int, float)):
            step_reward = float(info_dict['reward_step'])
            if not (step_reward != step_reward):  # Check for NaN
                self.state.step_reward = step_reward

        # Update timestamp and calculate market time
        if 'timestamp_iso' in info_dict and isinstance(info_dict['timestamp_iso'], str) and info_dict['timestamp_iso']:
            self.state.timestamp = info_dict['timestamp_iso']
            try:
                dt = datetime.fromisoformat(info_dict['timestamp_iso'].replace('Z', '+00:00'))
                from zoneinfo import ZoneInfo
                et_time = dt.astimezone(ZoneInfo('America/New_York'))
                self.state.market_time = et_time.strftime("%H:%M:%S ET")
            except Exception:
                self.state.market_time = "N/A"

        # Update rewards with validation
        if 'episode_cumulative_reward' in info_dict and isinstance(info_dict['episode_cumulative_reward'],
                                                                   (int, float)):
            reward = float(info_dict['episode_cumulative_reward'])
            if not (reward != reward):
                self.state.episode_reward = reward

        # Portfolio metrics with validation
        portfolio_updates = {
            'portfolio_equity': 'total_equity',
            'portfolio_cash': 'cash',
            'portfolio_unrealized_pnl': 'unrealized_pnl',
            'portfolio_realized_pnl_session_net': 'realized_pnl'
        }

        for info_key, state_key in portfolio_updates.items():
            if info_key in info_dict and isinstance(info_dict[info_key], (int, float)):
                value = float(info_dict[info_key])
                if not (value != value):
                    setattr(self.state, state_key, value)

        # Calculate session PnL only if we have valid values
        if self.state.total_equity > 0 and self.state.initial_capital > 0:
            self.state.session_pnl = self.state.total_equity - self.state.initial_capital
            self.state.session_pnl_pct = (self.state.session_pnl / self.state.initial_capital) * 100

        # Position data with validation
        if self.state.symbol != "N/A":
            symbol = self.state.symbol
            position_updates = {
                f'position_{symbol}_qty': 'position_qty',
                f'position_{symbol}_side': 'position_side',
                f'position_{symbol}_avg_entry': 'position_avg_entry'
            }

            for info_key, state_key in position_updates.items():
                if info_key in info_dict and info_dict[info_key] is not None:
                    value = info_dict[info_key]
                    if state_key == 'position_side':
                        if hasattr(value, 'value'):
                            setattr(self.state, state_key, str(value.value))
                        else:
                            setattr(self.state, state_key, str(value))
                    elif isinstance(value, (int, float)):
                        float_val = float(value)
                        if not (float_val != float_val):
                            setattr(self.state, state_key, float_val)

        # Action info with validation
        if 'action_decoded' in info_dict and isinstance(info_dict['action_decoded'], dict):
            action_decoded = info_dict['action_decoded']

            action_type = action_decoded.get('type')
            if action_type:
                self.state.last_action_type = action_type.name if hasattr(action_type, 'name') else str(action_type)

            size_enum = action_decoded.get('size_enum')
            if size_enum:
                self.state.last_action_size = size_enum.name if hasattr(size_enum, 'name') else str(size_enum)

            if 'invalid_reason' in action_decoded:
                self.state.last_action_invalid = bool(action_decoded['invalid_reason'])

            # FIXED: Add to recent actions (now maxlen=5)
            new_action = {
                'step': self.state.episode_step,
                'global_step': self.state.step,
                'type': self.state.last_action_type,
                'size': self.state.last_action_size,
                'invalid': self.state.last_action_invalid,
                'reward': self.state.step_reward,
                'timestamp': time.time()
            }

            should_add = True
            if self.state.recent_actions:
                last_action = self.state.recent_actions[-1]
                if (last_action['step'] == new_action['step'] and
                        last_action['type'] == new_action['type'] and
                        last_action['size'] == new_action['size']):
                    should_add = False

            if should_add:
                self.state.recent_actions.append(new_action)

        # Market data with validation
        if market_state and isinstance(market_state, dict):
            market_updates = {
                'current_price': 'current_price',
                'best_bid_price': 'bid_price',
                'best_ask_price': 'ask_price',
                'market_session': 'market_session'
            }

            for market_key, state_key in market_updates.items():
                if market_key in market_state and market_state[market_key] is not None:
                    value = market_state[market_key]
                    if state_key == 'market_session':
                        setattr(self.state, state_key, str(value))
                    elif isinstance(value, (int, float)):
                        float_val = float(value)
                        if not (float_val != float_val) and float_val >= 0:
                            setattr(self.state, state_key, float_val)

        # FIXED: Fills with proper validation (now maxlen=5)
        if 'fills_step' in info_dict and isinstance(info_dict['fills_step'], list):
            for fill in info_dict['fills_step']:
                if isinstance(fill, dict):
                    side = fill.get('order_side', 'N/A')
                    if hasattr(side, 'value'):
                        side_str = side.value
                    else:
                        side_str = str(side)

                    new_fill = {
                        'step': self.state.episode_step,
                        'global_step': self.state.step,
                        'side': side_str,
                        'quantity': float(fill.get('executed_quantity', 0.0)),
                        'price': float(fill.get('executed_price', 0.0)),
                        'timestamp': time.time()
                    }

                    if (new_fill['quantity'] > 0 and new_fill['price'] > 0):
                        should_add = True
                        if self.state.recent_fills:
                            last_fill = self.state.recent_fills[-1]
                            if (abs(last_fill['timestamp'] - new_fill['timestamp']) < 1.0 and
                                    last_fill['side'] == new_fill['side'] and
                                    abs(last_fill['quantity'] - new_fill['quantity']) < 0.01 and
                                    abs(last_fill['price'] - new_fill['price']) < 0.0001):
                                should_add = False

                        if should_add:
                            self.state.recent_fills.append(new_fill)

        # FIXED: Status updates including termination reasons
        if 'termination_reason' in info_dict:
            termination_reason = info_dict['termination_reason']
            self.state.is_terminated = termination_reason is not None
            if termination_reason:
                self.state.current_termination_reason = str(termination_reason)
            else:
                self.state.current_termination_reason = ""

        if 'TimeLimit.truncated' in info_dict:
            self.state.is_truncated = bool(info_dict['TimeLimit.truncated'])

        # Track invalid actions
        if 'invalid_actions_total_episode' in info_dict:
            self.state.invalid_actions_count = int(info_dict['invalid_actions_total_episode'])

        self.state.state_version += 1

    def _safe_throttled_update(self):
        """Safely update layout with enhanced throttling and error handling"""
        if not self._running or not self.live:
            return

        current_time = time.time()
        if current_time - self._last_update_time >= self._update_throttle:
            try:
                self._update_layout()
                self._last_update_time = current_time
                self._last_valid_update = current_time
            except Exception as e:
                logging.error(f"Error in safe throttled update: {e}")

    def _update_layout(self):
        """Update all layout components with enhanced metrics"""
        try:
            with self._state_lock:
                self.layout["header"].update(self._create_header())
                self.layout["market"].update(self._create_market_panel())
                self.layout["position"].update(self._create_position_panel())
                self.layout["portfolio"].update(self._create_portfolio_panel())
                self.layout["training_stage"].update(self._create_training_stage_panel())
                self.layout["training_metrics_main"].update(self._create_training_metrics_main_panel())
                self.layout["training_metrics_advanced"].update(self._create_training_metrics_advanced_panel())
                self.layout["reward_system"].update(self._create_reward_system_panel())
                self.layout["actions"].update(self._create_actions_panel())
                self.layout["episode_status"].update(self._create_episode_status_panel())
                self.layout["footer"].update(self._create_footer())
                self.layout["logs"].update(self._create_enhanced_logs_panel())
        except Exception as e:
            logging.error(f"Error updating layout components: {e}")

    def _create_enhanced_logs_panel(self) -> Panel:
        """Create enhanced logs panel with Rich formatting and colors"""
        recent_logs = list(self.log_buffer)[-(self.log_height - 2):]

        if recent_logs:
            log_display = "\n".join([entry['text'] for entry in recent_logs])
        else:
            log_display = "[dim]Waiting for log messages...[/dim]"

        return Panel(
            Text.from_markup(log_display),
            title="[bold]System Logs",
            border_style="cyan",
            height=self.log_height
        )

    def _create_header(self) -> Panel:
        """FIXED: Create header with proper global step counter display"""
        # Training status
        if self.state.is_training:
            status = f"[bold green]TRAINING[/bold green]"
        elif self.state.is_evaluating:
            status = f"[bold blue]EVALUATING[/bold blue]"
        elif self.state.is_terminated:
            status = "[bold red]TERMINATED[/bold red]"
        else:
            status = "[yellow]IDLE[/yellow]"

        # FIXED: Show global step counter prominently
        header_text = (f"[bold cyan]FX-AI Trading Dashboard[/bold cyan] | "
                       f"{self.state.symbol} | Episode {self.state.episode_number} | "
                       f"Step {self.state.episode_step} | [bold yellow]Global: {self.state.total_steps}[/bold yellow] | "
                       f"{status} - {self.state.current_stage.value} | "
                       f"Market: {self.state.market_time} | Equity: [bold]${self.state.total_equity:.2f}[/bold]")

        return Panel(Align.center(Text.from_markup(header_text)), style="bright_blue", box=box.HEAVY)

    def _create_training_stage_panel(self) -> Panel:
        """Create training stage panel with properly aligned progress bars"""

        def create_progress_bar(progress: float, style: str = "green") -> Text:
            progress = max(0.0, min(1.0, progress))
            filled_chars = int(progress * 25)
            empty_chars = 25 - filled_chars

            bar_text = Text()
            if filled_chars > 0:
                bar_text.append("█" * filled_chars, style=style)
            if empty_chars > 0:
                bar_text.append("░" * empty_chars, style="dim white")

            return bar_text

        stage_text = Text()
        stage_text.append("Stage: ", style="white")
        stage_text.append(f"{self.state.current_stage.value}", style="bold yellow")

        overall_progress_line = Text()
        overall_progress_line.append(f"Overall: {self.state.stage_progress:6.1%} ", style="white")
        overall_progress_line.append_text(create_progress_bar(self.state.stage_progress, "green"))

        substage_progress_line = Text()
        substage_progress_line.append(f"Current: {self.state.substage_progress:6.1%} ", style="white")
        substage_progress_line.append_text(create_progress_bar(self.state.substage_progress, "yellow"))

        details_text = Text(self.state.stage_details or "Ready", style="cyan")

        content = Group(stage_text, overall_progress_line, substage_progress_line, details_text)
        return Panel(content, title="[bold]Training Progress", border_style="cyan")

    def _create_training_metrics_main_panel(self) -> Panel:
        """FIXED: Create main training metrics panel with key metrics"""
        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        table.add_column("Metric", width=16)
        table.add_column("Value", width=12)

        # Key training metrics
        table.add_row("Update", Text(f"{self.state.update_count}", style="bold cyan"))
        table.add_row("Learning Rate", Text(f"{self.state.learning_rate:.2e}", style="white"))
        table.add_row("Global Steps", Text(f"{self.state.total_steps}", style="bold yellow"))
        table.add_row("Episode", Text(f"{self.state.episode_number}", style="white"))

        if self.state.collected_steps > 0 and self.state.rollout_steps > 0:
            collection_pct = (self.state.collected_steps / self.state.rollout_steps) * 100
            table.add_row("Collection",
                          Text(f"{self.state.collected_steps}/{self.state.rollout_steps} ({collection_pct:.1f}%)",
                               style="yellow"))

        table.add_row("Mean Reward", Text(f"{self.state.mean_episode_reward:.4f}",
                                          style="green" if self.state.mean_episode_reward > 0 else "red"))

        # Performance metrics
        if self.state.steps_per_second > 0:
            table.add_row("Steps/Sec", Text(f"{self.state.steps_per_second:.1f}", style="cyan"))

        table.add_row("Episode Length", Text(f"{self.state.mean_episode_length:.1f}", style="white"))

        return Panel(table, title="[bold]Training Metrics", border_style="green")

    def _create_training_metrics_advanced_panel(self) -> Panel:
        """FIXED: Create advanced training metrics panel"""
        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        table.add_column("Metric", width=16)
        table.add_column("Value", width=12)

        # Loss metrics
        if self.state.actor_loss != 0:
            table.add_row("Actor Loss", Text(f"{self.state.actor_loss:.4f}", style="red"))
        if self.state.critic_loss != 0:
            table.add_row("Critic Loss", Text(f"{self.state.critic_loss:.4f}", style="red"))
        if self.state.entropy != 0:
            table.add_row("Entropy", Text(f"{self.state.entropy:.4f}", style="magenta"))

        # Advanced metrics
        if self.state.clipfrac != 0:
            table.add_row("Clip Fraction", Text(f"{self.state.clipfrac:.3f}", style="orange"))
        if self.state.approx_kl != 0:
            table.add_row("Approx KL", Text(f"{self.state.approx_kl:.4f}", style="purple"))
        if self.state.value_function_explained_variance != 0:
            table.add_row("Value Expl Var", Text(f"{self.state.value_function_explained_variance:.3f}", style="blue"))

        # Performance metrics
        table.add_row("Episodes/Update", Text(f"{self.state.episodes_per_update}", style="white"))
        if self.state.time_per_update > 0:
            table.add_row("Time/Update", Text(f"{self.state.time_per_update:.1f}s", style="white"))

        # Batch info
        table.add_row("Batch Size", Text(f"{self.state.batch_size}", style="cyan"))
        table.add_row("Buffer Size", Text(f"{self.state.buffer_size}", style="cyan"))

        return Panel(table, title="[bold]Advanced Metrics", border_style="purple")

    def _create_reward_system_panel(self) -> Panel:
        """FIXED: Create focused reward system panel showing reward purpose and impact"""
        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        table.add_column("Component", width=14)
        table.add_column("Value", width=8)

        # Current step reward
        step_reward_style = "green" if self.state.step_reward > 0 else "red" if self.state.step_reward < 0 else "white"
        table.add_row("Step Reward", Text(f"{self.state.step_reward:.4f}", style=step_reward_style))

        # Episode total
        episode_reward_style = "green" if self.state.episode_reward > 0 else "red" if self.state.episode_reward < 0 else "white"
        table.add_row("Episode Total", Text(f"{self.state.episode_reward:.4f}", style=episode_reward_style))

        # Reward impact categories
        table.add_row("", Text("", style="white"))  # Separator
        table.add_row("[bold]Impact Type", Text("[bold]Value", style="white"))

        for impact_type, value in self.state.reward_impact_tracking.items():
            impact_name = impact_type.replace('_', ' ').title()[:12]
            value_style = "green" if value > 0 else "red" if value < 0 else "white"
            table.add_row(impact_name, Text(f"{value:.3f}", style=value_style))

        # Show most significant reward components
        if self.state.reward_breakdown:
            table.add_row("", Text("", style="white"))  # Separator
            sorted_components = sorted(self.state.reward_breakdown.items(), key=lambda x: abs(x[1]), reverse=True)
            for i, (component, value) in enumerate(sorted_components[:3]):  # Top 3 components
                component_name = component.replace('_', ' ').title()[:12]
                value_style = "green" if value > 0 else "red" if value < 0 else "white"
                table.add_row(component_name, Text(f"{value:.3f}", style=value_style))

        return Panel(table, title="[bold]Reward System", border_style="yellow")

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

        table.add_row("Session", Text(self.state.market_session, style="cyan"))

        return Panel(table, title="[bold]Market Data", border_style="yellow")

    def _create_position_panel(self) -> Panel:
        """Create position panel"""
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

        if self.state.current_price and self.state.position_avg_entry > 0 and self.state.position_qty != 0:
            price_diff = self.state.current_price - self.state.position_avg_entry
            if self.state.position_side == "SHORT":
                price_diff = -price_diff
            price_diff_pct = (price_diff / self.state.position_avg_entry) * 100

            diff_style = "green" if price_diff > 0 else "red" if price_diff < 0 else "white"
            diff_text = f"${price_diff:.4f} ({price_diff_pct:+.2f}%)"
            table.add_row("P&L vs Entry", Text(diff_text, style=diff_style))

        return Panel(table, title="[bold]Position", border_style="blue")

    def _create_portfolio_panel(self) -> Panel:
        """Create portfolio panel"""
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
        """FIXED: Create recent actions panel (now showing 5 actions)"""
        table = Table(box=box.SIMPLE, show_edge=False, padding=(0, 1))
        table.add_column("Step", width=6)
        table.add_column("Action", width=8)
        table.add_column("Size", width=8)
        table.add_column("Reward", width=8)

        recent_actions = list(self.state.recent_actions)
        for action in recent_actions:
            step = str(action['step'])
            action_type = action['type']
            size = action['size'].replace('SIZE_', '') if 'SIZE_' in action['size'] else action['size']

            reward_val = action.get('reward', 0.0)
            reward_style = "green" if reward_val > 0 else "red" if reward_val < 0 else "white"
            reward_text = Text(f"{reward_val:.3f}", style=reward_style)

            if action['invalid']:
                status = Text("INVALID", style="bold red")
                table.add_row(step, action_type, size, "")
            else:
                table.add_row(step, action_type, size, reward_text)

        # Fill remaining rows up to 5
        while len(table.rows) < 5:
            table.add_row("", "", "", "")

        return Panel(table, title="[bold]Recent Actions (5)", border_style="magenta")

    def _create_episode_status_panel(self) -> Panel:
        """FIXED: Create episode status panel with termination/truncation reasons"""
        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))

        # Episode status
        if self.state.is_terminated:
            status_text = "TERMINATED"
            status_style = "bold red"
        elif self.state.is_truncated:
            status_text = "TRUNCATED"
            status_style = "bold yellow"
        else:
            status_text = "RUNNING"
            status_style = "bold green"

        table.add_row("Status", Text(status_text, style=status_style))

        # FIXED: Show termination/truncation reason
        if self.state.current_termination_reason:
            reason_text = self.state.current_termination_reason.replace('_', ' ').title()
            table.add_row("Reason", Text(reason_text, style="yellow"))

        # Episode metrics
        table.add_row("Episode Step", Text(f"{self.state.episode_step}", style="white"))
        table.add_row("Invalid Actions", Text(f"{self.state.invalid_actions_count}",
                                              style="red" if self.state.invalid_actions_count > 0 else "white"))

        # Recent fills (now showing 5)
        table.add_row("", Text("", style="white"))  # Separator
        table.add_row("[bold]Recent Fills", Text("", style="white"))

        recent_fills = list(self.state.recent_fills)
        for i, fill in enumerate(recent_fills[-3:]):  # Show last 3 fills in the status panel
            side_style = "green" if fill['side'] == "BUY" else "red"
            fill_text = f"{fill['side']} {fill['quantity']:.1f}@${fill['price']:.2f}"
            table.add_row(f"Fill {i + 1}", Text(fill_text, style=side_style))

        return Panel(table, title="[bold]Episode Status", border_style="cyan")

    def _create_footer(self) -> Panel:
        """FIXED: Create footer with comprehensive status information"""
        # Dynamic status based on current state
        if self.state.is_terminated and self.state.current_termination_reason:
            status_text = f"[bold red]TERMINATED: {self.state.current_termination_reason.replace('_', ' ')}[/bold red]"
        elif self.state.is_truncated:
            status_text = "[bold yellow]TRUNCATED: Max steps reached[/bold yellow]"
        elif self.state.current_stage == TrainingStage.ERROR:
            status_text = "[bold red]ERROR OCCURRED - Check logs below[/bold red]"
        else:
            status_text = (f"[green]FX-AI Active[/green] | "
                           f"Global Step: [bold yellow]{self.state.total_steps}[/bold yellow] | "
                           f"Update: {self.state.update_count} | "
                           f"LR: {self.state.learning_rate:.2e} | "
                           f"Session P&L: [bold]${self.state.session_pnl:.2f}[/bold]")

        return Panel(Align.center(Text.from_markup(status_text)), style="bright_white")

    def set_symbol(self, symbol: str):
        """Set the trading symbol"""
        with self._state_lock:
            self.state.symbol = symbol

    def set_initial_capital(self, capital: float):
        """Set initial capital"""
        with self._state_lock:
            self.state.initial_capital = capital

    def set_training_info(self, episode_num: int = 0, total_episodes: int = 0,
                          total_steps: int = 0, update_count: int = 0,
                          buffer_size: int = 0, is_training: bool = True,
                          is_evaluating: bool = False, learning_rate: float = 0.0):
        """Set training information with thread safety and validation"""
        with self._state_lock:
            if episode_num > 0:
                self.state.episode_number = episode_num
            if total_steps >= 0:
                self.state.total_steps = total_steps
                self.state.step = total_steps  # FIXED: Keep both in sync
            if update_count >= 0:
                self.state.update_count = update_count

            self.state.total_episodes = max(0, total_episodes)
            self.state.buffer_size = max(0, buffer_size)
            self.state.is_training = is_training
            self.state.is_evaluating = is_evaluating
            if learning_rate > 0:
                self.state.learning_rate = learning_rate

            self.state.state_version += 1

        if self._running:
            self._safe_throttled_update()