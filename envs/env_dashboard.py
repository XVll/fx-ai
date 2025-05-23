# envs/env_dashboard.py - COMPREHENSIVE: Enhanced dashboard with full specification implementation

import logging
import threading
import time
from collections import deque, defaultdict
from typing import Any, Dict, Optional, Deque, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import io
import sys
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import statistics
import numpy as np

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
class EpisodeRecord:
    """Record of completed episode"""
    episode_number: int
    status: str  # "COMPLETED", "TERMINATED", "TRUNCATED"
    termination_reason: str
    episode_reward: float
    episode_length: int
    session_pnl: float
    final_equity: float
    timestamp: float


@dataclass
class TradeRecord:
    """Record of completed trade"""
    timestamp: str
    side: str  # "BUY", "SELL"
    quantity: float
    symbol: str
    entry_price: float
    exit_price: Optional[float]
    pnl: Optional[float]
    step: int


@dataclass
class ActionBiasRecord:
    """Record for action bias analysis"""
    action_type: str
    count: int
    total_reward: float
    positive_rewards: int
    step_count: int


@dataclass
class MetricHistory:
    """Track metric history for sparklines"""
    values: Deque[float] = field(default_factory=lambda: deque(maxlen=20))

    def add(self, value: float):
        self.values.append(value)

    def get_sparkline(self) -> str:
        """Generate simple text sparkline"""
        if len(self.values) < 2:
            return ""

        min_val = min(self.values)
        max_val = max(self.values)

        if max_val == min_val:
            return "▄" * min(len(self.values), 8)

        chars = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
        sparkline = ""

        for val in list(self.values)[-8:]:  # Last 8 values
            normalized = (val - min_val) / (max_val - min_val)
            char_idx = min(int(normalized * len(chars)), len(chars) - 1)
            sparkline += chars[char_idx]

        return sparkline


@dataclass
class DashboardState:
    """Comprehensive dashboard state with all specified metrics"""

    # Basic episode/step info
    step: int = 0
    episode_step: int = 0
    global_step_counter: int = 0
    episode_number: int = 0
    total_episodes: int = 0

    # Timing
    timestamp: str = "N/A"
    market_time: str = "N/A"
    session_start_time: float = 0.0
    session_elapsed_time: str = "00:00:00"

    # Model info
    model_name: str = "PPO_Transformer_v1.0"
    symbol: str = "N/A"

    # Market data
    current_price: Optional[float] = None
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None
    spread: Optional[float] = None
    price_change_direction: str = ""  # "↑", "↓", ""
    market_session: str = "UNKNOWN"

    # Position data
    position_qty: float = 0.0
    position_side: str = "FLAT"
    position_avg_entry: float = 0.0
    position_market_value: float = 0.0
    position_unrealized_pnl: float = 0.0
    position_pnl_vs_entry_usd: float = 0.0
    position_pnl_vs_entry_pct: float = 0.0

    # Portfolio metrics
    total_equity: float = 0.0
    cash: float = 0.0
    initial_capital: float = 25000.0
    session_pnl: float = 0.0
    session_pnl_pct: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    total_trades: int = 0

    # Reward system
    episode_reward: float = 0.0
    step_reward: float = 0.0
    last_step_reward: float = 0.0
    reward_breakdown: Dict[str, float] = field(default_factory=dict)

    # Action tracking
    last_action_type: str = "N/A"
    last_action_size: str = "N/A"
    last_action_invalid: bool = False
    invalid_actions_count: int = 0

    # Recent history (5 items each)
    recent_actions: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=5))
    recent_trades: Deque[TradeRecord] = field(default_factory=lambda: deque(maxlen=5))
    episode_history: Deque[EpisodeRecord] = field(default_factory=lambda: deque(maxlen=3))

    # Training state
    current_stage: TrainingStage = TrainingStage.INITIALIZING
    stage_progress: float = 0.0
    substage_progress: float = 0.0
    stage_details: str = ""
    training_mode: str = "Idle"  # "Training", "Evaluation", "Idle"

    # Training metrics
    update_count: int = 0
    learning_rate: float = 0.0
    batch_size: int = 0
    buffer_size: int = 0
    rollout_steps: int = 0
    collected_steps: int = 0

    # PPO Core Metrics with history
    mean_episode_reward: float = 0.0
    mean_episode_length: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    total_loss: float = 0.0
    entropy: float = 0.0
    clip_fraction: float = 0.0
    approx_kl: float = 0.0
    value_explained_variance: float = 0.0

    # Performance metrics
    steps_per_second: float = 0.0
    time_per_update: float = 0.0
    avg_time_per_episode: float = 0.0

    # Action bias analysis
    action_bias_records: Dict[str, ActionBiasRecord] = field(default_factory=dict)

    # Metric histories for sparklines
    metric_histories: Dict[str, MetricHistory] = field(default_factory=lambda: {
        'mean_reward': MetricHistory(),
        'policy_loss': MetricHistory(),
        'value_loss': MetricHistory(),
        'total_loss': MetricHistory(),
        'entropy': MetricHistory(),
        'clip_fraction': MetricHistory(),
        'approx_kl': MetricHistory(),
        'value_explained_variance': MetricHistory()
    })

    # Status flags
    is_training: bool = False
    is_evaluating: bool = False
    is_terminated: bool = False
    is_truncated: bool = False

    # Update tracking
    last_update_time: float = 0.0
    state_version: int = 0


class TradingDashboard:
    """
    COMPREHENSIVE: Enhanced trading dashboard with full specification implementation
    """

    def __init__(self, log_height: int = 15):
        """
        Initialize the comprehensive trading dashboard.

        Args:
            log_height: Height of the log section at bottom (in lines)
        """
        self.log_height = log_height
        self.state = DashboardState()

        # Thread safety and update throttling
        self._state_lock = threading.RLock()
        self._update_throttle = 0.25  # Update every 0.25 seconds
        self._last_update_time = 0.0
        self._update_in_progress = False

        # Performance tracking
        self._equity_history = deque(maxlen=100)
        self._episode_times = deque(maxlen=10)
        self._last_price = None

        # Create consoles
        self.console = Console()
        self.log_console = Console(
            width=140,
            height=log_height - 2,
            force_terminal=True,
            color_system="auto"
        )

        # Create layout
        self.layout = self._create_comprehensive_layout()

        # Live display
        self.live: Optional[Live] = None
        self._running = False

        # Enhanced log capture
        self.log_buffer = deque(maxlen=150)
        self._setup_dashboard_logging()

    def _create_comprehensive_layout(self) -> Layout:
        """Create the comprehensive 3-column layout as specified"""
        layout = Layout()

        # Main split: Header / Body / Footer / Logs
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=2),
            Layout(name="logs", size=self.log_height)
        )

        # Body: 3 main column groups
        layout["body"].split_row(
            Layout(name="column_group_1", ratio=1),  # Market, Position, Portfolio
            Layout(name="column_group_2", ratio=1),  # Actions, Trades, Episode
            Layout(name="column_group_3", ratio=1)  # Training, PPO, Rewards, Analysis
        )

        # Column Group 1: Market Data, Position, Portfolio
        layout["column_group_1"].split_column(
            Layout(name="market_data", ratio=1),
            Layout(name="current_position", ratio=1),
            Layout(name="portfolio", ratio=1)
        )

        # Column Group 2: Recent Actions, Recent Trades, Episode Status
        layout["column_group_2"].split_column(
            Layout(name="recent_actions", ratio=1),
            Layout(name="recent_trades", ratio=1),
            Layout(name="episode_status", ratio=1)
        )

        # Column Group 3: Training Progress, PPO Metrics, Reward System, Action Analysis
        layout["column_group_3"].split_column(
            Layout(name="training_progress", ratio=1),
            Layout(name="ppo_metrics", ratio=1),
            Layout(name="reward_system", ratio=1),
            Layout(name="action_analysis", ratio=1)
        )

        return layout

    def _setup_dashboard_logging(self):
        """Setup comprehensive logging with Rich formatting"""

        class ComprehensiveDashboardHandler(RichHandler):
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
                        temp_console = Console(file=buffer, width=140, force_terminal=False)
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

        self.dashboard_handler = ComprehensiveDashboardHandler(self)
        self.dashboard_handler.setLevel(logging.INFO)

    def start(self):
        """Start the comprehensive dashboard"""
        if self._running:
            return

        try:
            # Setup logging redirection
            root_logger = logging.getLogger()
            self.original_handlers = root_logger.handlers.copy()
            root_logger.handlers.clear()
            root_logger.addHandler(self.dashboard_handler)

            # Initialize session start time
            self.state.session_start_time = time.time()

            # Update layout and start Live display
            self._update_layout()

            self.live = Live(
                self.layout,
                console=self.console,
                refresh_per_second=4,  # Higher refresh rate for better responsiveness
                screen=False,
                auto_refresh=True,
                transient=False,
                redirect_stdout=False,
                redirect_stderr=False
            )

            self.live.start()
            self._running = True

            logging.info("📊 Comprehensive Trading Dashboard Started")

        except Exception as e:
            logging.error(f"Failed to start comprehensive dashboard: {e}")
            self._running = False

    def stop(self):
        """Stop the comprehensive dashboard"""
        if not self._running:
            return

        self._running = False

        try:
            # Restore original logging
            root_logger = logging.getLogger()
            root_logger.handlers.clear()

            if hasattr(self, 'original_handlers'):
                for handler in self.original_handlers:
                    root_logger.addHandler(handler)

            if self.live:
                logging.info("Stopping comprehensive trading dashboard...")
                self.live.stop()

        except Exception as e:
            print(f"Error stopping dashboard: {e}")

    def set_training_stage(self, stage: TrainingStage, overall_progress: float = None,
                           details: str = "", substage_progress: float = None):
        """Set training stage with comprehensive tracking"""
        with self._state_lock:
            self.state.current_stage = stage
            if overall_progress is not None:
                self.state.stage_progress = max(0.0, min(1.0, overall_progress))
            if substage_progress is not None:
                self.state.substage_progress = max(0.0, min(1.0, substage_progress))
            if details:
                self.state.stage_details = str(details)

            # Set training mode
            if stage in [TrainingStage.COLLECTING_ROLLOUT, TrainingStage.UPDATING_POLICY]:
                self.state.training_mode = "Training"
            elif stage == TrainingStage.EVALUATING:
                self.state.training_mode = "Evaluation"
            else:
                self.state.training_mode = "Idle"

            self.state.state_version += 1

        if self._running:
            self._safe_update()

    def update_training_metrics(self, metrics: Dict[str, Any]):
        """Update training metrics with comprehensive tracking and history"""
        if not metrics:
            return

        with self._state_lock:
            # Update basic metrics
            for key, value in metrics.items():
                if hasattr(self.state, key) and value is not None:
                    if isinstance(value, (int, float)) and not (isinstance(value, float) and (value != value)):
                        setattr(self.state, key, value)

            # Update metric histories for sparklines
            history_mapping = {
                'mean_episode_reward': 'mean_reward',
                'actor_loss': 'policy_loss',
                'critic_loss': 'value_loss',
                'entropy': 'entropy',
                'clipfrac': 'clip_fraction',
                'approx_kl': 'approx_kl',
                'value_function_explained_variance': 'value_explained_variance'
            }

            for metric_key, history_key in history_mapping.items():
                if metric_key in metrics and metrics[metric_key] is not None:
                    value = float(metrics[metric_key])
                    if not (value != value):  # Not NaN
                        self.state.metric_histories[history_key].add(value)

                        # Also update the current value
                        if history_key == 'policy_loss':
                            self.state.policy_loss = value
                        elif history_key == 'value_loss':
                            self.state.value_loss = value

            # Calculate total loss if we have both components
            if self.state.policy_loss != 0 and self.state.value_loss != 0:
                self.state.total_loss = self.state.policy_loss + self.state.value_loss
                self.state.metric_histories['total_loss'].add(self.state.total_loss)

            # Process reward breakdown
            if 'reward_components' in metrics and isinstance(metrics['reward_components'], dict):
                self.state.reward_breakdown = metrics['reward_components'].copy()

            self.state.state_version += 1

        if self._running:
            self._safe_update()

    def update_state(self, info_dict: Dict[str, Any], market_state: Optional[Dict[str, Any]] = None):
        """Update state with comprehensive tracking"""
        if not self._running or not self.live:
            return

        if not info_dict and not market_state:
            return

        try:
            with self._state_lock:
                if self._update_in_progress:
                    return

                self._update_in_progress = True
                self._update_comprehensive_state(info_dict, market_state)
                self._update_in_progress = False

            self._safe_update()

        except Exception as e:
            self._update_in_progress = False
            logging.error(f"Error updating comprehensive dashboard state: {e}")

    def _update_comprehensive_state(self, info_dict: Dict[str, Any], market_state: Optional[Dict[str, Any]] = None):
        """Comprehensive state update with all metrics"""

        # Update session elapsed time
        if self.state.session_start_time > 0:
            elapsed = time.time() - self.state.session_start_time
            hours, remainder = divmod(int(elapsed), 3600)
            minutes, seconds = divmod(remainder, 60)
            self.state.session_elapsed_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        # Update step counters
        if 'global_step_counter' in info_dict:
            self.state.global_step_counter = int(info_dict['global_step_counter'])
            self.state.step = self.state.global_step_counter

        if 'step' in info_dict:
            self.state.episode_step = int(info_dict['step'])

        if 'episode_number' in info_dict:
            self.state.episode_number = int(info_dict['episode_number'])

        # Update rewards
        if 'reward_step' in info_dict:
            new_step_reward = float(info_dict['reward_step'])
            self.state.last_step_reward = self.state.step_reward
            self.state.step_reward = new_step_reward

        if 'episode_cumulative_reward' in info_dict:
            self.state.episode_reward = float(info_dict['episode_cumulative_reward'])

        # Update timing and market data
        if 'timestamp_iso' in info_dict:
            self.state.timestamp = info_dict['timestamp_iso']
            try:
                dt = datetime.fromisoformat(info_dict['timestamp_iso'].replace('Z', '+00:00'))
                from zoneinfo import ZoneInfo
                et_time = dt.astimezone(ZoneInfo('America/New_York'))
                self.state.market_time = et_time.strftime("%H:%M:%S")
            except Exception:
                self.state.market_time = "N/A"

        # Update portfolio metrics
        portfolio_mapping = {
            'portfolio_equity': 'total_equity',
            'portfolio_cash': 'cash',
            'portfolio_unrealized_pnl': 'unrealized_pnl',
            'portfolio_realized_pnl_session_net': 'realized_pnl'
        }

        for info_key, state_key in portfolio_mapping.items():
            if info_key in info_dict and isinstance(info_dict[info_key], (int, float)):
                setattr(self.state, state_key, float(info_dict[info_key]))

        # Calculate derived portfolio metrics
        if self.state.total_equity > 0 and self.state.initial_capital > 0:
            self.state.session_pnl = self.state.total_equity - self.state.initial_capital
            self.state.session_pnl_pct = (self.state.session_pnl / self.state.initial_capital) * 100

            # Track equity for drawdown calculation
            self._equity_history.append(self.state.total_equity)
            self._calculate_performance_metrics()

        # Update position data
        if self.state.symbol != "N/A":
            symbol = self.state.symbol
            if f'position_{symbol}_qty' in info_dict:
                self.state.position_qty = float(info_dict[f'position_{symbol}_qty'])
            if f'position_{symbol}_side' in info_dict:
                side_value = info_dict[f'position_{symbol}_side']
                self.state.position_side = side_value.value if hasattr(side_value, 'value') else str(side_value)
            if f'position_{symbol}_avg_entry' in info_dict:
                self.state.position_avg_entry = float(info_dict[f'position_{symbol}_avg_entry'])

        # Update market data and calculate price changes
        if market_state:
            new_price = market_state.get('current_price')
            if new_price and isinstance(new_price, (int, float)):
                if self._last_price is not None:
                    if new_price > self._last_price:
                        self.state.price_change_direction = "↑"
                    elif new_price < self._last_price:
                        self.state.price_change_direction = "↓"
                    else:
                        self.state.price_change_direction = ""

                self.state.current_price = float(new_price)
                self._last_price = new_price

            # Update bid/ask and calculate spread
            if 'best_bid_price' in market_state:
                self.state.bid_price = float(market_state['best_bid_price'])
            if 'best_ask_price' in market_state:
                self.state.ask_price = float(market_state['best_ask_price'])

            if self.state.bid_price and self.state.ask_price:
                self.state.spread = self.state.ask_price - self.state.bid_price

            if 'market_session' in market_state:
                self.state.market_session = str(market_state['market_session'])

        # Calculate position P&L vs entry
        if (self.state.current_price and self.state.position_avg_entry > 0 and
                self.state.position_qty != 0):

            price_diff = self.state.current_price - self.state.position_avg_entry
            if self.state.position_side == "SHORT":
                price_diff = -price_diff

            self.state.position_pnl_vs_entry_usd = price_diff * self.state.position_qty
            self.state.position_pnl_vs_entry_pct = (price_diff / self.state.position_avg_entry) * 100

        # Update action tracking
        if 'action_decoded' in info_dict and isinstance(info_dict['action_decoded'], dict):
            self._update_action_tracking(info_dict['action_decoded'])

        # Update trade tracking
        if 'fills_step' in info_dict and isinstance(info_dict['fills_step'], list):
            self._update_trade_tracking(info_dict['fills_step'])

        # Check for episode completion
        if info_dict.get('termination_reason') or info_dict.get('TimeLimit.truncated'):
            self._handle_episode_completion(info_dict)

        self.state.state_version += 1

    def _update_action_tracking(self, action_decoded: Dict[str, Any]):
        """Update comprehensive action tracking"""
        action_type = action_decoded.get('type')
        if action_type:
            action_name = action_type.name if hasattr(action_type, 'name') else str(action_type)
            self.state.last_action_type = action_name

            # Update action bias tracking
            if action_name not in self.state.action_bias_records:
                self.state.action_bias_records[action_name] = ActionBiasRecord(
                    action_type=action_name, count=0, total_reward=0.0,
                    positive_rewards=0, step_count=0
                )

            bias_record = self.state.action_bias_records[action_name]
            bias_record.count += 1
            bias_record.step_count += 1
            bias_record.total_reward += self.state.step_reward

            if self.state.step_reward > 0:
                bias_record.positive_rewards += 1

        size_enum = action_decoded.get('size_enum')
        if size_enum:
            self.state.last_action_size = size_enum.name if hasattr(size_enum, 'name') else str(size_enum)

        self.state.last_action_invalid = bool(action_decoded.get('invalid_reason'))

        # Add to recent actions
        new_action = {
            'step': self.state.episode_step,
            'action_type': self.state.last_action_type,
            'size': self.state.last_action_size.replace('SIZE_', '') if 'SIZE_' in self.state.last_action_size else self.state.last_action_size,
            'step_reward': self.state.step_reward,
            'invalid': self.state.last_action_invalid
        }

        # Avoid duplicates
        if not self.state.recent_actions or self.state.recent_actions[-1] != new_action:
            self.state.recent_actions.append(new_action)

    def _update_trade_tracking(self, fills_list: List[Dict[str, Any]]):
        """Update comprehensive trade tracking"""
        for fill in fills_list:
            if isinstance(fill, dict):
                side = fill.get('order_side', 'N/A')
                side_str = side.value if hasattr(side, 'value') else str(side)

                trade_record = TradeRecord(
                    timestamp=self.state.market_time,
                    side=side_str,
                    quantity=float(fill.get('executed_quantity', 0.0)),
                    symbol=self.state.symbol,
                    entry_price=float(fill.get('executed_price', 0.0)),
                    exit_price=None,  # Will be filled for closing trades
                    pnl=None,  # Will be calculated
                    step=self.state.episode_step
                )

                if trade_record.quantity > 0 and trade_record.entry_price > 0:
                    self.state.recent_trades.append(trade_record)
                    self.state.total_trades += 1

    def _handle_episode_completion(self, info_dict: Dict[str, Any]):
        """Handle episode completion and update history"""
        termination_reason = info_dict.get('termination_reason', 'UNKNOWN')
        is_truncated = info_dict.get('TimeLimit.truncated', False)

        status = "TRUNCATED" if is_truncated else "TERMINATED" if termination_reason else "COMPLETED"

        episode_record = EpisodeRecord(
            episode_number=self.state.episode_number,
            status=status,
            termination_reason=str(termination_reason) if termination_reason else "Max Steps",
            episode_reward=self.state.episode_reward,
            episode_length=self.state.episode_step,
            session_pnl=self.state.session_pnl,
            final_equity=self.state.total_equity,
            timestamp=time.time()
        )

        self.state.episode_history.append(episode_record)

        # Track episode timing
        if len(self.state.episode_history) >= 2:
            prev_time = self.state.episode_history[-2].timestamp
            episode_duration = episode_record.timestamp - prev_time
            self._episode_times.append(episode_duration)

            if self._episode_times:
                self.state.avg_time_per_episode = statistics.mean(self._episode_times)

    def _calculate_performance_metrics(self):
        """Calculate advanced performance metrics"""
        if len(self._equity_history) < 2:
            return

        equity_values = list(self._equity_history)

        # Calculate max drawdown
        peak = equity_values[0]
        max_dd = 0.0

        for equity in equity_values:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, drawdown)

        self.state.max_drawdown_pct = max_dd * 100

        # Calculate simple Sharpe ratio (simplified)
        if len(equity_values) >= 10:
            returns = [(equity_values[i] - equity_values[i - 1]) / equity_values[i - 1]
                       for i in range(1, len(equity_values)) if equity_values[i - 1] > 0]

            if returns:
                mean_return = statistics.mean(returns)
                if len(returns) > 1:
                    std_return = statistics.stdev(returns)
                    self.state.sharpe_ratio = mean_return / std_return if std_return > 0 else 0.0

    def _safe_update(self):
        """Safely update layout with throttling"""
        if not self._running or not self.live:
            return

        current_time = time.time()
        if current_time - self._last_update_time >= self._update_throttle:
            try:
                self._update_layout()
                self._last_update_time = current_time
            except Exception as e:
                logging.error(f"Error in dashboard update: {e}")

    def _update_layout(self):
        """Update all layout components"""
        try:
            with self._state_lock:
                self.layout["header"].update(self._create_header())
                self.layout["market_data"].update(self._create_market_data_panel())
                self.layout["current_position"].update(self._create_current_position_panel())
                self.layout["portfolio"].update(self._create_portfolio_panel())
                self.layout["recent_actions"].update(self._create_recent_actions_panel())
                self.layout["recent_trades"].update(self._create_recent_trades_panel())
                self.layout["episode_status"].update(self._create_episode_status_panel())
                self.layout["training_progress"].update(self._create_training_progress_panel())
                self.layout["ppo_metrics"].update(self._create_ppo_metrics_panel())
                self.layout["reward_system"].update(self._create_reward_system_panel())
                self.layout["action_analysis"].update(self._create_action_analysis_panel())
                self.layout["footer"].update(self._create_footer())
                self.layout["logs"].update(self._create_logs_panel())
        except Exception as e:
            logging.error(f"Error updating layout components: {e}")

    # Panel creation methods
    def _create_header(self) -> Panel:
        """Create header with model info and session time"""
        left_text = f"Model: {self.state.model_name}_{self.state.symbol}"
        right_text = f"Session Time: {self.state.session_elapsed_time}"

        header_content = Text()
        header_content.append(left_text, style="bold cyan")
        header_content.append(" " * (80 - len(left_text) - len(right_text)))
        header_content.append(right_text, style="bold yellow")

        return Panel(Align.center(header_content), style="bright_blue", box=box.HEAVY)

    def _create_market_data_panel(self) -> Panel:
        """Create market data panel"""
        symbol_session = f"{self.state.symbol} - {self.state.market_session}"

        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        table.add_column("Item", width=12)
        table.add_column("Value", width=15)

        table.add_row("Time (NY)", Text(self.state.market_time, style="white"))

        price_text = f"${self.state.current_price:.4f} {self.state.price_change_direction}" if self.state.current_price else "N/A"
        table.add_row("Price", Text(price_text, style="bold yellow"))

        bid_text = f"${self.state.bid_price:.4f}" if self.state.bid_price else "N/A"
        ask_text = f"${self.state.ask_price:.4f}" if self.state.ask_price else "N/A"
        table.add_row("Bid", Text(bid_text, style="red"))
        table.add_row("Ask", Text(ask_text, style="green"))

        spread_text = f"${self.state.spread:.4f}" if self.state.spread else "N/A"
        table.add_row("Spread", Text(spread_text, style="white"))

        return Panel(table, title=f"📊 Market: {symbol_session}", border_style="yellow")

    def _create_current_position_panel(self) -> Panel:
        """Create current position panel"""
        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        table.add_column("Item", width=14)
        table.add_column("Value", width=15)

        side_style = {"LONG": "bold green", "SHORT": "bold red", "FLAT": "white"}.get(self.state.position_side, "white")
        table.add_row("Side", Text(self.state.position_side, style=side_style))

        table.add_row("Quantity", Text(f"{self.state.position_qty:.4f}", style="white"))

        entry_text = f"${self.state.position_avg_entry:.4f}" if self.state.position_avg_entry > 0 else "N/A"
        table.add_row("Avg Entry Price", Text(entry_text, style="white"))

        if abs(self.state.position_pnl_vs_entry_usd) > 0.01:
            pnl_style = "green" if self.state.position_pnl_vs_entry_usd > 0 else "red"
            pnl_text = f"${self.state.position_pnl_vs_entry_usd:.2f} ({self.state.position_pnl_vs_entry_pct:+.2f}%)"
            table.add_row("P&L vs Entry", Text(pnl_text, style=pnl_style))

        return Panel(table, title=f"💼 Position: {self.state.symbol}", border_style="blue")

    def _create_portfolio_panel(self) -> Panel:
        """Create comprehensive portfolio panel"""
        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        table.add_column("Item", width=16)
        table.add_column("Value", width=15)

        equity_style = "bold green" if self.state.total_equity >= self.state.initial_capital else "bold red"
        table.add_row("Total Equity", Text(f"${self.state.total_equity:.2f}", style=equity_style))

        table.add_row("Cash Balance", Text(f"${self.state.cash:.2f}", style="white"))

        session_style = "bold green" if self.state.session_pnl > 0 else "bold red" if self.state.session_pnl < 0 else "white"
        session_text = f"${self.state.session_pnl:.2f} ({self.state.session_pnl_pct:+.2f}%)"
        table.add_row("Session P&L", Text(session_text, style=session_style))

        realized_style = "green" if self.state.realized_pnl > 0 else "red" if self.state.realized_pnl < 0 else "white"
        table.add_row("Realized P&L", Text(f"${self.state.realized_pnl:.2f}", style=realized_style))

        unrealized_style = "green" if self.state.unrealized_pnl > 0 else "red" if self.state.unrealized_pnl < 0 else "white"
        table.add_row("Unrealized P&L", Text(f"${self.state.unrealized_pnl:.2f}", style=unrealized_style))

        table.add_row("Sharpe Ratio", Text(f"{self.state.sharpe_ratio:.2f}", style="cyan"))
        table.add_row("Max Drawdown", Text(f"{self.state.max_drawdown_pct:.2f}%", style="yellow"))
        table.add_row("Trades", Text(str(self.state.total_trades), style="white"))

        return Panel(table, title="📈 Portfolio", border_style="green")

    def _create_recent_actions_panel(self) -> Panel:
        """Create recent actions panel"""
        table = Table(box=box.SIMPLE, show_edge=False, padding=(0, 1))
        table.add_column("Step", width=6)
        table.add_column("Action", width=6)
        table.add_column("Size", width=6)
        table.add_column("Reward", width=8)

        for action in list(self.state.recent_actions):
            step_text = str(action['step'])
            action_type = action['action_type']
            size = action['size']
            reward = action['step_reward']

            reward_style = "green" if reward > 0 else "red" if reward < 0 else "white"
            reward_text = Text(f"{reward:.3f}", style=reward_style)

            if action.get('invalid', False):
                action_type = Text(action_type, style="bold red")

            table.add_row(step_text, action_type, size, reward_text)

        # Fill remaining rows
        while len(table.rows) < 5:
            table.add_row("", "", "", "")

        return Panel(table, title="⚡ Recent Actions", border_style="magenta")

    def _create_recent_trades_panel(self) -> Panel:
        """Create recent trades panel"""
        table = Table(box=box.SIMPLE, show_edge=False, padding=(0, 1))
        table.add_column("Time", width=8)
        table.add_column("Side", width=4)
        table.add_column("Qty", width=6)
        table.add_column("Price", width=8)

        for trade in list(self.state.recent_trades):
            side_style = "green" if trade.side == "BUY" else "red"
            side_text = Text(trade.side, style=side_style)

            table.add_row(
                trade.timestamp,
                side_text,
                f"{trade.quantity:.1f}",
                f"${trade.entry_price:.2f}"
            )

        # Fill remaining rows
        while len(table.rows) < 5:
            table.add_row("", "", "", "")

        return Panel(table, title="⚡ Recent Trades", border_style="blue")

    def _create_episode_status_panel(self) -> Panel:
        """Create episode status panel with history"""
        content = Group()

        # Current episode info
        episode_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        episode_table.add_column("Item", width=14)
        episode_table.add_column("Value", width=12)

        episode_table.add_row("Current Step", Text(str(self.state.episode_step), style="white"))
        episode_table.add_row("Cumulative Reward", Text(f"{self.state.episode_reward:.2f}", style="yellow"))
        episode_table.add_row("Last Step Reward", Text(f"{self.state.last_step_reward:.3f}", style="cyan"))

        content.renderables.append(episode_table)
        content.renderables.append(Text(""))  # Spacer

        # Episode history
        if self.state.episode_history:
            history_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
            history_table.add_column("Ep #", width=4)
            history_table.add_column("Status", width=8)
            history_table.add_column("Reason", width=10)
            history_table.add_column("Reward", width=8)

            for episode in list(self.state.episode_history):
                status_style = {"COMPLETED": "green", "TERMINATED": "red", "TRUNCATED": "yellow"}.get(episode.status, "white")
                reason_short = episode.termination_reason[:8] if len(episode.termination_reason) > 8 else episode.termination_reason

                history_table.add_row(
                    str(episode.episode_number),
                    Text(episode.status[:6], style=status_style),
                    reason_short,
                    f"{episode.episode_reward:.1f}"
                )

            content.renderables.append(Text("📚 Episode History (Last 3)", style="bold"))
            content.renderables.append(history_table)

        return Panel(content, title=f"🎬 Episode Analysis (Ep: {self.state.episode_number})", border_style="cyan")

    def _create_training_progress_panel(self) -> Panel:
        """Create training progress panel"""

        def create_progress_bar(progress: float, width: int = 20) -> Text:
            progress = max(0.0, min(1.0, progress))
            filled = int(progress * width)
            bar = "█" * filled + "░" * (width - filled)
            return Text(bar, style="green")

        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        table.add_column("Item", width=14)
        table.add_column("Value", width=20)

        table.add_row("Mode", Text(self.state.training_mode, style="bold yellow"))
        table.add_row("Current Stage", Text(self.state.current_stage.value, style="cyan"))

        # Progress bars
        overall_bar = create_progress_bar(self.state.stage_progress)
        table.add_row("Overall Progress", overall_bar)

        current_bar = create_progress_bar(self.state.substage_progress)
        table.add_row("Current Stage", current_bar)

        # Stage details (truncated)
        details = self.state.stage_details[:25] + "..." if len(self.state.stage_details) > 25 else self.state.stage_details
        table.add_row("Stage Status", Text(details, style="white"))

        table.add_row("Updates", Text(str(self.state.update_count), style="white"))
        table.add_row("Episode Counter", Text(str(self.state.episode_number), style="white"))
        table.add_row("Global Step", Text(str(self.state.global_step_counter), style="bold yellow"))

        return Panel(table, title="⚙️ Training Progress", border_style="cyan")

    def _create_ppo_metrics_panel(self) -> Panel:
        """Create PPO core metrics panel with sparklines"""
        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        table.add_column("Metric", width=14)
        table.add_column("Value", width=8)
        table.add_column("Trend", width=8)

        table.add_row("Learning Rate", Text(f"{self.state.learning_rate:.1e}", style="white"), Text("", style="white"))

        # Metrics with sparklines
        metrics_data = [
            ("Mean Reward", self.state.mean_episode_reward, "mean_reward"),
            ("Policy Loss", self.state.policy_loss, "policy_loss"),
            ("Value Loss", self.state.value_loss, "value_loss"),
            ("Total Loss", self.state.total_loss, "total_loss"),
            ("Entropy", self.state.entropy, "entropy"),
            ("Clip Fraction", self.state.clip_fraction, "clip_fraction"),
            ("Approx KL", self.state.approx_kl, "approx_kl"),
            ("Value Expl Var", self.state.value_explained_variance, "value_explained_variance")
        ]

        for metric_name, value, history_key in metrics_data:
            if value != 0:
                sparkline = self.state.metric_histories[history_key].get_sparkline()
                value_style = "green" if "Reward" in metric_name and value > 0 else "red" if "Loss" in metric_name else "white"

                table.add_row(
                    metric_name,
                    Text(f"{value:.3f}", style=value_style),
                    Text(sparkline, style="blue")
                )

        return Panel(table, title="🧠 PPO Core Metrics", border_style="green")

    def _create_reward_system_panel(self) -> Panel:
        """Create reward component breakdown panel"""
        table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
        table.add_column("Component", width=12)
        table.add_column("Value", width=8)
        table.add_column("Impact%", width=6)

        if self.state.reward_breakdown:
            total_abs_reward = sum(abs(v) for v in self.state.reward_breakdown.values())

            # Sort by absolute value
            sorted_components = sorted(
                self.state.reward_breakdown.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )

            for component, value in sorted_components[:6]:  # Top 6 components
                if total_abs_reward > 0:
                    impact_pct = (abs(value) / total_abs_reward) * 100
                else:
                    impact_pct = 0

                component_name = component.replace('_', ' ').title()[:12]
                value_style = "green" if value > 0 else "red" if value < 0 else "white"

                table.add_row(
                    component_name,
                    Text(f"{value:.3f}", style=value_style),
                    f"{impact_pct:.1f}%"
                )

        return Panel(table, title="🏆 Reward System", border_style="yellow")

    def _create_action_analysis_panel(self) -> Panel:
        """Create action analysis panel with bias summary"""
        content = Group()

        # Invalid actions count
        invalid_text = Text(f"Invalid Actions: {self.state.invalid_actions_count}",
                            style="red" if self.state.invalid_actions_count > 0 else "white")
        content.renderables.append(invalid_text)
        content.renderables.append(Text(""))  # Spacer

        # Action bias table
        if self.state.action_bias_records:
            bias_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
            bias_table.add_column("Action", width=6)
            bias_table.add_column("Count", width=5)
            bias_table.add_column("%", width=5)
            bias_table.add_column("Reward", width=8)
            bias_table.add_column("Win%", width=5)

            total_actions = sum(record.count for record in self.state.action_bias_records.values())

            for action_type, record in self.state.action_bias_records.items():
                if total_actions > 0:
                    percentage = (record.count / total_actions) * 100
                    win_rate = (record.positive_rewards / record.count) * 100 if record.count > 0 else 0

                    reward_style = "green" if record.total_reward > 0 else "red" if record.total_reward < 0 else "white"

                    bias_table.add_row(
                        action_type,
                        str(record.count),
                        f"{percentage:.1f}",
                        Text(f"{record.total_reward:.2f}", style=reward_style),
                        f"{win_rate:.0f}"
                    )

            content.renderables.append(Text("Action Bias:", style="bold"))
            content.renderables.append(bias_table)

        return Panel(content, title="🏆 Action Analysis", border_style="purple")

    def _create_footer(self) -> Panel:
        """Create footer with performance metrics"""
        footer_content = Text()

        # Performance metrics
        metrics = [
            f"Steps/Sec: {self.state.steps_per_second:.1f}",
            f"Time/Update: {self.state.time_per_update:.2f}s",
            f"Time/Episode (Avg): {self.state.avg_time_per_episode:.1f}s"
        ]

        footer_text = " | ".join(metrics)
        footer_content.append(footer_text, style="white")

        return Panel(Align.center(footer_content), style="bright_white")

    def _create_logs_panel(self) -> Panel:
        """Create enhanced logs panel"""
        recent_logs = list(self.log_buffer)[-(self.log_height - 2):]

        if recent_logs:
            log_display = "\n".join([entry['text'] for entry in recent_logs])
        else:
            log_display = "[dim]Waiting for log messages...[/dim]"

        return Panel(
            Text.from_markup(log_display),
            title="📜 System Logs (Real-time)",
            border_style="cyan",
            height=self.log_height
        )

    # Public interface methods
    def set_symbol(self, symbol: str):
        """Set the trading symbol"""
        with self._state_lock:
            self.state.symbol = symbol

    def set_initial_capital(self, capital: float):
        """Set initial capital"""
        with self._state_lock:
            self.state.initial_capital = capital

    def set_model_name(self, model_name: str):
        """Set the model name for display"""
        with self._state_lock:
            self.state.model_name = model_name

    def set_training_info(self, episode_num: int = 0, total_episodes: int = 0,
                          total_steps: int = 0, update_count: int = 0,
                          buffer_size: int = 0, is_training: bool = True,
                          is_evaluating: bool = False, learning_rate: float = 0.0):
        """Set comprehensive training information"""
        with self._state_lock:
            if episode_num > 0:
                self.state.episode_number = episode_num
            if total_steps >= 0:
                self.state.global_step_counter = total_steps
                self.state.step = total_steps
            if update_count >= 0:
                self.state.update_count = update_count

            self.state.total_episodes = max(0, total_episodes)
            self.state.buffer_size = max(0, buffer_size)
            self.state.is_training = is_training
            self.state.is_evaluating = is_evaluating

            if learning_rate > 0:
                self.state.learning_rate = learning_rate

            # Update training mode
            if is_training:
                self.state.training_mode = "Training"
            elif is_evaluating:
                self.state.training_mode = "Evaluation"
            else:
                self.state.training_mode = "Idle"

            self.state.state_version += 1

        if self._running:
            self._safe_update()