"""
Dashboard data structures and state management.
Clean architecture for extensible dashboard metrics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Deque
from collections import deque, defaultdict
from datetime import datetime
import time
import pytz
import pandas as pd


@dataclass
class MarketData:
    """Current market data"""
    symbol: str = "N/A"
    price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    spread: float = 0.0
    volume: float = 0.0
    market_session: str = "Regular"  # Pre/Regular/Post
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def time_ny(self) -> str:
        """Get NY time"""
        # Convert to NY timezone if not already
        ny_tz = pytz.timezone('America/New_York')
        if self.timestamp.tzinfo is None:
            # Assume UTC if no timezone
            utc_time = pytz.utc.localize(self.timestamp)
            ny_time = utc_time.astimezone(ny_tz)
        else:
            ny_time = self.timestamp.astimezone(ny_tz)
        return ny_time.strftime("%H:%M:%S")


@dataclass
class Position:
    """Current position data"""
    symbol: str = "N/A"
    side: str = "Flat"  # Long/Short/Flat
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    current_price: float = 0.0
    
    @property
    def pnl_dollars(self) -> float:
        """P&L in dollars"""
        if self.side == "Flat" or self.quantity == 0:
            return 0.0
        return (self.current_price - self.avg_entry_price) * self.quantity
    
    @property
    def pnl_percent(self) -> float:
        """P&L in percent"""
        if self.avg_entry_price == 0:
            return 0.0
        return (self.pnl_dollars / (self.avg_entry_price * abs(self.quantity))) * 100


@dataclass
class Portfolio:
    """Portfolio data"""
    total_equity: float = 25000.0
    cash_balance: float = 25000.0
    session_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    num_trades: int = 0
    initial_equity: float = 25000.0
    total_commission: float = 0.0
    total_slippage: float = 0.0
    total_fees: float = 0.0
    
    @property
    def session_pnl_percent(self) -> float:
        """Session P&L as percentage"""
        if self.initial_equity == 0:
            return 0.0
        return (self.session_pnl / self.initial_equity) * 100


@dataclass
class Action:
    """Single action record"""
    step: int
    action_type: str  # BUY/SELL/HOLD
    size_signal: str  # 25%/50%/75%/100%
    step_reward: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Trade:
    """Single trade record"""
    timestamp: datetime
    side: str  # LONG/SHORT (or BUY/SELL for legacy)
    quantity: float
    symbol: str
    entry_price: float
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    fees: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0


@dataclass
class Execution:
    """Single execution record (fill)"""
    timestamp: datetime
    side: str  # BUY/SELL
    quantity: float
    symbol: str
    price: float
    commission: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0


@dataclass
class EpisodeData:
    """Data for a single episode"""
    episode_num: int
    status: str = "Active"  # Active/Completed
    termination_reason: str = ""
    total_reward: float = 0.0
    total_pnl: float = 0.0
    steps: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    # Episode-specific data
    actions: Deque[Action] = field(default_factory=lambda: deque(maxlen=1000))
    trades: Deque[Trade] = field(default_factory=lambda: deque(maxlen=100))
    executions: Deque[Execution] = field(default_factory=lambda: deque(maxlen=100))
    price_history: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    reward_history: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    position_history: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    
    # Episode termination info
    truncated: bool = False
    truncation_reason: str = ""
    
    # Action statistics
    action_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    action_rewards: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))


@dataclass
class TrainingProgress:
    """Training progress data"""
    mode: str = "Idle"  # Training/Evaluation/Idle
    current_stage: str = "Initializing"
    overall_progress: float = 0.0
    stage_progress: float = 0.0
    stage_status: str = ""
    updates: int = 0
    total_episodes: int = 0
    global_steps: int = 0
    
    # Timing
    start_time: datetime = field(default_factory=datetime.now)
    steps_per_second: float = 0.0
    time_per_update: float = 0.0
    time_per_episode: float = 0.0
    
    # Enhanced stage-specific information
    rollout_steps: int = 0
    rollout_total: int = 2048  # Default PPO rollout size
    current_epoch: int = 0
    total_epochs: int = 10
    current_batch: int = 0
    total_batches: int = 32


@dataclass
class PPOMetrics:
    """PPO training metrics"""
    learning_rate: float = 0.0
    mean_reward_batch: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    total_loss: float = 0.0
    entropy: float = 0.0
    clip_fraction: float = 0.0
    approx_kl: float = 0.0
    explained_variance: float = 0.0
    
    # Historical data for sparklines
    policy_loss_history: Deque[float] = field(default_factory=lambda: deque(maxlen=20))
    value_loss_history: Deque[float] = field(default_factory=lambda: deque(maxlen=20))
    entropy_history: Deque[float] = field(default_factory=lambda: deque(maxlen=20))
    clip_fraction_history: Deque[float] = field(default_factory=lambda: deque(maxlen=20))


@dataclass
class RewardComponent:
    """Single reward component"""
    name: str
    component_type: str  # Reward/Penalty
    total_impact: float = 0.0
    percent_of_total: float = 0.0
    avg_magnitude: float = 0.0
    times_triggered: int = 0


@dataclass
class ActionAnalysis:
    """Action analysis data"""
    invalid_actions_count: int = 0
    action_bias: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # action_bias format: {
    #     'HOLD': {'count': 150, 'percent_steps': 60.0, 'mean_reward': 0.01, 'total_reward': 1.50, 'pos_reward_rate': 60.0},
    #     ...
    # }


class DashboardState:
    """Central dashboard state management"""
    
    def __init__(self):
        # Core data
        self.market_data = MarketData()
        self.position = Position()
        self.portfolio = Portfolio()
        self.training_progress = TrainingProgress()
        self.ppo_metrics = PPOMetrics()
        
        # Episode management
        self.current_episode: Optional[EpisodeData] = None
        self.episode_history: Deque[EpisodeData] = deque(maxlen=50)
        
        # Global tracking (across all episodes)
        self.global_trades: Deque[Trade] = deque(maxlen=500)
        self.global_executions: Deque[Execution] = deque(maxlen=500)
        self.global_actions: Deque[Action] = deque(maxlen=5000)
        
        # Reward components
        self.reward_components: Dict[str, RewardComponent] = {}
        
        # Action analysis - global (training-wide)
        self.action_analysis = ActionAnalysis()
        self.global_action_counts = defaultdict(int)
        self.global_action_rewards = defaultdict(list)
        
        # Model info
        self.model_name: str = "N/A"
        self.session_start_time: datetime = datetime.now()
        self.start_time = time.time()  # For uptime tracking
        
        # OHLC data for candlestick charts
        self.ohlc_data: Deque[Dict[str, Any]] = deque(maxlen=500)  # Keep last 500 bars for better visibility
        self.last_bar_timestamp = None  # Track last bar to avoid duplicates
        
        # Full day 1m bars data (loaded once per day)
        self.full_day_1m_bars: List[Dict[str, Any]] = []  # Complete day's 1m bars
        self.full_day_date = None  # Date of loaded full day data
        self.full_day_symbol = None  # Symbol of loaded full day data
        
    def start_new_episode(self, episode_num: int):
        """Start a new episode"""
        # Save current episode if exists
        if self.current_episode and self.current_episode.status == "Active":
            self.end_current_episode("Interrupted")
        
        # Create new episode
        self.current_episode = EpisodeData(episode_num=episode_num)
        
        # Don't reset action analysis - it's global now
        
    def end_current_episode(self, reason: str = "Completed", truncated: bool = False):
        """End current episode"""
        if self.current_episode:
            self.current_episode.status = "Completed"
            self.current_episode.termination_reason = reason
            self.current_episode.truncated = truncated
            # Set truncation reason for normal episode endings
            if truncated:
                # If truncated, set a meaningful truncation reason
                if reason == "TRUNCATED" or reason == "Completed":
                    self.current_episode.truncation_reason = "Max steps reached"
                else:
                    self.current_episode.truncation_reason = reason
            elif reason not in ['BANKRUPTCY', 'MAX_LOSS_REACHED', 'INVALID_ACTION_LIMIT_REACHED', 'END_OF_SESSION_DATA', 'OBSERVATION_FAILURE']:
                # For other normal completions
                self.current_episode.truncation_reason = "Normal completion"
            self.current_episode.end_time = datetime.now()
            self.episode_history.append(self.current_episode)
            self.current_episode = None
    
    def update_market(self, data: Dict[str, Any]):
        """Update market data"""
        self.market_data.symbol = data.get('symbol', self.market_data.symbol)
        self.market_data.price = data.get('price', self.market_data.price)
        self.market_data.bid = data.get('bid', self.market_data.bid)
        self.market_data.ask = data.get('ask', self.market_data.ask)
        self.market_data.spread = self.market_data.ask - self.market_data.bid
        self.market_data.volume = data.get('volume', self.market_data.volume)
        
        # Use provided timestamp or current time
        if 'timestamp' in data and data['timestamp']:
            self.market_data.timestamp = data['timestamp']
        else:
            self.market_data.timestamp = datetime.now()
        
        # Update position current price
        self.position.current_price = self.market_data.price
        
        # Track in current episode
        if self.current_episode:
            self.current_episode.price_history.append(self.market_data.price)
    
    def update_position(self, data: Dict[str, Any]):
        """Update position data"""
        self.position.symbol = data.get('symbol', self.position.symbol)
        self.position.quantity = data.get('quantity', 0.0)
        self.position.avg_entry_price = data.get('avg_entry_price', self.position.avg_entry_price)
        
        # Determine side
        if self.position.quantity > 0:
            self.position.side = "Long"
        elif self.position.quantity < 0:
            self.position.side = "Short"
        else:
            self.position.side = "Flat"
        
        # Track in current episode
        if self.current_episode:
            self.current_episode.position_history.append(self.position.quantity)
    
    def update_portfolio(self, data: Dict[str, Any]):
        """Update portfolio data"""
        self.portfolio.total_equity = data.get('equity', self.portfolio.total_equity)
        self.portfolio.cash_balance = data.get('cash', self.portfolio.cash_balance)
        self.portfolio.realized_pnl = data.get('realized_pnl', self.portfolio.realized_pnl)
        self.portfolio.unrealized_pnl = data.get('unrealized_pnl', self.portfolio.unrealized_pnl)
        self.portfolio.session_pnl = self.portfolio.total_equity - self.portfolio.initial_equity
        
        # Update costs
        if 'total_commission' in data:
            self.portfolio.total_commission = data['total_commission']
        if 'total_slippage' in data:
            self.portfolio.total_slippage = data['total_slippage']
        if 'total_fees' in data:
            self.portfolio.total_fees = data['total_fees']
        
        # Update metrics
        if 'sharpe_ratio' in data:
            self.portfolio.sharpe_ratio = data['sharpe_ratio']
        if 'max_drawdown' in data:
            self.portfolio.max_drawdown = data['max_drawdown']
    
    def add_action(self, step: int, action_type: str, size: float, reward: float):
        """Add an action"""
        size_map = {0.25: "25%", 0.5: "50%", 0.75: "75%", 1.0: "100%"}
        size_signal = size_map.get(size, f"{size*100:.0f}%")
        
        action = Action(
            step=step,
            action_type=action_type,
            size_signal=size_signal,
            step_reward=reward
        )
        
        # Add to global
        self.global_actions.append(action)
        self.global_action_counts[action_type] += 1
        self.global_action_rewards[action_type].append(reward)
        
        # Add to current episode
        if self.current_episode:
            self.current_episode.actions.append(action)
            self.current_episode.reward_history.append(reward)
            self.current_episode.total_reward += reward
            self.current_episode.steps = step
            
            # Update action stats
            self.current_episode.action_counts[action_type] += 1
            self.current_episode.action_rewards[action_type].append(reward)
    
    def add_trade(self, trade_data: Dict[str, Any]):
        """Add a completed trade"""
        trade = Trade(
            timestamp=datetime.now(),
            side=trade_data.get('side', 'UNKNOWN'),
            quantity=trade_data.get('quantity', 0),
            symbol=trade_data.get('symbol', self.market_data.symbol),
            entry_price=trade_data.get('entry_price', 0),
            exit_price=trade_data.get('exit_price'),
            pnl=trade_data.get('pnl', 0),
            fees=trade_data.get('fees', 0),
            commission=trade_data.get('commission', 0),
            slippage=trade_data.get('slippage', 0)
        )
        
        # Add to global
        self.global_trades.append(trade)
        self.portfolio.num_trades += 1
        
        # Add to current episode
        if self.current_episode:
            self.current_episode.trades.append(trade)
            if trade.pnl:
                self.current_episode.total_pnl += trade.pnl
    
    def add_execution(self, execution_data: Dict[str, Any]):
        """Add an execution (fill)"""
        # Use provided timestamp or current time
        timestamp = execution_data.get('timestamp', datetime.now())
        if timestamp and not isinstance(timestamp, datetime):
            # Convert if needed
            try:
                timestamp = pd.to_datetime(timestamp)
            except:
                timestamp = datetime.now()
                
        execution = Execution(
            timestamp=timestamp,
            side=execution_data.get('side', 'UNKNOWN'),
            quantity=execution_data.get('quantity', 0),
            symbol=execution_data.get('symbol', self.market_data.symbol),
            price=execution_data.get('price', 0),
            commission=execution_data.get('commission', 0),
            fees=execution_data.get('fees', 0),
            slippage=execution_data.get('slippage', 0)
        )
        
        # Add to global
        self.global_executions.append(execution)
        
        # Add to current episode
        if self.current_episode:
            self.current_episode.executions.append(execution)
    
    def update_training_progress(self, data: Dict[str, Any]):
        """Update training progress"""
        self.training_progress.mode = data.get('mode', self.training_progress.mode)
        self.training_progress.current_stage = data.get('stage', self.training_progress.current_stage)
        self.training_progress.updates = data.get('updates', self.training_progress.updates)
        self.training_progress.global_steps = data.get('global_steps', self.training_progress.global_steps)
        self.training_progress.total_episodes = data.get('total_episodes', self.training_progress.total_episodes)
        
        # Update timing metrics if provided
        if 'steps_per_second' in data:
            self.training_progress.steps_per_second = data['steps_per_second']
        elif self.training_progress.global_steps > 0:
            elapsed = (datetime.now() - self.training_progress.start_time).total_seconds()
            self.training_progress.steps_per_second = self.training_progress.global_steps / elapsed
            
        # Update other timing metrics
        if 'overall_progress' in data:
            self.training_progress.overall_progress = data['overall_progress']
        if 'stage_progress' in data:
            self.training_progress.stage_progress = data['stage_progress']
        if 'stage_status' in data:
            self.training_progress.stage_status = data['stage_status']
            
        # Update enhanced stage-specific information
        if 'rollout_steps' in data:
            self.training_progress.rollout_steps = data['rollout_steps']
        if 'rollout_total' in data:
            self.training_progress.rollout_total = data['rollout_total']
        if 'current_epoch' in data:
            self.training_progress.current_epoch = data['current_epoch']
        if 'total_epochs' in data:
            self.training_progress.total_epochs = data['total_epochs']
        if 'current_batch' in data:
            self.training_progress.current_batch = data['current_batch']
        if 'total_batches' in data:
            self.training_progress.total_batches = data['total_batches']
    
    def update_ppo_metrics(self, data: Dict[str, Any]):
        """Update PPO metrics"""
        self.ppo_metrics.learning_rate = data.get('lr', self.ppo_metrics.learning_rate)
        self.ppo_metrics.mean_reward_batch = data.get('mean_reward', self.ppo_metrics.mean_reward_batch)
        
        # Update losses and add to history
        if 'policy_loss' in data:
            self.ppo_metrics.policy_loss = data['policy_loss']
            self.ppo_metrics.policy_loss_history.append(data['policy_loss'])
        
        if 'value_loss' in data:
            self.ppo_metrics.value_loss = data['value_loss']
            self.ppo_metrics.value_loss_history.append(data['value_loss'])
        
        if 'entropy' in data:
            self.ppo_metrics.entropy = data['entropy']
            self.ppo_metrics.entropy_history.append(data['entropy'])
        
        self.ppo_metrics.total_loss = data.get('total_loss', 
                                               self.ppo_metrics.policy_loss + self.ppo_metrics.value_loss)
        self.ppo_metrics.clip_fraction = data.get('clip_fraction', self.ppo_metrics.clip_fraction)
        self.ppo_metrics.approx_kl = data.get('approx_kl', self.ppo_metrics.approx_kl)
        self.ppo_metrics.explained_variance = data.get('explained_variance', self.ppo_metrics.explained_variance)
    
    def update_reward_components(self, components_data: Dict[str, float]):
        """Update reward components from raw data"""
        # Calculate total impact for this step
        step_total_impact = sum(abs(v) for v in components_data.values() if v != 0)
        
        for name, value in components_data.items():
            if value == 0:  # Skip zero values
                continue
                
            if name not in self.reward_components:
                # Create new component
                self.reward_components[name] = RewardComponent(
                    name=name,
                    component_type="Reward" if value >= 0 else "Penalty"
                )
            
            component = self.reward_components[name]
            component.times_triggered += 1
            
            # Update running average
            if component.times_triggered == 1:
                component.avg_magnitude = value
            else:
                # Exponential moving average for smoother updates
                alpha = 0.1  # Smoothing factor
                component.avg_magnitude = alpha * value + (1 - alpha) * component.avg_magnitude
            
            # Update total impact (cumulative)
            component.total_impact += value
        
        # Calculate overall percentages based on total cumulative impact
        total_cumulative_impact = sum(abs(comp.total_impact) for comp in self.reward_components.values())
        if total_cumulative_impact > 0:
            for comp in self.reward_components.values():
                comp.percent_of_total = (abs(comp.total_impact) / total_cumulative_impact) * 100
    
    def get_recent_actions(self, n: int = 5) -> List[Action]:
        """Get recent actions from current episode"""
        if self.current_episode:
            return list(self.current_episode.actions)[-n:]
        return []
    
    def get_recent_trades(self, n: int = 5) -> List[Trade]:
        """Get recent completed trades from current episode"""
        if self.current_episode:
            return list(self.current_episode.trades)[-n:]
        return []
    
    def get_recent_executions(self, n: int = 5) -> List[Execution]:
        """Get recent executions from current episode"""
        if self.current_episode:
            return list(self.current_episode.executions)[-n:]
        return []
    
    def get_episode_history(self, n: int = 3) -> List[EpisodeData]:
        """Get recent episode history"""
        return list(self.episode_history)[-n:]
    
    def calculate_action_bias(self):
        """Calculate action bias statistics globally (training-wide)"""
        total_actions = sum(self.global_action_counts.values())
        
        if total_actions == 0:
            return
        
        action_bias = {}
        for action_type in ['HOLD', 'BUY', 'SELL']:
            count = self.global_action_counts.get(action_type, 0)
            rewards = self.global_action_rewards.get(action_type, [])
            
            action_bias[action_type] = {
                'count': count,
                'percent_steps': (count / total_actions) * 100 if total_actions > 0 else 0,
                'mean_reward': sum(rewards) / len(rewards) if rewards else 0,
                'total_reward': sum(rewards),
                'pos_reward_rate': (len([r for r in rewards if r > 0]) / len(rewards) * 100) if rewards else 0
            }
        
        self.action_analysis.action_bias = action_bias
    
    def update_trade(self, trade_data: Dict[str, Any]):
        """Update/add a trade - alias for add_trade with side mapping"""
        # Map action to side for compatibility
        if 'action' in trade_data and 'side' not in trade_data:
            trade_data = trade_data.copy()
            trade_data['side'] = trade_data['action']
        self.add_trade(trade_data)
    
    @property
    def session_elapsed_time(self) -> str:
        """Get elapsed session time as HH:MM:SS"""
        elapsed = datetime.now() - self.session_start_time
        hours = int(elapsed.total_seconds() // 3600)
        minutes = int((elapsed.total_seconds() % 3600) // 60)
        seconds = int(elapsed.total_seconds() % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"