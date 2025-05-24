"""
Dashboard data structures and state management.
Clean architecture for extensible dashboard metrics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Deque
from collections import deque, defaultdict
from datetime import datetime
import time


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
        return self.timestamp.strftime("%H:%M:%S")


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
    side: str  # BUY/SELL
    quantity: float
    symbol: str
    entry_price: float
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    fees: float = 0.0


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
    price_history: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    reward_history: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    position_history: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    
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
        self.global_actions: Deque[Action] = deque(maxlen=5000)
        
        # Reward components
        self.reward_components: Dict[str, RewardComponent] = {}
        
        # Action analysis
        self.action_analysis = ActionAnalysis()
        
        # Model info
        self.model_name: str = "N/A"
        self.session_start_time: datetime = datetime.now()
        
    def start_new_episode(self, episode_num: int):
        """Start a new episode"""
        # Save current episode if exists
        if self.current_episode and self.current_episode.status == "Active":
            self.end_current_episode("Interrupted")
        
        # Create new episode
        self.current_episode = EpisodeData(episode_num=episode_num)
        
    def end_current_episode(self, reason: str = "Completed"):
        """End current episode"""
        if self.current_episode:
            self.current_episode.status = "Completed"
            self.current_episode.termination_reason = reason
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
        """Add a trade"""
        trade = Trade(
            timestamp=datetime.now(),
            side=trade_data.get('side', 'UNKNOWN'),
            quantity=trade_data.get('quantity', 0),
            symbol=trade_data.get('symbol', self.market_data.symbol),
            entry_price=trade_data.get('entry_price', 0),
            exit_price=trade_data.get('exit_price'),
            pnl=trade_data.get('pnl', 0),
            fees=trade_data.get('fees', 0)
        )
        
        # Add to global
        self.global_trades.append(trade)
        self.portfolio.num_trades += 1
        
        # Add to current episode
        if self.current_episode:
            self.current_episode.trades.append(trade)
            if trade.pnl:
                self.current_episode.total_pnl += trade.pnl
    
    def update_training_progress(self, data: Dict[str, Any]):
        """Update training progress"""
        self.training_progress.mode = data.get('mode', self.training_progress.mode)
        self.training_progress.current_stage = data.get('stage', self.training_progress.current_stage)
        self.training_progress.updates = data.get('updates', self.training_progress.updates)
        self.training_progress.global_steps = data.get('global_steps', self.training_progress.global_steps)
        
        # Calculate timing metrics
        if self.training_progress.global_steps > 0:
            elapsed = (datetime.now() - self.training_progress.start_time).total_seconds()
            self.training_progress.steps_per_second = self.training_progress.global_steps / elapsed
    
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
    
    def get_recent_actions(self, n: int = 5) -> List[Action]:
        """Get recent actions from current episode"""
        if self.current_episode:
            return list(self.current_episode.actions)[-n:]
        return []
    
    def get_recent_trades(self, n: int = 5) -> List[Trade]:
        """Get recent trades from current episode"""
        if self.current_episode:
            return list(self.current_episode.trades)[-n:]
        return []
    
    def get_episode_history(self, n: int = 3) -> List[EpisodeData]:
        """Get recent episode history"""
        return list(self.episode_history)[-n:]
    
    def calculate_action_bias(self):
        """Calculate action bias statistics for current episode"""
        if not self.current_episode:
            return
        
        episode = self.current_episode
        total_actions = sum(episode.action_counts.values())
        
        if total_actions == 0:
            return
        
        action_bias = {}
        for action_type in ['HOLD', 'BUY', 'SELL']:
            count = episode.action_counts.get(action_type, 0)
            rewards = episode.action_rewards.get(action_type, [])
            
            action_bias[action_type] = {
                'count': count,
                'percent_steps': (count / total_actions) * 100 if total_actions > 0 else 0,
                'mean_reward': sum(rewards) / len(rewards) if rewards else 0,
                'total_reward': sum(rewards),
                'pos_reward_rate': (len([r for r in rewards if r > 0]) / len(rewards) * 100) if rewards else 0
            }
        
        self.action_analysis.action_bias = action_bias
    
    @property
    def session_elapsed_time(self) -> str:
        """Get elapsed session time as HH:MM:SS"""
        elapsed = datetime.now() - self.session_start_time
        hours = int(elapsed.total_seconds() // 3600)
        minutes = int((elapsed.total_seconds() % 3600) // 60)
        seconds = int(elapsed.total_seconds() % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"