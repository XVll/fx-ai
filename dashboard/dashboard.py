# dashboard/dashboard.py - Live trading dashboard with dark mode

import threading
import time
import logging
import webbrowser
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import deque
import numpy as np

import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd


# Dark theme colors
DARK_THEME = {
    'bg_primary': '#0d1117',      # Main background
    'bg_secondary': '#161b22',    # Cards/panels
    'bg_tertiary': '#21262d',     # Inputs/tables
    'border': '#30363d',          # Borders
    'text_primary': '#f0f6fc',    # Main text
    'text_secondary': '#8b949e',  # Secondary text
    'text_muted': '#6e7681',      # Muted text
    'accent_blue': '#58a6ff',     # Links/accents
    'accent_green': '#56d364',    # Success/profit
    'accent_red': '#f85149',      # Error/loss
    'accent_orange': '#d29922',   # Warning
    'accent_purple': '#bc8cff',   # Special
}

@dataclass
class DashboardState:
    """Enhanced state container for comprehensive trading dashboard"""
    
    # Session info
    session_start_time: datetime = field(default_factory=datetime.now)
    model_name: str = "MLGO_v1"
    symbol: str = "MLGO"
    
    # Market data
    ny_time: str = ""
    trading_hours: str = "PRE-MARKET"
    current_price: float = 0.0
    bid_price: float = 0.0
    ask_price: float = 0.0
    spread: float = 0.0
    spread_pct: float = 0.0
    volume: int = 0
    
    # Position data
    position_side: str = "FLAT"
    position_qty: int = 0
    avg_entry_price: float = 0.0
    position_pnl_dollar: float = 0.0
    position_pnl_percent: float = 0.0
    
    # Portfolio data
    total_equity: float = 100000.0
    cash_balance: float = 100000.0
    session_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    
    # Episode info
    current_step: int = 0
    max_steps: int = 0
    cumulative_reward: float = 0.0
    last_step_reward: float = 0.0
    episode_number: int = 0
    
    # Training state
    mode: str = "Idle"
    stage: str = "Not Started"
    stage_status: str = ""
    overall_progress: float = 0.0
    stage_progress: float = 0.0
    
    # Counters
    updates: int = 0
    global_steps: int = 0
    total_episodes: int = 0
    
    # Performance metrics
    steps_per_second: float = 0.0
    episodes_per_hour: float = 0.0
    updates_per_second: float = 0.0
    time_per_update: float = 0.0
    time_per_episode: float = 0.0
    
    # PPO metrics with history
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    clip_fraction: float = 0.0
    approx_kl: float = 0.0
    explained_variance: float = 0.0
    learning_rate: float = 0.0
    
    # Trading metrics
    mean_episode_reward: float = 0.0
    total_pnl: float = 0.0
    
    # Data collections
    episode_rewards: deque = field(default_factory=lambda: deque(maxlen=100))
    episode_lengths: deque = field(default_factory=lambda: deque(maxlen=100))
    episode_data: deque = field(default_factory=lambda: deque(maxlen=50))
    
    # Reward components
    reward_components: Dict[str, float] = field(default_factory=dict)
    reward_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Action tracking
    action_counts: Dict[str, int] = field(default_factory=lambda: {'HOLD': 0, 'BUY': 0, 'SELL': 0})
    recent_actions: deque = field(default_factory=lambda: deque(maxlen=20))
    
    # Trade data with enhanced tracking
    trades: deque = field(default_factory=lambda: deque(maxlen=200))
    recent_trades: deque = field(default_factory=lambda: deque(maxlen=10))
    
    # Time series for sparklines and charts
    ppo_loss_history: deque = field(default_factory=lambda: deque(maxlen=200))
    value_loss_history: deque = field(default_factory=lambda: deque(maxlen=200))
    entropy_history: deque = field(default_factory=lambda: deque(maxlen=200))
    reward_history_ts: deque = field(default_factory=lambda: deque(maxlen=200))
    price_history: deque = field(default_factory=lambda: deque(maxlen=500))
    
    # Curriculum learning info
    curriculum_stage: str = "Basic"
    curriculum_progress: float = 0.0
    difficulty_level: float = 0.5
    learning_focus: str = "Entry timing"
    
    # Environment info
    environment_step: int = 0
    data_quality: float = 1.0
    momentum_score: float = 0.0
    volatility: float = 0.0
    
    def update_training_state(self, data: Dict[str, Any]):
        """Update training state from metrics"""
        timestamp = datetime.now()
        
        # Update scalar values
        for key, value in data.items():
            if hasattr(self, key) and not isinstance(getattr(self, key), (list, deque, dict)):
                setattr(self, key, value)
        
        # Update time series data
        if 'policy_loss' in data:
            self.ppo_loss_history.append({'time': timestamp, 'value': data['policy_loss']})
        if 'value_loss' in data:
            self.value_loss_history.append({'time': timestamp, 'value': data['value_loss']})
        if 'entropy' in data:
            self.entropy_history.append({'time': timestamp, 'value': data['entropy']})
        if 'mean_episode_reward' in data:
            self.reward_history_ts.append({'time': timestamp, 'value': data['mean_episode_reward']})
        if 'current_price' in data:
            self.price_history.append({'time': timestamp, 'price': data['current_price']})
                
    def update_episode_data(self, episode_data: Dict[str, Any]):
        """Update episode tracking"""
        if 'episode_reward' in episode_data:
            self.episode_rewards.append(episode_data['episode_reward'])
        if 'episode_length' in episode_data:
            self.episode_lengths.append(episode_data['episode_length'])
                
        self.episode_data.append({
            'timestamp': datetime.now(),
            **episode_data
        })
            
    def update_reward_components(self, components: Dict[str, float]):
        """Update reward component tracking"""
        self.reward_components.update(components)
        self.reward_history.append({
            'timestamp': datetime.now(),
            **components
        })
        
    def update_trade_data(self, trade_data: Dict[str, Any]):
        """Update trade tracking with enhanced data"""
        trade_entry = {
            'timestamp': datetime.now(),
            **trade_data
        }
        self.trades.append(trade_entry)
        self.recent_trades.append(trade_entry)
            
    def update_action_data(self, action_data: Dict[str, Any]):
        """Update action tracking"""
        action_type = action_data.get('action_type', 'HOLD')
        if action_type in self.action_counts:
            self.action_counts[action_type] += 1
        
        self.recent_actions.append({
            'timestamp': datetime.now(),
            **action_data
        })
            
    def update_market_data(self, market_data: Dict[str, Any]):
        """Update real-time market data"""
        for key, value in market_data.items():
            if hasattr(self, key):
                setattr(self, key, value)
                
        # Update NY time
        now_ny = datetime.now(timezone.utc).astimezone()
        self.ny_time = now_ny.strftime("%H:%M:%S")
        
        # Determine trading hours
        hour = now_ny.hour
        if 4 <= hour < 9.5:
            self.trading_hours = "PRE-MARKET"
        elif 9.5 <= hour < 16:
            self.trading_hours = "MARKET HOURS"
        elif 16 <= hour < 20:
            self.trading_hours = "AFTER HOURS"
        else:
            self.trading_hours = "CLOSED"
            
    def update_reset_point_performance(self, reset_idx: int, performance: Dict[str, float]):
        """Update reset point performance"""
        pass


@dataclass 
class MomentumDay:
    """Momentum day data structure"""
    date: Any
    symbol: str
    activity_score: float
    max_intraday_move: float = 0.0
    volume_multiplier: float = 0.0
    reset_points: list = field(default_factory=list)
    is_front_side: bool = False
    is_back_side: bool = False
    halt_count: int = 0


class MomentumDashboard:
    """Comprehensive live trading dashboard with dark theme"""
    
    def __init__(self, port: int = 8050, update_interval: int = 500):
        self.port = port
        self.update_interval = update_interval
        self.logger = logging.getLogger(__name__)
        
        # State
        self.state = DashboardState()
        self.is_running = False
        self._momentum_day: Optional[MomentumDay] = None
        self._curriculum_progress: float = 0.0
        self._curriculum_strategy: Optional[str] = None
        
        # Dash app
        self.app = None
        self._server_thread = None
        self._lock = threading.Lock()
        
    def _create_app(self):
        """Create the comprehensive Dash application with dark theme"""
        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)
        
        # Define comprehensive layout
        self.app.layout = html.Div([
            # Header section
            html.Div([
                html.Div([
                    html.H1("FxAI Live Trading Dashboard", style={
                        'color': DARK_THEME['text_primary'], 
                        'margin': '0',
                        'fontSize': '28px',
                        'fontWeight': 'bold'
                    }),
                    html.P(id='header-info', style={
                        'color': DARK_THEME['text_secondary'],
                        'margin': '5px 0 0 0',
                        'fontSize': '14px'
                    })
                ], style={'textAlign': 'center', 'padding': '15px'})
            ], style={
                'backgroundColor': DARK_THEME['bg_secondary'],
                'border': f"1px solid {DARK_THEME['border']}",
                'borderRadius': '8px',
                'marginBottom': '15px'
            }),
            
            # Top row: Market Info, Position, Portfolio
            html.Div([
                # Market Info
                html.Div([
                    html.H3("Market Info", style={'color': DARK_THEME['text_primary'], 'fontSize': '16px', 'marginBottom': '10px'}),
                    html.Div(id='market-info', style={'fontSize': '13px'})
                ], style={
                    'backgroundColor': DARK_THEME['bg_secondary'],
                    'border': f"1px solid {DARK_THEME['border']}",
                    'borderRadius': '6px',
                    'padding': '12px',
                    'width': '32%',
                    'display': 'inline-block',
                    'verticalAlign': 'top',
                    'marginRight': '2%'
                }),
                
                # Position
                html.Div([
                    html.H3("Position", style={'color': DARK_THEME['text_primary'], 'fontSize': '16px', 'marginBottom': '10px'}),
                    html.Div(id='position-info', style={'fontSize': '13px'})
                ], style={
                    'backgroundColor': DARK_THEME['bg_secondary'],
                    'border': f"1px solid {DARK_THEME['border']}",
                    'borderRadius': '6px',
                    'padding': '12px',
                    'width': '32%',
                    'display': 'inline-block',
                    'verticalAlign': 'top',
                    'marginRight': '2%'
                }),
                
                # Portfolio
                html.Div([
                    html.H3("Portfolio", style={'color': DARK_THEME['text_primary'], 'fontSize': '16px', 'marginBottom': '10px'}),
                    html.Div(id='portfolio-info', style={'fontSize': '13px'})
                ], style={
                    'backgroundColor': DARK_THEME['bg_secondary'],
                    'border': f"1px solid {DARK_THEME['border']}",
                    'borderRadius': '6px',
                    'padding': '12px',
                    'width': '32%',
                    'display': 'inline-block',
                    'verticalAlign': 'top'
                })
            ], style={'marginBottom': '15px'}),
            
            # Middle row: Trades table and Actions panel
            html.Div([
                # Recent Trades
                html.Div([
                    html.H3("Recent Trades", style={'color': DARK_THEME['text_primary'], 'fontSize': '16px', 'marginBottom': '10px'}),
                    html.Div(id='trades-table')
                ], style={
                    'backgroundColor': DARK_THEME['bg_secondary'],
                    'border': f"1px solid {DARK_THEME['border']}",
                    'borderRadius': '6px',
                    'padding': '12px',
                    'width': '60%',
                    'display': 'inline-block',
                    'verticalAlign': 'top',
                    'marginRight': '2%'
                }),
                
                # Actions Analysis
                html.Div([
                    html.H3("Actions Analysis", style={'color': DARK_THEME['text_primary'], 'fontSize': '16px', 'marginBottom': '10px'}),
                    html.Div(id='actions-analysis', style={'fontSize': '13px'})
                ], style={
                    'backgroundColor': DARK_THEME['bg_secondary'],
                    'border': f"1px solid {DARK_THEME['border']}",
                    'borderRadius': '6px',
                    'padding': '12px',
                    'width': '36%',
                    'display': 'inline-block',
                    'verticalAlign': 'top'
                })
            ], style={'marginBottom': '15px'}),
            
            # Episode and Training row
            html.Div([
                # Episode Info
                html.Div([
                    html.H3("Episode Info", style={'color': DARK_THEME['text_primary'], 'fontSize': '16px', 'marginBottom': '10px'}),
                    html.Div(id='episode-info', style={'fontSize': '13px'})
                ], style={
                    'backgroundColor': DARK_THEME['bg_secondary'],
                    'border': f"1px solid {DARK_THEME['border']}",
                    'borderRadius': '6px',
                    'padding': '12px',
                    'width': '32%',
                    'display': 'inline-block',
                    'verticalAlign': 'top',
                    'marginRight': '2%'
                }),
                
                # Training Progress
                html.Div([
                    html.H3("Training Progress", style={'color': DARK_THEME['text_primary'], 'fontSize': '16px', 'marginBottom': '10px'}),
                    html.Div(id='training-progress', style={'fontSize': '13px'})
                ], style={
                    'backgroundColor': DARK_THEME['bg_secondary'],
                    'border': f"1px solid {DARK_THEME['border']}",
                    'borderRadius': '6px',
                    'padding': '12px',
                    'width': '32%',
                    'display': 'inline-block',
                    'verticalAlign': 'top',
                    'marginRight': '2%'
                }),
                
                # PPO Metrics
                html.Div([
                    html.H3("PPO Metrics", style={'color': DARK_THEME['text_primary'], 'fontSize': '16px', 'marginBottom': '10px'}),
                    html.Div(id='ppo-metrics', style={'fontSize': '13px'})
                ], style={
                    'backgroundColor': DARK_THEME['bg_secondary'],
                    'border': f"1px solid {DARK_THEME['border']}",
                    'borderRadius': '6px',
                    'padding': '12px',
                    'width': '32%',
                    'display': 'inline-block',
                    'verticalAlign': 'top'
                })
            ], style={'marginBottom': '15px'}),
            
            # Reward components and Environment info
            html.Div([
                # Reward Breakdown
                html.Div([
                    html.H3("Reward Components", style={'color': DARK_THEME['text_primary'], 'fontSize': '16px', 'marginBottom': '10px'}),
                    html.Div(id='reward-breakdown', style={'fontSize': '13px'})
                ], style={
                    'backgroundColor': DARK_THEME['bg_secondary'],
                    'border': f"1px solid {DARK_THEME['border']}",
                    'borderRadius': '6px',
                    'padding': '12px',
                    'width': '48%',
                    'display': 'inline-block',
                    'verticalAlign': 'top',
                    'marginRight': '4%'
                }),
                
                # Environment Info
                html.Div([
                    html.H3("Environment & Learning", style={'color': DARK_THEME['text_primary'], 'fontSize': '16px', 'marginBottom': '10px'}),
                    html.Div(id='environment-info', style={'fontSize': '13px'})
                ], style={
                    'backgroundColor': DARK_THEME['bg_secondary'],
                    'border': f"1px solid {DARK_THEME['border']}",
                    'borderRadius': '6px',
                    'padding': '12px',
                    'width': '48%',
                    'display': 'inline-block',
                    'verticalAlign': 'top'
                })
            ], style={'marginBottom': '15px'}),
            
            # Performance footer
            html.Div([
                html.Div(id='performance-footer', style={
                    'color': DARK_THEME['text_secondary'],
                    'fontSize': '12px',
                    'textAlign': 'center',
                    'padding': '10px'
                })
            ], style={
                'backgroundColor': DARK_THEME['bg_secondary'],
                'border': f"1px solid {DARK_THEME['border']}",
                'borderRadius': '6px',
                'marginBottom': '15px'
            }),
            
            # Full-width candlestick chart
            html.Div([
                dcc.Graph(id='candlestick-chart', style={'height': '400px'})
            ], style={
                'backgroundColor': DARK_THEME['bg_secondary'],
                'border': f"1px solid {DARK_THEME['border']}",
                'borderRadius': '6px',
                'padding': '15px'
            }),
            
            # Auto-refresh
            dcc.Interval(
                id='interval-component',
                interval=self.update_interval,
                n_intervals=0
            ),
        ], style={
            'backgroundColor': DARK_THEME['bg_primary'],
            'minHeight': '100vh',
            'padding': '15px',
            'fontFamily': 'Arial, sans-serif'
        })
        
        # Define callbacks
        self._setup_callbacks()
        
    def _setup_callbacks(self):
        """Setup comprehensive Dash callbacks for all dashboard components"""
        
        def format_color(value, positive_good=True):
            """Helper to format values with appropriate colors"""
            if positive_good:
                return DARK_THEME['accent_green'] if value >= 0 else DARK_THEME['accent_red']
            else:
                return DARK_THEME['accent_red'] if value >= 0 else DARK_THEME['accent_green']
        
        def format_pnl(value):
            """Format P&L with color and sign"""
            color = format_color(value)
            sign = '+' if value > 0 else ''
            return f"{sign}${value:.2f}", color
        
        def format_percentage(value):
            """Format percentage with color and sign"""
            color = format_color(value)
            sign = '+' if value > 0 else ''
            return f"{sign}{value:.2f}%", color
        
        @self.app.callback(
            Output('header-info', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_header(n):
            with self._lock:
                session_duration = datetime.now() - self.state.session_start_time
                hours, remainder = divmod(session_duration.total_seconds(), 3600)
                minutes, seconds = divmod(remainder, 60)
                duration_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
                
                return f"Model: {self.state.model_name} | Session: {duration_str} | Symbol: {self.state.symbol}"
        
        @self.app.callback(
            Output('market-info', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_market_info(n):
            with self._lock:
                return [
                    html.P([
                        html.Span("NY Time: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(self.state.ny_time or "00:00:00", style={'color': DARK_THEME['text_primary'], 'fontWeight': 'bold'})
                    ], style={'margin': '3px 0'}),
                    html.P([
                        html.Span("Status: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(self.state.trading_hours, style={
                            'color': DARK_THEME['accent_green'] if self.state.trading_hours == "MARKET HOURS" else DARK_THEME['accent_orange'],
                            'fontWeight': 'bold'
                        })
                    ], style={'margin': '3px 0'}),
                    html.P([
                        html.Span("Price: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(f"${self.state.current_price:.2f}", style={'color': DARK_THEME['text_primary'], 'fontWeight': 'bold'})
                    ], style={'margin': '3px 0'}),
                    html.P([
                        html.Span("Bid/Ask: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(f"${self.state.bid_price:.2f} / ${self.state.ask_price:.2f}", style={'color': DARK_THEME['text_primary']})
                    ], style={'margin': '3px 0'}),
                    html.P([
                        html.Span("Spread: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(f"${self.state.spread:.2f} ({self.state.spread_pct:.2f}%)", style={'color': DARK_THEME['text_primary']})
                    ], style={'margin': '3px 0'}),
                    html.P([
                        html.Span("Volume: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(f"{self.state.volume:,}", style={'color': DARK_THEME['text_primary']})
                    ], style={'margin': '3px 0'})
                ]
        
        @self.app.callback(
            Output('position-info', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_position_info(n):
            with self._lock:
                pnl_text, pnl_color = format_pnl(self.state.position_pnl_dollar)
                pnl_pct_text, _ = format_percentage(self.state.position_pnl_percent)
                
                side_color = {
                    'LONG': DARK_THEME['accent_green'],
                    'SHORT': DARK_THEME['accent_red'],
                    'FLAT': DARK_THEME['text_secondary']
                }.get(self.state.position_side, DARK_THEME['text_primary'])
                
                return [
                    html.P([
                        html.Span("Side: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(self.state.position_side, style={'color': side_color, 'fontWeight': 'bold'})
                    ], style={'margin': '3px 0'}),
                    html.P([
                        html.Span("Quantity: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(f"{self.state.position_qty:,}", style={'color': DARK_THEME['text_primary'], 'fontWeight': 'bold'})
                    ], style={'margin': '3px 0'}),
                    html.P([
                        html.Span("Avg Entry: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(f"${self.state.avg_entry_price:.2f}", style={'color': DARK_THEME['text_primary']})
                    ], style={'margin': '3px 0'}),
                    html.P([
                        html.Span("P&L: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(pnl_text, style={'color': pnl_color, 'fontWeight': 'bold'})
                    ], style={'margin': '3px 0'}),
                    html.P([
                        html.Span("P&L %: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(pnl_pct_text, style={'color': pnl_color, 'fontWeight': 'bold'})
                    ], style={'margin': '3px 0'})
                ]
        
        @self.app.callback(
            Output('portfolio-info', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_portfolio_info(n):
            with self._lock:
                session_pnl_text, session_color = format_pnl(self.state.session_pnl)
                realized_pnl_text, realized_color = format_pnl(self.state.realized_pnl)
                unrealized_pnl_text, unrealized_color = format_pnl(self.state.unrealized_pnl)
                
                return [
                    html.P([
                        html.Span("Total Equity: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(f"${self.state.total_equity:,.2f}", style={'color': DARK_THEME['text_primary'], 'fontWeight': 'bold'})
                    ], style={'margin': '3px 0'}),
                    html.P([
                        html.Span("Cash: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(f"${self.state.cash_balance:,.2f}", style={'color': DARK_THEME['text_primary']})
                    ], style={'margin': '3px 0'}),
                    html.P([
                        html.Span("Session P&L: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(session_pnl_text, style={'color': session_color, 'fontWeight': 'bold'})
                    ], style={'margin': '3px 0'}),
                    html.P([
                        html.Span("Realized: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(realized_pnl_text, style={'color': realized_color})
                    ], style={'margin': '3px 0'}),
                    html.P([
                        html.Span("Unrealized: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(unrealized_pnl_text, style={'color': unrealized_color})
                    ], style={'margin': '3px 0'}),
                    html.P([
                        html.Span("Max DD: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(f"{self.state.max_drawdown:.2f}%", style={'color': DARK_THEME['accent_red'] if self.state.max_drawdown > 0 else DARK_THEME['text_primary']})
                    ], style={'margin': '3px 0'}),
                    html.P([
                        html.Span("Sharpe: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(f"{self.state.sharpe_ratio:.2f}", style={'color': format_color(self.state.sharpe_ratio)})
                    ], style={'margin': '3px 0'}),
                    html.P([
                        html.Span("Win Rate: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(f"{self.state.win_rate:.1%}", style={'color': format_color(self.state.win_rate - 0.5)})
                    ], style={'margin': '3px 0'})
                ]
        
        @self.app.callback(
            Output('trades-table', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_trades_table(n):
            with self._lock:
                trades = list(self.state.recent_trades)
            
            if not trades:
                return html.P("No trades yet", style={'color': DARK_THEME['text_secondary'], 'textAlign': 'center'})
            
            # Create table data
            table_data = []
            for trade in trades[-10:]:  # Last 10 trades
                pnl = trade.get('pnl', 0)
                pnl_color = format_color(pnl)
                table_data.append({
                    'Time': trade['timestamp'].strftime('%H:%M:%S'),
                    'Side': trade.get('side', 'N/A'),
                    'Qty': f"{trade.get('quantity', 0):,}",
                    'Price': f"${trade.get('price', 0):.2f}",
                    'P&L': f"${pnl:.2f}"
                })
            
            return dash_table.DataTable(
                data=table_data,
                columns=[
                    {'name': 'Time', 'id': 'Time'},
                    {'name': 'Side', 'id': 'Side'},
                    {'name': 'Qty', 'id': 'Qty'},
                    {'name': 'Price', 'id': 'Price'},
                    {'name': 'P&L', 'id': 'P&L'}
                ],
                style_cell={
                    'backgroundColor': DARK_THEME['bg_tertiary'],
                    'color': DARK_THEME['text_primary'],
                    'fontSize': '11px',
                    'textAlign': 'center',
                    'border': f"1px solid {DARK_THEME['border']}",
                    'padding': '4px'
                },
                style_header={
                    'backgroundColor': DARK_THEME['bg_secondary'],
                    'color': DARK_THEME['text_primary'],
                    'fontWeight': 'bold',
                    'fontSize': '11px'
                },
                style_data_conditional=[
                    {
                        'if': {'filter_query': '{P&L} contains -'},
                        'backgroundColor': f'{DARK_THEME["accent_red"]}20',
                        'color': DARK_THEME['accent_red']
                    },
                    {
                        'if': {'filter_query': '{P&L} contains + || {P&L} > 0'},
                        'backgroundColor': f'{DARK_THEME["accent_green"]}20',
                        'color': DARK_THEME['accent_green']
                    }
                ]
            )
        
        @self.app.callback(
            Output('actions-analysis', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_actions_analysis(n):
            with self._lock:
                total_actions = sum(self.state.action_counts.values()) or 1
                hold_pct = (self.state.action_counts.get('HOLD', 0) / total_actions) * 100
                buy_pct = (self.state.action_counts.get('BUY', 0) / total_actions) * 100
                sell_pct = (self.state.action_counts.get('SELL', 0) / total_actions) * 100
                
                recent_actions = list(self.state.recent_actions)[-5:]  # Last 5 actions
                
                return [
                    html.P("Action Distribution:", style={'color': DARK_THEME['text_primary'], 'fontWeight': 'bold', 'margin': '5px 0'}),
                    html.P([
                        html.Span("HOLD: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(f"{hold_pct:.1f}%", style={'color': DARK_THEME['accent_blue']})
                    ], style={'margin': '3px 0'}),
                    html.P([
                        html.Span("BUY: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(f"{buy_pct:.1f}%", style={'color': DARK_THEME['accent_green']})
                    ], style={'margin': '3px 0'}),
                    html.P([
                        html.Span("SELL: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(f"{sell_pct:.1f}%", style={'color': DARK_THEME['accent_red']})
                    ], style={'margin': '3px 0'}),
                    html.Hr(style={'border': f"1px solid {DARK_THEME['border']}", 'margin': '10px 0'}),
                    html.P("Recent Actions:", style={'color': DARK_THEME['text_primary'], 'fontWeight': 'bold', 'margin': '5px 0'}),
                    *([
                        html.P(f"{action.get('action_type', 'HOLD')} @ {action['timestamp'].strftime('%H:%M:%S')}", 
                               style={'color': DARK_THEME['text_secondary'], 'fontSize': '11px', 'margin': '2px 0'})
                        for action in recent_actions
                    ] if recent_actions else [html.P("No actions yet", style={'color': DARK_THEME['text_muted'], 'fontSize': '11px'})])
                ]
        
        @self.app.callback(
            [Output('episode-info', 'children'),
             Output('training-progress', 'children'),
             Output('ppo-metrics', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_training_sections(n):
            with self._lock:
                # Episode info
                episode_progress = (self.state.current_step / max(self.state.max_steps, 1)) * 100
                episode_info = [
                    html.P([
                        html.Span("Episode: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(f"#{self.state.episode_number}", style={'color': DARK_THEME['text_primary'], 'fontWeight': 'bold'})
                    ], style={'margin': '3px 0'}),
                    html.P([
                        html.Span("Step: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(f"{self.state.current_step}/{self.state.max_steps}", style={'color': DARK_THEME['text_primary']})
                    ], style={'margin': '3px 0'}),
                    html.P([
                        html.Span("Progress: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(f"{episode_progress:.1f}%", style={'color': DARK_THEME['accent_blue']})
                    ], style={'margin': '3px 0'}),
                    html.P([
                        html.Span("Cum. Reward: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(f"{self.state.cumulative_reward:.2f}", style={'color': format_color(self.state.cumulative_reward), 'fontWeight': 'bold'})
                    ], style={'margin': '3px 0'}),
                    html.P([
                        html.Span("Last Reward: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(f"{self.state.last_step_reward:.3f}", style={'color': format_color(self.state.last_step_reward)})
                    ], style={'margin': '3px 0'})
                ]
                
                # Training progress
                training_progress = [
                    html.P([
                        html.Span("Mode: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(self.state.mode, style={'color': DARK_THEME['accent_green'] if self.state.mode == 'Training' else DARK_THEME['accent_orange'], 'fontWeight': 'bold'})
                    ], style={'margin': '3px 0'}),
                    html.P([
                        html.Span("Stage: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(self.state.stage, style={'color': DARK_THEME['text_primary']})
                    ], style={'margin': '3px 0'}),
                    html.P([
                        html.Span("Progress: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(f"{self.state.overall_progress:.1f}%", style={'color': DARK_THEME['accent_blue']})
                    ], style={'margin': '3px 0'}),
                    html.P([
                        html.Span("Episodes: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(f"{self.state.total_episodes:,}", style={'color': DARK_THEME['text_primary']})
                    ], style={'margin': '3px 0'}),
                    html.P([
                        html.Span("Updates: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(f"{self.state.updates:,}", style={'color': DARK_THEME['text_primary']})
                    ], style={'margin': '3px 0'}),
                    html.P([
                        html.Span("Global Steps: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(f"{self.state.global_steps:,}", style={'color': DARK_THEME['text_primary']})
                    ], style={'margin': '3px 0'})
                ]
                
                # PPO metrics
                ppo_metrics = [
                    html.P([
                        html.Span("Policy Loss: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(f"{self.state.policy_loss:.4f}", style={'color': DARK_THEME['text_primary']})
                    ], style={'margin': '3px 0'}),
                    html.P([
                        html.Span("Value Loss: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(f"{self.state.value_loss:.4f}", style={'color': DARK_THEME['text_primary']})
                    ], style={'margin': '3px 0'}),
                    html.P([
                        html.Span("Entropy: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(f"{self.state.entropy:.4f}", style={'color': DARK_THEME['text_primary']})
                    ], style={'margin': '3px 0'}),
                    html.P([
                        html.Span("Clip Fraction: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(f"{self.state.clip_fraction:.3f}", style={'color': DARK_THEME['text_primary']})
                    ], style={'margin': '3px 0'}),
                    html.P([
                        html.Span("Approx KL: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(f"{self.state.approx_kl:.4f}", style={'color': DARK_THEME['text_primary']})
                    ], style={'margin': '3px 0'}),
                    html.P([
                        html.Span("Learning Rate: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(f"{self.state.learning_rate:.2e}", style={'color': DARK_THEME['text_primary']})
                    ], style={'margin': '3px 0'})
                ]
                
                return episode_info, training_progress, ppo_metrics
        
        @self.app.callback(
            [Output('reward-breakdown', 'children'),
             Output('environment-info', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_reward_and_environment(n):
            with self._lock:
                # Reward breakdown
                components = dict(self.state.reward_components)
                if components:
                    reward_breakdown = [
                        html.P("Component Values:", style={'color': DARK_THEME['text_primary'], 'fontWeight': 'bold', 'margin': '5px 0'}),
                        *[
                            html.P([
                                html.Span(f"{comp}: ", style={'color': DARK_THEME['text_secondary']}),
                                html.Span(f"{value:.3f}", style={'color': format_color(value), 'fontSize': '11px'})
                            ], style={'margin': '2px 0'})
                            for comp, value in list(components.items())[:8]  # Show top 8 components
                        ]
                    ]
                else:
                    reward_breakdown = [html.P("No reward data yet", style={'color': DARK_THEME['text_muted']})]
                
                # Environment info
                environment_info = [
                    html.P("Curriculum Learning:", style={'color': DARK_THEME['text_primary'], 'fontWeight': 'bold', 'margin': '5px 0'}),
                    html.P([
                        html.Span("Stage: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(self.state.curriculum_stage, style={'color': DARK_THEME['accent_purple']})
                    ], style={'margin': '3px 0'}),
                    html.P([
                        html.Span("Focus: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(self.state.learning_focus, style={'color': DARK_THEME['text_primary']})
                    ], style={'margin': '3px 0'}),
                    html.P([
                        html.Span("Difficulty: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(f"{self.state.difficulty_level:.1%}", style={'color': DARK_THEME['accent_blue']})
                    ], style={'margin': '3px 0'}),
                    html.Hr(style={'border': f"1px solid {DARK_THEME['border']}", 'margin': '10px 0'}),
                    html.P("Environment:", style={'color': DARK_THEME['text_primary'], 'fontWeight': 'bold', 'margin': '5px 0'}),
                    html.P([
                        html.Span("Data Quality: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(f"{self.state.data_quality:.1%}", style={'color': format_color(self.state.data_quality - 0.9)})
                    ], style={'margin': '3px 0'}),
                    html.P([
                        html.Span("Momentum: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(f"{self.state.momentum_score:.2f}", style={'color': format_color(self.state.momentum_score)})
                    ], style={'margin': '3px 0'}),
                    html.P([
                        html.Span("Volatility: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(f"{self.state.volatility:.1%}", style={'color': DARK_THEME['accent_orange']})
                    ], style={'margin': '3px 0'})
                ]
                
                return reward_breakdown, environment_info
        
        @self.app.callback(
            Output('performance-footer', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_performance_footer(n):
            with self._lock:
                return f"Performance: {self.state.steps_per_second:.1f} steps/sec | {self.state.updates_per_second:.2f} updates/sec | {self.state.episodes_per_hour:.1f} episodes/hr | Avg Update: {self.state.time_per_update:.2f}s | Avg Episode: {self.state.time_per_episode:.1f}s"
        
        @self.app.callback(
            Output('candlestick-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_candlestick_chart(n):
            with self._lock:
                price_data = list(self.state.price_history)
                trades = list(self.state.trades)
                rewards = list(self.state.reward_history_ts)
            
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=('Price & Trades', 'Reward History'),
                row_heights=[0.7, 0.3]
            )
            
            if price_data:
                # Extract price data
                times = [d['time'] for d in price_data]
                prices = [d['price'] for d in price_data]
                
                # Price line
                fig.add_trace(go.Scatter(
                    x=times, y=prices,
                    mode='lines',
                    name='Price',
                    line=dict(color=DARK_THEME['accent_blue'], width=2)
                ), row=1, col=1)
                
                # Add trade markers
                if trades:
                    trade_times = [t['timestamp'] for t in trades if 'timestamp' in t]
                    trade_prices = [t.get('price', 0) for t in trades if 'timestamp' in t]
                    trade_sides = [t.get('side', 'HOLD') for t in trades if 'timestamp' in t]
                    
                    buy_times = [t for i, t in enumerate(trade_times) if trade_sides[i] == 'BUY']
                    buy_prices = [p for i, p in enumerate(trade_prices) if trade_sides[i] == 'BUY']
                    sell_times = [t for i, t in enumerate(trade_times) if trade_sides[i] == 'SELL']
                    sell_prices = [p for i, p in enumerate(trade_prices) if trade_sides[i] == 'SELL']
                    
                    if buy_times:
                        fig.add_trace(go.Scatter(
                            x=buy_times, y=buy_prices,
                            mode='markers',
                            name='Buy',
                            marker=dict(color=DARK_THEME['accent_green'], size=8, symbol='triangle-up')
                        ), row=1, col=1)
                    
                    if sell_times:
                        fig.add_trace(go.Scatter(
                            x=sell_times, y=sell_prices,
                            mode='markers',
                            name='Sell',
                            marker=dict(color=DARK_THEME['accent_red'], size=8, symbol='triangle-down')
                        ), row=1, col=1)
            
            # Reward history
            if rewards:
                reward_times = [r['time'] for r in rewards]
                reward_values = [r['value'] for r in rewards]
                
                fig.add_trace(go.Scatter(
                    x=reward_times, y=reward_values,
                    mode='lines',
                    name='Episode Reward',
                    line=dict(color=DARK_THEME['accent_purple'], width=2)
                ), row=2, col=1)
            
            # Update layout with dark theme
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor=DARK_THEME['bg_secondary'],
                plot_bgcolor=DARK_THEME['bg_tertiary'],
                font_color=DARK_THEME['text_primary'],
                showlegend=True,
                height=400,
                margin=dict(l=40, r=40, t=40, b=40)
            )
            
            fig.update_xaxes(gridcolor=DARK_THEME['border'])
            fig.update_yaxes(gridcolor=DARK_THEME['border'])
            
            return fig
        
    def start(self, open_browser: bool = True):
        """Start the dashboard"""
        if self.is_running:
            return
            
        try:
            # Create Dash app
            self._create_app()
            
            # Start server in background thread
            self._server_thread = threading.Thread(
                target=self._run_server,
                daemon=True
            )
            self._server_thread.start()
            
            self.is_running = True
            time.sleep(1)  # Give server time to start
            
            self.logger.info(f"📊 Dashboard started on port {self.port}")
            self.logger.info(f"🌐 Access at http://localhost:{self.port}")
            
            if open_browser:
                webbrowser.open(f"http://localhost:{self.port}")
                
        except Exception as e:
            self.logger.error(f"Failed to start dashboard: {e}")
            self.is_running = False
            
    def _run_server(self):
        """Run the Dash server"""
        try:
            self.app.run(
                host='0.0.0.0',
                port=self.port,
                debug=False,
                use_reloader=False,
                dev_tools_silence_routes_logging=True
            )
        except Exception as e:
            self.logger.error(f"Dashboard server error: {e}")
            self.is_running = False
            
    def stop(self):
        """Stop the dashboard"""
        if self.is_running:
            self.is_running = False
            self.logger.info("📊 Dashboard stopped")
            
    def update_training_state(self, data: Dict[str, Any]):
        """Update training state"""
        if self.is_running:
            with self._lock:
                self.state.update_training_state(data)
    
    def update_trade_data(self, trade_data: Dict[str, Any]):
        """Update trade data"""
        if self.is_running:
            with self._lock:
                self.state.update_trade_data(trade_data)
    
    def update_episode_data(self, episode_data: Dict[str, Any]):
        """Update episode data"""
        if self.is_running:
            with self._lock:
                self.state.update_episode_data(episode_data)
    
    def update_reward_components(self, components: Dict[str, float]):
        """Update reward components"""
        if self.is_running:
            with self._lock:
                self.state.update_reward_components(components)
            
    def update_momentum_day(self, momentum_day: MomentumDay):
        """Update momentum day"""
        if self.is_running:
            with self._lock:
                self._momentum_day = momentum_day
            
    def update_curriculum_progress(self, progress: float, strategy: Optional[str] = None):
        """Update curriculum progress"""
        if self.is_running:
            with self._lock:
                self._curriculum_progress = progress
                self._curriculum_strategy = strategy
                self.state.curriculum_progress = progress
                if strategy:
                    self.state.learning_focus = strategy
    
    def update_market_data(self, market_data: Dict[str, Any]):
        """Update market data from external sources"""
        if self.is_running:
            with self._lock:
                self.state.update_market_data(market_data)
    
    def update_position_data(self, position_data: Dict[str, Any]):
        """Update position data"""
        if self.is_running:
            with self._lock:
                for key, value in position_data.items():
                    if hasattr(self.state, key):
                        setattr(self.state, key, value)
    
    def update_portfolio_data(self, portfolio_data: Dict[str, Any]):
        """Update portfolio data"""
        if self.is_running:
            with self._lock:
                for key, value in portfolio_data.items():
                    if hasattr(self.state, key):
                        setattr(self.state, key, value)
    
    def update_episode_info(self, episode_info: Dict[str, Any]):
        """Update episode information"""
        if self.is_running:
            with self._lock:
                for key, value in episode_info.items():
                    if hasattr(self.state, key):
                        setattr(self.state, key, value)
    
    def update_action_data(self, action_data: Dict[str, Any]):
        """Update action tracking data"""
        if self.is_running:
            with self._lock:
                self.state.update_action_data(action_data)
    
    def update_environment_data(self, env_data: Dict[str, Any]):
        """Update environment data"""
        if self.is_running:
            with self._lock:
                for key, value in env_data.items():
                    if hasattr(self.state, key):
                        setattr(self.state, key, value)
            
    def is_dashboard_running(self) -> bool:
        """Check if dashboard is running"""
        return self.is_running
        
    def get_state_summary(self) -> Dict[str, Any]:
        """Get current state summary"""
        with self._lock:
            return {
                'mode': self.state.mode,
                'stage': self.state.stage,
                'updates': self.state.updates,
                'global_steps': self.state.global_steps,
                'total_episodes': self.state.total_episodes,
                'mean_reward': self.state.mean_episode_reward,
                'momentum_day': self._momentum_day.symbol if self._momentum_day else None,
                'curriculum_progress': self._curriculum_progress,
                'is_running': self.is_running,
                'current_price': self.state.current_price,
                'position_side': self.state.position_side,
                'session_pnl': self.state.session_pnl,
                'total_trades': len(self.state.trades)
            }