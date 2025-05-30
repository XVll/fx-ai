# dashboard/dashboard_server.py - Dash web server implementation

import logging
import threading
import webbrowser
from typing import Dict, Any, Optional, List
from datetime import datetime

import dash
from dash import dcc, html, Input, Output, State, dash_table, callback_context
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

from .shared_state import dashboard_state

# Dark theme colors (GitHub-inspired)
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

class DashboardServer:
    """Dash-based dashboard server reading from shared state"""
    
    def __init__(self, port: int = 8050):
        self.port = port
        self.app = None
        self.logger = logging.getLogger(__name__)
        
    def create_app(self) -> dash.Dash:
        """Create and configure the Dash application"""
        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)
        
        # Configure app
        self.app.title = "FxAI Trading Dashboard"
        
        # Create layout
        self.app.layout = self._create_layout()
        
        # Setup callbacks
        self._setup_callbacks()
        
        return self.app
        
    def _create_layout(self) -> html.Div:
        """Create the dashboard layout with dark theme"""
        return html.Div([
            # Header
            html.Div([
                html.Div([
                    html.H2("FxAI Trading Dashboard", style={'margin': '0', 'color': DARK_THEME['text_primary'], 'fontSize': '18px'}),
                    html.Div(id='header-info', style={'color': DARK_THEME['text_secondary'], 'fontSize': '12px'})
                ], style={'flex': '1'}),
                html.Div([
                    html.Span("Session Time: ", style={'color': DARK_THEME['text_secondary']}),
                    html.Span(id='session-time', style={'color': DARK_THEME['accent_blue']})
                ])
            ], style={
                'display': 'flex',
                'justifyContent': 'space-between',
                'alignItems': 'center',
                'padding': '8px 12px',
                'backgroundColor': DARK_THEME['bg_secondary'],
                'borderBottom': f"1px solid {DARK_THEME['border']}"
            }),
            
            # Main content area with 4-column grid
            html.Div([
                # Row 1: 4-column layout - Market Info, Position, Portfolio, Actions
                html.Div([
                    # Market Info Card
                    html.Div([
                        html.H4("Market Info", style={'color': DARK_THEME['text_primary'], 'marginBottom': '4px', 'fontSize': '12px', 'fontWeight': 'bold'}),
                        html.Div(id='market-info-content')
                    ], style=self._card_style()),
                    
                    # Position Card
                    html.Div([
                        html.H4("Position", style={'color': DARK_THEME['text_primary'], 'marginBottom': '4px', 'fontSize': '12px', 'fontWeight': 'bold'}),
                        html.Div(id='position-content')
                    ], style=self._card_style()),
                    
                    # Portfolio Card
                    html.Div([
                        html.H4("Portfolio", style={'color': DARK_THEME['text_primary'], 'marginBottom': '4px', 'fontSize': '12px', 'fontWeight': 'bold'}),
                        html.Div(id='portfolio-content')
                    ], style=self._card_style()),
                    
                    # Actions Card
                    html.Div([
                        html.H4("Actions", style={'color': DARK_THEME['text_primary'], 'marginBottom': '4px', 'fontSize': '12px', 'fontWeight': 'bold'}),
                        html.Div(id='actions-content')
                    ], style=self._card_style()),
                ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr 1fr 1fr', 'gap': '6px', 'marginBottom': '6px'}),
                
                # Row 2: 4-column layout - Episode, Training, PPO, Environment
                html.Div([
                    # Episode Info Card
                    html.Div([
                        html.H4("Episode", style={'color': DARK_THEME['text_primary'], 'marginBottom': '4px', 'fontSize': '12px', 'fontWeight': 'bold'}),
                        html.Div(id='episode-content')
                    ], style=self._card_style()),
                    
                    # Training Progress Card
                    html.Div([
                        html.H4("Training", style={'color': DARK_THEME['text_primary'], 'marginBottom': '4px', 'fontSize': '12px', 'fontWeight': 'bold'}),
                        html.Div(id='training-content')
                    ], style=self._card_style()),
                    
                    # PPO Metrics Card
                    html.Div([
                        html.H4("PPO", style={'color': DARK_THEME['text_primary'], 'marginBottom': '4px', 'fontSize': '12px', 'fontWeight': 'bold'}),
                        html.Div(id='ppo-metrics-content')
                    ], style=self._card_style()),
                    
                    # Environment/Curriculum Card
                    html.Div([
                        html.H4("Quality", style={'color': DARK_THEME['text_primary'], 'marginBottom': '4px', 'fontSize': '12px', 'fontWeight': 'bold'}),
                        html.Div(id='env-content')
                    ], style=self._card_style()),
                ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr 1fr 1fr', 'gap': '6px', 'marginBottom': '6px'}),
                
                # Row 3: Split layout - Trades table + Reward chart
                html.Div([
                    # Recent Trades Card
                    html.Div([
                        html.H4("Trades", style={'color': DARK_THEME['text_primary'], 'marginBottom': '4px', 'fontSize': '12px', 'fontWeight': 'bold'}),
                        html.Div(id='trades-table-container')
                    ], style=self._card_style()),
                    
                    # Reward Components Card
                    html.Div([
                        html.H4("Rewards", style={'color': DARK_THEME['text_primary'], 'marginBottom': '4px', 'fontSize': '12px', 'fontWeight': 'bold'}),
                        html.Div(id='reward-table-container')
                    ], style=self._card_style()),
                ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '6px', 'marginBottom': '6px'}),
                
                # Row 4: Full-width Chart
                html.Div([
                    html.H4("Price & Volume", style={'color': DARK_THEME['text_primary'], 'marginBottom': '4px', 'fontSize': '12px', 'fontWeight': 'bold'}),
                    dcc.Graph(id='candlestick-chart', style={'height': '300px'})
                ], style=self._card_style()),
                
            ], style={'padding': '6px', 'backgroundColor': DARK_THEME['bg_primary']}),
            
            # Footer
            html.Div([
                html.Div(id='performance-footer', style={'color': DARK_THEME['text_secondary']})
            ], style={
                'padding': '4px 8px',
                'backgroundColor': DARK_THEME['bg_secondary'],
                'borderTop': f"1px solid {DARK_THEME['border']}",
                'textAlign': 'center',
                'fontSize': '11px'
            }),
            
            # Auto-refresh interval
            dcc.Interval(id='interval-component', interval=500, n_intervals=0)  # 500ms refresh
            
        ], style={'backgroundColor': DARK_THEME['bg_primary'], 'minHeight': '100vh'})
        
    def _card_style(self) -> Dict[str, str]:
        """Standard card styling"""
        return {
            'backgroundColor': DARK_THEME['bg_secondary'],
            'border': f"1px solid {DARK_THEME['border']}",
            'borderRadius': '3px',
            'padding': '6px 8px'
        }
        
    def _setup_callbacks(self):
        """Setup all dashboard callbacks"""
        
        @self.app.callback(
            [Output('header-info', 'children'),
             Output('session-time', 'children'),
             Output('market-info-content', 'children'),
             Output('position-content', 'children'),
             Output('portfolio-content', 'children'),
             Output('trades-table-container', 'children'),
             Output('actions-content', 'children'),
             Output('episode-content', 'children'),
             Output('training-content', 'children'),
             Output('ppo-metrics-content', 'children'),
             Output('reward-table-container', 'children'),
             Output('env-content', 'children'),
             Output('candlestick-chart', 'figure'),
             Output('performance-footer', 'children')],
            Input('interval-component', 'n_intervals')
        )
        def update_dashboard(n):
            """Main update callback"""
            state = dashboard_state.get_state()
            
            # Calculate session time
            session_duration = datetime.now() - state.session_start_time
            hours, remainder = divmod(int(session_duration.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            session_time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            # Header info
            header_info = f"Model: {state.model_name} | Symbol: {state.symbol}"
            
            # Market info
            spread = state.ask_price - state.bid_price
            ny_time = getattr(state, 'ny_time', datetime.now().strftime('%H:%M:%S'))
            trading_hours = getattr(state, 'trading_hours', 'MARKET')
            market_info = html.Div([
                self._info_row("Time", ny_time),
                self._info_row("Session", trading_hours, color=DARK_THEME['accent_orange']),
                self._info_row("Price", f"${state.current_price:.2f}", color=DARK_THEME['accent_blue']),
                self._info_row("Bid", f"${state.bid_price:.2f}"),
                self._info_row("Ask", f"${state.ask_price:.2f}"),
                self._info_row("Spread", f"${spread:.3f}"),
                self._info_row("Volume", f"{state.volume:,}"),
            ])
            
            # Position info
            if state.position_side != "FLAT":
                position_pnl = getattr(state, 'position_pnl_dollar', 0.0)
                pnl_color = DARK_THEME['accent_green'] if position_pnl >= 0 else DARK_THEME['accent_red']
                position_info = html.Div([
                    self._info_row("Side", state.position_side, color=DARK_THEME['accent_blue']),
                    self._info_row("Qty", f"{state.position_qty:,}"),
                    self._info_row("Entry", f"${state.avg_entry_price:.3f}"),
                    self._info_row("P&L $", f"${position_pnl:.2f}", color=pnl_color),
                    self._info_row("P&L %", f"{state.position_pnl_percent:.1f}%", color=pnl_color),
                    self._info_row("Hold Time", f"{getattr(state, 'position_hold_time_minutes', 0):.0f}m"),
                ])
            else:
                position_info = html.Div([
                    self._info_row("Side", "FLAT", color=DARK_THEME['text_muted']),
                    self._info_row("Qty", "0"),
                    self._info_row("Entry", "-"),
                    self._info_row("P&L $", "$0.00"),
                    self._info_row("P&L %", "0.0%"),
                    self._info_row("Hold Time", "0m"),
                ])
            
            # Portfolio info
            session_pnl_color = DARK_THEME['accent_green'] if state.session_pnl >= 0 else DARK_THEME['accent_red']
            portfolio_info = html.Div([
                self._info_row("Equity", f"${state.total_equity:.0f}"),
                self._info_row("Cash", f"${state.cash_balance:.0f}"),
                self._info_row("Session P&L", f"${state.session_pnl:.2f}", color=session_pnl_color),
                self._info_row("Realized", f"${state.realized_pnl:.2f}"),
                self._info_row("Drawdown", f"{state.max_drawdown:.1%}", color=DARK_THEME['accent_orange']),
                self._info_row("Sharpe", f"{state.sharpe_ratio:.2f}"),
                self._info_row("Win Rate", f"{state.win_rate:.0%}"),
            ])
            
            # Recent trades table
            if state.recent_trades:
                trades_df = pd.DataFrame(list(state.recent_trades)[-10:])  # Last 10 trades
                trades_table = dash_table.DataTable(
                    data=trades_df.to_dict('records'),
                    columns=[
                        {'name': 'Time', 'id': 'timestamp'},
                        {'name': 'Side', 'id': 'side'},
                        {'name': 'Qty', 'id': 'quantity'},
                        {'name': 'Price', 'id': 'fill_price', 'type': 'numeric', 'format': {'specifier': '$.3f'}},
                        {'name': 'P&L', 'id': 'pnl', 'type': 'numeric', 'format': {'specifier': '$.2f'}}
                    ],
                    style_cell={
                        'backgroundColor': DARK_THEME['bg_tertiary'],
                        'color': DARK_THEME['text_primary'],
                        'border': f"1px solid {DARK_THEME['border']}"
                    },
                    style_data_conditional=[
                        {
                            'if': {'column_id': 'pnl', 'filter_query': '{pnl} > 0'},
                            'color': DARK_THEME['accent_green']
                        },
                        {
                            'if': {'column_id': 'pnl', 'filter_query': '{pnl} < 0'},
                            'color': DARK_THEME['accent_red']
                        },
                        {
                            'if': {'column_id': 'side', 'filter_query': '{side} = BUY'},
                            'color': DARK_THEME['accent_green']
                        },
                        {
                            'if': {'column_id': 'side', 'filter_query': '{side} = SELL'},
                            'color': DARK_THEME['accent_red']
                        }
                    ],
                    style_header={
                        'backgroundColor': DARK_THEME['bg_secondary'],
                        'color': DARK_THEME['text_secondary'],
                        'fontWeight': 'bold'
                    },
                    page_size=10
                )
            else:
                trades_table = html.Div("No trades yet", style={'color': DARK_THEME['text_muted'], 'textAlign': 'center', 'padding': '20px'})
            
            # Actions table similar to rewards
            episode_actions = getattr(state, 'episode_action_distribution', state.action_distribution)
            session_actions = getattr(state, 'session_action_distribution', state.action_distribution)
            
            action_data = []
            for action_type in ['HOLD', 'BUY', 'SELL']:
                episode_count = episode_actions.get(action_type, 0)
                session_count = session_actions.get(action_type, 0)
                
                # Calculate percentages
                episode_total = sum(episode_actions.values()) or 1
                session_total = sum(session_actions.values()) or 1
                episode_pct = episode_count / episode_total * 100
                session_pct = session_count / session_total * 100
                
                action_data.append({
                    'Action': action_type,
                    'Episode': f"{episode_count}",
                    'Episode %': f"{episode_pct:.1f}%",
                    'Session': f"{session_count}",
                    'Session %': f"{session_pct:.1f}%"
                })
            
            actions_table = dash_table.DataTable(
                data=action_data,
                columns=[
                    {'name': 'Action', 'id': 'Action'},
                    {'name': 'Episode', 'id': 'Episode', 'type': 'numeric'},
                    {'name': 'Ep %', 'id': 'Episode %'},
                    {'name': 'Session', 'id': 'Session', 'type': 'numeric'},
                    {'name': 'Sess %', 'id': 'Session %'}
                ],
                style_cell={
                    'backgroundColor': DARK_THEME['bg_tertiary'],
                    'color': DARK_THEME['text_primary'],
                    'border': f"1px solid {DARK_THEME['border']}",
                    'fontSize': '11px',
                    'padding': '4px 6px',
                    'textAlign': 'left'
                },
                style_data_conditional=[
                    {
                        'if': {'column_id': 'Action', 'filter_query': '{Action} = HOLD'},
                        'color': DARK_THEME['accent_blue']
                    },
                    {
                        'if': {'column_id': 'Action', 'filter_query': '{Action} = BUY'},
                        'color': DARK_THEME['accent_green']
                    },
                    {
                        'if': {'column_id': 'Action', 'filter_query': '{Action} = SELL'},
                        'color': DARK_THEME['accent_red']
                    }
                ],
                style_header={
                    'backgroundColor': DARK_THEME['bg_secondary'],
                    'color': DARK_THEME['text_secondary'],
                    'fontWeight': 'bold',
                    'fontSize': '10px'
                },
                page_size=10
            )
            
            actions_content = actions_table
            
            # Episode info - more compact
            progress = (state.current_step / state.max_steps * 100) if state.max_steps > 0 else 0
            episode_content = html.Div([
                self._info_row("Episode", f"{state.episode_number}"),
                self._info_row("Step", f"{state.current_step:,}/{state.max_steps:,}"),
                self._info_row("Progress", f"{progress:.1f}%"),
                self._info_row("Cum. Reward", f"{state.cumulative_reward:.2f}"),
                self._info_row("Step Reward", f"{state.last_step_reward:.3f}"),
                # Compact progress bar
                html.Div([
                    html.Div(style={
                        'backgroundColor': DARK_THEME['bg_tertiary'],
                        'height': '6px',
                        'borderRadius': '3px',
                        'marginTop': '4px'
                    }, children=[
                        html.Div(style={
                            'backgroundColor': DARK_THEME['accent_blue'],
                            'height': '100%',
                            'width': f"{progress}%",
                            'borderRadius': '3px',
                            'transition': 'width 0.3s ease'
                        })
                    ])
                ])
            ])
            
            # Training progress
            invalid_actions = getattr(state, 'invalid_actions', 0)
            eps_per_hour = getattr(state, 'episodes_per_hour', 0.0)
            training_content = html.Div([
                self._info_row("Mode", state.mode, color=DARK_THEME['accent_purple']),
                self._info_row("Stage", state.stage),
                self._info_row("Episodes", f"{state.total_episodes:,}"),
                self._info_row("Global Steps", f"{state.global_steps:,}"),
                self._info_row("Eps/Hour", f"{eps_per_hour:.1f}"),
                self._info_row("Invalid", f"{invalid_actions}", color=DARK_THEME['accent_orange'] if invalid_actions > 0 else None),
            ])
            
            # PPO Metrics - simplified
            ppo_content = html.Div([
                self._info_row("Policy Loss", f"{state.policy_loss:.4f}"),
                self._info_row("Value Loss", f"{state.value_loss:.4f}"),
                self._info_row("Entropy", f"{state.entropy:.4f}"),
                self._info_row("Learn Rate", f"{state.learning_rate:.2e}"),
                self._info_row("KL Div", f"{getattr(state, 'kl_divergence', 0.0):.4f}"),
                self._info_row("Clip Frac", f"{getattr(state, 'clip_fraction', 0.0):.2f}"),
            ])
            
            # Reward components table with episode vs session stats
            if state.reward_components:
                # Get episode and session reward data
                episode_rewards = getattr(state, 'episode_reward_components', {})
                session_rewards = getattr(state, 'session_reward_components', {})
                
                # Use all components from either current, episode, or session
                all_components = set(state.reward_components.keys()) | set(episode_rewards.keys()) | set(session_rewards.keys())
                
                reward_data = []
                for component in sorted(all_components):
                    episode_value = episode_rewards.get(component, 0.0)
                    session_total = session_rewards.get(component, 0.0)
                    
                    # Estimate episode count for mean calculation (rough approximation)
                    episodes_estimate = max(1, state.total_episodes)
                    session_mean = session_total / episodes_estimate if episodes_estimate > 0 else session_total
                    
                    reward_data.append({
                        'Component': component,
                        'Episode': f"{episode_value:.3f}",
                        'Session Total': f"{session_total:.2f}",
                        'Session Mean': f"{session_mean:.3f}",
                        'Count': str(episodes_estimate)
                    })
                
                reward_table = dash_table.DataTable(
                    data=reward_data,
                    columns=[
                        {'name': 'Component', 'id': 'Component'},
                        {'name': 'Episode', 'id': 'Episode', 'type': 'numeric'},
                        {'name': 'Sess Total', 'id': 'Session Total', 'type': 'numeric'},
                        {'name': 'Sess Mean', 'id': 'Session Mean', 'type': 'numeric'},
                        {'name': 'Count', 'id': 'Count', 'type': 'numeric'}
                    ],
                    style_cell={
                        'backgroundColor': DARK_THEME['bg_tertiary'],
                        'color': DARK_THEME['text_primary'],
                        'border': f"1px solid {DARK_THEME['border']}",
                        'fontSize': '11px',
                        'padding': '4px 6px',
                        'textAlign': 'left'
                    },
                    style_data_conditional=[
                        {
                            'if': {'column_id': 'Episode', 'filter_query': '{Episode} > 0'},
                            'color': DARK_THEME['accent_green']
                        },
                        {
                            'if': {'column_id': 'Episode', 'filter_query': '{Episode} < 0'},
                            'color': DARK_THEME['accent_red']
                        },
                        {
                            'if': {'column_id': 'Session Mean', 'filter_query': '{Session Mean} > 0'},
                            'color': DARK_THEME['accent_green']
                        },
                        {
                            'if': {'column_id': 'Session Mean', 'filter_query': '{Session Mean} < 0'},
                            'color': DARK_THEME['accent_red']
                        }
                    ],
                    style_header={
                        'backgroundColor': DARK_THEME['bg_secondary'],
                        'color': DARK_THEME['text_secondary'],
                        'fontWeight': 'bold',
                        'fontSize': '10px'
                    },
                    page_size=10
                )
            else:
                reward_table = html.Div("No reward data", style={'color': DARK_THEME['text_muted'], 'textAlign': 'center', 'padding': '20px'})
            
            # Environment/Curriculum info with expanded details
            avg_spread = getattr(state, 'avg_spread', getattr(state, 'spread', 0.001))
            volume_ratio = getattr(state, 'volume_ratio', 1.0)
            halt_count = getattr(state, 'halt_count', 0)
            is_front_side = getattr(state, 'is_front_side', False)
            is_back_side = getattr(state, 'is_back_side', False)
            day_activity_score = getattr(state, 'day_activity_score', 0.0)
            reset_point_quality = getattr(state, 'reset_point_quality', 0.0)
            intraday_move = getattr(state, 'max_intraday_move', 0.0)
            curriculum_stage = getattr(state, 'curriculum_stage', 'unknown')
            
            # Determine momentum direction
            momentum_direction = 'Front' if is_front_side else ('Back' if is_back_side else 'Mixed')
            momentum_color = DARK_THEME['accent_green'] if is_front_side else (DARK_THEME['accent_red'] if is_back_side else DARK_THEME['text_muted'])
            
            env_content = html.Div([
                self._info_row("Data Quality", f"{state.data_quality:.1%}"),
                self._info_row("Momentum Score", f"{state.momentum_score:.2f}"),
                self._info_row("Day Activity", f"{day_activity_score:.2f}"),
                self._info_row("Reset Quality", f"{reset_point_quality:.2f}"),
                self._info_row("Direction", momentum_direction, color=momentum_color),
                self._info_row("Volatility", f"{state.volatility:.1%}"),
                self._info_row("Intraday Move", f"{intraday_move:.1%}"),
                self._info_row("Volume Ratio", f"{volume_ratio:.1f}x"),
                self._info_row("Spread", f"${avg_spread:.3f}"),
                self._info_row("Halts", f"{halt_count}", color=DARK_THEME['accent_orange'] if halt_count > 0 else None),
            ])
            
            # Custom candlestick chart
            candlestick_chart = self._create_price_chart(state)
            
            # Performance footer
            footer = f"Steps/sec: {state.steps_per_second:.1f} | Updates/sec: {state.updates_per_second:.1f} | Episodes/hr: {state.episodes_per_hour:.1f}"
            
            return (header_info, session_time_str, market_info, position_info, portfolio_info,
                   trades_table, actions_content, episode_content, training_content, ppo_content,
                   reward_table, env_content, candlestick_chart, footer)
            
    def _info_row(self, label: str, value: str, color: Optional[str] = None) -> html.Div:
        """Create an info row with label left-aligned and value right-aligned"""
        value_color = color or DARK_THEME['text_primary']
        return html.Div([
            html.Span(label, style={'color': DARK_THEME['text_secondary'], 'fontSize': '12px'}),
            html.Span(value, style={'color': value_color, 'fontWeight': 'bold', 'fontSize': '12px'})
        ], style={
            'display': 'flex', 
            'justifyContent': 'space-between', 
            'alignItems': 'center',
            'marginBottom': '2px',
            'minHeight': '16px'
        })
        
    def _metric_with_sparkline(self, label: str, value: float, history: List[float]) -> html.Div:
        """Create a metric with inline sparkline"""
        # Create mini sparkline
        if len(history) > 1:
            fig = go.Figure()
            # Convert to list if it's a deque to handle slicing
            history_list = list(history) if hasattr(history, 'popleft') else history
            fig.add_trace(go.Scatter(
                y=history_list[-20:],  # Last 20 values
                mode='lines',
                line=dict(color=DARK_THEME['accent_blue'], width=1),
                showlegend=False
            ))
            fig.update_layout(
                height=30,
                margin=dict(l=0, r=0, t=0, b=0),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(visible=False),
                yaxis=dict(visible=False)
            )
            
            return html.Div([
                html.Div([
                    html.Span(f"{label}: ", style={'color': DARK_THEME['text_secondary']}),
                    html.Span(f"{value:.4f}", style={'color': DARK_THEME['text_primary'], 'fontWeight': 'bold'})
                ]),
                dcc.Graph(figure=fig, style={'height': '30px'}, config={'displayModeBar': False})
            ], style={'marginBottom': '10px'})
        else:
            return self._info_row(label, f"{value:.4f}")
            
    def _create_price_chart(self, state) -> go.Figure:
        """Create candlestick chart with 1m bars using Plotly"""
        # Get 1m candle data from state
        candle_data = getattr(state, 'candle_data_1m', [])
        trades_data = list(state.recent_trades) if state.recent_trades else []
        
        # Create subplots with candlestick and volume
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=(None, None)
        )
        
        if not candle_data:
            # Return empty figure with message
            fig.add_annotation(
                text="No market data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14, color=DARK_THEME['text_muted'])
            )
        else:
            # Use ALL candles for the full trading day (4 AM - 8 PM = 16 hours = 960 minutes)
            candles = candle_data  # Use all available data
            
            if candles:
                # Convert to dataframe for easier handling
                df = pd.DataFrame(candles)
                
                # Parse timestamps and ensure timezone-naive
                df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
                
                # Add candlestick trace
                fig.add_trace(
                    go.Candlestick(
                        x=df['timestamp'],
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name='Price',
                        increasing_line_color=DARK_THEME['accent_green'],
                        decreasing_line_color=DARK_THEME['accent_red'],
                        increasing_fillcolor=DARK_THEME['accent_green'],
                        decreasing_fillcolor=DARK_THEME['accent_red']
                    ),
                    row=1, col=1
                )
                
                # Add volume bars
                colors = [DARK_THEME['accent_green'] if close >= open else DARK_THEME['accent_red'] 
                         for open, close in zip(df['open'], df['close'])]
                
                fig.add_trace(
                    go.Bar(
                        x=df['timestamp'],
                        y=df['volume'],
                        name='Volume',
                        marker_color=colors,
                        opacity=0.5
                    ),
                    row=2, col=1
                )
                
                # Add trade markers
                if trades_data:
                    for trade in trades_data[-10:]:  # Last 10 trades
                        trade_time = trade.get('timestamp')
                        trade_price = trade.get('fill_price', 0)
                        
                        if trade_time and trade_price > 0:
                            # Parse trade timestamp and ensure timezone-naive
                            try:
                                trade_dt = pd.to_datetime(trade_time)
                                # Remove timezone if present
                                if trade_dt.tz is not None:
                                    trade_dt = trade_dt.tz_localize(None)
                            except:
                                continue
                            
                            # Only show trades within the chart time range
                            if trade_dt >= df['timestamp'].min() and trade_dt <= df['timestamp'].max():
                                is_buy = trade.get('side') == 'BUY'
                                marker_color = DARK_THEME['accent_green'] if is_buy else DARK_THEME['accent_red']
                                marker_symbol = 'triangle-up' if is_buy else 'triangle-down'
                                
                                fig.add_trace(
                                    go.Scatter(
                                        x=[trade_dt],
                                        y=[trade_price],
                                        mode='markers',
                                        marker=dict(
                                            size=12,
                                            color=marker_color,
                                            symbol=marker_symbol,
                                            line=dict(width=1, color=DARK_THEME['text_primary'])
                                        ),
                                        name=trade.get('side', 'Trade'),
                                        showlegend=False,
                                        hovertemplate=f"{trade.get('side', 'Trade')}<br>Price: ${trade_price:.3f}<br>Qty: {trade.get('quantity', 0)}<extra></extra>"
                                    ),
                                    row=1, col=1
                                )
        
        # Update layout
        fig.update_layout(
            template='plotly_dark',
            plot_bgcolor=DARK_THEME['bg_tertiary'],
            paper_bgcolor=DARK_THEME['bg_secondary'],
            font_color=DARK_THEME['text_primary'],
            xaxis=dict(
                gridcolor=DARK_THEME['border'], 
                rangeslider=dict(visible=False),
                # Force show all data without zoom
                autorange=True,
                fixedrange=True  # Disable zoom/pan to see all data
            ),
            yaxis=dict(gridcolor=DARK_THEME['border'], title='Price'),
            xaxis2=dict(gridcolor=DARK_THEME['border'], fixedrange=True),
            yaxis2=dict(gridcolor=DARK_THEME['border'], title='Volume'),
            margin=dict(l=60, r=40, t=20, b=40),
            showlegend=False,
            hovermode='x unified',
            height=300
        )
        
        # Update x-axis to show time nicely
        fig.update_xaxes(
            tickformat='%H:%M',
            tickmode='auto',
            nticks=16,  # Show more ticks for full day
            type='date'
        )
        
        return fig
        
    def run(self):
        """Run the dashboard server"""
        if self.app is None:
            self.create_app()
            
        self.logger.info(f"Starting dashboard server on port {self.port}")
        self.app.run(host='0.0.0.0', port=self.port, debug=False, use_reloader=False)


def start_dashboard(port: int = 8050, open_browser: bool = True) -> None:
    """Start the dashboard server
    
    Args:
        port: Port to run the server on
        open_browser: Whether to open browser automatically
    """
    server = DashboardServer(port=port)
    
    # Open browser if requested
    if open_browser:
        def open_browser_delayed():
            import time
            time.sleep(1.5)  # Give server time to start
            webbrowser.open(f'http://localhost:{port}')
            
        browser_thread = threading.Thread(target=open_browser_delayed, daemon=True)
        browser_thread.start()
    
    # Run server (blocking)
    server.run()