# " dashboard/dashboard_server.py - Dash web server implementation

import logging
import threading
import webbrowser
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum

import dash
from dash import dcc, html, Input, Output, State, dash_table, callback_context
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from collections import deque

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
                # Row 1: 4-column layout - Market Info, Portfolio & Position, Performance Metrics, Actions
                html.Div([
                    # Market Info Card
                    html.Div([
                        html.H4("Market Info", style={'color': DARK_THEME['text_primary'], 'marginBottom': '4px', 'fontSize': '12px', 'fontWeight': 'bold'}),
                        html.Div(id='market-info-content')
                    ], style=self._card_style()),
                    
                    # Combined Portfolio & Position Card
                    html.Div([
                        html.H4("Portfolio & Position", style={'color': DARK_THEME['text_primary'], 'marginBottom': '4px', 'fontSize': '12px', 'fontWeight': 'bold'}),
                        html.Div(id='portfolio-position-content')
                    ], style=self._card_style()),
                    
                    # Performance Metrics Card
                    html.Div([
                        html.H4("Performance", style={'color': DARK_THEME['text_primary'], 'marginBottom': '4px', 'fontSize': '12px', 'fontWeight': 'bold'}),
                        html.Div(id='performance-metrics-content')
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
                        html.H4("Curriculum", style={'color': DARK_THEME['text_primary'], 'marginBottom': '4px', 'fontSize': '12px', 'fontWeight': 'bold'}),
                        html.Div(id='env-content')
                    ], style=self._card_style()),
                ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr 1fr 1fr', 'gap': '6px', 'marginBottom': '6px'}),
                
                # Row 3: Split layout - Trades table + Reward chart
                html.Div([
                    # Recent Trades Card
                    html.Div([
                        html.Div([
                            html.H4("Trades", style={'color': DARK_THEME['text_primary'], 'marginBottom': '0px', 'fontSize': '12px', 'fontWeight': 'bold', 'display': 'inline-block'}),
                            html.Span(id='trade-counter', style={'color': DARK_THEME['text_secondary'], 'fontSize': '10px', 'marginLeft': '8px'})
                        ], style={'marginBottom': '4px'}),
                        html.Div(id='trades-table-container')
                    ], style=self._card_style()),
                    
                    # Reward Components Card
                    html.Div([
                        html.H4("Rewards", style={'color': DARK_THEME['text_primary'], 'marginBottom': '4px', 'fontSize': '12px', 'fontWeight': 'bold'}),
                        html.Div(id='reward-table-container')
                    ], style=self._card_style()),
                ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '6px', 'marginBottom': '6px'}),
                
                # Row 4: Feature Attribution Panel (2-column layout)
                html.Div([
                    # Top Features by Branch
                    html.Div([
                        html.H4("🔍 Top Features", style={'color': DARK_THEME['text_primary'], 'marginBottom': '4px', 'fontSize': '12px', 'fontWeight': 'bold'}),
                        html.Div(id='attribution-features-content')
                    ], style=self._card_style()),
                    
                    # Attribution Quality Metrics
                    html.Div([
                        html.H4("📊 Attribution Quality", style={'color': DARK_THEME['text_primary'], 'marginBottom': '4px', 'fontSize': '12px', 'fontWeight': 'bold'}),
                        html.Div(id='attribution-quality-content')
                    ], style=self._card_style()),
                ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '6px', 'marginBottom': '6px'}),
                
                # Row 5: Full-width Chart
                html.Div([
                    html.H4("Price & Volume (Focused View)", style={'color': DARK_THEME['text_primary'], 'marginBottom': '4px', 'fontSize': '12px', 'fontWeight': 'bold'}),
                    dcc.Graph(id='candlestick-chart', style={'height': '500px'})
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
                'fontSize': '12px'
            }),
            
            # Auto-refresh interval
            dcc.Interval(id='interval-component', interval=1000, n_intervals=0)  # 1s refresh (was 500ms)
            
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
             Output('portfolio-position-content', 'children'),
             Output('performance-metrics-content', 'children'),
             Output('trades-table-container', 'children'),
             Output('trade-counter', 'children'),
             Output('actions-content', 'children'),
             Output('episode-content', 'children'),
             Output('training-content', 'children'),
             Output('ppo-metrics-content', 'children'),
             Output('reward-table-container', 'children'),
             Output('env-content', 'children'),
             Output('attribution-features-content', 'children'),
             Output('attribution-quality-content', 'children'),
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
            
            # Header info with momentum day
            momentum_day_info = ""
            if state.current_momentum_day_date:
                momentum_day_info = f" | Day: {state.current_momentum_day_date} (Q: {state.current_momentum_day_quality:.2f})"
            header_info = f"Model: {state.model_name} | Symbol: {state.symbol}{momentum_day_info}"
            
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
            
            # Combined Portfolio & Position info
            if state.position_side != "FLAT":
                position_pnl = getattr(state, 'position_pnl_dollar', 0.0)
                pnl_color = DARK_THEME['accent_green'] if position_pnl >= 0 else DARK_THEME['accent_red']
                position_section = [
                    self._info_row("Side", state.position_side, color=DARK_THEME['accent_blue']),
                    self._info_row("Qty", f"{state.position_qty:,}"),
                    self._info_row("Entry", f"${state.avg_entry_price:.3f}"),
                    self._info_row("P&L $", f"${position_pnl:.2f}", color=pnl_color),
                    self._info_row("P&L %", f"{state.position_pnl_percent:.1f}%", color=pnl_color),
                    self._info_row("Hold Time", f"{getattr(state, 'position_hold_time_seconds', 0)//60:.0f}m" if getattr(state, 'position_hold_time_seconds', 0) > 0 else "0m"),
                ]
            else:
                position_section = [
                    self._info_row("Side", "FLAT", color=DARK_THEME['text_muted']),
                    self._info_row("Qty", "0"),
                    self._info_row("Entry", "-"),
                    self._info_row("P&L $", "$0.00"),
                    self._info_row("P&L %", "0.0%"),
                    self._info_row("Hold Time", "0m"),
                ]
            
            session_pnl_color = DARK_THEME['accent_green'] if state.session_pnl >= 0 else DARK_THEME['accent_red']
            portfolio_section = [
                html.Hr(style={'margin': '4px 0', 'borderColor': DARK_THEME['border']}),
                self._info_row("Equity", f"${state.total_equity:.0f}"),
                self._info_row("Cash", f"${state.cash_balance:.0f}"),
                self._info_row("Session P&L", f"${state.session_pnl:.2f}", color=session_pnl_color),
                self._info_row("Realized", f"${state.realized_pnl:.2f}"),
            ]
            
            portfolio_position_info = html.Div(position_section + portfolio_section)
            
            # Performance Metrics info - Table format with Episode/Session columns
            # Calculate episode metrics
            episode_wins = getattr(state, 'episode_winning_trades', 0)
            episode_losses = getattr(state, 'episode_losing_trades', 0)
            episode_total = episode_wins + episode_losses
            
            # Calculate session metrics  
            session_wins = getattr(state, 'session_winning_trades', 0)
            session_losses = getattr(state, 'session_losing_trades', 0)
            session_total = session_wins + session_losses
            
            # Calculate win/loss ratios
            episode_win_loss_ratio = episode_wins / episode_losses if episode_losses > 0 else float('inf') if episode_wins > 0 else 0
            session_win_loss_ratio = session_wins / session_losses if session_losses > 0 else float('inf') if session_wins > 0 else 0
            
            # Format win/loss ratios
            episode_wl_display = f"{episode_win_loss_ratio:.2f}" if episode_win_loss_ratio != float('inf') else "∞" if episode_wins > 0 else "0.00"
            session_wl_display = f"{session_win_loss_ratio:.2f}" if session_win_loss_ratio != float('inf') else "∞" if session_wins > 0 else "0.00"
            
            # Calculate episode and session win rates
            episode_win_rate = (episode_wins / episode_total * 100) if episode_total > 0 else 0
            session_win_rate = (session_wins / session_total * 100) if session_total > 0 else 0
            
            # Format profit factor (handle infinity case)
            profit_factor = getattr(state, 'profit_factor', 0.0)
            profit_factor_display = f"{profit_factor:.2f}" if profit_factor != float('inf') else "∞" if profit_factor > 0 else "0.00"
            
            # Performance data for table
            performance_data = [
                {
                    'Metric': 'Win Rate %',
                    'Episode': f"{episode_win_rate:.1f}%",
                    'Session': f"{session_win_rate:.1f}%"
                },
                {
                    'Metric': 'W/L Ratio',
                    'Episode': episode_wl_display,
                    'Session': session_wl_display
                },
                {
                    'Metric': 'Profit Factor',
                    'Episode': "N/A",
                    'Session': profit_factor_display
                }
            ]
            
            performance_metrics_info = dash_table.DataTable(
                data=performance_data,
                columns=[
                    {'name': 'Metric', 'id': 'Metric'},
                    {'name': 'Episode', 'id': 'Episode'},
                    {'name': 'Session', 'id': 'Session'}
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
                    # Highlight win rates
                    {
                        'if': {'column_id': 'Episode', 'filter_query': '{Metric} = "Win Rate %"'},
                        'color': DARK_THEME['accent_green'] if episode_win_rate >= 50 else DARK_THEME['accent_red']
                    },
                    {
                        'if': {'column_id': 'Session', 'filter_query': '{Metric} = "Win Rate %"'},
                        'color': DARK_THEME['accent_green'] if session_win_rate >= 50 else DARK_THEME['accent_red']
                    },
                    # Highlight drawdown
                    {
                        'if': {'filter_query': '{Metric} = "Drawdown %"'},
                        'color': DARK_THEME['accent_red'] if state.max_drawdown > 10 else DARK_THEME['accent_orange']
                    },
                    # Highlight profit factor
                    {
                        'if': {'filter_query': '{Metric} = "Profit Factor"'},
                        'color': DARK_THEME['accent_green'] if profit_factor > 1.0 else DARK_THEME['accent_red']
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
            
            # Recent trades table (completed trades only)
            if state.recent_trades:
                trades_df = pd.DataFrame(list(state.recent_trades)[-10:])  # Last 10 trades
                
                # Convert any enum values to strings for JSON serialization
                trades_data = trades_df.to_dict('records')
                for trade in trades_data:
                    for key, value in trade.items():
                        if isinstance(value, Enum):  # Check if it's an enum
                            trade[key] = value.value
                
                trades_table = dash_table.DataTable(
                    data=trades_data,
                    columns=[
                        {'name': 'Entry', 'id': 'entry_time'},
                        {'name': 'Exit', 'id': 'exit_time'},
                        {'name': 'Side', 'id': 'side'},
                        {'name': 'Qty', 'id': 'quantity'},
                        {'name': 'Entry $', 'id': 'entry_price', 'type': 'numeric', 'format': {'specifier': '$.3f'}},
                        {'name': 'Exit $', 'id': 'exit_price', 'type': 'numeric', 'format': {'specifier': '$.3f'}},
                        {'name': 'P&L', 'id': 'pnl', 'type': 'numeric', 'format': {'specifier': '$.2f'}},
                        {'name': 'Hold', 'id': 'hold_time'},
                        {'name': 'Status', 'id': 'status'}
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
            
            # Trade counter text
            episode_trades = getattr(state, 'episode_total_trades', 0)
            episode_wins = getattr(state, 'episode_winning_trades', 0)
            episode_losses = getattr(state, 'episode_losing_trades', 0)
            session_trades = getattr(state, 'session_total_trades', 0)
            
            if episode_trades > 0:
                win_rate = (episode_wins / episode_trades) * 100
                trade_counter_text = f"Episode: {episode_trades} ({episode_wins}W/{episode_losses}L - {win_rate:.0f}%) | Session: {session_trades}"
            else:
                trade_counter_text = f"Episode: 0 | Session: {session_trades}"
            
            # Actions table - check for proper action distributions
            episode_actions = getattr(state, 'episode_action_distribution', {'HOLD': 0, 'BUY': 0, 'SELL': 0})
            session_actions = getattr(state, 'session_action_distribution', {'HOLD': 0, 'BUY': 0, 'SELL': 0})
            
            # Fallback logic for both episode and session actions
            event_stream_actions = getattr(state, 'action_distribution', {'HOLD': 0, 'BUY': 0, 'SELL': 0})
            
            # If session actions are empty but event stream has data, use event stream as session fallback
            if all(v == 0 for v in session_actions.values()) and sum(event_stream_actions.values()) > 0:
                session_actions = event_stream_actions.copy()
            
            # If episode actions are empty but event stream has data, show partial episode progress
            # This handles the case where metrics haven't arrived yet but actions are being tracked
            if all(v == 0 for v in episode_actions.values()) and sum(event_stream_actions.values()) > 0:
                # Use event stream as episode actions if we're in an active episode
                if state.current_step > 0:  # Only if we're actually in an episode
                    episode_actions = event_stream_actions.copy()
            
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
            
            # Invalid action tracking removed - action masking prevents invalid actions
            # episode_invalid = getattr(state, 'episode_invalid_actions', 0)
            # session_invalid = getattr(state, 'session_invalid_actions', 0)
            
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
            
            # Invalid actions display removed - action masking prevents invalid actions
            # invalid_actions_row = html.Div([
            #     html.Hr(style={'margin': '4px 0', 'borderColor': DARK_THEME['border']}),
            #     self._info_row("Invalid Actions", f"Ep: {episode_invalid} | Sess: {session_invalid}", color=DARK_THEME['accent_orange'])
            # ], style={'marginTop': '4px'})
            
            actions_content = html.Div([actions_table])
            
            # Episode info - more compact
            progress = (state.current_step / state.max_steps * 100) if state.max_steps > 0 else 0
            
            # Handle display when values are zero or missing
            episode_display = state.episode_number if state.episode_number > 0 else "-"
            step_display = f"{state.current_step:,}/{state.max_steps:,}" if state.max_steps > 0 else f"{state.current_step:,}/∞"
            progress_display = f"{progress:.1f}%" if state.max_steps > 0 else "∞"
            
            # Combine episode number with progress percentage
            episode_with_progress = f"Episode {episode_display} ({progress_display})"
            
            # Calculate current episode reward (sum of episode components)
            episode_reward_total = sum(getattr(state, 'episode_reward_components', {}).values())
            
            episode_content = html.Div([
                self._info_row("Episode", episode_with_progress),
                self._info_row("Step", step_display),
                self._info_row("Episode Reward", f"{episode_reward_total:.3f}"),
                self._info_row("Step Reward", f"{state.last_step_reward:.3f}"),
                # Compact progress bar with numbers
                html.Div([
                    html.Div(style={
                        'backgroundColor': DARK_THEME['bg_tertiary'],
                        'height': '8px',
                        'borderRadius': '4px',
                        'marginTop': '4px',
                        'marginBottom': '2px'
                    }, children=[
                        html.Div(style={
                            'backgroundColor': DARK_THEME['accent_blue'],
                            'height': '100%',
                            'width': f"{progress}%",
                            'borderRadius': '4px',
                            'transition': 'width 0.3s ease'
                        })
                    ]),
                    html.Div(step_display, style={
                        'color': DARK_THEME['text_primary'], 
                        'fontSize': '11px', 
                        'textAlign': 'center',
                        'marginTop': '2px'
                    })
                ])
            ])
            
            # Training progress
            max_updates = getattr(state, 'max_updates', 0)
            update_display = f"{state.updates:,}" + (f"/{max_updates:,}" if max_updates > 0 else "")
            
            # Calculate updates per hour (transmitter sends updates_per_second)
            updates_per_hour = getattr(state, 'updates_per_second', 0.0) * 3600
            
            training_children = [
                self._info_row("Mode", state.mode, color=DARK_THEME['accent_purple']),
                self._info_row("Stage", state.stage),
                self._info_row("Episodes", f"{state.total_episodes:,}"),
                self._info_row("Updates", update_display),
                self._info_row("Global Steps", f"{state.global_steps:,}"),
                self._info_row("Steps/sec", f"{state.steps_per_second:.1f}"),
                self._info_row("Eps/hour", f"{state.episodes_per_hour:.0f}"),
                self._info_row("Updates/hour", f"{updates_per_hour:.0f}"),
            ]
            
            # Add training completion progress bar (if max_updates is available)
            if max_updates > 0:
                progress_section = self._create_progress_section(
                    "Training Progress", 
                    state.updates, 
                    max_updates,
                    f"Update {state.updates:,}/{max_updates:,}"
                )
                if progress_section:
                    training_children.append(progress_section)
            
            # Add current stage progress bar
            stage_progress = self._create_stage_progress_section(state)
            if stage_progress:
                training_children.append(stage_progress)
                
            training_content = html.Div(training_children)
            
            # PPO Metrics - using correct attribute names from ppo_agent.py
            # The PPO agent sends metrics with these exact names in the ppo_data dict
            
            ppo_metrics_data = [
                ("Policy Loss", state.policy_loss, state.policy_loss_history),
                ("Value Loss", state.value_loss, state.value_loss_history),
                ("Entropy", state.entropy, state.entropy_history),
                ("KL Divergence", getattr(state, 'kl_divergence', 0.0), getattr(state, 'kl_divergence_history', deque())),
                ("Clip Fraction", getattr(state, 'clip_fraction', 0.0), getattr(state, 'clip_fraction_history', deque())),
                ("Learning Rate", getattr(state, 'learning_rate', 0.0), getattr(state, 'learning_rate_history', deque())),
                ("Explained Var", getattr(state, 'explained_variance', 0.0), getattr(state, 'explained_variance_history', deque())),
                ("Mean Reward", getattr(state, 'mean_episode_reward', 0.0), getattr(state, 'mean_episode_reward_history', deque())),
            ]
            
            ppo_content = html.Div([
                self._metric_with_sparkline(
                    label, value, history, 
                    *self._get_ppo_metric_guidance(label, value, list(history) if history else [])
                ) for label, value, history in ppo_metrics_data
            ])
            
            # Reward components table - redesigned to match actions panel format
            # Define all active reward components with their types (new percentage-based system)
            all_reward_components = {
                'pnl': 'foundational',
                'holding_penalty': 'shaping',
                'drawdown_penalty': 'shaping',
                'profit_giveback_penalty': 'shaping',
                'max_drawdown_penalty': 'shaping',
                'profit_closing_bonus': 'shaping',
                'clean_trade_bonus': 'shaping',
                'trading_activity_bonus': 'shaping',
                'inactivity_penalty': 'shaping',
                'bankruptcy_penalty': 'terminal'
            }
            
            # Color mapping for component types
            component_type_colors = {
                'foundational': DARK_THEME['accent_blue'],
                'shaping': DARK_THEME['accent_orange'],
                'terminal': DARK_THEME['accent_red'],
                'trade': DARK_THEME['accent_purple'],
                'summary': DARK_THEME['accent_green']
            }
            
            # Get reward data and counts
            episode_rewards = getattr(state, 'episode_reward_components', {})
            session_rewards = getattr(state, 'session_reward_components', {})
            episode_counts = getattr(state, 'episode_reward_component_counts', {})
            session_counts = getattr(state, 'session_reward_component_counts', {})
            
            # Show ALL active components, including those with zero values
            reward_data = []
            episode_total = 0.0
            session_total = 0.0
            
            for component in sorted(all_reward_components.keys()):
                episode_value = episode_rewards.get(component, 0.0)
                session_value = session_rewards.get(component, 0.0)
                episode_count = episode_counts.get(component, 0)
                session_count = session_counts.get(component, 0)
                
                episode_total += episode_value
                session_total += session_value
                
                # Calculate percentages based on absolute values to show component activity
                ep_abs_total = sum(abs(v) for v in episode_rewards.values()) or 1
                sess_abs_total = sum(abs(v) for v in session_rewards.values()) or 1
                ep_pct = abs(episode_value) / ep_abs_total * 100
                sess_pct = abs(session_value) / sess_abs_total * 100
                
                reward_data.append({
                    'Component': component,
                    'Type': all_reward_components[component],
                    'Episode': f"{episode_value:.3f}",
                    'Ep %': f"{ep_pct:.1f}%",
                    'Session': f"{session_value:.2f}",
                    'Sess %': f"{sess_pct:.1f}%",
                    'Ep Count': f"{episode_count}",
                    'Sess Count': f"{session_count}"
                })
            
            # Add a total row for validation
            reward_data.append({
                'Component': "TOTAL",
                'Type': 'summary',
                'Episode': f"{episode_total:.3f}",
                'Ep %': "100%",
                'Session': f"{session_total:.2f}",
                'Sess %': "100%",
                'Ep Count': f"{sum(episode_counts.values())}",
                'Sess Count': f"{sum(session_counts.values())}"
            })
            
            reward_table = dash_table.DataTable(
                data=reward_data,
                columns=[
                    {'name': 'Component', 'id': 'Component'},
                    {'name': 'Episode', 'id': 'Episode', 'type': 'numeric'},
                    {'name': 'Ep %', 'id': 'Ep %'},
                    {'name': 'Session', 'id': 'Session', 'type': 'numeric'},
                    {'name': 'Sess %', 'id': 'Sess %'},
                    {'name': 'Ep Cnt', 'id': 'Ep Count'},
                    {'name': 'S Cnt', 'id': 'Sess Count'}
                ],
                style_cell={
                    'backgroundColor': DARK_THEME['bg_tertiary'],
                    'color': DARK_THEME['text_primary'],
                    'border': f"1px solid {DARK_THEME['border']}",
                    'fontSize': '10px',
                    'padding': '4px 6px',
                    'textAlign': 'left'
                },
                style_data_conditional=[
                    # Color component names by type
                    *[{
                        'if': {'column_id': 'Component', 'filter_query': f'{{Type}} = {comp_type}'},
                        'color': color
                    } for comp_type, color in component_type_colors.items()],
                    # Color episode values 
                    {
                        'if': {'column_id': 'Episode', 'filter_query': '{Episode} > 0'},
                        'color': DARK_THEME['accent_green']
                    },
                    {
                        'if': {'column_id': 'Episode', 'filter_query': '{Episode} < 0'},
                        'color': DARK_THEME['accent_red']
                    },
                    # Color session values
                    {
                        'if': {'column_id': 'Session', 'filter_query': '{Session} > 0'},
                        'color': DARK_THEME['accent_green']
                    },
                    {
                        'if': {'column_id': 'Session', 'filter_query': '{Session} < 0'},
                        'color': DARK_THEME['accent_red']
                    },
                    # Highlight total row
                    {
                        'if': {'row_index': len(reward_data) - 1},
                        'backgroundColor': DARK_THEME['bg_secondary'],
                        'fontWeight': 'bold'
                    }
                ],
                style_header={
                    'backgroundColor': DARK_THEME['bg_secondary'],
                    'color': DARK_THEME['text_secondary'],
                    'fontWeight': 'bold',
                    'fontSize': '10px'
                },
                page_size=12
            )
            
            # Environment/Curriculum info with expanded details
            avg_spread = getattr(state, 'avg_spread', getattr(state, 'spread', 0.001))
            volume_ratio = getattr(state, 'volume_ratio', 1.0)
            halt_count = getattr(state, 'halt_count', 0)
            # 3-Component Sniper Curriculum
            curriculum_stage = getattr(state, 'curriculum_stage', 'stage_1_beginner')
            total_episodes_curriculum = getattr(state, 'total_episodes_for_curriculum', state.total_episodes)
            
            # 2-component scores from current reset point
            current_roc_score = getattr(state, 'current_roc_score', 0.0)
            current_activity_score = getattr(state, 'current_activity_score', 0.0)
            
            # Curriculum ranges (match PPO agent field names)
            roc_range = getattr(state, 'roc_range', [0.0, 1.0])
            activity_range = getattr(state, 'activity_range', [0.0, 1.0])
            
            episode_length = getattr(state, 'curriculum_episode_length', 256)
            
            # Determine curriculum stage color
            stage_colors = {
                'stage_1_beginner': DARK_THEME['accent_green'],
                'stage_2_intermediate': DARK_THEME['accent_blue'],
                'stage_3_advanced': DARK_THEME['accent_orange'],
                'stage_4_specialization': DARK_THEME['accent_purple']
            }
            stage_names = {
                'stage_1_beginner': 'Beginner',
                'stage_2_intermediate': 'Intermediate', 
                'stage_3_advanced': 'Advanced',
                'stage_4_specialization': 'Master'
            }
            stage_display = stage_names.get(curriculum_stage, curriculum_stage.replace('_', ' ').title())
            curriculum_color = stage_colors.get(curriculum_stage, DARK_THEME['text_muted'])
            
            # Get progress percentage from curriculum system (transmitted from agent)
            progress_pct = getattr(state, 'curriculum_progress', 0.0)
            if progress_pct == 0.0:
                # Fallback calculation if not received from agent
                stage_thresholds = {
                    'stage_1_beginner': 2000,
                    'stage_2_intermediate': 5000, 
                    'stage_3_advanced': 8000,
                    'stage_4_specialization': float('inf')
                }
                current_threshold = stage_thresholds.get(curriculum_stage, float('inf'))
                progress_pct = (total_episodes_curriculum / current_threshold * 100) if current_threshold != float('inf') else 100
                progress_pct = min(100, progress_pct)
                
            # Determine momentum direction
            
            # Add reset points info (show filtered count)
            reset_points_data = getattr(state, 'reset_points_data', [])
            total_reset_points = len(reset_points_data)
            
            # Count filtered reset points using curriculum ranges
            # (range filtering happens in PPO agent)
            # is centralized in the momentum scanner
            filtered_count = total_reset_points  # Show all available points
            
            # Add day information with color coding
            day_info_color = DARK_THEME['accent_green'] if state.current_momentum_day_quality >= 0.7 else DARK_THEME['accent_blue'] if state.current_momentum_day_quality >= 0.5 else DARK_THEME['text_muted']
            
            # Enhanced curriculum tracking
            episodes_to_next = getattr(state, 'episodes_to_next_stage', 0)
            next_stage_name = getattr(state, 'next_stage_name', '')
            episodes_per_day_config = getattr(state, 'episodes_per_day_config', 10)
            
            # Reset point cycle tracking
            total_available = getattr(state, 'total_available_points', 0)
            points_used = getattr(state, 'points_used_in_cycle', 0)
            points_remaining = getattr(state, 'points_remaining_in_cycle', 0)
            cycles_completed = getattr(state, 'cycles_completed', getattr(state, 'reset_point_cycles_completed', 0))
            target_cycles = getattr(state, 'target_cycles_per_day', 10)
            day_switch_progress = getattr(state, 'day_switch_progress_pct', 0.0)
            
            # Fallback calculation for day switch progress if not received from events
            if day_switch_progress == 0.0 and target_cycles > 0:
                day_switch_progress = (cycles_completed / target_cycles) * 100
                day_switch_progress = min(100, day_switch_progress)
            
            # Debug: Check if we received curriculum updates
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Curriculum values - progress: {progress_pct}%, cycles: {cycles_completed}/{target_cycles}, day_switch: {day_switch_progress}%, roc_range: {roc_range}, activity_range: {activity_range}")
            logger.debug(f"Reset point tracking - total_available: {total_available}, points_used: {points_used}, episodes_on_day: {getattr(state, 'episodes_on_current_day', 0)}")
            
            env_content = html.Div([
                # Curriculum Stage Progress
                self._info_row("Stage", stage_display, color=curriculum_color),
                self._info_row("Progress", f"{progress_pct:.1f}%", color=curriculum_color),
                self._info_row("To Next Stage", f"{episodes_to_next:,} eps" if episodes_to_next > 0 else "Max Stage", 
                              color=DARK_THEME['accent_orange'] if episodes_to_next > 0 else DARK_THEME['accent_green']),
                self._info_row("Episode Len", f"{episode_length} steps"),
                
                # Reset Point Cycle Tracking
                html.Hr(style={'margin': '4px 0', 'borderColor': DARK_THEME['border']}),
                self._info_row("Available Points", f"{total_available}", color=DARK_THEME['accent_blue']),
                self._info_row("Points Used", f"{points_used}/{total_available}" if total_available > 0 else "0/0", 
                              color=DARK_THEME['accent_green'] if points_used < total_available else DARK_THEME['accent_orange']),
                self._info_row("Cycle Progress", f"{cycles_completed}/{target_cycles}", 
                              color=DARK_THEME['accent_blue']),
                self._info_row("Day Switch", f"{day_switch_progress:.0f}%", 
                              color=DARK_THEME['accent_green'] if day_switch_progress < 90 else DARK_THEME['accent_orange']),
                
                # Day information
                html.Hr(style={'margin': '4px 0', 'borderColor': DARK_THEME['border']}),
                self._info_row("Day", state.current_momentum_day_date or "N/A", color=day_info_color),
                self._info_row("Day Score", f"{state.current_momentum_day_quality:.3f}" if state.current_momentum_day_quality > 0 else "N/A", color=day_info_color),
                # 2-Component Scores with range indicators
                html.Hr(style={'margin': '4px 0', 'borderColor': DARK_THEME['border']}),
                self._info_row("ROC", f"{current_roc_score:.2f}", 
                             color=DARK_THEME['accent_blue'] if roc_range[0] <= current_roc_score <= roc_range[1] else DARK_THEME['text_muted']),
                self._info_row("Activity", f"{current_activity_score:.2f}", 
                             color=DARK_THEME['accent_orange'] if activity_range[0] <= current_activity_score <= activity_range[1] else DARK_THEME['text_muted']),
                self._info_row("ROC Range", f"[{roc_range[0]:.2f}, {roc_range[1]:.2f}]", 
                             color=DARK_THEME['text_muted']),
                self._info_row("Activity Range", f"[{activity_range[0]:.2f}, {activity_range[1]:.2f}]", 
                             color=DARK_THEME['text_muted']),
            ])
            
            # Custom candlestick chart
            candlestick_chart = self._create_price_chart(state)
            
            # Attribution panels
            attribution_features_content = self._create_attribution_features_content(state)
            attribution_quality_content = self._create_attribution_quality_content(state)
            
            # Performance footer with eps/Hour moved here
            eps_per_hour = getattr(state, 'episodes_per_hour', 0.0)
            updates_per_hour = state.updates_per_second * 3600
            footer = f"Steps/sec: {state.steps_per_second:.1f} | Updates/hr: {updates_per_hour:.1f} | Episodes/hr: {eps_per_hour:.1f}"
            
            return (header_info, session_time_str, market_info, portfolio_position_info, performance_metrics_info,
                   trades_table, trade_counter_text, actions_content, episode_content, training_content, ppo_content,
                   reward_table, env_content, attribution_features_content, attribution_quality_content, 
                   candlestick_chart, footer)
            
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
    
    def _create_progress_section(self, title: str, current: int, total: int, text: str) -> Optional[html.Div]:
        """Create a progress section with title, progress bar, and text"""
        if total <= 0:
            return None
            
        progress_pct = min(100.0, (current / total) * 100.0)
        
        return html.Div([
            html.Div(title, style={'color': DARK_THEME['text_secondary'], 'fontSize': '10px', 'marginBottom': '2px'}),
            html.Div([
                html.Div(style={
                    'backgroundColor': DARK_THEME['bg_tertiary'],
                    'height': '8px',
                    'borderRadius': '4px',
                    'marginBottom': '2px'
                }, children=[
                    html.Div(style={
                        'backgroundColor': DARK_THEME['accent_green'],
                        'height': '100%',
                        'width': f"{progress_pct}%",
                        'borderRadius': '4px',
                        'transition': 'width 0.3s ease'
                    })
                ])
            ]),
            html.Div(text, style={'color': DARK_THEME['text_primary'], 'fontSize': '11px', 'textAlign': 'center'})
        ], style={'marginTop': '8px'})
    
    def _create_stage_progress_section(self, state) -> Optional[html.Div]:
        """Create current stage progress section"""
        stage = getattr(state, 'stage', '')
        stage_status = getattr(state, 'stage_status', '')
        
        # Handle rollout collection progress
        if 'rollout' in stage.lower():
            rollout_steps = getattr(state, 'rollout_steps', 0)
            rollout_total = getattr(state, 'rollout_total', 0)
            if rollout_total > 0:
                progress_pct = min(100, (rollout_steps / rollout_total) * 100)
                text = f"{rollout_steps:,}/{rollout_total:,} steps collected"
                return self._create_progress_bar_with_text("Current Stage", progress_pct, text)
        
        # Handle PPO update progress  
        elif 'update' in stage.lower() or 'ppo' in stage.lower():
            current_epoch = getattr(state, 'current_epoch', 0)
            total_epochs = getattr(state, 'total_epochs', 0)
            current_batch = getattr(state, 'current_batch', 0)
            total_batches = getattr(state, 'total_batches', 0)
            
            if total_epochs > 0:
                epoch_progress = min(100, (current_epoch / total_epochs) * 100)
                if total_batches > 0:
                    # Calculate global batch number to fix non-linear display
                    global_batch = (current_epoch - 1) * total_batches + current_batch if current_epoch > 0 else current_batch
                    global_total_batches = total_epochs * total_batches
                    batch_progress = min(100, (current_batch / total_batches) * 100)
                    text = f"Epoch {current_epoch}/{total_epochs}, Batch {global_batch}/{global_total_batches}"
                else:
                    text = f"Epoch {current_epoch}/{total_epochs}"
                    
                # Use epoch progress primarily, with some batch detail
                progress_pct = epoch_progress
                return self._create_progress_bar_with_text("Current Stage", progress_pct, text)
        
        # Fallback to stage status if available
        if stage_status:
            return html.Div([
                html.Div("Current Stage", style={'color': DARK_THEME['text_secondary'], 'fontSize': '10px', 'marginBottom': '2px'}),
                html.Div(stage_status, style={'color': DARK_THEME['text_primary'], 'fontSize': '10px', 'textAlign': 'center'})
            ], style={'marginTop': '8px'})
        
        return None
    
    def _create_progress_bar_with_text(self, title: str, progress_pct: float, text: str) -> html.Div:
        """Create a progress bar with title and text"""
        return html.Div([
            html.Div(title, style={'color': DARK_THEME['text_secondary'], 'fontSize': '11px', 'marginBottom': '2px'}),
            html.Div([
                html.Div(style={
                    'backgroundColor': DARK_THEME['bg_tertiary'],
                    'height': '8px',
                    'borderRadius': '4px',
                    'marginBottom': '2px'
                }, children=[
                    html.Div(style={
                        'backgroundColor': DARK_THEME['accent_blue'],
                        'height': '100%',
                        'width': f"{progress_pct}%",
                        'borderRadius': '4px',
                        'transition': 'width 0.3s ease'
                    })
                ])
            ]),
            html.Div(text, style={'color': DARK_THEME['text_primary'], 'fontSize': '12px', 'textAlign': 'center'})
        ], style={'marginTop': '8px'})
        
    def _metric_with_sparkline(self, label: str, value: float, history: List[float], tooltip: str = None, health_status: str = "good") -> html.Div:
        """Create a compact one-liner metric with inline sparkline, tooltip, and health indicator"""
        # Format value based on label type
        if "Learning Rate" in label:
            value_str = f"{value:.2e}"
        elif "Clip Fraction" in label or "Explained Var" in label:
            value_str = f"{value:.3f}"
        else:
            value_str = f"{value:.4f}"
        
        # Determine health status color
        health_colors = {
            "good": DARK_THEME['accent_green'],
            "warning": DARK_THEME['accent_orange'], 
            "bad": DARK_THEME['accent_red']
        }
        health_color = health_colors.get(health_status, DARK_THEME['text_primary'])
        
        # Create mini sparkline - always show a chart area
        sparkline_element = None
        
        # Convert history to list and get data points
        if history and hasattr(history, '__iter__'):
            try:
                history_list = list(history) if hasattr(history, 'popleft') else history
                if len(history_list) >= 1:  # Show chart even with just 1 point
                    # Pad with current value if we have less than 2 points
                    if len(history_list) == 1:
                        chart_data = [history_list[0], history_list[0]]
                    else:
                        chart_data = history_list[-20:]  # Last 20 values for better trend visibility
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=chart_data,
                        mode='lines+markers',
                        line=dict(color=DARK_THEME['accent_blue'], width=2),
                        marker=dict(size=3, color=DARK_THEME['accent_blue']),
                        showlegend=False,
                        hoverinfo='y',
                        hovertemplate=f'{label}: %{{y}}<extra></extra>'
                    ))
                    fig.update_layout(
                        height=25,
                        width=80,
                        margin=dict(l=2, r=2, t=2, b=2),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(visible=False, showgrid=False),
                        yaxis=dict(visible=False, showgrid=False)
                    )
                    
                    sparkline_element = dcc.Graph(
                        figure=fig, 
                        style={'height': '25px', 'width': '80px', 'flexShrink': '0'}, 
                        config={'displayModeBar': False, 'staticPlot': False}
                    )
                else:
                    # Show placeholder when no data
                    sparkline_element = html.Div("--", style={
                        'height': '25px', 'width': '80px', 'display': 'flex', 
                        'alignItems': 'center', 'justifyContent': 'center',
                        'color': DARK_THEME['text_muted'], 'fontSize': '10px',
                        'border': f"1px dashed {DARK_THEME['border']}"
                    })
            except:
                # Fallback placeholder on any error
                sparkline_element = html.Div("--", style={
                    'height': '25px', 'width': '80px', 'display': 'flex', 
                    'alignItems': 'center', 'justifyContent': 'center',
                    'color': DARK_THEME['text_muted'], 'fontSize': '10px',
                    'border': f"1px dashed {DARK_THEME['border']}"
                })
        else:
            # Show placeholder when no history data - indicate missing data
            sparkline_element = html.Div("NO DATA", style={
                'height': '25px', 'width': '80px', 'display': 'flex', 
                'alignItems': 'center', 'justifyContent': 'center',
                'color': DARK_THEME['accent_red'], 'fontSize': '8px',
                'border': f"1px dashed {DARK_THEME['accent_red']}",
                'fontWeight': 'bold'
            })
        
        # Create compact layout with right-aligned values
        return html.Div([
            # Label on the left with health indicator
            html.Div([
                html.Span("●", style={
                    'color': health_color, 
                    'fontSize': '12px', 
                    'marginRight': '4px',
                    'fontWeight': 'bold'
                }),
                html.Span(f"{label}:", style={'color': DARK_THEME['text_secondary'], 'fontSize': '11px'})
            ], style={'display': 'flex', 'alignItems': 'center', 'flex': '0 0 auto'}),
            # Spacer to push value and chart to the right
            html.Div(style={'flex': '1'}),
            # Value right-aligned next to chart
            html.Span(value_str, style={
                'color': DARK_THEME['text_primary'], 
                'fontWeight': 'bold', 
                'fontSize': '11px',
                'marginRight': '8px',
                'flex': '0 0 auto'
            }),
            # Chart on the far right
            sparkline_element
        ], style={
            'display': 'flex', 
            'alignItems': 'center', 
            'marginBottom': '3px',
            'minHeight': '25px'
        }, title=tooltip or f"{label}: {value_str}")
    
    def _get_ppo_metric_guidance(self, label: str, value: float, history: List[float]) -> tuple[str, str]:
        """Get health status and tooltip for PPO metrics"""
        # Calculate trend direction from history
        trend = "stable"
        if history and len(history) >= 2:
            recent_avg = sum(history[-3:]) / len(history[-3:]) if len(history) >= 3 else history[-1]
            older_avg = sum(history[-6:-3]) / 3 if len(history) >= 6 else recent_avg
            if recent_avg > older_avg * 1.05:
                trend = "increasing"
            elif recent_avg < older_avg * 0.95:
                trend = "decreasing"
        
        if label == "Policy Loss":
            if abs(value) < 0.01:
                health = "good"
                tooltip = f"Policy Loss: {value:.4f} - GOOD: Small loss indicates stable policy. Continue current settings."
            elif abs(value) < 0.05:
                health = "warning" 
                tooltip = f"Policy Loss: {value:.4f} - OK: Moderate loss. Monitor for stability. Consider slight LR reduction if trend={trend}."
            else:
                health = "bad"
                tooltip = f"Policy Loss: {value:.4f} - HIGH: Large loss indicates unstable policy. REDUCE learning rate significantly."
                
        elif label == "Value Loss":
            if value < 0.5:
                health = "good"
                tooltip = f"Value Loss: {value:.4f} - GOOD: Low critic loss, good value prediction. Current settings working well."
            elif value < 1.0:
                health = "warning"
                tooltip = f"Value Loss: {value:.4f} - OK: Moderate critic loss. Watch for convergence. Trend: {trend}."
            else:
                health = "bad"
                tooltip = f"Value Loss: {value:.4f} - HIGH: Poor value estimation. Check reward scaling or reduce learning rate."
                
        elif label == "Entropy":
            if 1.5 <= value <= 2.5:
                health = "good"
                tooltip = f"Entropy: {value:.4f} - GOOD: Healthy exploration level. Policy is exploring appropriately."
            elif 1.0 <= value < 1.5 or 2.5 < value <= 3.0:
                health = "warning"
                tooltip = f"Entropy: {value:.4f} - {'LOW' if value < 1.5 else 'HIGH'}: {'Policy converging, reduce entropy coef if needed' if value < 1.5 else 'High exploration, consider reducing entropy coef'}."
            else:
                health = "bad"
                tooltip = f"Entropy: {value:.4f} - {'VERY LOW' if value < 1.0 else 'VERY HIGH'}: {'Policy too deterministic, increase entropy coef' if value < 1.0 else 'Policy too random, decrease entropy coef significantly'}."
                
        elif label == "KL Divergence":
            if value < 0.01:
                health = "good"
                tooltip = f"KL Divergence: {value:.4f} - GOOD: Stable policy updates. Changes are conservative and safe."
            elif value < 0.1:
                health = "warning"
                tooltip = f"KL Divergence: {value:.4f} - ELEVATED: Moderate policy changes. Monitor for stability."
            else:
                health = "bad"
                tooltip = f"KL Divergence: {value:.4f} - HIGH: Large policy changes. REDUCE learning rate immediately to prevent instability."
                
        elif label == "Clip Fraction":
            if value < 0.1:
                health = "good"
                tooltip = f"Clip Fraction: {value:.3f} - GOOD: Few clipped updates. Policy changes are appropriate size."
            elif value < 0.3:
                health = "warning"
                tooltip = f"Clip Fraction: {value:.3f} - ELEVATED: Some clipping occurring. Consider reducing learning rate slightly."
            else:
                health = "bad"
                tooltip = f"Clip Fraction: {value:.3f} - HIGH: Many updates clipped (target <30%). REDUCE learning rate significantly."
                
        elif label == "Learning Rate":
            if 1e-5 <= value <= 5e-4:
                health = "good"
                tooltip = f"Learning Rate: {value:.2e} - GOOD: Appropriate range for PPO. Should work well for most cases."
            elif 5e-4 < value <= 1e-3:
                health = "warning"
                tooltip = f"Learning Rate: {value:.2e} - HIGH: Consider reducing if seeing high clip rates or KL divergence."
            else:
                health = "bad" if value > 1e-3 else "warning"
                tooltip = f"Learning Rate: {value:.2e} - {'VERY HIGH' if value > 1e-3 else 'LOW'}: {'Reduce to prevent instability' if value > 1e-3 else 'May slow learning, consider increasing if stable'}."
                
        elif label == "Explained Var":
            if value > 0.7:
                health = "good"
                tooltip = f"Explained Variance: {value:.3f} - GOOD: Critic predicting values well (>70%). Value function is effective."
            elif value > 0.4:
                health = "warning"
                tooltip = f"Explained Variance: {value:.3f} - OK: Moderate value prediction. May improve with more training."
            else:
                health = "bad"
                tooltip = f"Explained Variance: {value:.3f} - LOW: Poor value prediction (<40%). Check reward signal or network architecture."
                
        elif label == "Mean Reward":
            if value > 0:
                health = "good"
                tooltip = f"Mean Reward: {value:.4f} - GOOD: Positive average return. Policy is learning profitable strategies."
            elif value > -1:
                health = "warning"
                tooltip = f"Mean Reward: {value:.4f} - LOW: Small negative returns. Monitor progress, may need more training."
            else:
                health = "bad"
                tooltip = f"Mean Reward: {value:.4f} - POOR: Large negative returns. Check reward function or reset training."
        else:
            health = "good"
            tooltip = f"{label}: {value:.4f}"
            
        return health, tooltip
            
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
        
        # Initialize candles to ensure it's always defined
        candles = candle_data
        
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
            # Get episode and reset information to determine chart window
            episode_start_time = getattr(state, 'episode_start_time', None)
            current_timestamp = getattr(state, 'current_timestamp', None)
            
            # Determine chart window based on episode state
            if episode_start_time and current_timestamp:
                try:
                    # Parse timestamps
                    episode_start = pd.to_datetime(episode_start_time)
                    current_time = pd.to_datetime(current_timestamp)
                    
                    # Remove timezone info if present for consistent handling
                    if episode_start.tz is not None:
                        episode_start = episode_start.tz_localize(None)
                    if current_time.tz is not None:
                        current_time = current_time.tz_localize(None)
                    
                    # Calculate time windows:
                    # - Show 1 hour before episode start (reset point)
                    # - Show up to 30 minutes after episode start
                    window_start = episode_start - pd.Timedelta(hours=1)
                    window_end = episode_start + pd.Timedelta(minutes=30)
                    
                    # Extend window if current time is beyond the 30-minute mark
                    if current_time > window_end:
                        window_end = current_time + pd.Timedelta(minutes=5)  # Small buffer
                    
                    # Filter candle data to the focused window
                    candle_df = pd.DataFrame(candle_data)
                    candle_df['timestamp'] = pd.to_datetime(candle_df['timestamp']).dt.tz_localize(None)
                    
                    # Filter to window
                    mask = (candle_df['timestamp'] >= window_start) & (candle_df['timestamp'] <= window_end)
                    filtered_candles = candle_df[mask]
                    candles = filtered_candles.to_dict('records') if not filtered_candles.empty else candle_data
                    
                except Exception:
                    # Fallback to all data if timestamp parsing fails - candles already set above
                    pass
            
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
                
                # Add horizontal line for current price
                current_price = getattr(state, 'current_price', 0)
                if current_price > 0:
                    fig.add_hline(
                        y=current_price,
                        line_dash="dash",
                        line_color=DARK_THEME['accent_blue'],
                        line_width=1,
                        annotation_text=f"${current_price:.2f}",
                        annotation_position="right",
                        annotation_font_size=10,
                        annotation_font_color=DARK_THEME['accent_blue'],
                        row=1, col=1
                    )
                
                # Add vertical line for episode start (reset point)
                episode_start_time = getattr(state, 'episode_start_time', None)
                if episode_start_time:
                    try:
                        # Convert to pandas Timestamp and ensure timezone-naive
                        if isinstance(episode_start_time, str):
                            episode_start_ts = pd.to_datetime(episode_start_time).tz_localize(None)
                        else:
                            episode_start_ts = pd.to_datetime(episode_start_time)
                            if hasattr(episode_start_ts, 'tz') and episode_start_ts.tz is not None:
                                episode_start_ts = episode_start_ts.tz_localize(None)
                        
                        # Check if episode start is within chart range
                        if episode_start_ts >= df['timestamp'].min() and episode_start_ts <= df['timestamp'].max():
                            # Add vertical line spanning both price and volume subplots
                            fig.add_shape(
                                type="line",
                                x0=episode_start_ts, x1=episode_start_ts,
                                y0=0, y1=1,
                                yref="paper",
                                line=dict(
                                    color=DARK_THEME['accent_purple'],
                                    width=3,
                                    dash="solid"
                                )
                            )
                            # Add annotation for episode start
                            price_mid = (df['high'].max() + df['low'].min()) / 2
                            fig.add_annotation(
                                x=episode_start_ts,
                                y=price_mid,
                                yref="y",
                                text="Reset",
                                showarrow=True,
                                arrowhead=2,
                                arrowsize=1,
                                arrowcolor=DARK_THEME['accent_purple'],
                                font=dict(size=10, color=DARK_THEME['accent_purple']),
                                bgcolor="rgba(0,0,0,0.7)",
                                bordercolor=DARK_THEME['accent_purple'],
                                borderwidth=1,
                                xshift=-20
                            )
                    except Exception:
                        # Skip episode start line if timestamp conversion fails
                        pass
                
                # Add vertical line for current trading time
                current_timestamp = getattr(state, 'current_timestamp', None)
                if current_timestamp:
                    try:
                        # Convert to pandas Timestamp and ensure timezone-naive
                        if isinstance(current_timestamp, str):
                            current_ts = pd.to_datetime(current_timestamp).tz_localize(None)
                        else:
                            current_ts = pd.to_datetime(current_timestamp)
                            if hasattr(current_ts, 'tz') and current_ts.tz is not None:
                                current_ts = current_ts.tz_localize(None)
                        
                        # Check if current timestamp is within chart range
                        if current_ts >= df['timestamp'].min() and current_ts <= df['timestamp'].max():
                            # Add vertical line spanning both price and volume subplots
                            fig.add_shape(
                                type="line",
                                x0=current_ts, x1=current_ts,
                                y0=0, y1=1,
                                yref="paper",
                                line=dict(
                                    color=DARK_THEME['accent_orange'],
                                    width=2,
                                    dash="dash"
                                )
                            )
                            # Add annotation for current time at top of price chart
                            price_max = df['high'].max()
                            fig.add_annotation(
                                x=current_ts,
                                y=price_max,
                                yref="y",
                                text=f"Now: {state.ny_time}",
                                showarrow=False,
                                font=dict(size=10, color=DARK_THEME['accent_orange']),
                                bgcolor="rgba(0,0,0,0.7)",
                                bordercolor=DARK_THEME['accent_orange'],
                                borderwidth=1,
                                yshift=10
                            )
                    except Exception:
                        # Skip vertical line if timestamp conversion fails
                        pass
                
                # Add reset point markers (show all for now - strategy filtering in agent)
                reset_points_data = getattr(state, 'reset_points_data', [])
                if reset_points_data:
                    # Note: Strategy-based filtering is now handled in the PPO agent
                    # Dashboard shows all reset points for visibility
                    
                    for reset_point in reset_points_data:
                        # Parse reset point timestamp
                        reset_time = reset_point.get('timestamp')
                        reset_price = reset_point.get('price', 0)
                        activity_score = reset_point.get('activity_score', 0)
                        roc_score = reset_point.get('roc_score', 0)
                        combined_score = reset_point.get('combined_score', 0)
                        
                        if reset_time and reset_price > 0:
                            try:
                                reset_dt = pd.to_datetime(reset_time)
                                # Convert UTC reset points to ET for proper chart display
                                if reset_dt.tz is not None:
                                    # Convert from UTC to ET, then remove timezone for chart compatibility
                                    reset_dt = reset_dt.tz_convert('America/New_York').tz_localize(None)
                                else:
                                    # If no timezone, assume UTC and convert to ET
                                    reset_dt = reset_dt.tz_localize('UTC').tz_convert('America/New_York').tz_localize(None)
                            except:
                                continue
                            
                            # Show reset points within full trading session (4 AM to 8 PM ET)
                            chart_date = df['timestamp'].iloc[0].date()
                            session_start = pd.Timestamp(f"{chart_date} 04:00:00").tz_localize(None)
                            session_end = pd.Timestamp(f"{chart_date} 20:00:00").tz_localize(None)
                            
                            if session_start <= reset_dt <= session_end:
                                # Color based on combined score - rank-based system
                                combined_score = reset_point.get('combined_score', 0.5)
                                if combined_score >= 0.8:
                                    marker_color = DARK_THEME['accent_purple']  # Very high quality
                                    marker_size = 10
                                elif combined_score >= 0.6:
                                    marker_color = DARK_THEME['accent_blue']    # High quality
                                    marker_size = 8
                                elif combined_score >= 0.4:
                                    marker_color = DARK_THEME['accent_orange']  # Medium quality
                                    marker_size = 7
                                else:
                                    marker_color = DARK_THEME['text_muted']     # Low quality
                                    marker_size = 6
                                
                                # Place reset points at bottom of volume chart for better visibility
                                # Get the volume range to position markers consistently
                                volume_max = df['volume'].max() if 'volume' in df.columns and not df['volume'].empty else 1000
                                marker_y_volume = -volume_max * 0.1  # 10% below the volume chart
                                
                                fig.add_trace(
                                    go.Scatter(
                                        x=[reset_dt],
                                        y=[marker_y_volume],
                                        mode='markers',
                                        marker=dict(
                                            size=marker_size,
                                            color=marker_color,
                                            symbol='diamond',
                                            line=dict(width=1, color=DARK_THEME['text_primary'])
                                        ),
                                        name='Reset Point',
                                        showlegend=False,
                                        hovertemplate=f"Reset Point<br>Time: {reset_dt.strftime('%H:%M')}<br>Price: ${reset_price:.3f}<br>Activity: {activity_score:.3f}<br>ROC: {roc_score:.3f}<br>Combined: {combined_score:.3f}<extra></extra>"
                                    ),
                                    row=2, col=1  # Place on volume chart (row 2)
                                )

                # Add execution markers (not completed trades)
                executions_data = list(state.recent_executions) if state.recent_executions else []
                if executions_data:
                    for execution in executions_data[-20:]:  # Last 20 executions
                        # Use raw timestamp for chart plotting
                        exec_time = execution.get('timestamp_raw', execution.get('timestamp'))
                        exec_price = execution.get('fill_price', 0)
                        
                        if exec_time and exec_price > 0:
                            # Parse execution timestamp and ensure timezone-naive
                            try:
                                exec_dt = pd.to_datetime(exec_time)
                                # Remove timezone if present
                                if exec_dt.tz is not None:
                                    exec_dt = exec_dt.tz_localize(None)
                            except:
                                continue
                            
                            # Only show executions within the chart time range
                            if exec_dt >= df['timestamp'].min() and exec_dt <= df['timestamp'].max():
                                is_buy = execution.get('side') == 'BUY'
                                marker_color = DARK_THEME['accent_green'] if is_buy else DARK_THEME['accent_red']
                                marker_symbol = 'triangle-up' if is_buy else 'triangle-down'
                                
                                fig.add_trace(
                                    go.Scatter(
                                        x=[exec_dt],
                                        y=[exec_price],
                                        mode='markers',
                                        marker=dict(
                                            size=12,
                                            color=marker_color,
                                            symbol=marker_symbol,
                                            line=dict(width=1, color=DARK_THEME['text_primary'])
                                        ),
                                        name=execution.get('side', 'Execution'),
                                        showlegend=False,
                                        hovertemplate=f"{execution.get('side', 'Execution')}<br>Price: ${exec_price:.3f}<br>Qty: {execution.get('quantity', 0)}<extra></extra>"
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
            height=500
        )
        
        # Update x-axis to show time nicely in NY time
        fig.update_xaxes(
            tickformat='%H:%M',
            tickmode='auto',
            nticks=16,  # Show more ticks for full day
            type='date'
        )
        
        # Set x-axis range based on filtered candle data for focused view
        if candles and len(candles) > 0:
            # Use the filtered candles data range for focused view
            df_filtered = pd.DataFrame(candles)
            df_filtered['timestamp'] = pd.to_datetime(df_filtered['timestamp']).dt.tz_localize(None)
            
            # Set range based on actual filtered data with small buffer
            start_time = df_filtered['timestamp'].min() - pd.Timedelta(minutes=5)
            end_time = df_filtered['timestamp'].max() + pd.Timedelta(minutes=5)
            
            fig.update_xaxes(
                range=[start_time, end_time]
            )
        
        return fig
        
    def _create_attribution_features_content(self, state) -> html.Div:
        """Create top features attribution content"""
        try:
            # Get attribution summary from state
            attribution_summary = getattr(state, 'attribution_summary', None)
            
            if not attribution_summary or not attribution_summary.get('top_features_by_branch'):
                return html.Div([
                    html.Div("🔍 Analyzing features...", style={'color': DARK_THEME['text_secondary'], 'fontSize': '11px', 'textAlign': 'center', 'padding': '20px'})
                ])
            
            top_features = attribution_summary['top_features_by_branch']
            
            feature_elements = []
            branch_colors = {
                'hf': DARK_THEME['accent_red'], 
                'mf': DARK_THEME['accent_blue'], 
                'lf': DARK_THEME['accent_green'], 
                'portfolio': DARK_THEME['accent_orange']
            }
            
            for branch, features in top_features.items():
                if features:  # List of {'name': str, 'importance': float}
                    # Branch header
                    branch_color = branch_colors.get(branch, DARK_THEME['accent_purple'])
                    feature_elements.append(
                        html.Div(f"{branch.upper()}", style={
                            'color': branch_color, 
                            'fontSize': '11px', 
                            'fontWeight': 'bold', 
                            'marginBottom': '2px'
                        })
                    )
                    
                    # Top 3 features for this branch
                    for feature in features[:3]:
                        importance = feature['importance']
                        # Scale importance to percentage for display
                        importance_pct = min(100, abs(importance) * 1000)  # Scale for visibility
                        bar_color = branch_color if importance > 0 else DARK_THEME['text_muted']
                        
                        feature_elements.append(
                            html.Div([
                                html.Div([
                                    html.Span(feature['name'], style={
                                        'color': DARK_THEME['text_primary'], 
                                        'fontSize': '10px',
                                        'display': 'inline-block',
                                        'width': '60%',
                                        'whiteSpace': 'nowrap',
                                        'overflow': 'hidden',
                                        'textOverflow': 'ellipsis'
                                    }),
                                    html.Span(f"{importance:.3f}", style={
                                        'color': branch_color, 
                                        'fontSize': '10px',
                                        'fontWeight': 'bold',
                                        'float': 'right'
                                    })
                                ], style={'marginBottom': '1px'}),
                                html.Div(style={
                                    'height': '2px',
                                    'backgroundColor': DARK_THEME['border'],
                                    'marginBottom': '3px',
                                    'position': 'relative'
                                }, children=[
                                    html.Div(style={
                                        'height': '100%',
                                        'width': f"{importance_pct}%",
                                        'backgroundColor': bar_color,
                                        'transition': 'width 0.3s ease'
                                    })
                                ])
                            ], style={'marginBottom': '4px'})
                        )
                    
                    # Add spacing between branches
                    feature_elements.append(html.Div(style={'height': '4px'}))
            
            if not feature_elements:
                return html.Div([
                    html.Div("No feature data available", style={'color': DARK_THEME['text_secondary'], 'fontSize': '11px', 'textAlign': 'center'})
                ])
            
            return html.Div(feature_elements, style={'maxHeight': '120px', 'overflowY': 'auto'})
            
        except Exception as e:
            return html.Div([
                html.Div(f"Error: {str(e)}", style={'color': DARK_THEME['accent_red'], 'fontSize': '10px'})
            ])
    
    def _create_attribution_quality_content(self, state) -> html.Div:
        """Create attribution quality metrics content"""
        try:
            # Get attribution summary from state
            attribution_summary = getattr(state, 'attribution_summary', None)
            
            if not attribution_summary:
                return html.Div([
                    html.Div("📊 Waiting for attribution analysis...", style={'color': DARK_THEME['text_secondary'], 'fontSize': '11px', 'textAlign': 'center', 'padding': '20px'})
                ])
            
            quality_elements = []
            
            # Consensus score
            consensus = attribution_summary.get('consensus_mean_correlation', 0.0)
            consensus_color = DARK_THEME['accent_green'] if consensus > 0.7 else DARK_THEME['accent_orange'] if consensus > 0.4 else DARK_THEME['accent_red']
            quality_elements.append(
                self._info_row("Consensus", f"{consensus:.3f}", consensus_color)
            )
            
            # Sparsity
            sparsity = attribution_summary.get('quality_sparsity', 0.0)
            sparsity_color = DARK_THEME['accent_blue'] if sparsity > 0.5 else DARK_THEME['text_primary']
            quality_elements.append(
                self._info_row("Sparsity", f"{sparsity:.3f}", sparsity_color)
            )
            
            # Dead features count
            dead_count = attribution_summary.get('dead_features_count', 0)
            dead_color = DARK_THEME['accent_red'] if dead_count > 10 else DARK_THEME['text_primary']
            quality_elements.append(
                self._info_row("Dead Features", str(dead_count), dead_color)
            )
            
            # Branch max attributions
            max_attrs = {}
            for key, value in attribution_summary.items():
                if key.startswith('branch_') and key.endswith('_max_attribution'):
                    branch = key.replace('branch_', '').replace('_max_attribution', '')
                    max_attrs[branch] = value
            
            if max_attrs:
                quality_elements.append(html.Div(style={'height': '4px'}))  # Spacer
                quality_elements.append(
                    html.Div("Max Attribution:", style={'color': DARK_THEME['text_secondary'], 'fontSize': '10px', 'fontWeight': 'bold'})
                )
                
                for branch, max_attr in max_attrs.items():
                    branch_colors = {'hf': DARK_THEME['accent_red'], 'mf': DARK_THEME['accent_blue'], 'lf': DARK_THEME['accent_green'], 'portfolio': DARK_THEME['accent_orange']}
                    branch_color = branch_colors.get(branch, DARK_THEME['text_primary'])
                    quality_elements.append(
                        self._info_row(f"{branch.upper()}", f"{max_attr:.3f}", branch_color)
                    )
            
            return html.Div(quality_elements)
            
        except Exception as e:
            return html.Div([
                html.Div(f"Error: {str(e)}", style={'color': DARK_THEME['accent_red'], 'fontSize': '10px'})
            ])

    def run(self):
        """Run the dashboard server"""
        if self.app is None:
            self.create_app()
        
        # Suppress Werkzeug INFO logs
        werkzeug_logger = logging.getLogger('werkzeug')
        werkzeug_logger.setLevel(logging.WARNING)
            
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