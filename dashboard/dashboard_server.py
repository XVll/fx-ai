# dashboard/dashboard_server.py - Dash web server implementation

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
                
                # Row 4: Full-width Chart
                html.Div([
                    html.H4("Price & Volume", style={'color': DARK_THEME['text_primary'], 'marginBottom': '4px', 'fontSize': '12px', 'fontWeight': 'bold'}),
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
             Output('trade-counter', 'children'),
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
            
            # Add invalid actions row
            invalid_actions = getattr(state, 'invalid_actions', 0)
            action_data.append({
                'Action': 'INVALID',
                'Episode': f"{invalid_actions}",
                'Episode %': "—",
                'Session': f"{invalid_actions}",
                'Session %': "—"
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
                    },
                    {
                        'if': {'column_id': 'Action', 'filter_query': '{Action} = INVALID'},
                        'color': DARK_THEME['accent_orange']
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
                        'fontSize': '10px', 
                        'textAlign': 'center',
                        'marginTop': '2px'
                    })
                ])
            ])
            
            # Training progress
            max_updates = getattr(state, 'max_updates', 0)
            update_display = f"{state.updates:,}" + (f"/{max_updates:,}" if max_updates > 0 else "")
            
            training_children = [
                self._info_row("Mode", state.mode, color=DARK_THEME['accent_purple']),
                self._info_row("Stage", state.stage),
                self._info_row("Episodes", f"{state.total_episodes:,}"),
                self._info_row("Updates", update_display),
                self._info_row("Global Steps", f"{state.global_steps:,}"),
            ]
            
            # Add momentum day information if available
            if state.current_momentum_day_date:
                training_children.extend([
                    html.Hr(style={'border': '1px solid ' + DARK_THEME['border'], 'margin': '8px 0'}),
                    self._info_row("Current Day", state.current_momentum_day_date, color=DARK_THEME['accent_blue']),
                    self._info_row("Day Quality", f"{state.current_momentum_day_quality:.3f}"),
                    self._info_row("Episodes on Day", f"{state.episodes_on_current_day}"),
                    self._info_row("Cycles Complete", f"{state.reset_point_cycles_completed}"),
                ])
            
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
            
            # PPO Metrics - simplified
            ppo_content = html.Div([
                self._info_row("Policy Loss", f"{state.policy_loss:.4f}"),
                self._info_row("Value Loss", f"{state.value_loss:.4f}"),
                self._info_row("Entropy", f"{state.entropy:.4f}"),
                self._info_row("Learn Rate", f"{state.learning_rate:.2e}"),
                self._info_row("KL Div", f"{getattr(state, 'kl_divergence', 0.0):.4f}"),
                self._info_row("Clip Frac", f"{getattr(state, 'clip_fraction', 0.0):.2f}"),
            ])
            
            # Reward components table - redesigned to match actions panel format
            # Define all active reward components with their types
            all_reward_components = {
                'realized_pnl': 'foundational',
                'holding_time_penalty': 'shaping',
                'overtrading_penalty': 'shaping',
                'quick_profit_incentive': 'shaping',
                'drawdown_penalty': 'shaping',
                'terminal_penalty': 'terminal',
                'mark_to_market': 'foundational',
                'mae_penalty': 'trade',
                'mfe_penalty': 'trade'
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
            
            # 3-component scores from current reset point
            current_direction_score = getattr(state, 'current_direction_score', 0.0)
            current_roc_score = getattr(state, 'current_roc_score', 0.0)
            current_activity_score = getattr(state, 'current_activity_score', 0.0)
            
            # Curriculum thresholds
            min_roc_score = getattr(state, 'curriculum_min_roc', 0.0)
            min_activity_score = getattr(state, 'curriculum_min_activity', 0.0)
            min_direction_score = getattr(state, 'curriculum_min_direction', 0.0)
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
            
            # Calculate next stage progress
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
            
            # Add reset points info
            reset_points_count = len(getattr(state, 'reset_points_data', []))
            
            env_content = html.Div([
                self._info_row("Stage", stage_display, color=curriculum_color),
                self._info_row("Progress", f"{progress_pct:.1f}%", color=curriculum_color),
                self._info_row("Episode Len", f"{episode_length} steps"),
                self._info_row("Reset Points", f"{reset_points_count}", color=DARK_THEME['accent_blue']),
                self._info_row("Total Episodes", f"{total_episodes_curriculum:,}"),
                # 3-Component Scores with threshold indicators
                html.Hr(style={'margin': '4px 0', 'borderColor': DARK_THEME['border']}),
                self._info_row("Direction", f"{current_direction_score:.2f}", 
                             color=DARK_THEME['accent_green'] if abs(current_direction_score) >= min_direction_score else DARK_THEME['text_muted']),
                self._info_row("ROC", f"{current_roc_score:.2f}", 
                             color=DARK_THEME['accent_blue'] if current_roc_score >= min_roc_score else DARK_THEME['text_muted']),
                self._info_row("Activity", f"{current_activity_score:.2f}", 
                             color=DARK_THEME['accent_orange'] if current_activity_score >= min_activity_score else DARK_THEME['text_muted']),
            ])
            
            # Custom candlestick chart
            candlestick_chart = self._create_price_chart(state)
            
            # Performance footer with eps/Hour moved here
            eps_per_hour = getattr(state, 'episodes_per_hour', 0.0)
            updates_per_hour = state.updates_per_second * 3600
            footer = f"Steps/sec: {state.steps_per_second:.1f} | Updates/hr: {updates_per_hour:.1f} | Episodes/hr: {eps_per_hour:.1f}"
            
            return (header_info, session_time_str, market_info, position_info, portfolio_info,
                   trades_table, trade_counter_text, actions_content, episode_content, training_content, ppo_content,
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
            html.Div(text, style={'color': DARK_THEME['text_primary'], 'fontSize': '10px', 'textAlign': 'center'})
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
                    batch_progress = min(100, (current_batch / total_batches) * 100)
                    text = f"Epoch {current_epoch}/{total_epochs}, Batch {current_batch}/{total_batches}"
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
            html.Div(title, style={'color': DARK_THEME['text_secondary'], 'fontSize': '10px', 'marginBottom': '2px'}),
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
            html.Div(text, style={'color': DARK_THEME['text_primary'], 'fontSize': '10px', 'textAlign': 'center'})
        ], style={'marginTop': '8px'})
        
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
                
                # Add reset point markers
                reset_points_data = getattr(state, 'reset_points_data', [])
                if reset_points_data:
                    for reset_point in reset_points_data:
                        # Parse reset point timestamp
                        reset_time = reset_point.get('timestamp')
                        reset_price = reset_point.get('price', 0)
                        activity_score = reset_point.get('activity_score', 0)
                        combined_score = reset_point.get('combined_score', 0)
                        
                        if reset_time and reset_price > 0:
                            try:
                                reset_dt = pd.to_datetime(reset_time)
                                # Remove timezone if present
                                if reset_dt.tz is not None:
                                    reset_dt = reset_dt.tz_localize(None)
                            except:
                                continue
                            
                            # Show reset points within full trading session (4 AM to 8 PM ET)
                            chart_date = df['timestamp'].iloc[0].date()
                            session_start = pd.Timestamp(f"{chart_date} 04:00:00").tz_localize(None)
                            session_end = pd.Timestamp(f"{chart_date} 20:00:00").tz_localize(None)
                            
                            if session_start <= reset_dt <= session_end:
                                # Color based on activity score - higher score = more blue/purple
                                if activity_score >= 0.7:
                                    marker_color = DARK_THEME['accent_purple']  # High activity
                                    marker_size = 10
                                elif activity_score >= 0.5:
                                    marker_color = DARK_THEME['accent_blue']    # Medium activity
                                    marker_size = 8
                                else:
                                    marker_color = DARK_THEME['text_muted']     # Low activity
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
                                        hovertemplate=f"Reset Point<br>Time: {reset_dt.strftime('%H:%M')}<br>Price: ${reset_price:.3f}<br>Activity: {activity_score:.3f}<br>Combined: {combined_score:.3f}<extra></extra>"
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
        
        # If we have data, set the x-axis range to show full trading day (4 AM to 8 PM)
        if candle_data and len(candle_data) > 0:
            # Get the date from the first candle
            first_candle = candle_data[0]
            if 'timestamp' in first_candle:
                # Parse the date
                first_ts = pd.to_datetime(first_candle['timestamp'])
                date_str = first_ts.strftime('%Y-%m-%d')
                
                # Set range from 4 AM to 8 PM for that date to include all reset points
                fig.update_xaxes(
                    range=[f'{date_str} 04:00:00', f'{date_str} 20:00:00']
                )
        
        return fig
        
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