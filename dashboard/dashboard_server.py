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
import numpy as np

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
                    html.H1("FxAI Trading Dashboard", style={'margin': '0', 'color': DARK_THEME['text_primary']}),
                    html.Div(id='header-info', style={'color': DARK_THEME['text_secondary']})
                ], style={'flex': '1'}),
                html.Div([
                    html.Span("Session Time: ", style={'color': DARK_THEME['text_secondary']}),
                    html.Span(id='session-time', style={'color': DARK_THEME['accent_blue']})
                ])
            ], style={
                'display': 'flex',
                'justifyContent': 'space-between',
                'alignItems': 'center',
                'padding': '20px',
                'backgroundColor': DARK_THEME['bg_secondary'],
                'borderBottom': f"1px solid {DARK_THEME['border']}"
            }),
            
            # Main content area
            html.Div([
                # Row 1: Market Info and Position
                html.Div([
                    # Market Info Card
                    html.Div([
                        html.H3("Market Info", style={'color': DARK_THEME['text_primary'], 'marginBottom': '15px'}),
                        html.Div(id='market-info-content')
                    ], style=self._card_style(), className='col-6'),
                    
                    # Position Card
                    html.Div([
                        html.H3("Position", style={'color': DARK_THEME['text_primary'], 'marginBottom': '15px'}),
                        html.Div(id='position-content')
                    ], style=self._card_style(), className='col-6'),
                ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}),
                
                # Row 2: Portfolio and Recent Trades
                html.Div([
                    # Portfolio Card
                    html.Div([
                        html.H3("Portfolio", style={'color': DARK_THEME['text_primary'], 'marginBottom': '15px'}),
                        html.Div(id='portfolio-content')
                    ], style=self._card_style(), className='col-6'),
                    
                    # Recent Trades Card
                    html.Div([
                        html.H3("Recent Trades", style={'color': DARK_THEME['text_primary'], 'marginBottom': '15px'}),
                        html.Div(id='trades-table-container')
                    ], style=self._card_style(), className='col-6'),
                ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}),
                
                # Row 3: Actions and Episode Info
                html.Div([
                    # Actions Card
                    html.Div([
                        html.H3("Actions Analysis", style={'color': DARK_THEME['text_primary'], 'marginBottom': '15px'}),
                        html.Div(id='actions-content')
                    ], style=self._card_style(), className='col-6'),
                    
                    # Episode Info Card
                    html.Div([
                        html.H3("Episode Info", style={'color': DARK_THEME['text_primary'], 'marginBottom': '15px'}),
                        html.Div(id='episode-content')
                    ], style=self._card_style(), className='col-6'),
                ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}),
                
                # Row 4: Training Progress and PPO Metrics
                html.Div([
                    # Training Progress Card
                    html.Div([
                        html.H3("Training Progress", style={'color': DARK_THEME['text_primary'], 'marginBottom': '15px'}),
                        html.Div(id='training-content')
                    ], style=self._card_style(), className='col-6'),
                    
                    # PPO Metrics Card
                    html.Div([
                        html.H3("PPO Metrics", style={'color': DARK_THEME['text_primary'], 'marginBottom': '15px'}),
                        html.Div(id='ppo-metrics-content')
                    ], style=self._card_style(), className='col-6'),
                ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}),
                
                # Row 5: Reward Components and Environment Info
                html.Div([
                    # Reward Components Card
                    html.Div([
                        html.H3("Reward Components", style={'color': DARK_THEME['text_primary'], 'marginBottom': '15px'}),
                        dcc.Graph(id='reward-components-chart', style={'height': '300px'})
                    ], style=self._card_style(), className='col-8'),
                    
                    # Environment Info Card
                    html.Div([
                        html.H3("Environment", style={'color': DARK_THEME['text_primary'], 'marginBottom': '15px'}),
                        html.Div(id='env-content')
                    ], style=self._card_style(), className='col-4'),
                ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}),
                
                # Row 6: Full-width Chart
                html.Div([
                    html.H3("Price Action & Trades", style={'color': DARK_THEME['text_primary'], 'marginBottom': '15px'}),
                    dcc.Graph(id='price-chart', style={'height': '400px'})
                ], style=self._card_style()),
                
            ], style={'padding': '20px', 'backgroundColor': DARK_THEME['bg_primary']}),
            
            # Footer
            html.Div([
                html.Div(id='performance-footer', style={'color': DARK_THEME['text_secondary']})
            ], style={
                'padding': '10px 20px',
                'backgroundColor': DARK_THEME['bg_secondary'],
                'borderTop': f"1px solid {DARK_THEME['border']}",
                'textAlign': 'center'
            }),
            
            # Auto-refresh interval
            dcc.Interval(id='interval-component', interval=500, n_intervals=0)  # 500ms refresh
            
        ], style={'backgroundColor': DARK_THEME['bg_primary'], 'minHeight': '100vh'})
        
    def _card_style(self) -> Dict[str, str]:
        """Standard card styling"""
        return {
            'backgroundColor': DARK_THEME['bg_secondary'],
            'border': f"1px solid {DARK_THEME['border']}",
            'borderRadius': '6px',
            'padding': '20px',
            'flex': '1'
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
             Output('reward-components-chart', 'figure'),
             Output('env-content', 'children'),
             Output('price-chart', 'figure'),
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
                self._info_row("NY Time", ny_time),
                self._info_row("Session", trading_hours, color=DARK_THEME['accent_orange']),
                self._info_row("Price", f"${state.current_price:.2f}", color=DARK_THEME['accent_blue']),
                self._info_row("Bid/Ask", f"${state.bid_price:.2f} / ${state.ask_price:.2f}"),
                self._info_row("Spread", f"${spread:.3f}"),
                self._info_row("Volume", f"{state.volume:,}"),
            ])
            
            # Position info
            if state.position_side != "FLAT":
                position_pnl = getattr(state, 'position_pnl_dollar', 0.0)
                pnl_color = DARK_THEME['accent_green'] if position_pnl >= 0 else DARK_THEME['accent_red']
                position_info = html.Div([
                    self._info_row("Side", state.position_side, color=DARK_THEME['accent_blue']),
                    self._info_row("Quantity", f"{state.position_qty:,}"),
                    self._info_row("Avg Entry", f"${state.avg_entry_price:.3f}"),
                    self._info_row("P&L", f"${position_pnl:.2f}", color=pnl_color),
                    self._info_row("P&L %", f"{state.position_pnl_percent:.2f}%", color=pnl_color),
                ])
            else:
                position_info = html.Div([
                    html.Div("No Position", style={'color': DARK_THEME['text_muted'], 'textAlign': 'center', 'padding': '40px'})
                ])
            
            # Portfolio info
            session_pnl_color = DARK_THEME['accent_green'] if state.session_pnl >= 0 else DARK_THEME['accent_red']
            portfolio_info = html.Div([
                self._info_row("Total Equity", f"${state.total_equity:.2f}"),
                self._info_row("Cash", f"${state.cash_balance:.2f}"),
                self._info_row("Session P&L", f"${state.session_pnl:.2f}", color=session_pnl_color),
                self._info_row("Realized P&L", f"${state.realized_pnl:.2f}"),
                self._info_row("Max Drawdown", f"{state.max_drawdown:.2%}", color=DARK_THEME['accent_orange']),
                self._info_row("Sharpe Ratio", f"{state.sharpe_ratio:.2f}"),
                self._info_row("Win Rate", f"{state.win_rate:.1%}"),
            ])
            
            # Recent trades table
            if state.recent_trades:
                trades_df = pd.DataFrame(list(state.recent_trades)[-10:])  # Last 10 trades
                trades_table = dash_table.DataTable(
                    data=trades_df.to_dict('records'),
                    columns=[
                        {'name': 'Time', 'id': 'time'},
                        {'name': 'Side', 'id': 'side'},
                        {'name': 'Qty', 'id': 'quantity'},
                        {'name': 'Price', 'id': 'price', 'type': 'numeric', 'format': {'specifier': '$.3f'}},
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
            
            # Actions analysis
            actions_content = html.Div([
                html.Div("Action Distribution:", style={'color': DARK_THEME['text_secondary'], 'marginBottom': '10px'}),
                html.Div([
                    html.Div([
                        html.Span("HOLD: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(f"{state.action_distribution.get('HOLD', 0)}", style={'color': DARK_THEME['accent_blue']})
                    ], style={'flex': '1'}),
                    html.Div([
                        html.Span("BUY: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(f"{state.action_distribution.get('BUY', 0)}", style={'color': DARK_THEME['accent_green']})
                    ], style={'flex': '1'}),
                    html.Div([
                        html.Span("SELL: ", style={'color': DARK_THEME['text_secondary']}),
                        html.Span(f"{state.action_distribution.get('SELL', 0)}", style={'color': DARK_THEME['accent_red']})
                    ], style={'flex': '1'}),
                ], style={'display': 'flex', 'marginBottom': '15px'}),
                html.Div("Recent Actions:", style={'color': DARK_THEME['text_secondary'], 'marginBottom': '10px'}),
                html.Div([
                    html.Div(f"{a['time']}: {a['action']} ({a['confidence']:.1%})", 
                            style={'color': DARK_THEME['text_muted'], 'fontSize': '12px', 'marginBottom': '5px'})
                    for a in list(state.recent_actions)[-5:]
                ])
            ])
            
            # Episode info
            progress = (state.current_step / state.max_steps * 100) if state.max_steps > 0 else 0
            episode_content = html.Div([
                self._info_row("Episode", f"{state.episode_number}"),
                self._info_row("Step", f"{state.current_step:,} / {state.max_steps:,}"),
                self._info_row("Progress", f"{progress:.1f}%"),
                self._info_row("Cumulative Reward", f"{state.cumulative_reward:.2f}"),
                self._info_row("Last Step Reward", f"{state.last_step_reward:.3f}"),
                html.Div([
                    html.Div(style={
                        'backgroundColor': DARK_THEME['bg_tertiary'],
                        'height': '10px',
                        'borderRadius': '5px',
                        'marginTop': '10px'
                    }, children=[
                        html.Div(style={
                            'backgroundColor': DARK_THEME['accent_blue'],
                            'height': '100%',
                            'width': f"{progress}%",
                            'borderRadius': '5px',
                            'transition': 'width 0.3s ease'
                        })
                    ])
                ])
            ])
            
            # Training progress
            invalid_actions = getattr(state, 'invalid_actions', 0)
            training_content = html.Div([
                self._info_row("Mode", state.mode, color=DARK_THEME['accent_purple']),
                self._info_row("Stage", state.stage),
                self._info_row("Episodes", f"{state.total_episodes:,}"),
                self._info_row("Steps", f"{state.global_steps:,}"),
                self._info_row("Invalid Actions", f"{invalid_actions}"),
            ])
            
            # PPO Metrics with sparklines
            policy_loss_history = getattr(state, 'policy_loss_history', [])
            value_loss_history = getattr(state, 'value_loss_history', [])
            entropy_history = getattr(state, 'entropy_history', [])
            clip_range = getattr(state, 'clip_range', 0.2)
            ppo_content = html.Div([
                self._metric_with_sparkline("Policy Loss", state.policy_loss, policy_loss_history),
                self._metric_with_sparkline("Value Loss", state.value_loss, value_loss_history),
                self._metric_with_sparkline("Entropy", state.entropy, entropy_history),
                self._info_row("Learning Rate", f"{state.learning_rate:.2e}"),
                self._info_row("Clip Range", f"{clip_range:.3f}"),
            ])
            
            # Reward components chart
            if state.reward_components:
                components = list(state.reward_components.keys())
                values = list(state.reward_components.values())
                colors = [DARK_THEME['accent_green'] if v >= 0 else DARK_THEME['accent_red'] for v in values]
                
                reward_fig = go.Figure([go.Bar(
                    x=components,
                    y=values,
                    marker_color=colors,
                    text=[f"{v:.2f}" for v in values],
                    textposition='auto',
                )])
                
                reward_fig.update_layout(
                    plot_bgcolor=DARK_THEME['bg_tertiary'],
                    paper_bgcolor=DARK_THEME['bg_secondary'],
                    font_color=DARK_THEME['text_primary'],
                    xaxis=dict(gridcolor=DARK_THEME['border']),
                    yaxis=dict(gridcolor=DARK_THEME['border']),
                    margin=dict(l=40, r=40, t=40, b=40),
                    showlegend=False
                )
            else:
                reward_fig = {}
            
            # Environment info
            avg_spread = getattr(state, 'avg_spread', state.spread)
            volume_ratio = getattr(state, 'volume_ratio', 1.0)
            env_content = html.Div([
                self._info_row("Data Quality", f"{state.data_quality:.1%}"),
                self._info_row("Momentum Score", f"{state.momentum_score:.2f}"),
                self._info_row("Volatility", f"{state.volatility:.2%}"),
                self._info_row("Bid-Ask Spread", f"${avg_spread:.3f}"),
                self._info_row("Volume Ratio", f"{volume_ratio:.1f}x"),
            ])
            
            # Price chart with trades
            price_fig = self._create_price_chart(state)
            
            # Performance footer
            footer = f"Steps/sec: {state.steps_per_second:.1f} | Updates/sec: {state.updates_per_second:.1f} | Episodes/hr: {state.episodes_per_hour:.1f}"
            
            return (header_info, session_time_str, market_info, position_info, portfolio_info,
                   trades_table, actions_content, episode_content, training_content, ppo_content,
                   reward_fig, env_content, price_fig, footer)
            
    def _info_row(self, label: str, value: str, color: Optional[str] = None) -> html.Div:
        """Create an info row with label and value"""
        value_color = color or DARK_THEME['text_primary']
        return html.Div([
            html.Span(f"{label}: ", style={'color': DARK_THEME['text_secondary']}),
            html.Span(value, style={'color': value_color, 'fontWeight': 'bold'})
        ], style={'marginBottom': '8px'})
        
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
        """Create candlestick chart with trade markers"""
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.03, row_heights=[0.7, 0.3])
        
        if state.price_history:
            # Price line
            price_history_list = list(state.price_history)[-200:]  # Last 200 points
            times = [p['time'] for p in price_history_list]
            prices = [p['price'] for p in price_history_list]
            
            fig.add_trace(go.Scatter(
                x=times, y=prices,
                mode='lines',
                name='Price',
                line=dict(color=DARK_THEME['accent_blue'], width=2)
            ), row=1, col=1)
            
            # Add trade markers
            for trade in list(state.recent_trades)[-20:]:  # Last 20 trades
                color = DARK_THEME['accent_green'] if trade['side'] == 'BUY' else DARK_THEME['accent_red']
                fig.add_trace(go.Scatter(
                    x=[trade['time']],
                    y=[trade['price']],
                    mode='markers',
                    marker=dict(size=10, color=color, symbol='triangle-up' if trade['side'] == 'BUY' else 'triangle-down'),
                    name=trade['side'],
                    showlegend=False
                ), row=1, col=1)
            
            # Volume bars
            volume_history = getattr(state, 'volume_history', [])
            if volume_history:
                vol_times = [v['time'] for v in volume_history[-200:]]
                volumes = [v['volume'] for v in volume_history[-200:]]
                
                fig.add_trace(go.Bar(
                    x=vol_times, y=volumes,
                    name='Volume',
                    marker_color=DARK_THEME['accent_purple']
                ), row=2, col=1)
        
        # Update layout
        fig.update_layout(
            plot_bgcolor=DARK_THEME['bg_tertiary'],
            paper_bgcolor=DARK_THEME['bg_secondary'],
            font_color=DARK_THEME['text_primary'],
            xaxis=dict(gridcolor=DARK_THEME['border']),
            yaxis=dict(gridcolor=DARK_THEME['border'], title='Price'),
            xaxis2=dict(gridcolor=DARK_THEME['border']),
            yaxis2=dict(gridcolor=DARK_THEME['border'], title='Volume'),
            margin=dict(l=60, r=40, t=40, b=40),
            showlegend=False,
            hovermode='x unified'
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