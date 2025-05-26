"""
Improved Live Trading Dashboard with better layout and visualization.
Enhanced candlestick charts, compact design, and better progress tracking.
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import threading
import webbrowser
import logging
from typing import Dict, List, Any, Optional
import queue
import time
import os
from datetime import datetime

from .dashboard_data import DashboardState

logger = logging.getLogger(__name__)

# Suppress Dash/Flask/Werkzeug logs
logging.getLogger('werkzeug').setLevel(logging.ERROR)
logging.getLogger('dash').setLevel(logging.ERROR)
os.environ['DASH_SILENCE_ROUTES_LOGGING'] = 'true'


class LiveTradingDashboard:
    """Real-time trading dashboard with improved layout"""
    
    def __init__(self, port: int = 8050, update_interval: int = 500):
        self.port = port
        self.update_interval = update_interval
        
        # Central state management
        self.state = DashboardState()
        
        # Update queue for thread safety
        self.update_queue = queue.Queue()
        
        # Dashboard app
        self.app = None
        self.server_thread = None
        self.is_running = False
        
        self._create_app()
    
    def _create_app(self):
        """Create the Dash application with improved layout"""
        self.app = dash.Dash(__name__)
        
        # Custom CSS for compact layout
        self.app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>FX-AI Trading Dashboard</title>
                {%favicon%}
                {%css%}
                <style>
                    body {
                        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
                        background-color: #0e1117;
                        color: #e6e6e6;
                        margin: 0;
                        padding: 0;
                        font-size: 14px;
                        line-height: 1.6;
                    }
                    .main-header {
                        background-color: #1e2130;
                        padding: 3px 10px;
                        border-bottom: 1px solid #3d4158;
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        height: 25px;
                    }
                    .main-container {
                        padding: 10px 15px;
                        display: grid;
                        grid-template-columns: 40% 30% 30%;
                        gap: 10px;
                        max-width: 100%;
                        height: auto;
                        min-height: 500px;
                    }
                    .panel {
                        background-color: #1e2130;
                        border-radius: 3px;
                        padding: 10px;
                        overflow: hidden;
                        border: 1px solid #3d4158;
                    }
                    .panel-title {
                        font-size: 14px;
                        font-weight: bold;
                        margin: 0 0 8px 0;
                        color: #00d084;
                        border-bottom: 1px solid #3d4158;
                        padding-bottom: 4px;
                        padding-bottom: 2px;
                    }
                    .metric-row {
                        display: flex;
                        justify-content: space-between;
                        margin: 3px 0;
                        font-size: 12px;
                    }
                    .metric-label {
                        color: #999;
                    }
                    .metric-value {
                        font-weight: bold;
                    }
                    .positive { color: #00d084; }
                    .negative { color: #ff4757; }
                    .neutral { color: #ffd32a; }
                    .flat { color: #999; }
                    table {
                        width: 100%;
                        font-size: 11px;
                        border-collapse: collapse;
                    }
                    th {
                        background-color: #262837;
                        padding: 2px;
                        text-align: left;
                        font-weight: bold;
                        border-bottom: 1px solid #3d4158;
                    }
                    td {
                        padding: 2px;
                        border-bottom: 1px solid #2a2b3d;
                    }
                    .chart-container {
                        margin-top: 3px;
                    }
                    .progress-bar {
                        background-color: #3d4158;
                        height: 10px;
                        border-radius: 2px;
                        overflow: hidden;
                        margin: 2px 0;
                        position: relative;
                    }
                    .progress-fill {
                        background-color: #00d084;
                        height: 100%;
                        transition: width 0.3s ease;
                        position: absolute;
                        left: 0;
                        top: 0;
                    }
                    .progress-text {
                        position: absolute;
                        width: 100%;
                        text-align: center;
                        color: white;
                        font-size: 8px;
                        line-height: 10px;
                        font-weight: bold;
                        text-shadow: 1px 1px 1px rgba(0,0,0,0.5);
                    }
                    .footer {
                        background-color: #1e2130;
                        padding: 5px 10px;
                        border-top: 1px solid #3d4158;
                        font-size: 11px;
                        color: #999;
                        display: flex;
                        justify-content: space-between;
                        height: 25px;
                        align-items: center;
                    }
                    .modebar {
                        display: none !important;
                    }
                    .js-plotly-plot .plotly .gtitle {
                        font-size: 10px !important;
                    }
                    /* Compact chart styling */
                    .js-plotly-plot {
                        margin: 0 !important;
                    }
                    .main-svg {
                        overflow: visible !important;
                    }
                    /* Remove chart container borders */
                    .chart-panel {
                        padding: 0 !important;
                        background: transparent !important;
                        box-shadow: none !important;
                    }
                    .js-plotly-plot .plotly {
                        border-radius: 3px;
                        overflow: hidden;
                    }
                </style>
            </head>
            <body>
                {%app_entry%}
                <footer>
                    {%config%}
                    {%scripts%}
                    {%renderer%}
                </footer>
            </body>
        </html>
        '''
        
        # App layout with improved chart placement
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.Div([
                    html.Span("🤖 FX-AI Trading System", style={'fontWeight': 'bold', 'fontSize': '12px'}),
                    html.Span(id='model-info', style={'fontSize': '10px', 'color': '#999'})
                ]),
                html.Div(id='live-time', style={'fontFamily': 'monospace', 'fontSize': '10px'})
            ], className='main-header'),
            
            # Main content area
            html.Div([
                # Top section - 3 column grid for panels
                html.Div([
                    # Column 1 - Market & Trading
                    html.Div([
                        # Market Data Panel
                        html.Div([
                            html.H3("📊 Market Data", className='panel-title'),
                            html.Div(id='market-content')
                        ], className='panel'),
                        
                        # Position Status Panel
                        html.Div([
                            html.H3("💼 Position Status", className='panel-title'),
                            html.Div(id='position-content')
                        ], className='panel'),
                        
                        # Portfolio Panel
                        html.Div([
                            html.H3("📈 Portfolio", className='panel-title'),
                            html.Div(id='portfolio-content')
                        ], className='panel'),
                    ]),
                
                # Column 2 - Actions & Analysis
                html.Div([
                    # Reward System Panel - moved here from column 3
                    html.Div([
                        html.H3("🏆 Reward Components", className='panel-title'),
                        html.Div(id='reward-system')
                    ], className='panel'),
                    
                    # Action Bias Summary Panel
                    html.Div([
                        html.H3("🎯 Action Bias Summary", className='panel-title'),
                        html.Div(id='action-bias-summary')
                    ], className='panel'),
                    
                    # Episode Analysis Panel (includes Trading Activity)
                    html.Div([
                        html.H3("🎬 Episode Analysis", className='panel-title', id='episode-title'),
                        html.Div(id='episode-content'),
                        html.Hr(style={'margin': '8px 0', 'borderColor': '#3d4158'}),
                        
                        # Trading Activity section
                        html.Div([
                            html.H4("Recent Executions", style={'fontSize': '12px', 'margin': '0 0 5px 0', 'color': '#ffd32a'}),
                            html.Div(id='recent-executions', style={'marginBottom': '10px'}),
                            
                            html.H4("Completed Trades", style={'fontSize': '12px', 'margin': '0 0 5px 0', 'color': '#00d084'}),
                            html.Div(id='recent-trades', style={'marginBottom': '10px'}),
                            
                            html.H4("Episode History", style={'fontSize': '12px', 'margin': '0 0 5px 0', 'color': '#999'}),
                            html.Div(id='episode-history')
                        ])
                    ], className='panel', style={'gridRow': 'span 2'}),  # Make it taller
                ]),
                
                # Column 3 - Training & Metrics
                html.Div([
                    # Training Progress with enhanced display
                    html.Div([
                        html.H3("⚙️ Training Progress", className='panel-title'),
                        html.Div(id='training-progress')
                    ], className='panel'),
                    
                    # Training Chart (compact)
                    html.Div([
                        html.H3("📈 Episode Rewards", className='panel-title'),
                        dcc.Graph(
                            id='training-chart', 
                            style={'height': '120px'},
                            config={'displayModeBar': False}
                        )
                    ], className='panel'),
                    
                    # PPO Metrics Panel
                    html.Div([
                        html.H3("🧠 PPO Core Metrics", className='panel-title'),
                        html.Div(id='ppo-metrics')
                    ], className='panel'),
                    
                    # Model Internals Panel - NEW
                    html.Div([
                        html.H3("🔍 Model Internals", className='panel-title'),
                        html.Div(id='model-internals')
                    ], className='panel'),
                ]),
                ], className='main-container'),
                
                # Bottom section - Full-width chart
                html.Div([
                    html.Div([
                        html.H3("📈 Market Analysis", className='panel-title'),
                        dcc.Graph(
                            id='full-day-chart', 
                            style={'height': '400px'},
                            config={'displayModeBar': False}
                        )
                    ], className='panel')
                ], style={'margin': '10px 20px', 'width': 'calc(100% - 40px)'}),
            ]),
            
            # Footer
            html.Div(id='footer-stats', className='footer'),
            
            # Auto-update interval
            dcc.Interval(id='interval-component', interval=self.update_interval),
            
            # Hidden div for data storage
            html.Div(id='hidden-data', style={'display': 'none'})
        ])
        
        # Setup callbacks
        self._setup_callbacks()
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks with improved functionality"""
        
        @self.app.callback(
            [Output('model-info', 'children'),
             Output('live-time', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_header(n):
            model_info = f"Model: {self.state.model_name}"
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            return model_info, current_time
        
        @self.app.callback(
            Output('market-content', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_market_panel(n):
            m = self.state.market_data
            
            content = html.Div([
                html.Div([
                    html.Span("Symbol:", className='metric-label'),
                    html.Span(m.symbol, className='metric-value')
                ], className='metric-row'),
                html.Div([
                    html.Span("Price:", className='metric-label'),
                    html.Span(f"${m.price:.2f}", className='metric-value')
                ], className='metric-row'),
                html.Div([
                    html.Span("Bid/Ask:", className='metric-label'),
                    html.Span(f"${m.bid:.2f} / ${m.ask:.2f}", className='metric-value')
                ], className='metric-row'),
                html.Div([
                    html.Span("Spread:", className='metric-label'),
                    html.Span(f"${m.spread:.3f}", className='metric-value')
                ], className='metric-row'),
                html.Div([
                    html.Span("Volume:", className='metric-label'),
                    html.Span(f"{m.volume:,.0f}", className='metric-value')
                ], className='metric-row'),
                html.Div([
                    html.Span("Time (NY):", className='metric-label'),
                    html.Span(m.time_ny, className='metric-value')
                ], className='metric-row'),
            ])
            
            return content
        
        @self.app.callback(
            Output('position-content', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_position_panel(n):
            pos = self.state.position
            
            side_color = {'Long': 'positive', 'Short': 'negative', 'Flat': 'flat'}.get(pos.side, 'neutral')
            pnl_color = 'positive' if pos.pnl_dollars >= 0 else 'negative'
            
            content = html.Div([
                html.Div([
                    html.Span("Side:", className='metric-label'),
                    html.Span(pos.side, className=f'metric-value {side_color}')
                ], className='metric-row'),
                html.Div([
                    html.Span("Quantity:", className='metric-label'),
                    html.Span(f"{abs(pos.quantity):,.0f}", className='metric-value')
                ], className='metric-row'),
                html.Div([
                    html.Span("Avg Price:", className='metric-label'),
                    html.Span(f"${pos.avg_entry_price:.2f}", className='metric-value')
                ], className='metric-row'),
                html.Div([
                    html.Span("Current Price:", className='metric-label'),
                    html.Span(f"${pos.current_price:.2f}", className='metric-value')
                ], className='metric-row'),
                html.Div([
                    html.Span("P&L:", className='metric-label'),
                    html.Span([
                        f"${pos.pnl_dollars:+,.2f} ",
                        html.Span(f"({pos.pnl_percent:+.2f}%)", style={'fontSize': '10px'})
                    ], className=f'metric-value {pnl_color}')
                ], className='metric-row'),
            ])
            
            return content
        
        @self.app.callback(
            Output('portfolio-content', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_portfolio_panel(n):
            p = self.state.portfolio
            equity_color = 'positive' if p.session_pnl >= 0 else 'negative'
            
            content = html.Div([
                html.Div([
                    html.Span("Equity:", className='metric-label'),
                    html.Span(f"${p.total_equity:,.2f}", className='metric-value')
                ], className='metric-row'),
                html.Div([
                    html.Span("Cash:", className='metric-label'),
                    html.Span(f"${p.cash_balance:,.2f}", className='metric-value')
                ], className='metric-row'),
                html.Div([
                    html.Span("Session P&L:", className='metric-label'),
                    html.Span([
                        f"${p.session_pnl:+,.2f} ",
                        html.Span(f"({p.session_pnl_percent:+.2f}%)", style={'fontSize': '10px'})
                    ], className=f'metric-value {equity_color}')
                ], className='metric-row'),
                html.Div([
                    html.Span("Realized:", className='metric-label'),
                    html.Span(f"${p.realized_pnl:+,.2f}", 
                             className=f'metric-value {"positive" if p.realized_pnl >= 0 else "negative"}')
                ], className='metric-row'),
                html.Div([
                    html.Span("Unrealized:", className='metric-label'),
                    html.Span(f"${p.unrealized_pnl:+,.2f}", 
                             className=f'metric-value {"positive" if p.unrealized_pnl >= 0 else "negative"}')
                ], className='metric-row'),
                html.Hr(style={'margin': '3px 0', 'borderColor': '#3d4158'}),
                html.Div([
                    html.Span("Trades:", className='metric-label'),
                    html.Span(f"{p.num_trades}", className='metric-value')
                ], className='metric-row'),
                html.Div([
                    html.Span("Sharpe:", className='metric-label'),
                    html.Span(f"{p.sharpe_ratio:.2f}", className='metric-value', style={'fontSize': '11px'})
                ], className='metric-row'),
                html.Div([
                    html.Span("Sortino:", className='metric-label'),
                    html.Span(f"{getattr(p, 'sortino_ratio', 0.0):.2f}", className='metric-value', style={'fontSize': '11px'})
                ], className='metric-row'),
                html.Div([
                    html.Span("Avg Hold:", className='metric-label'),
                    html.Span(f"{getattr(p, 'avg_holding_time_formatted', '--')}", className='metric-value', style={'fontSize': '11px'})
                ], className='metric-row'),
                html.Hr(style={'margin': '3px 0', 'borderColor': '#3d4158'}),
                html.Div([
                    html.Span("Total Costs:", className='metric-label'),
                    html.Span(f"${p.total_commission + p.total_fees + p.total_slippage:.2f}", 
                             className='metric-value negative', style={'fontSize': '10px'})
                ], className='metric-row'),
                html.Div([
                    html.Span("", className='metric-label'),
                    html.Span(f"Slip: ${p.total_slippage:.2f} | Comm: ${p.total_commission:.2f}", 
                             style={'fontSize': '9px', 'color': '#666'})
                ], className='metric-row', style={'marginTop': '-2px'}),
            ])
            
            return content
        
        @self.app.callback(
            Output('action-bias-summary', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_action_bias_summary(n):
            # Calculate action bias if we have a current episode
            if self.state.current_episode:
                self.state.calculate_action_bias()
            
            content = []
            
            # Invalid actions count
            invalid_count = self.state.action_analysis.invalid_actions_count
            content.append(html.Div([
                html.Span("Invalid Actions:", className='metric-label'),
                html.Span(f"{invalid_count}", className='metric-value negative' if invalid_count > 0 else 'metric-value')
            ], className='metric-row'))
            
            content.append(html.Hr(style={'margin': '5px 0', 'borderColor': '#3d4158'}))
            
            # Action bias table
            if self.state.action_analysis.action_bias:
                # Header
                content.append(html.Div([
                    html.Span("Action", style={'width': '15%', 'fontSize': '10px', 'color': '#999'}),
                    html.Span("Count", style={'width': '15%', 'fontSize': '10px', 'color': '#999'}),
                    html.Span("%Steps", style={'width': '15%', 'fontSize': '10px', 'color': '#999'}),
                    html.Span("AvgRew", style={'width': '20%', 'fontSize': '10px', 'color': '#999'}),
                    html.Span("TotRew", style={'width': '20%', 'fontSize': '10px', 'color': '#999'}),
                    html.Span("Win%", style={'width': '15%', 'fontSize': '10px', 'color': '#999', 'textAlign': 'right'}),
                ], style={'display': 'flex', 'borderBottom': '1px solid #3d4158', 'paddingBottom': '2px', 'marginBottom': '3px'}))
                
                # Data rows
                for action_type in ['HOLD', 'BUY', 'SELL']:
                    data = self.state.action_analysis.action_bias.get(action_type, {})
                    if data:
                        color = {'BUY': 'positive', 'SELL': 'negative', 'HOLD': 'flat'}.get(action_type, 'neutral')
                        content.append(html.Div([
                            html.Span(action_type, className=color, style={'width': '15%', 'fontSize': '10px', 'fontWeight': 'bold'}),
                            html.Span(f"{data.get('count', 0)}", style={'width': '15%', 'fontSize': '10px'}),
                            html.Span(f"{data.get('percent_steps', 0):.0f}%", style={'width': '15%', 'fontSize': '10px'}),
                            html.Span(f"{data.get('mean_reward', 0):.3f}", style={'width': '20%', 'fontSize': '10px'}),
                            html.Span(f"{data.get('total_reward', 0):.2f}", style={'width': '20%', 'fontSize': '10px'}),
                            html.Span(f"{data.get('pos_reward_rate', 0):.0f}%", style={'width': '15%', 'fontSize': '10px', 'textAlign': 'right'}),
                        ], style={'display': 'flex', 'marginBottom': '2px'}))
            else:
                content.append(html.Div("Calculating action bias...", style={'color': '#999', 'fontSize': '10px', 'marginTop': '10px'}))
            
            return html.Div(content)
        
        @self.app.callback(
            Output('training-progress', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_training_progress(n):
            prog = self.state.training_progress
            
            # Enhanced stage display with specific information
            stage_info = prog.stage_status or ""
            stage_progress = 0
            
            if prog.current_stage:
                if "rollout" in prog.current_stage.lower() or "collecting" in prog.current_stage.lower():
                    # Rollout/Collection phase
                    if prog.rollout_steps > 0 and prog.rollout_total > 0:
                        stage_info = f"Collecting: {prog.rollout_steps}/{prog.rollout_total} steps"
                        stage_progress = (prog.rollout_steps / prog.rollout_total) * 100
                    else:
                        stage_info = "Starting rollout collection..."
                        stage_progress = 0
                        
                elif "update" in prog.current_stage.lower() or "training" in prog.current_stage.lower():
                    # PPO Update phase
                    if prog.current_epoch > 0 and prog.total_epochs > 0:
                        if prog.current_batch > 0 and prog.total_batches > 0:
                            stage_info = f"Epoch {prog.current_epoch}/{prog.total_epochs}, Batch {prog.current_batch}/{prog.total_batches}"
                            # Calculate progress based on epochs and batches
                            epoch_progress = (prog.current_epoch - 1) / prog.total_epochs
                            batch_progress = prog.current_batch / prog.total_batches / prog.total_epochs
                            stage_progress = (epoch_progress + batch_progress) * 100
                        else:
                            stage_info = f"Epoch {prog.current_epoch}/{prog.total_epochs}"
                            stage_progress = ((prog.current_epoch - 1) / prog.total_epochs) * 100
                    else:
                        stage_info = "Preparing PPO update..."
                        stage_progress = 0
                        
                elif "data" in prog.current_stage.lower() or "loading" in prog.current_stage.lower():
                    stage_info = "Loading market data..."
                    # Use stage_status for detailed progress if available
                    if prog.stage_status and "/" in prog.stage_status:
                        stage_info = f"Loading data: {prog.stage_status}"
                    stage_progress = prog.stage_progress if prog.stage_progress > 0 else 0
                    
                elif "precompute" in prog.current_stage.lower():
                    stage_info = "Precomputing features..."
                    if prog.stage_status and "/" in prog.stage_status:
                        stage_info = f"Precomputing: {prog.stage_status}"
                    stage_progress = prog.stage_progress if prog.stage_progress > 0 else 0
                    
                elif "setup" in prog.current_stage.lower() or "init" in prog.current_stage.lower():
                    stage_info = "Initializing environment..."
                    stage_progress = prog.stage_progress if prog.stage_progress > 0 else 0
                else:
                    # Default case
                    stage_info = prog.current_stage
                    stage_progress = prog.stage_progress
            
            # Ensure progress percentages are valid
            overall_progress = max(0, min(100, prog.overall_progress))
            stage_progress = max(0, min(100, stage_progress))
            
            content = html.Div([
                html.Div([
                    html.Span("Mode:", className='metric-label'),
                    html.Span(prog.mode or "Idle", className='metric-value')
                ], className='metric-row'),
                html.Div([
                    html.Span("Current Stage:", className='metric-label'),
                    html.Span(prog.current_stage or "Waiting", className='metric-value', style={'fontSize': '11px'})
                ], className='metric-row'),
                html.Div([
                    html.Span("Overall:", className='metric-label'),
                    html.Div([
                        html.Div(style={'width': f"{overall_progress}%"}, className='progress-fill'),
                        html.Div(f"{overall_progress:.0f}%", className='progress-text')
                    ], className='progress-bar', style={'flex': 1, 'marginLeft': '5px'})
                ], className='metric-row', style={'alignItems': 'center'}),
                html.Div([
                    html.Span("Stage Progress:", className='metric-label'),
                    html.Div([
                        html.Div(style={'width': f"{stage_progress}%"}, className='progress-fill'),
                        html.Div(f"{stage_progress:.0f}%", className='progress-text')
                    ], className='progress-bar', style={'flex': 1, 'marginLeft': '5px'})
                ], className='metric-row', style={'alignItems': 'center'}),
                html.Div([
                    html.Span("Stage Details:", className='metric-label'),
                    html.Span(stage_info, className='metric-value', style={'fontSize': '11px', 'fontFamily': 'monospace'})
                ], className='metric-row'),
                html.Hr(style={'margin': '3px 0', 'borderColor': '#3d4158'}),
                html.Div([
                    html.Span("Updates:", className='metric-label'),
                    html.Span(f"{prog.updates:,}", className='metric-value')
                ], className='metric-row'),
                html.Div([
                    html.Span("Episodes:", className='metric-label'),
                    html.Span(f"{prog.total_episodes:,}", className='metric-value')
                ], className='metric-row'),
                html.Div([
                    html.Span("Steps:", className='metric-label'),
                    html.Span(f"{prog.global_steps:,}", className='metric-value')
                ], className='metric-row'),
            ])
            
            return content
        
        @self.app.callback(
            Output('episode-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_episode_chart(n):
            if not self.state.current_episode:
                return go.Figure()
            
            episode = self.state.current_episode
            
            # Create subplots with more space for price chart
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.8, 0.2],  # Give more space to price chart
                subplot_titles=('', '')
            )
            
            # Collect all available 1m bars - prefer full day data if available
            if hasattr(self.state, 'full_day_1m_bars') and self.state.full_day_1m_bars:
                all_bars = list(self.state.full_day_1m_bars)
            else:
                all_bars = list(self.state.ohlc_data) if hasattr(self.state, 'ohlc_data') else []
            
            # Initialize variables to avoid warnings
            timestamps = []
            
            if all_bars:
                # Sort bars by timestamp to ensure correct order
                all_bars.sort(key=lambda x: x.get('timestamp', datetime.min))
                
                # Create candlestick chart with all available bars
                opens = []
                highs = []
                lows = []
                closes = []
                time_labels = []
                
                for bar in all_bars:
                    ts = bar.get('timestamp')
                    if ts:
                        timestamps.append(ts)
                        opens.append(bar.get('open', 0))
                        highs.append(bar.get('high', 0))
                        lows.append(bar.get('low', 0))
                        closes.append(bar.get('close', 0))
                        
                        # Create time label
                        if isinstance(ts, datetime):
                            time_labels.append(ts.strftime("%H:%M"))
                        else:
                            time_labels.append(str(ts))
                
                if timestamps:
                    # Add candlestick chart
                    fig.add_trace(
                        go.Candlestick(
                            x=timestamps,
                            open=opens,
                            high=highs,
                            low=lows,
                            close=closes,
                            name='Price',
                            increasing= dict(line=dict(color='#00d084')),
                            decreasing=dict(line=dict(color='#ff4757')),
                            showlegend=False

                        ),
                        row=1, col=1
                    )
                    
                    # Add execution markers if we have executions
                    if episode.executions:
                        exec_timestamps = []
                        exec_prices = []
                        exec_colors = []
                        exec_symbols = []
                        exec_text = []
                        
                        for execution in episode.executions:
                            # Use execution timestamp directly
                            exec_ts = execution.timestamp
                            if exec_ts:
                                exec_timestamps.append(exec_ts)
                                exec_prices.append(execution.price)
                                exec_colors.append('#00d084' if execution.side == 'BUY' else '#ff4757')
                                exec_symbols.append('triangle-up' if execution.side == 'BUY' else 'triangle-down')
                                exec_text.append(f"{execution.side} {execution.quantity:.0f} @ ${execution.price:.2f}")
                        
                        if exec_timestamps:
                            fig.add_trace(
                                go.Scatter(
                                    x=exec_timestamps,
                                    y=exec_prices,
                                    mode='markers',
                                    marker=dict(
                                        size=12,
                                        color=exec_colors,
                                        symbol=exec_symbols,
                                        line=dict(width=2, color='white')
                                    ),
                                    showlegend=False,
                                    name='Executions',
                                    hovertext=exec_text,
                                    hoverinfo='text'
                                ),
                                row=1, col=1
                            )
            
            # Cumulative reward chart
            display_steps = []  # Initialize to avoid warnings
            if episode.reward_history:
                # Create time-based x-axis for rewards
                reward_steps = list(range(len(episode.reward_history)))
                cumulative_rewards = []
                cum_sum = 0
                for r in episode.reward_history:
                    cum_sum += r
                    cumulative_rewards.append(cum_sum)
                
                # Show all reward data to match price chart timeline  
                display_steps = reward_steps
                display_rewards = cumulative_rewards
                
                fig.add_trace(
                    go.Scatter(
                        x=display_steps,
                        y=display_rewards,
                        fill='tozeroy',
                        fillcolor='rgba(0, 208, 132, 0.1)',
                        line=dict(color='#00d084', width=2),
                        showlegend=False,
                        name='Cumulative Reward',
                        hovertemplate='Step: %{x}<br>Reward: %{y:.3f}<extra></extra>'
                    ),
                    row=2, col=1
                )
            
            # Update layout to fill bottom space completely
            fig.update_layout(
                height=600,  # Fill more vertical space
                margin=dict(l=60, r=20, t=20, b=40),
                paper_bgcolor='#1e2130',
                plot_bgcolor='#0e1117',
                font=dict(size=10, color='#e6e6e6'),
                showlegend=False,
                xaxis2_title="Step",
                yaxis1_title="Price ($)",
                yaxis2_title="Cumulative Reward",
                hovermode='x unified',
                autosize=True  # Allow chart to resize
            )
            
            # Configure x-axis to show full day without scrolling
            if timestamps:
                fig.update_xaxes(
                    rangeslider_visible=False,
                    range=[timestamps[0], timestamps[-1]],  # Show full range
                    fixedrange=True,  # Disable zoom/pan
                    autorange=False,  # Disable auto-ranging
                    row=1, col=1
                )
                # Also apply to reward chart x-axis if we have reward data
                if 'display_steps' in locals() and display_steps:
                    fig.update_xaxes(
                        range=[0, len(display_steps) - 1],
                        autorange=False,
                        row=2, col=1
                    )
            else:
                fig.update_xaxes(
                    rangeslider_visible=False,
                    autorange=False,
                    row=1, col=1
                )
            
            # Update axes styling
            for i in range(1, 3):
                fig.update_xaxes(gridcolor='#3d4158', gridwidth=0.5, row=i, col=1)
                fig.update_yaxes(gridcolor='#3d4158', gridwidth=0.5, row=i, col=1)
            
            return fig
        
        @self.app.callback(
            Output('training-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_training_chart(n):
            episodes = list(self.state.episode_history)[-50:]  # Last 50 episodes
            
            if not episodes:
                return go.Figure()
            
            # Extract data
            episode_nums = [e.episode_num for e in episodes]
            rewards = [e.total_reward for e in episodes]
            
            # Create simple line chart
            fig = go.Figure()
            
            fig.add_trace(
                go.Scatter(
                    x=episode_nums,
                    y=rewards,
                    mode='lines+markers',
                    line=dict(color='#00d084', width=1),
                    marker=dict(size=3),
                    fill='tozeroy',
                    fillcolor='rgba(0, 208, 132, 0.1)'
                )
            )
            
            fig.update_layout(
                height=150,
                margin=dict(l=40, r=10, t=20, b=20),
                paper_bgcolor='#1e2130',
                plot_bgcolor='#0e1117',
                font=dict(size=9, color='#e6e6e6'),
                title="Episode Rewards",
                title_font_size=10,
                xaxis_title="Episode",
                yaxis_title="Reward",
                showlegend=False
            )
            
            fig.update_xaxes(gridcolor='#3d4158', gridwidth=0.5)
            fig.update_yaxes(gridcolor='#3d4158', gridwidth=0.5)
            
            return fig
        
        # Continue with other callbacks...
        # (Include all other callbacks from the original file with similar improvements)
        
        @self.app.callback(
            Output('reward-system', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_reward_system(n):
            if not self.state.reward_components:
                return html.Div([
                    html.P("No reward components data", 
                          style={'color': '#999', 'fontSize': '11px', 'textAlign': 'center'})
                ])
            
            # Get all components, not just active ones
            all_components = list(self.state.reward_components.items())
            
            if not all_components:
                return html.Div([
                    html.P("Waiting for rewards...", 
                          style={'color': '#999', 'fontSize': '11px', 'textAlign': 'center'})
                ])
            
            # Sort by absolute total impact (so most impactful components are at top)
            sorted_components = sorted(
                all_components,
                key=lambda x: abs(x[1].total_impact) if x[1].times_triggered > 0 else 0,
                reverse=True
            )
            
            # Add header row with TotRew column and better spacing
            rows = [
                html.Div([
                    html.Span("Component", style={'flex': '1', 'fontSize': '10px', 'color': '#999', 'paddingRight': '10px'}),
                    html.Span("Avg", style={'fontSize': '10px', 'color': '#999', 'width': '60px', 'textAlign': 'right', 'paddingRight': '10px'}),
                    html.Span("Total", style={'fontSize': '10px', 'color': '#999', 'width': '60px', 'textAlign': 'right', 'paddingRight': '10px'}),
                    html.Span("Count", style={'fontSize': '10px', 'color': '#999', 'width': '45px', 'textAlign': 'center', 'paddingRight': '10px'}),
                    html.Span("%", style={'fontSize': '10px', 'color': '#999', 'width': '35px', 'textAlign': 'right'})
                ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '3px', 'borderBottom': '1px solid #3d4158', 'paddingBottom': '2px'})
            ]
            
            for name, comp in sorted_components:
                display_name = name.replace('_', ' ').title()
                
                # Determine color based on the actual value or component type
                if comp.times_triggered > 0:
                    # Use actual average magnitude to determine color
                    color = '#00d084' if comp.avg_magnitude >= 0 else '#ff4757'
                else:
                    # For components that haven't triggered, use a neutral color
                    color = '#666'
                
                # Format values - show "--" for components that haven't triggered
                avg_display = f"{comp.avg_magnitude:+.3f}" if comp.times_triggered > 0 else "--"
                total_display = f"{comp.total_impact:+.3f}" if comp.times_triggered > 0 else "--"
                
                rows.append(html.Div([
                    html.Span(display_name, style={'flex': '1', 'fontSize': '11px', 'paddingRight': '10px'}),
                    html.Span(avg_display, 
                             style={'color': color, 'fontSize': '11px', 'fontFamily': 'monospace', 'width': '60px', 'textAlign': 'right', 'paddingRight': '10px'}),
                    html.Span(total_display, 
                             style={'color': color, 'fontSize': '11px', 'fontFamily': 'monospace', 'width': '60px', 'textAlign': 'right', 'paddingRight': '10px'}),
                    html.Span(f"{comp.times_triggered}", 
                             style={'color': '#666', 'fontSize': '10px', 'width': '45px', 'textAlign': 'center', 'paddingRight': '10px'}),
                    html.Span(f"{comp.percent_of_total:.0f}%" if comp.times_triggered > 0 else "0%", 
                             style={'color': '#666', 'fontSize': '10px', 'width': '35px', 'textAlign': 'right'})
                ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '1px'}))
            
            return html.Div(rows)
        
        @self.app.callback(
            Output('recent-executions', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_recent_executions(n):
            executions = self.state.get_recent_executions(5)
            
            if not executions:
                return html.Div("No executions yet", style={'color': '#999', 'fontSize': '11px'})
            
            rows = []
            for execution in executions[-5:]:
                side_class = 'positive' if execution.side == 'BUY' else 'negative'
                rows.append(html.Div([
                    html.Span(execution.timestamp.strftime("%H:%M:%S"), style={'fontSize': '11px', 'color': '#666'}),
                    html.Span(execution.side, className=side_class, style={'fontWeight': 'bold'}),
                    html.Span(f"{execution.quantity:.0f}", style={'fontSize': '11px'}),
                    html.Span(f"${execution.price:.2f}", style={'fontSize': '11px'}),
                ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '1px'}))
            
            return html.Div(rows)
        
        @self.app.callback(
            Output('recent-trades', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_recent_trades(n):
            trades = self.state.get_recent_trades(3)
            
            if not trades:
                return html.Div("No completed trades", style={'color': '#999', 'fontSize': '11px'})
            
            rows = []
            # Add header with status and holding time columns
            rows.append(html.Div([
                html.Span("Status", style={'fontSize': '10px', 'color': '#999', 'width': '12%'}),
                html.Span("Side", style={'fontSize': '10px', 'color': '#999', 'width': '8%'}),
                html.Span("Qty", style={'fontSize': '10px', 'color': '#999', 'width': '8%'}),
                html.Span("Entry", style={'fontSize': '10px', 'color': '#999', 'width': '12%'}),
                html.Span("Exit", style={'fontSize': '10px', 'color': '#999', 'width': '12%'}),
                html.Span("Hold Time", style={'fontSize': '10px', 'color': '#999', 'width': '15%'}),
                html.Span("P&L", style={'fontSize': '10px', 'color': '#999', 'width': '33%', 'textAlign': 'right'}),
            ], style={'display': 'flex', 'marginBottom': '3px', 'borderBottom': '1px solid #3d4158', 'paddingBottom': '2px'}))
            
            for trade in trades[-3:]:
                # Determine status based on exit_price
                status = "CLOSED" if trade.exit_price else "OPEN"
                status_color = '#00d084' if status == "CLOSED" else '#ffd32a'
                
                # Only show PnL for closed trades
                pnl_display = f"${trade.pnl:+.2f}" if trade.exit_price and trade.pnl is not None else "--"
                pnl_class = 'positive' if trade.pnl and trade.pnl >= 0 else 'negative' if trade.pnl and trade.pnl < 0 else ''
                
                side_class = 'positive' if trade.side in ['BUY', 'LONG'] else 'negative'
                
                # Main trade row
                rows.append(html.Div([
                    html.Span(status, style={'fontSize': '10px', 'color': status_color, 'fontWeight': 'bold', 'width': '12%'}),
                    html.Span(f"{trade.side}", className=side_class, style={'fontSize': '10px', 'width': '8%'}),
                    html.Span(f"{trade.quantity:.0f}", style={'fontSize': '10px', 'width': '8%'}),
                    html.Span(f"${trade.entry_price:.2f}", style={'fontSize': '10px', 'width': '12%'}),
                    html.Span(f"${trade.exit_price:.2f}" if trade.exit_price else "--", style={'fontSize': '10px', 'width': '12%'}),
                    html.Span(trade.holding_time_formatted if hasattr(trade, 'holding_time_formatted') else "--", 
                             style={'fontSize': '10px', 'width': '15%', 'color': '#999'}),
                    html.Span(pnl_display, className=pnl_class, style={'fontWeight': 'bold', 'fontSize': '10px', 'width': '33%', 'textAlign': 'right'}),
                ], style={'display': 'flex', 'marginBottom': '1px'}))
                
                # Add costs detail row if trade is closed and has costs
                total_costs = trade.commission + trade.fees + trade.slippage
                if status == "CLOSED" and total_costs > 0:
                    rows.append(html.Div([
                        html.Span("", style={'width': '12%'}),  # Empty space under status
                        html.Span("Costs:", style={'fontSize': '9px', 'color': '#666', 'width': '8%'}),
                        html.Span(f"Slip: ${trade.slippage:.2f}", style={'fontSize': '9px', 'color': '#999', 'width': '20%'}),
                        html.Span(f"Comm: ${trade.commission:.2f}", style={'fontSize': '9px', 'color': '#999', 'width': '20%'}),
                        html.Span("", style={'width': '15%'}),  # Empty space under hold time
                        html.Span(f"Total: ${total_costs:.2f}", style={'fontSize': '9px', 'color': '#ff6b6b', 'width': '25%', 'textAlign': 'right'}),
                    ], style={'display': 'flex', 'marginBottom': '3px', 'paddingLeft': '10px'}))
            
            return html.Div(rows)
        
        @self.app.callback(
            [Output('episode-title', 'children'),
             Output('episode-content', 'children'),
             Output('episode-history', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_episode_panel(n):
            current = self.state.current_episode
            
            if current:
                title = f"🎬 Episode Analysis (Ep: {current.episode_num})"
                last_reward = current.reward_history[-1] if current.reward_history else 0
                
                content = html.Div([
                    html.Div([
                        html.Span("Steps:", className='metric-label'),
                        html.Span(f"{current.steps}", className='metric-value')
                    ], className='metric-row'),
                    html.Div([
                        html.Span("Total Reward:", className='metric-label'),
                        html.Span(f"{current.total_reward:.2f}", className='metric-value')
                    ], className='metric-row'),
                    html.Div([
                        html.Span("Last Reward:", className='metric-label'),
                        html.Span(f"{last_reward:.3f}", className='metric-value')
                    ], className='metric-row'),
                ])
            else:
                title = "🎬 Episode Analysis"
                content = html.Div("Waiting for episode...", style={'color': '#999', 'fontSize': '11px'})
            
            # Episode history
            history = self.state.get_episode_history(3)
            if history:
                history_rows = []
                for ep in history[-3:]:
                    reason = (ep.termination_reason or "Complete")[:10]
                    color = 'positive' if ep.total_reward > 0 else 'negative'
                    history_rows.append(html.Div([
                        html.Span(f"Ep {ep.episode_num}", style={'fontSize': '11px'}),
                        html.Span(reason, style={'fontSize': '11px', 'color': '#666'}),
                        html.Span(f"{ep.total_reward:.1f}", className=color, style={'fontSize': '11px'})
                    ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '1px'}))
                
                history_content = html.Div([
                    html.H4("Recent Episodes", style={'fontSize': '11px', 'margin': '5px 0 3px 0', 'color': '#999'}),
                    html.Div(history_rows)
                ])
            else:
                history_content = html.Div()
            
            return title, content, history_content
        
        @self.app.callback(
            Output('ppo-metrics', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_ppo_metrics(n):
            ppo = self.state.ppo_metrics
            
            def create_sparkline(values):
                """Create a simple text-based sparkline"""
                if not values or len(values) < 2:
                    return ""
                
                # Convert to list if needed
                vals = list(values)
                
                # Normalize values to 0-7 range for 8 spark characters
                min_val = min(vals)
                max_val = max(vals)
                if max_val == min_val:
                    return "▁" * len(vals)
                
                spark_chars = "▁▂▃▄▅▆▇█"
                sparkline = ""
                
                for val in vals:
                    # Normalize to 0-7
                    normalized = int((val - min_val) / (max_val - min_val) * 7)
                    sparkline += spark_chars[normalized]
                
                return sparkline
            
            content = html.Div([
                html.Div([
                    html.Span("Learning Rate:", className='metric-label'),
                    html.Span([
                        f"{ppo.learning_rate:.1e} ",
                        html.Span(create_sparkline(ppo.learning_rate_history), 
                                 style={'color': '#666', 'fontFamily': 'monospace', 'fontSize': '10px'})
                    ], className='metric-value')
                ], className='metric-row'),
                html.Div([
                    html.Span("Batch Mean Reward:", className='metric-label'),
                    html.Span([
                        f"{ppo.mean_reward_batch:.4f} ",
                        html.Span(create_sparkline(ppo.mean_reward_history), 
                                 style={'color': '#666', 'fontFamily': 'monospace', 'fontSize': '10px'})
                    ], className='metric-value')
                ], className='metric-row'),
                html.Hr(style={'margin': '3px 0', 'borderColor': '#3d4158'}),
                html.Div([
                    html.Span("Policy Loss:", className='metric-label'),
                    html.Span([
                        f"{ppo.policy_loss:.4f} ",
                        html.Span(create_sparkline(ppo.policy_loss_history), 
                                 style={'color': '#666', 'fontFamily': 'monospace', 'fontSize': '10px'})
                    ], className='metric-value')
                ], className='metric-row'),
                html.Div([
                    html.Span("Value Loss:", className='metric-label'),
                    html.Span([
                        f"{ppo.value_loss:.4f} ",
                        html.Span(create_sparkline(ppo.value_loss_history), 
                                 style={'color': '#666', 'fontFamily': 'monospace', 'fontSize': '10px'})
                    ], className='metric-value')
                ], className='metric-row'),
                html.Div([
                    html.Span("Total Loss:", className='metric-label'),
                    html.Span([
                        f"{ppo.total_loss:.4f} ",
                        html.Span(create_sparkline(ppo.total_loss_history), 
                                 style={'color': '#666', 'fontFamily': 'monospace', 'fontSize': '10px'})
                    ], className='metric-value')
                ], className='metric-row'),
                html.Hr(style={'margin': '3px 0', 'borderColor': '#3d4158'}),
                html.Div([
                    html.Span("Entropy:", className='metric-label'),
                    html.Span([
                        f"{ppo.entropy:.4f} ",
                        html.Span(create_sparkline(ppo.entropy_history), 
                                 style={'color': '#666', 'fontFamily': 'monospace', 'fontSize': '10px'})
                    ], className='metric-value')
                ], className='metric-row'),
                html.Div([
                    html.Span("Clip Fraction:", className='metric-label'),
                    html.Span([
                        f"{ppo.clip_fraction:.3f} ",
                        html.Span(create_sparkline(ppo.clip_fraction_history), 
                                 style={'color': '#666', 'fontFamily': 'monospace', 'fontSize': '10px'})
                    ], className='metric-value')
                ], className='metric-row'),
                html.Div([
                    html.Span("KL Divergence:", className='metric-label'),
                    html.Span([
                        f"{ppo.approx_kl:.4f} ",
                        html.Span(create_sparkline(ppo.approx_kl_history), 
                                 style={'color': '#666', 'fontFamily': 'monospace', 'fontSize': '10px'})
                    ], className='metric-value')
                ], className='metric-row'),
                html.Div([
                    html.Span("Explained Var:", className='metric-label'),
                    html.Span([
                        f"{ppo.explained_variance:.3f} ",
                        html.Span(create_sparkline(ppo.explained_variance_history), 
                                 style={'color': '#666', 'fontFamily': 'monospace', 'fontSize': '10px'})
                    ], className='metric-value')
                ], className='metric-row'),
            ])
            
            return content
        
        @self.app.callback(
            Output('full-day-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_full_day_chart(n):
            """Create simplified full-width chart"""
            # Create empty figure first
            fig = go.Figure()
            
            # Try to use actual OHLC data if available
            if hasattr(self.state, 'ohlc_data') and self.state.ohlc_data:
                all_bars = list(self.state.ohlc_data)
                if all_bars:
                    # Sort by timestamp
                    all_bars.sort(key=lambda x: x.get('timestamp', datetime.min))
                    
                    # Extract data
                    timestamps = []
                    opens = []
                    highs = []
                    lows = []
                    closes = []
                    
                    for bar in all_bars:
                        if all(k in bar for k in ['timestamp', 'open', 'high', 'low', 'close']):
                            timestamps.append(bar['timestamp'])
                            opens.append(bar['open'])
                            highs.append(bar['high'])
                            lows.append(bar['low'])
                            closes.append(bar['close'])
                    
                    if timestamps:
                        # Add candlestick
                        fig.add_trace(
                            go.Candlestick(
                                x=timestamps,
                                open=opens,
                                high=highs,
                                low=lows,
                                close=closes,
                                name='Price',
                                increasing_line_color='#00d084',
                                decreasing_line_color='#ff4757'
                            )
                        )
                        
                        # Add executions if available
                        if self.state.current_episode and self.state.current_episode.executions:
                            exec_data = []
                            for e in self.state.current_episode.executions:
                                exec_data.append({
                                    'time': e.timestamp,
                                    'price': e.price,
                                    'side': e.side,
                                    'qty': e.quantity
                                })
                            
                            if exec_data:
                                buy_execs = [e for e in exec_data if e['side'] == 'BUY']
                                sell_execs = [e for e in exec_data if e['side'] == 'SELL']
                                
                                # Add buy markers
                                if buy_execs:
                                    fig.add_trace(
                                        go.Scatter(
                                            x=[e['time'] for e in buy_execs],
                                            y=[e['price'] for e in buy_execs],
                                            mode='markers',
                                            marker=dict(
                                                symbol='triangle-up',
                                                size=12,
                                                color='#00d084',
                                                line=dict(width=2, color='white')
                                            ),
                                            name='Buy',
                                            text=[f"Buy {e['qty']:.0f}" for e in buy_execs],
                                            hoverinfo='text+y'
                                        )
                                    )
                                
                                # Add sell markers
                                if sell_execs:
                                    fig.add_trace(
                                        go.Scatter(
                                            x=[e['time'] for e in sell_execs],
                                            y=[e['price'] for e in sell_execs],
                                            mode='markers',
                                            marker=dict(
                                                symbol='triangle-down',
                                                size=12,
                                                color='#ff4757',
                                                line=dict(width=2, color='white')
                                            ),
                                            name='Sell',
                                            text=[f"Sell {e['qty']:.0f}" for e in sell_execs],
                                            hoverinfo='text+y'
                                        )
                                    )
            
            # Update layout
            fig.update_layout(
                height=400,
                margin=dict(l=60, r=20, t=40, b=40),
                paper_bgcolor='#1e2130',
                plot_bgcolor='#0e1117',
                font=dict(size=11, color='#e6e6e6'),
                showlegend=False,
                title=f"{self.state.market_data.symbol} - Trading Day",
                xaxis_title="Time",
                yaxis_title="Price ($)",
                hovermode='x unified'
            )
            
            # Style axes
            fig.update_xaxes(
                rangeslider_visible=False,
                gridcolor='#3d4158',
                gridwidth=0.5
            )
            
            fig.update_yaxes(
                gridcolor='#3d4158',
                gridwidth=0.5
            )
            
            # Update layout
            fig.update_layout(
                height=400,
                margin=dict(l=60, r=20, t=30, b=40),
                paper_bgcolor='#1e2130',
                plot_bgcolor='#0e1117',
                font=dict(size=11, color='#e6e6e6'),
                showlegend=False,
                title=dict(
                    text=f"{self.state.market_data.symbol} - Full Trading Day",
                    font=dict(size=14),
                    x=0.5,
                    xanchor='center'
                ),
                hovermode='x unified',
                xaxis_title="Time (ET)",
                yaxis_title="Price ($)"
            )
            
            # Configure axes
            fig.update_xaxes(
                rangeslider_visible=False,
                gridcolor='#3d4158',
                gridwidth=0.5,
                tickformat='%H:%M',
                dtick=3600000  # Show hourly ticks
            )
            
            fig.update_yaxes(
                gridcolor='#3d4158',
                gridwidth=0.5,
                tickformat='$,.2f'
            )
            
            return fig
        
        @self.app.callback(
            Output('model-internals', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_model_internals(n):
            """Display model internals metrics including attention weights and feature stats"""
            internals = self.state.model_internals
            
            content = []
            
            # Attention Metrics
            if internals.attention_weights:
                content.append(html.H4("Attention Analysis", style={'fontSize': '12px', 'margin': '0 0 5px 0', 'color': '#ffd32a'}))
                
                # Attention entropy and focus
                content.append(html.Div([
                    html.Span("Attention Entropy:", className='metric-label'),
                    html.Span(f"{internals.attention_entropy:.4f}", className='metric-value')
                ], className='metric-row'))
                
                content.append(html.Div([
                    html.Span("Max Weight:", className='metric-label'),
                    html.Span(f"{internals.attention_max_weight:.3f}", className='metric-value')
                ], className='metric-row'))
                
                # Branch focus
                branch_names = ['HF', 'MF', 'LF', 'Portfolio', 'Static']
                focus_branch = branch_names[internals.attention_focus_branch] if 0 <= internals.attention_focus_branch < 5 else 'Unknown'
                content.append(html.Div([
                    html.Span("Focus Branch:", className='metric-label'),
                    html.Span(focus_branch, className='metric-value', style={'color': '#00d084'})
                ], className='metric-row'))
                
                # Attention weights distribution
                if len(internals.attention_weights) == 5:
                    content.append(html.Div([
                        html.Span("Weights:", className='metric-label', style={'fontSize': '10px'}),
                    ], className='metric-row'))
                    for i, (branch, weight) in enumerate(zip(branch_names, internals.attention_weights)):
                        bar_width = f"{weight * 100:.1f}%"
                        content.append(html.Div([
                            html.Span(f"{branch}:", style={'width': '60px', 'fontSize': '10px', 'color': '#999'}),
                            html.Div([
                                html.Div(style={'width': bar_width, 'backgroundColor': '#00d084', 'height': '8px'}),
                            ], style={'flex': 1, 'backgroundColor': '#3d4158', 'height': '8px', 'marginLeft': '5px', 'position': 'relative'}),
                            html.Span(f"{weight:.3f}", style={'width': '50px', 'fontSize': '10px', 'textAlign': 'right', 'marginLeft': '5px'})
                        ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '2px'}))
                
                content.append(html.Hr(style={'margin': '5px 0', 'borderColor': '#3d4158'}))
            
            # Action Probabilities
            content.append(html.H4("Action Analysis", style={'fontSize': '12px', 'margin': '0 0 5px 0', 'color': '#00d084'}))
            
            content.append(html.Div([
                html.Span("Action Entropy:", className='metric-label'),
                html.Span(f"{internals.action_entropy:.4f}", className='metric-value')
            ], className='metric-row'))
            
            content.append(html.Div([
                html.Span("Action Confidence:", className='metric-label'),
                html.Span(f"{internals.action_confidence:.3f}", className='metric-value')
            ], className='metric-row'))
            
            content.append(html.Div([
                html.Span("Type Entropy:", className='metric-label'),
                html.Span(f"{internals.action_type_entropy:.4f}", className='metric-value', style={'fontSize': '10px'})
            ], className='metric-row'))
            
            content.append(html.Div([
                html.Span("Size Entropy:", className='metric-label'),
                html.Span(f"{internals.action_size_entropy:.4f}", className='metric-value', style={'fontSize': '10px'})
            ], className='metric-row'))
            
            # Feature Statistics
            if internals.feature_stats:
                content.append(html.Hr(style={'margin': '5px 0', 'borderColor': '#3d4158'}))
                content.append(html.H4("Feature Stats", style={'fontSize': '12px', 'margin': '0 0 5px 0', 'color': '#999'}))
                
                # Create mini table for feature stats
                stats_rows = []
                for branch, stats in internals.feature_stats.items():
                    if stats:
                        sparsity_color = '#ff4757' if stats.get('sparsity', 0) > 0.5 else '#00d084'
                        stats_rows.append(html.Div([
                            html.Span(branch, style={'width': '60px', 'fontSize': '10px'}),
                            html.Span(f"{stats.get('mean', 0):.3f}", style={'width': '50px', 'fontSize': '10px', 'textAlign': 'right'}),
                            html.Span(f"±{stats.get('std', 0):.3f}", style={'width': '50px', 'fontSize': '10px', 'textAlign': 'right', 'color': '#666'}),
                            html.Span(f"{stats.get('sparsity', 0)*100:.0f}%", style={'width': '40px', 'fontSize': '10px', 'textAlign': 'right', 'color': sparsity_color})
                        ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '1px'}))
                
                if stats_rows:
                    # Header
                    content.append(html.Div([
                        html.Span("Branch", style={'width': '60px', 'fontSize': '9px', 'color': '#666'}),
                        html.Span("Mean", style={'width': '50px', 'fontSize': '9px', 'color': '#666', 'textAlign': 'right'}),
                        html.Span("Std", style={'width': '50px', 'fontSize': '9px', 'color': '#666', 'textAlign': 'right'}),
                        html.Span("Sparse", style={'width': '40px', 'fontSize': '9px', 'color': '#666', 'textAlign': 'right'})
                    ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '2px', 'borderBottom': '1px solid #3d4158'}))
                    
                    content.extend(stats_rows)
            
            if not content:
                return html.Div("Waiting for model data...", style={'color': '#999', 'fontSize': '11px', 'textAlign': 'center'})
            
            return html.Div(content)
        
        @self.app.callback(
            Output('footer-stats', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_footer(n):
            # Calculate uptime
            elapsed = time.time() - self.state.start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            uptime = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            # Get stats
            total_trades = sum(len(ep.trades) for ep in self.state.episode_history)
            
            # Performance metrics
            steps_per_sec = self.state.training_progress.steps_per_second
            time_per_update = self.state.training_progress.time_per_update
            time_per_episode = self.state.training_progress.time_per_episode
            
            footer_content = [
                html.Span(f"Uptime: {uptime}"),
                html.Span(f"Total Trades: {total_trades}"),
                html.Span(f"Steps/Sec: {steps_per_sec:.1f}"),
                html.Span(f"Time/Update: {time_per_update:.2f}s"),
                html.Span(f"Time/Episode: {time_per_episode:.1f}s")
            ]
            
            return footer_content
    
    # Include all other methods from the original class...
    def _process_updates(self):
        """Process queued updates"""
        while self.is_running:
            try:
                update = self.update_queue.get(timeout=0.1)
                self._process_update(update)
            except queue.Empty:
                continue
    
    def _process_update(self, update: Dict[str, Any]):
        """Process a single update"""
        update_type = update.get('type')
        data = update.get('data', {})
        
        # Process based on type (same as original)
        if update_type == 'market':
            self.state.update_market(data)
        elif update_type == 'position':
            self.state.update_position(data)
        elif update_type == 'portfolio':
            self.state.update_portfolio(data)
        elif update_type == 'action':
            self.state.add_action(data['step'], data['action_type'], 
                                data.get('size', 1.0), data.get('reward', 0))
        elif update_type == 'trade':
            self.state.add_trade(data)
        elif update_type == 'episode_start':
            self.state.start_new_episode(data.get('episode_num', 0))
        elif update_type == 'episode_end':
            self.state.end_current_episode(data.get('reason', 'Completed'))
        elif update_type == 'training':
            self.state.update_training_progress(data)
        elif update_type == 'ppo_metrics':
            self.state.update_ppo_metrics(data)
        elif update_type == 'model_info':
            self.state.model_name = data.get('name', 'N/A')
        elif update_type == 'reward_components':
            self.state.update_reward_components(data)
        elif update_type == 'model_internals':
            self.state.update_model_internals(data)
    
    # Public API methods (same as original)
    def update_market(self, data: Dict[str, Any]):
        self.update_queue.put({'type': 'market', 'data': data})
    
    def update_position(self, data: Dict[str, Any]):
        self.update_queue.put({'type': 'position', 'data': data})
    
    def update_portfolio(self, data: Dict[str, Any]):
        self.update_queue.put({'type': 'portfolio', 'data': data})
    
    def update_action(self, step: int, action_type: str, size: float, reward: float):
        self.update_queue.put({'type': 'action', 'data': {
            'step': step, 'action_type': action_type, 'size': size, 'reward': reward
        }})
    
    def update_trade(self, trade_data: Dict[str, Any]):
        self.update_queue.put({'type': 'trade', 'data': trade_data})
    
    def start_episode(self, episode_num: int):
        self.update_queue.put({'type': 'episode_start', 'data': {'episode_num': episode_num}})
    
    def end_episode(self, reason: str):
        self.update_queue.put({'type': 'episode_end', 'data': {'reason': reason}})
    
    def update_training_progress(self, data: Dict[str, Any]):
        self.update_queue.put({'type': 'training', 'data': data})
    
    def update_ppo_metrics(self, data: Dict[str, Any]):
        self.update_queue.put({'type': 'ppo_metrics', 'data': data})
    
    def update_reward_components(self, components: Dict[str, float]):
        self.update_queue.put({'type': 'reward_components', 'data': components})
    
    def set_model_info(self, name: str):
        self.update_queue.put({'type': 'model_info', 'data': {'name': name}})
    
    def update_model_internals(self, data: Dict[str, Any]):
        self.update_queue.put({'type': 'model_internals', 'data': data})
    
    def start(self, open_browser: bool = True):
        """Start the dashboard server"""
        if self.is_running:
            logger.warning("Dashboard is already running")
            return
        
        self.is_running = True
        
        # Start update processing thread
        self.update_thread = threading.Thread(target=self._process_updates)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        # Start Dash server
        self.server_thread = threading.Thread(
            target=self.app.run_server,
            kwargs={'debug': False, 'port': self.port, 'host': '0.0.0.0'}
        )
        self.server_thread.daemon = True
        self.server_thread.start()
        
        # Open browser
        if open_browser:
            time.sleep(1)
            webbrowser.open(f'http://localhost:{self.port}')
        
        logger.info(f"Dashboard started on http://localhost:{self.port}")
    
    def stop(self):
        """Stop the dashboard server"""
        self.is_running = False
        logger.info("Dashboard stopped")