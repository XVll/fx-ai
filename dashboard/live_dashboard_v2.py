"""
Refactored Live Trading Dashboard with clean architecture.
Compact layout with separate current episode and overall training views.
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
import threading
import webbrowser
import logging
from typing import Dict, List, Any, Optional
import queue
import time
import os

from .dashboard_data import DashboardState

logger = logging.getLogger(__name__)

# Suppress Dash/Flask/Werkzeug logs
logging.getLogger('werkzeug').setLevel(logging.ERROR)
logging.getLogger('dash').setLevel(logging.ERROR)
os.environ['DASH_SILENCE_ROUTES_LOGGING'] = 'true'


class LiveTradingDashboard:
    """Real-time trading dashboard with clean architecture"""
    
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
        """Create the Dash application with compact layout"""
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
                        font-size: 11px;
                        line-height: 1.3;
                    }
                    .main-header {
                        background-color: #1e2130;
                        padding: 5px 15px;
                        border-bottom: 1px solid #3d4158;
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        height: 30px;
                    }
                    .main-container {
                        padding: 10px;
                        display: grid;
                        grid-template-columns: repeat(3, 1fr);
                        gap: 10px;
                        max-width: 100%;
                    }
                    .panel {
                        background-color: #1e2130;
                        border-radius: 4px;
                        padding: 8px;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
                        overflow: hidden;
                    }
                    .panel-title {
                        font-size: 12px;
                        font-weight: bold;
                        margin: 0 0 5px 0;
                        color: #00d084;
                        border-bottom: 1px solid #3d4158;
                        padding-bottom: 3px;
                    }
                    .metric-row {
                        display: flex;
                        justify-content: space-between;
                        margin: 2px 0;
                        font-size: 11px;
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
                        font-size: 10px;
                        border-collapse: collapse;
                    }
                    th {
                        background-color: #262837;
                        padding: 4px;
                        text-align: left;
                        font-weight: bold;
                        border-bottom: 1px solid #3d4158;
                    }
                    td {
                        padding: 3px 4px;
                        border-bottom: 1px solid #2a2b3d;
                    }
                    .chart-container {
                        margin-top: 5px;
                    }
                    .progress-bar {
                        background-color: #3d4158;
                        height: 12px;
                        border-radius: 2px;
                        overflow: hidden;
                        margin: 3px 0;
                    }
                    .progress-fill {
                        background-color: #00d084;
                        height: 100%;
                        transition: width 0.3s ease;
                    }
                    .sparkline {
                        font-family: monospace;
                        font-size: 10px;
                        color: #00d084;
                    }
                    .footer {
                        background-color: #1e2130;
                        padding: 5px 15px;
                        border-top: 1px solid #3d4158;
                        display: flex;
                        justify-content: space-between;
                        font-size: 10px;
                        position: fixed;
                        bottom: 0;
                        width: 100%;
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
        
        # Layout
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.Div(id='model-info', style={'fontSize': '12px'}),
                html.Div(id='session-time', style={'fontSize': '12px'})
            ], className='main-header'),
            
            # Main container with 3 columns
            html.Div([
                # Column 1
                html.Div([
                    # Market Data Panel
                    html.Div([
                        html.H3("📊 Market: ", className='panel-title', id='market-title'),
                        html.Div(id='market-content')
                    ], className='panel'),
                    
                    # Current Position Panel
                    html.Div([
                        html.H3("💼 Position: ", className='panel-title', id='position-title'),
                        html.Div(id='position-content')
                    ], className='panel'),
                    
                    # Portfolio Panel
                    html.Div([
                        html.H3("📈 Portfolio", className='panel-title'),
                        html.Div(id='portfolio-content')
                    ], className='panel'),
                ]),
                
                # Column 2
                html.Div([
                    # Recent Actions Panel
                    html.Div([
                        html.H3("⚡ Recent Actions", className='panel-title'),
                        html.Div(id='recent-actions')
                    ], className='panel'),
                    
                    # Recent Trades Panel
                    html.Div([
                        html.H3("⚡ Recent Trades", className='panel-title'),
                        html.Div(id='recent-trades')
                    ], className='panel'),
                    
                    # Episode Analysis Panel
                    html.Div([
                        html.H3("🎬 Episode Analysis", className='panel-title', id='episode-title'),
                        html.Div(id='episode-content'),
                        html.Div(id='episode-history')
                    ], className='panel'),
                ]),
                
                # Column 3
                html.Div([
                    # Training Progress Panel
                    html.Div([
                        html.H3("⚙️ Training Progress", className='panel-title'),
                        html.Div(id='training-progress')
                    ], className='panel'),
                    
                    # PPO Metrics Panel
                    html.Div([
                        html.H3("🧠 PPO Core Metrics", className='panel-title'),
                        html.Div(id='ppo-metrics')
                    ], className='panel'),
                    
                    # Reward System Panel
                    html.Div([
                        html.H3("🏆 Reward System", className='panel-title'),
                        html.Div(id='reward-system')
                    ], className='panel'),
                    
                    # Action Analysis Panel
                    html.Div([
                        html.H3("🎯 Action Analysis", className='panel-title'),
                        html.Div(id='action-analysis')
                    ], className='panel'),
                ]),
            ], className='main-container'),
            
            # Charts Section (below panels)
            html.Div([
                html.Div([
                    dcc.Graph(id='episode-chart', style={'height': '250px'})
                ], className='panel', style={'gridColumn': 'span 2'}),
                
                html.Div([
                    dcc.Graph(id='training-chart', style={'height': '250px'})
                ], className='panel'),
            ], style={'display': 'grid', 'gridTemplateColumns': '2fr 1fr', 'gap': '10px', 'padding': '0 10px'}),
            
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
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            [Output('model-info', 'children'),
             Output('session-time', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_header(n):
            # Process updates
            self._process_queue_updates()
            
            model_text = f"Model: {self.state.model_name}"
            session_text = f"Session Time: {self.state.session_elapsed_time}"
            
            return model_text, session_text
        
        @self.app.callback(
            [Output('market-title', 'children'),
             Output('market-content', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_market_panel(n):
            market = self.state.market_data
            
            title = f"📊 Market: {market.symbol} - {market.market_session}"
            
            content = html.Div([
                html.Div([
                    html.Span("Time (NY):", className='metric-label'),
                    html.Span(market.time_ny, className='metric-value')
                ], className='metric-row'),
                html.Div([
                    html.Span("Price:", className='metric-label'),
                    html.Span(f"${market.price:.2f}", className='metric-value neutral')
                ], className='metric-row'),
                html.Div([
                    html.Span("Bid:", className='metric-label'),
                    html.Span(f"${market.bid:.2f}", className='metric-value')
                ], className='metric-row'),
                html.Div([
                    html.Span("Ask:", className='metric-label'),
                    html.Span(f"${market.ask:.2f}", className='metric-value')
                ], className='metric-row'),
                html.Div([
                    html.Span("Spread:", className='metric-label'),
                    html.Span(f"${market.spread:.2f}", className='metric-value')
                ], className='metric-row'),
            ])
            
            return title, content
        
        @self.app.callback(
            [Output('position-title', 'children'),
             Output('position-content', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_position_panel(n):
            pos = self.state.position
            
            title = f"💼 Position: {pos.symbol}"
            
            side_class = 'positive' if pos.side == 'Long' else 'negative' if pos.side == 'Short' else 'flat'
            pnl_class = 'positive' if pos.pnl_dollars > 0 else 'negative' if pos.pnl_dollars < 0 else 'neutral'
            
            content = html.Div([
                html.Div([
                    html.Span("Side:", className='metric-label'),
                    html.Span(pos.side, className=f'metric-value {side_class}')
                ], className='metric-row'),
                html.Div([
                    html.Span("Quantity:", className='metric-label'),
                    html.Span(f"{pos.quantity:.4f}", className='metric-value')
                ], className='metric-row'),
                html.Div([
                    html.Span("Avg Entry:", className='metric-label'),
                    html.Span(f"${pos.avg_entry_price:.2f}", className='metric-value')
                ], className='metric-row'),
                html.Div([
                    html.Span("P&L vs Entry:", className='metric-label'),
                    html.Span(f"${pos.pnl_dollars:.2f} ({pos.pnl_percent:.2f}%)", 
                             className=f'metric-value {pnl_class}')
                ], className='metric-row'),
            ])
            
            return title, content
        
        @self.app.callback(
            Output('portfolio-content', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_portfolio_panel(n):
            port = self.state.portfolio
            
            session_pnl_class = 'positive' if port.session_pnl > 0 else 'negative' if port.session_pnl < 0 else 'neutral'
            unrealized_class = 'positive' if port.unrealized_pnl > 0 else 'negative' if port.unrealized_pnl < 0 else 'neutral'
            
            content = html.Div([
                html.Div([
                    html.Span("Total Equity:", className='metric-label'),
                    html.Span(f"${port.total_equity:.2f}", className='metric-value')
                ], className='metric-row'),
                html.Div([
                    html.Span("Cash Balance:", className='metric-label'),
                    html.Span(f"${port.cash_balance:.2f}", className='metric-value')
                ], className='metric-row'),
                html.Div([
                    html.Span("Session P&L:", className='metric-label'),
                    html.Span(f"${port.session_pnl:.2f} ({port.session_pnl_percent:.2f}%)", 
                             className=f'metric-value {session_pnl_class}')
                ], className='metric-row'),
                html.Div([
                    html.Span("Realized P&L:", className='metric-label'),
                    html.Span(f"${port.realized_pnl:.2f}", className='metric-value')
                ], className='metric-row'),
                html.Div([
                    html.Span("Unrealized P&L:", className='metric-label'),
                    html.Span(f"${port.unrealized_pnl:.2f}", className=f'metric-value {unrealized_class}')
                ], className='metric-row'),
                html.Div([
                    html.Span("Sharpe Ratio:", className='metric-label'),
                    html.Span(f"{port.sharpe_ratio:.2f}", className='metric-value')
                ], className='metric-row'),
                html.Div([
                    html.Span("Max Drawdown:", className='metric-label'),
                    html.Span(f"{port.max_drawdown:.2f}%", className='metric-value')
                ], className='metric-row'),
                html.Div([
                    html.Span("Trades:", className='metric-label'),
                    html.Span(f"{port.num_trades}", className='metric-value')
                ], className='metric-row'),
            ])
            
            return content
        
        @self.app.callback(
            Output('recent-actions', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_recent_actions(n):
            actions = self.state.get_recent_actions(5)
            
            if not actions:
                return html.Div("No actions yet", style={'color': '#999', 'textAlign': 'center'})
            
            rows = []
            for action in reversed(actions):
                reward_class = 'positive' if action.step_reward > 0 else 'negative'
                row = html.Tr([
                    html.Td(f"{action.step}"),
                    html.Td(action.action_type),
                    html.Td(action.size_signal),
                    html.Td(f"{action.step_reward:+.3f}", className=reward_class)
                ])
                rows.append(row)
            
            return html.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Step"),
                        html.Th("Action"),
                        html.Th("Size"),
                        html.Th("Reward")
                    ])
                ]),
                html.Tbody(rows)
            ])
        
        @self.app.callback(
            Output('recent-trades', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_recent_trades(n):
            trades = self.state.get_recent_trades(5)
            
            if not trades:
                return html.Div("No trades yet", style={'color': '#999', 'textAlign': 'center'})
            
            rows = []
            for trade in reversed(trades):
                pnl_class = 'positive' if trade.pnl and trade.pnl > 0 else 'negative'
                row = html.Tr([
                    html.Td(trade.timestamp.strftime("%H:%M:%S")),
                    html.Td(trade.side),
                    html.Td(f"{trade.quantity:.2f}"),
                    html.Td(trade.symbol),
                    html.Td(f"${trade.entry_price:.2f}"),
                    html.Td(f"${trade.exit_price:.2f}" if trade.exit_price else "-"),
                    html.Td(f"${trade.pnl:.2f}" if trade.pnl else "-", className=pnl_class)
                ])
                rows.append(row)
            
            return html.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Time"),
                        html.Th("Side"),
                        html.Th("Qty"),
                        html.Th("Sym"),
                        html.Th("Entry"),
                        html.Th("Exit"),
                        html.Th("P&L")
                    ])
                ]),
                html.Tbody(rows)
            ])
        
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
                        html.Span("Current Step:", className='metric-label'),
                        html.Span(f"{current.steps}", className='metric-value')
                    ], className='metric-row'),
                    html.Div([
                        html.Span("Cumulative Reward:", className='metric-label'),
                        html.Span(f"{current.total_reward:.2f}", className='metric-value')
                    ], className='metric-row'),
                    html.Div([
                        html.Span("Last Step Reward:", className='metric-label'),
                        html.Span(f"{last_reward:.3f}", className='metric-value')
                    ], className='metric-row'),
                ])
            else:
                title = "🎬 Episode Analysis (No Active Episode)"
                content = html.Div("No active episode", style={'color': '#999'})
            
            # Episode history
            history = self.state.get_episode_history(3)
            if history:
                history_rows = []
                for ep in reversed(history):
                    row = html.Tr([
                        html.Td(f"{ep.episode_num}"),
                        html.Td(ep.status),
                        html.Td(ep.termination_reason[:15] + "..." if len(ep.termination_reason) > 15 else ep.termination_reason),
                        html.Td(f"{ep.total_reward:.2f}")
                    ])
                    history_rows.append(row)
                
                history_content = html.Div([
                    html.H4("📚 Episode History (Last 3)", style={'fontSize': '11px', 'margin': '10px 0 5px 0'}),
                    html.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Ep #"),
                                html.Th("Status"),
                                html.Th("Reason"),
                                html.Th("Reward")
                            ])
                        ]),
                        html.Tbody(history_rows)
                    ])
                ])
            else:
                history_content = html.Div()
            
            return title, content, history_content
        
        @self.app.callback(
            Output('training-progress', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_training_progress(n):
            prog = self.state.training_progress
            
            content = html.Div([
                html.Div([
                    html.Span("Mode:", className='metric-label'),
                    html.Span(prog.mode, className='metric-value')
                ], className='metric-row'),
                html.Div([
                    html.Span("Current Stage:", className='metric-label'),
                    html.Span(prog.current_stage, className='metric-value')
                ], className='metric-row'),
                html.Div([
                    html.Span("Overall Progress:", className='metric-label'),
                    html.Div([
                        html.Div(style={'width': f"{prog.overall_progress}%"}, className='progress-fill')
                    ], className='progress-bar', style={'flex': 1, 'marginLeft': '10px'})
                ], className='metric-row', style={'alignItems': 'center'}),
                html.Div([
                    html.Span("Stage Progress:", className='metric-label'),
                    html.Div([
                        html.Div(style={'width': f"{prog.stage_progress}%"}, className='progress-fill')
                    ], className='progress-bar', style={'flex': 1, 'marginLeft': '10px'})
                ], className='metric-row', style={'alignItems': 'center'}),
                html.Div([
                    html.Span("Stage Status:", className='metric-label'),
                    html.Span(prog.stage_status, className='metric-value', style={'fontSize': '10px'})
                ], className='metric-row'),
                html.Div([
                    html.Span("Updates:", className='metric-label'),
                    html.Span(f"{prog.updates}", className='metric-value')
                ], className='metric-row'),
                html.Div([
                    html.Span("Episodes:", className='metric-label'),
                    html.Span(f"{prog.total_episodes}", className='metric-value')
                ], className='metric-row'),
                html.Div([
                    html.Span("Global Steps:", className='metric-label'),
                    html.Span(f"{prog.global_steps:,}", className='metric-value')
                ], className='metric-row'),
            ])
            
            return content
        
        @self.app.callback(
            Output('ppo-metrics', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_ppo_metrics(n):
            ppo = self.state.ppo_metrics
            
            def make_sparkline(values):
                if not values:
                    return ""
                # Simple ASCII sparkline
                chars = "▁▂▃▄▅▆▇█"
                min_val = min(values)
                max_val = max(values)
                if max_val == min_val:
                    return "─" * len(values)
                
                sparkline = ""
                for v in values:
                    idx = int((v - min_val) / (max_val - min_val) * (len(chars) - 1))
                    sparkline += chars[idx]
                return sparkline
            
            content = html.Div([
                html.Div([
                    html.Span("Learning Rate:", className='metric-label'),
                    html.Span(f"{ppo.learning_rate:.1e}", className='metric-value')
                ], className='metric-row'),
                html.Div([
                    html.Span("Mean Reward:", className='metric-label'),
                    html.Span(f"{ppo.mean_reward_batch:.2f}", className='metric-value')
                ], className='metric-row'),
                html.Div([
                    html.Span("Policy Loss:", className='metric-label'),
                    html.Span([
                        f"{ppo.policy_loss:.3f} ",
                        html.Span(make_sparkline(list(ppo.policy_loss_history)), className='sparkline')
                    ], className='metric-value')
                ], className='metric-row'),
                html.Div([
                    html.Span("Value Loss:", className='metric-label'),
                    html.Span([
                        f"{ppo.value_loss:.3f} ",
                        html.Span(make_sparkline(list(ppo.value_loss_history)), className='sparkline')
                    ], className='metric-value')
                ], className='metric-row'),
                html.Div([
                    html.Span("Total Loss:", className='metric-label'),
                    html.Span(f"{ppo.total_loss:.3f}", className='metric-value')
                ], className='metric-row'),
                html.Div([
                    html.Span("Entropy:", className='metric-label'),
                    html.Span([
                        f"{ppo.entropy:.3f} ",
                        html.Span(make_sparkline(list(ppo.entropy_history)), className='sparkline')
                    ], className='metric-value')
                ], className='metric-row'),
                html.Div([
                    html.Span("Clip Fraction:", className='metric-label'),
                    html.Span(f"{ppo.clip_fraction:.3f}", className='metric-value')
                ], className='metric-row'),
                html.Div([
                    html.Span("Approx KL:", className='metric-label'),
                    html.Span(f"{ppo.approx_kl:.3f}", className='metric-value')
                ], className='metric-row'),
                html.Div([
                    html.Span("Explained Var:", className='metric-label'),
                    html.Span(f"{ppo.explained_variance:.3f}", className='metric-value')
                ], className='metric-row'),
            ])
            
            return content
        
        @self.app.callback(
            Output('reward-system', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_reward_system(n):
            # Placeholder for reward component breakdown
            # This would need integration with the reward calculator
            content = html.Div([
                html.P("Reward components tracking coming soon...", 
                      style={'color': '#999', 'fontSize': '10px', 'textAlign': 'center'})
            ])
            return content
        
        @self.app.callback(
            Output('action-analysis', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_action_analysis(n):
            # Calculate action bias
            self.state.calculate_action_bias()
            
            analysis = self.state.action_analysis
            
            content = [
                html.Div([
                    html.Span("Invalid Actions:", className='metric-label'),
                    html.Span(f"{analysis.invalid_actions_count}", className='metric-value negative')
                ], className='metric-row'),
                html.H4("Action Bias:", style={'fontSize': '11px', 'margin': '10px 0 5px 0'})
            ]
            
            if analysis.action_bias:
                rows = []
                for action, stats in analysis.action_bias.items():
                    row = html.Tr([
                        html.Td(action),
                        html.Td(f"{stats['count']}"),
                        html.Td(f"{stats['percent_steps']:.1f}"),
                        html.Td(f"{stats['mean_reward']:.3f}"),
                        html.Td(f"{stats['total_reward']:.2f}"),
                        html.Td(f"{stats['pos_reward_rate']:.1f}")
                    ])
                    rows.append(row)
                
                table = html.Table([
                    html.Thead([
                        html.Tr([
                            html.Th("Action"),
                            html.Th("Count"),
                            html.Th("%"),
                            html.Th("Mean R"),
                            html.Th("Total R"),
                            html.Th("Win%")
                        ])
                    ]),
                    html.Tbody(rows)
                ])
                content.append(table)
            
            return html.Div(content)
        
        @self.app.callback(
            Output('episode-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_episode_chart(n):
            if not self.state.current_episode:
                return go.Figure()
            
            episode = self.state.current_episode
            
            # Create subplots for current episode
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.5, 0.25, 0.25],
                subplot_titles=('Price & Trades (Current Episode)', 'Position', 'Reward')
            )
            
            steps = list(range(len(episode.price_history)))
            
            # Price line
            if episode.price_history:
                fig.add_trace(
                    go.Scatter(x=steps, y=list(episode.price_history), 
                              name='Price', line=dict(color='#00d084', width=1)),
                    row=1, col=1
                )
            
            # Add trades for current episode only
            for trade in episode.trades:
                idx = len(episode.price_history) - 1  # Approximate position
                color = '#00d084' if trade.side == 'BUY' else '#ff4757'
                symbol = 'triangle-up' if trade.side == 'BUY' else 'triangle-down'
                
                fig.add_trace(
                    go.Scatter(
                        x=[idx], y=[trade.entry_price],
                        mode='markers',
                        marker=dict(size=8, color=color, symbol=symbol),
                        showlegend=False,
                        hovertext=f"{trade.side} @ ${trade.entry_price:.2f}"
                    ),
                    row=1, col=1
                )
            
            # Position
            if episode.position_history:
                fig.add_trace(
                    go.Scatter(x=steps, y=list(episode.position_history),
                              name='Position', fill='tozeroy',
                              line=dict(color='#ffd32a', width=1)),
                    row=2, col=1
                )
            
            # Rewards
            if episode.reward_history:
                cumulative_rewards = np.cumsum(list(episode.reward_history))
                fig.add_trace(
                    go.Scatter(x=steps, y=cumulative_rewards.tolist(),
                              name='Cumulative Reward',
                              line=dict(color='#00d084', width=2)),
                    row=3, col=1
                )
            
            # Update layout
            fig.update_layout(
                template='plotly_dark',
                height=250,
                showlegend=False,
                margin=dict(l=40, r=10, t=30, b=30),
                font=dict(size=10)
            )
            
            fig.update_xaxes(title_text="Step", row=3, col=1)
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="Pos", row=2, col=1)
            fig.update_yaxes(title_text="Reward", row=3, col=1)
            
            return fig
        
        @self.app.callback(
            Output('training-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_training_chart(n):
            # Overall training metrics across episodes
            history = list(self.state.episode_history)
            
            if not history:
                return go.Figure()
            
            episodes = [ep.episode_num for ep in history]
            rewards = [ep.total_reward for ep in history]
            pnls = [ep.total_pnl for ep in history]
            
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=('Episode Rewards', 'Episode P&L')
            )
            
            # Rewards
            fig.add_trace(
                go.Scatter(x=episodes, y=rewards, name='Reward',
                          line=dict(color='#00d084', width=2)),
                row=1, col=1
            )
            
            # P&L
            fig.add_trace(
                go.Bar(x=episodes, y=pnls, name='P&L',
                      marker_color=['#00d084' if p > 0 else '#ff4757' for p in pnls]),
                row=2, col=1
            )
            
            fig.update_layout(
                template='plotly_dark',
                height=250,
                showlegend=False,
                margin=dict(l=40, r=10, t=30, b=30),
                font=dict(size=10)
            )
            
            fig.update_xaxes(title_text="Episode", row=2, col=1)
            fig.update_yaxes(title_text="Reward", row=1, col=1)
            fig.update_yaxes(title_text="P&L ($)", row=2, col=1)
            
            return fig
        
        @self.app.callback(
            Output('footer-stats', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_footer(n):
            prog = self.state.training_progress
            
            return html.Div([
                html.Span(f"Steps/Sec: {prog.steps_per_second:.1f}"),
                html.Span(f"Time/Update: {prog.time_per_update:.2f}s"),
                html.Span(f"Time/Episode: {prog.time_per_episode:.1f}s")
            ], style={'display': 'flex', 'gap': '20px'})
    
    def _process_queue_updates(self):
        """Process all pending updates from queue"""
        processed = 0
        while not self.update_queue.empty() and processed < 50:  # Limit to prevent blocking
            try:
                update = self.update_queue.get_nowait()
                self._process_update(update)
                processed += 1
            except queue.Empty:
                break
    
    def _process_update(self, update: Dict[str, Any]):
        """Process a single update"""
        update_type = update.get('type')
        data = update.get('data', {})
        
        if update_type == 'market':
            self.state.update_market(data)
        elif update_type == 'position':
            self.state.update_position(data)
        elif update_type == 'portfolio':
            self.state.update_portfolio(data)
        elif update_type == 'action':
            self.state.add_action(
                data.get('step', 0),
                data.get('action_type', 'HOLD'),
                data.get('size', 1.0),
                data.get('reward', 0.0)
            )
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
    
    # Public update methods
    def update_market(self, data: Dict[str, Any]):
        """Update market data"""
        self.update_queue.put({'type': 'market', 'data': data})
    
    def update_position(self, data: Dict[str, Any]):
        """Update position data"""
        self.update_queue.put({'type': 'position', 'data': data})
    
    def update_portfolio(self, data: Dict[str, Any]):
        """Update portfolio data"""
        self.update_queue.put({'type': 'portfolio', 'data': data})
    
    def update_action(self, step: int, action_type: str, size: float, reward: float):
        """Update action data"""
        self.update_queue.put({
            'type': 'action',
            'data': {'step': step, 'action_type': action_type, 'size': size, 'reward': reward}
        })
    
    def update_trade(self, trade_data: Dict[str, Any]):
        """Update trade data"""
        self.update_queue.put({'type': 'trade', 'data': trade_data})
    
    def start_episode(self, episode_num: int):
        """Start new episode"""
        self.update_queue.put({'type': 'episode_start', 'data': {'episode_num': episode_num}})
    
    def end_episode(self, reason: str = 'Completed'):
        """End current episode"""
        self.update_queue.put({'type': 'episode_end', 'data': {'reason': reason}})
    
    def update_training_progress(self, data: Dict[str, Any]):
        """Update training progress"""
        self.update_queue.put({'type': 'training', 'data': data})
    
    def update_ppo_metrics(self, data: Dict[str, Any]):
        """Update PPO metrics"""
        self.update_queue.put({'type': 'ppo_metrics', 'data': data})
    
    def set_model_info(self, name: str):
        """Set model information"""
        self.update_queue.put({'type': 'model_info', 'data': {'name': name}})
    
    def start(self, open_browser: bool = True):
        """Start the dashboard server"""
        if self.is_running:
            logger.warning("Dashboard is already running")
            return
        
        self.is_running = True
        
        # Start server in separate thread
        def run_server():
            self.app.run_server(
                host='127.0.0.1',
                port=self.port,
                debug=False,
                use_reloader=False
            )
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        
        # Wait for server to start
        time.sleep(2)
        
        # Open browser
        if open_browser:
            url = f"http://127.0.0.1:{self.port}"
            webbrowser.open(url)
            logger.info(f"Dashboard started at {url}")
    
    def stop(self):
        """Stop the dashboard server"""
        self.is_running = False
        logger.info("Dashboard stopped")