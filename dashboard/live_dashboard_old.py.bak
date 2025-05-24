import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
from collections import deque
import threading
import webbrowser
import logging
from typing import Dict, List, Any, Optional
import queue
import time

logger = logging.getLogger(__name__)


class LiveTradingDashboard:
    """Real-time trading dashboard using Plotly Dash"""
    
    def __init__(self, port: int = 8050, update_interval: int = 1000, max_points: int = 1000):
        self.port = port
        self.update_interval = update_interval  # milliseconds
        self.max_points = max_points
        
        # Data storage with thread-safe deques
        self.price_data = deque(maxlen=max_points)
        self.volume_data = deque(maxlen=max_points)
        self.position_data = deque(maxlen=max_points)
        self.reward_data = deque(maxlen=max_points)
        self.equity_data = deque(maxlen=max_points)
        self.trades = deque(maxlen=100)
        self.episode_stats = deque(maxlen=50)
        
        # Current state
        self.current_step = 0
        self.current_episode = 0
        self.current_price = 0
        self.current_position = 0
        self.current_equity = 0
        self.total_pnl = 0
        
        # Action tracking
        self.action_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
        self.win_loss_counts = {"wins": 0, "losses": 0}
        
        # Feature data for heatmap
        self.feature_data = {}
        
        # Thread-safe queue for updates
        self.update_queue = queue.Queue()
        
        # Dashboard app
        self.app = None
        self.server_thread = None
        self.is_running = False
        
        self._create_app()
        
    def _create_app(self):
        """Create the Dash application"""
        self.app = dash.Dash(__name__)
        
        # Define CSS styles
        self.app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>{%title%}</title>
                {%favicon%}
                {%css%}
                <style>
                    body {
                        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                        background-color: #0e1117;
                        color: #e6e6e6;
                        margin: 0;
                        padding: 0;
                    }
                    .main-header {
                        background-color: #1e2130;
                        padding: 20px;
                        text-align: center;
                        border-bottom: 2px solid #3d4158;
                    }
                    .metric-card {
                        background-color: #1e2130;
                        border-radius: 8px;
                        padding: 15px;
                        margin: 10px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
                    }
                    .metric-value {
                        font-size: 28px;
                        font-weight: bold;
                        margin: 5px 0;
                    }
                    .metric-label {
                        font-size: 14px;
                        color: #999;
                        text-transform: uppercase;
                    }
                    .positive { color: #00d084; }
                    .negative { color: #ff4757; }
                    .neutral { color: #ffd32a; }
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
                html.H1("🚀 FX-AI Live Trading Dashboard", style={'color': '#e6e6e6', 'margin': 0}),
                html.P(f"Real-time monitoring of trading agent performance", 
                      style={'color': '#999', 'margin': 0})
            ], className='main-header'),
            
            # Metrics Row
            html.Div([
                html.Div([
                    html.Div("Current Price", className='metric-label'),
                    html.Div(id='current-price', className='metric-value neutral'),
                ], className='metric-card', style={'flex': 1}),
                
                html.Div([
                    html.Div("Position", className='metric-label'),
                    html.Div(id='current-position', className='metric-value'),
                ], className='metric-card', style={'flex': 1}),
                
                html.Div([
                    html.Div("Equity", className='metric-label'),
                    html.Div(id='current-equity', className='metric-value'),
                ], className='metric-card', style={'flex': 1}),
                
                html.Div([
                    html.Div("Total P&L", className='metric-label'),
                    html.Div(id='total-pnl', className='metric-value'),
                ], className='metric-card', style={'flex': 1}),
                
                html.Div([
                    html.Div("Episode", className='metric-label'),
                    html.Div(id='episode-step', className='metric-value neutral'),
                ], className='metric-card', style={'flex': 1}),
            ], style={'display': 'flex', 'padding': '20px'}),
            
            # Main Charts
            html.Div([
                # Price and Position Chart
                dcc.Graph(id='price-chart', style={'height': '400px'}),
                
                # Volume and Actions Chart
                html.Div([
                    dcc.Graph(id='volume-chart', style={'flex': 1, 'height': '200px'}),
                    dcc.Graph(id='action-distribution', style={'flex': 1, 'height': '200px'}),
                ], style={'display': 'flex'}),
                
                # Reward and Equity Charts
                html.Div([
                    dcc.Graph(id='reward-chart', style={'flex': 1, 'height': '300px'}),
                    dcc.Graph(id='equity-chart', style={'flex': 1, 'height': '300px'}),
                ], style={'display': 'flex'}),
                
                # Feature Heatmap
                dcc.Graph(id='feature-heatmap', style={'height': '300px'}),
                
                # Recent Trades Table
                html.Div(id='trades-table', style={'padding': '20px'}),
                
            ], style={'padding': '20px'}),
            
            # Auto-update interval
            dcc.Interval(id='interval-component', interval=self.update_interval),
            
            # Hidden div to store data
            html.Div(id='hidden-data', style={'display': 'none'})
        ])
        
        # Callbacks
        self._setup_callbacks()
        
    def _setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            [Output('current-price', 'children'),
             Output('current-price', 'className'),
             Output('current-position', 'children'),
             Output('current-position', 'className'),
             Output('current-equity', 'children'),
             Output('current-equity', 'className'),
             Output('total-pnl', 'children'),
             Output('total-pnl', 'className'),
             Output('episode-step', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_metrics(n):
            # Process updates from queue
            while not self.update_queue.empty():
                try:
                    update = self.update_queue.get_nowait()
                    self._process_update(update)
                except queue.Empty:
                    break
            
            # Format metrics
            price_class = 'metric-value neutral'
            position_class = 'metric-value positive' if self.current_position > 0 else \
                           'metric-value negative' if self.current_position < 0 else 'metric-value neutral'
            equity_class = 'metric-value positive' if self.current_equity > 25000 else 'metric-value negative'
            pnl_class = 'metric-value positive' if self.total_pnl > 0 else 'metric-value negative'
            
            return (
                f"${self.current_price:.4f}",
                price_class,
                f"{self.current_position:+.2f}",
                position_class,
                f"${self.current_equity:,.2f}",
                equity_class,
                f"${self.total_pnl:+,.2f}",
                pnl_class,
                f"Ep {self.current_episode} | Step {self.current_step}"
            )
        
        @self.app.callback(
            Output('price-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_price_chart(n):
            if not self.price_data:
                return go.Figure()
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3],
                subplot_titles=('Price & Trades', 'Position')
            )
            
            # Price line
            steps = list(range(len(self.price_data)))
            prices = list(self.price_data)
            
            fig.add_trace(
                go.Scatter(x=steps, y=prices, name='Price', 
                          line=dict(color='#00d084', width=2)),
                row=1, col=1
            )
            
            # Add trades
            for trade in self.trades:
                if trade['step'] < len(steps):
                    color = '#00d084' if trade['action'] == 'BUY' else '#ff4757'
                    symbol = 'triangle-up' if trade['action'] == 'BUY' else 'triangle-down'
                    fig.add_trace(
                        go.Scatter(
                            x=[trade['step']], 
                            y=[trade['price']],
                            mode='markers',
                            marker=dict(size=12, color=color, symbol=symbol),
                            name=trade['action'],
                            showlegend=False,
                            hovertext=f"{trade['action']} @ ${trade['price']:.4f}<br>P&L: ${trade.get('pnl', 0):.2f}"
                        ),
                        row=1, col=1
                    )
            
            # Position line
            positions = list(self.position_data) if self.position_data else []
            if positions:
                fig.add_trace(
                    go.Scatter(x=steps[-len(positions):], y=positions, 
                              name='Position', fill='tozeroy',
                              line=dict(color='#ffd32a', width=2)),
                    row=2, col=1
                )
            
            # Update layout
            fig.update_layout(
                template='plotly_dark',
                height=400,
                showlegend=True,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            fig.update_xaxes(title_text="Step", row=2, col=1)
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Position", row=2, col=1)
            
            return fig
        
        @self.app.callback(
            Output('volume-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_volume_chart(n):
            if not self.volume_data:
                return go.Figure()
            
            steps = list(range(len(self.volume_data)))
            volumes = list(self.volume_data)
            
            fig = go.Figure()
            fig.add_trace(
                go.Bar(x=steps, y=volumes, name='Volume',
                      marker_color='#3d4158')
            )
            
            fig.update_layout(
                template='plotly_dark',
                title='Volume',
                height=200,
                margin=dict(l=0, r=0, t=40, b=0),
                showlegend=False
            )
            
            return fig
        
        @self.app.callback(
            Output('action-distribution', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_action_distribution(n):
            # Create pie chart of actions
            labels = list(self.action_counts.keys())
            values = list(self.action_counts.values())
            colors = ['#00d084', '#ff4757', '#ffd32a']
            
            fig = go.Figure(data=[
                go.Pie(labels=labels, values=values, hole=.3,
                      marker=dict(colors=colors))
            ])
            
            fig.update_layout(
                template='plotly_dark',
                title='Action Distribution',
                height=200,
                margin=dict(l=0, r=0, t=40, b=0),
                showlegend=True
            )
            
            return fig
        
        @self.app.callback(
            Output('reward-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_reward_chart(n):
            if not self.reward_data:
                return go.Figure()
            
            steps = list(range(len(self.reward_data)))
            rewards = list(self.reward_data)
            cumulative_rewards = np.cumsum(rewards).tolist()
            
            fig = go.Figure()
            
            # Step rewards
            fig.add_trace(
                go.Bar(x=steps, y=rewards, name='Step Reward',
                      marker_color=['#00d084' if r > 0 else '#ff4757' for r in rewards],
                      opacity=0.6)
            )
            
            # Cumulative reward
            fig.add_trace(
                go.Scatter(x=steps, y=cumulative_rewards, name='Cumulative',
                          line=dict(color='#ffd32a', width=3),
                          yaxis='y2')
            )
            
            fig.update_layout(
                template='plotly_dark',
                title='Rewards',
                height=300,
                margin=dict(l=0, r=0, t=40, b=0),
                yaxis=dict(title='Step Reward'),
                yaxis2=dict(title='Cumulative', overlaying='y', side='right'),
                showlegend=True
            )
            
            return fig
        
        @self.app.callback(
            Output('equity-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_equity_chart(n):
            if not self.equity_data:
                return go.Figure()
            
            steps = list(range(len(self.equity_data)))
            equity = list(self.equity_data)
            
            fig = go.Figure()
            
            # Equity line
            fig.add_trace(
                go.Scatter(x=steps, y=equity, name='Equity',
                          line=dict(color='#00d084', width=2),
                          fill='tozeroy', fillcolor='rgba(0, 208, 132, 0.1)')
            )
            
            # Add baseline
            if equity:
                fig.add_hline(y=25000, line_dash="dash", line_color="gray",
                             annotation_text="Initial Capital")
            
            fig.update_layout(
                template='plotly_dark',
                title='Portfolio Equity',
                height=300,
                margin=dict(l=0, r=0, t=40, b=0),
                yaxis=dict(title='Equity ($)'),
                showlegend=False
            )
            
            return fig
        
        @self.app.callback(
            Output('feature-heatmap', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_feature_heatmap(n):
            if not self.feature_data:
                return go.Figure()
            
            # Get last N steps of features
            n_steps = min(50, self.current_step)
            feature_names = list(self.feature_data.keys())[:20]  # Top 20 features
            
            if not feature_names:
                return go.Figure()
            
            # Create matrix
            matrix = []
            for fname in feature_names:
                if fname in self.feature_data:
                    values = list(self.feature_data[fname])[-n_steps:]
                    matrix.append(values)
            
            if not matrix:
                return go.Figure()
            
            fig = go.Figure(data=go.Heatmap(
                z=matrix,
                x=list(range(len(matrix[0]))),
                y=feature_names,
                colorscale='RdBu',
                zmid=0
            ))
            
            fig.update_layout(
                template='plotly_dark',
                title='Feature Activity Heatmap (Recent Steps)',
                height=300,
                margin=dict(l=100, r=0, t=40, b=0),
                xaxis=dict(title='Step'),
                yaxis=dict(title='Feature')
            )
            
            return fig
        
        @self.app.callback(
            Output('trades-table', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_trades_table(n):
            if not self.trades:
                return html.Div("No trades yet", style={'color': '#999', 'textAlign': 'center'})
            
            # Create table
            recent_trades = list(self.trades)[-10:]  # Last 10 trades
            recent_trades.reverse()  # Most recent first
            
            rows = []
            for trade in recent_trades:
                pnl = trade.get('pnl', 0)
                pnl_class = 'positive' if pnl > 0 else 'negative'
                
                row = html.Tr([
                    html.Td(f"Step {trade['step']}"),
                    html.Td(trade['action'], className=pnl_class),
                    html.Td(f"${trade['price']:.4f}"),
                    html.Td(f"{trade.get('quantity', 1):.2f}"),
                    html.Td(f"${pnl:+.2f}", className=pnl_class),
                    html.Td(trade.get('time', ''))
                ])
                rows.append(row)
            
            table = html.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Step"),
                        html.Th("Action"),
                        html.Th("Price"),
                        html.Th("Quantity"),
                        html.Th("P&L"),
                        html.Th("Time")
                    ])
                ]),
                html.Tbody(rows)
            ], style={
                'width': '100%',
                'borderCollapse': 'collapse',
                'backgroundColor': '#1e2130',
                'borderRadius': '8px',
                'overflow': 'hidden'
            })
            
            return html.Div([
                html.H3("Recent Trades", style={'color': '#e6e6e6'}),
                table
            ])
    
    def _process_update(self, update: Dict[str, Any]):
        """Process update from queue"""
        update_type = update.get('type')
        
        if update_type == 'step':
            self._update_step_data(update.get('data', {}))
        elif update_type == 'trade':
            self._update_trade_data(update.get('data', {}))
        elif update_type == 'episode':
            self._update_episode_data(update.get('data', {}))
        elif update_type == 'features':
            self._update_feature_data(update.get('data', {}))
    
    def _update_step_data(self, data: Dict[str, Any]):
        """Update step-by-step data"""
        self.current_step = data.get('step', self.current_step)
        self.current_price = data.get('price', self.current_price)
        self.current_position = data.get('position', self.current_position)
        self.current_equity = data.get('equity', self.current_equity)
        
        # Append to deques
        if 'price' in data:
            self.price_data.append(data['price'])
        if 'volume' in data:
            self.volume_data.append(data['volume'])
        if 'position' in data:
            self.position_data.append(data['position'])
        if 'reward' in data:
            self.reward_data.append(data['reward'])
        if 'equity' in data:
            self.equity_data.append(data['equity'])
        
        # Update action counts
        if 'action' in data:
            action = data['action'].upper()
            if action in self.action_counts:
                self.action_counts[action] += 1
    
    def _update_trade_data(self, data: Dict[str, Any]):
        """Update trade data"""
        trade_info = {
            'step': data.get('step', self.current_step),
            'action': data.get('action', 'UNKNOWN'),
            'price': data.get('price', 0),
            'quantity': data.get('quantity', 0),
            'pnl': data.get('pnl', 0),
            'time': datetime.now().strftime('%H:%M:%S')
        }
        self.trades.append(trade_info)
        
        # Update P&L
        self.total_pnl += trade_info.get('pnl', 0)
        
        # Update win/loss
        if trade_info['pnl'] > 0:
            self.win_loss_counts['wins'] += 1
        elif trade_info['pnl'] < 0:
            self.win_loss_counts['losses'] += 1
    
    def _update_episode_data(self, data: Dict[str, Any]):
        """Update episode data"""
        self.current_episode = data.get('episode', self.current_episode)
        
        # Store episode stats
        self.episode_stats.append({
            'episode': self.current_episode,
            'total_reward': data.get('total_reward', 0),
            'total_pnl': data.get('total_pnl', 0),
            'steps': data.get('steps', 0),
            'win_rate': data.get('win_rate', 0),
            'sharpe': data.get('sharpe', 0)
        })
        
        # Reset step counter if new episode
        if data.get('reset', False):
            self.current_step = 0
    
    def _update_feature_data(self, data: Dict[str, Any]):
        """Update feature data for heatmap"""
        for feature_name, value in data.items():
            if feature_name not in self.feature_data:
                self.feature_data[feature_name] = deque(maxlen=self.max_points)
            self.feature_data[feature_name].append(value)
    
    # Public methods for external updates
    def update_step(self, step_data: Dict[str, Any]):
        """Update dashboard with step data"""
        self.update_queue.put({'type': 'step', 'data': step_data})
    
    def update_trade(self, trade_data: Dict[str, Any]):
        """Update dashboard with trade data"""
        self.update_queue.put({'type': 'trade', 'data': trade_data})
    
    def update_episode(self, episode_data: Dict[str, Any]):
        """Update dashboard with episode data"""
        self.update_queue.put({'type': 'episode', 'data': episode_data})
    
    def update_features(self, feature_data: Dict[str, Any]):
        """Update dashboard with feature data"""
        self.update_queue.put({'type': 'features', 'data': feature_data})
    
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
        
        # Wait a bit for server to start
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


# Convenience function
def create_dashboard(port: int = 8050, update_interval: int = 1000) -> LiveTradingDashboard:
    """Create and return a dashboard instance"""
    return LiveTradingDashboard(port=port, update_interval=update_interval)