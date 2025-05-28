"""
Momentum-Based Trading Dashboard v3
Designed specifically for momentum trading with curriculum learning and multi-day training.
Features momentum day tracking, curriculum progress, reset point analysis, and real-time metrics.
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np
import threading
import webbrowser
import logging
from typing import Dict, List, Any, Optional, Tuple
import queue
import time
import os
from datetime import datetime, timedelta
import json

from .dashboard_data import DashboardState

logger = logging.getLogger(__name__)

# Suppress unnecessary logs
logging.getLogger('werkzeug').setLevel(logging.ERROR)
logging.getLogger('dash').setLevel(logging.ERROR)
os.environ['DASH_SILENCE_ROUTES_LOGGING'] = 'true'

# Color schemes for momentum dashboard
COLORS = {
    'primary': '#1f77b4',      # Blue
    'success': '#2ca02c',      # Green
    'warning': '#ff7f0e',      # Orange
    'danger': '#d62728',       # Red
    'info': '#17becf',         # Cyan
    'dark': '#2c3e50',         # Dark blue-gray
    'light': '#ecf0f1',        # Light gray
    'momentum_high': '#e74c3c', # Red for high momentum
    'momentum_med': '#f39c12',  # Orange for medium momentum
    'momentum_low': '#3498db',  # Blue for low momentum
    'background': '#ffffff',    # White background
    'card_bg': '#f8f9fa',      # Light background for cards
}

class MomentumDashboard:
    """Advanced dashboard for momentum-based algorithmic trading"""
    
    def __init__(self, port: int = 8050, update_interval: int = 1000):
        self.port = port
        self.update_interval = update_interval
        
        # State management
        self.state = DashboardState()
        self.update_queue = queue.Queue()
        
        # Momentum-specific tracking
        self.momentum_days = {}
        self.curriculum_history = []
        self.reset_point_performance = {}
        self.day_performance_stats = {}
        
        # Dashboard components
        self.app = None
        self.server_thread = None
        self.is_running = False
        
        self._create_app()
    
    def _create_app(self):
        """Create the Dash application with momentum-focused layout"""
        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)
        self.app.title = "Momentum Trading Dashboard"
        
        # Custom CSS
        self.app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>Momentum Trading Dashboard</title>
                {%favicon%}
                {%css%}
                <style>
                    body { margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
                    .dashboard-container { padding: 10px; background-color: #f8f9fa; }
                    .metric-card { 
                        background: white; 
                        border-radius: 8px; 
                        padding: 15px; 
                        margin: 5px; 
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        border-left: 4px solid #1f77b4;
                    }
                    .momentum-high { border-left-color: #e74c3c !important; }
                    .momentum-med { border-left-color: #f39c12 !important; }
                    .momentum-low { border-left-color: #3498db !important; }
                    .status-running { color: #2ca02c; font-weight: bold; }
                    .status-stopped { color: #d62728; font-weight: bold; }
                    .curriculum-progress { 
                        background: linear-gradient(90deg, #3498db 0%, #2ecc71 50%, #e74c3c 100%);
                        height: 10px;
                        border-radius: 5px;
                        overflow: hidden;
                    }
                    .small-metric { font-size: 0.9em; color: #666; }
                    .big-number { font-size: 2em; font-weight: bold; color: #2c3e50; }
                    .momentum-indicator {
                        display: inline-block;
                        width: 12px;
                        height: 12px;
                        border-radius: 50%;
                        margin-right: 8px;
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
        
        self.app.layout = self._create_layout()
        self._register_callbacks()
    
    def _create_layout(self):
        """Create the main dashboard layout"""
        return html.Div([
            # Header
            html.Div([
                html.H1("🎯 Momentum Trading Dashboard", 
                       style={'textAlign': 'center', 'color': COLORS['dark'], 'margin': '10px 0'}),
                html.Div(id='status-indicator', style={'textAlign': 'center', 'margin': '5px 0'})
            ], className='dashboard-header'),
            
            # Main content
            html.Div([
                # Top row - Key metrics
                html.Div([
                    self._create_metrics_row(),
                ], style={'margin': '10px 0'}),
                
                # Second row - Momentum day info and curriculum progress
                html.Div([
                    html.Div([
                        self._create_momentum_day_card(),
                    ], style={'width': '60%', 'display': 'inline-block', 'vertical-align': 'top'}),
                    html.Div([
                        self._create_curriculum_card(),
                    ], style={'width': '40%', 'display': 'inline-block', 'vertical-align': 'top'}),
                ], style={'margin': '10px 0'}),
                
                # Third row - Charts
                html.Div([
                    html.Div([
                        self._create_performance_chart(),
                    ], style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top'}),
                    html.Div([
                        self._create_curriculum_progress_chart(),
                    ], style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top'}),
                ], style={'margin': '10px 0'}),
                
                # Fourth row - Detailed charts
                html.Div([
                    html.Div([
                        self._create_momentum_days_overview(),
                    ], style={'width': '60%', 'display': 'inline-block', 'vertical-align': 'top'}),
                    html.Div([
                        self._create_reset_points_analysis(),
                    ], style={'width': '40%', 'display': 'inline-block', 'vertical-align': 'top'}),
                ], style={'margin': '10px 0'}),
                
                # Bottom row - Component analysis
                html.Div([
                    self._create_reward_components_chart(),
                ], style={'margin': '10px 0'}),
                
            ], className='dashboard-container'),
            
            # Auto-refresh component
            dcc.Interval(
                id='interval-component',
                interval=self.update_interval,
                n_intervals=0
            ),
            
            # Hidden divs for data storage
            html.Div(id='data-store', style={'display': 'none'}),
            
        ], style={'backgroundColor': COLORS['light']})
    
    def _create_metrics_row(self):
        """Create the top metrics row"""
        return html.Div([
            html.Div([
                html.Div("Training Status", className='small-metric'),
                html.Div(id='training-status', className='big-number'),
                html.Div(id='training-progress', style={'margin': '5px 0'}),
            ], className='metric-card', style={'width': '18%', 'display': 'inline-block'}),
            
            html.Div([
                html.Div("Episodes", className='small-metric'),
                html.Div(id='total-episodes', className='big-number'),
                html.Div(id='episode-rate', className='small-metric'),
            ], className='metric-card', style={'width': '15%', 'display': 'inline-block'}),
            
            html.Div([
                html.Div("Mean Reward", className='small-metric'),
                html.Div(id='mean-reward', className='big-number'),
                html.Div(id='reward-trend', className='small-metric'),
            ], className='metric-card', style={'width': '15%', 'display': 'inline-block'}),
            
            html.Div([
                html.Div("Learning Rate", className='small-metric'),
                html.Div(id='learning-rate', className='big-number'),
                html.Div(id='lr-schedule', className='small-metric'),
            ], className='metric-card', style={'width': '15%', 'display': 'inline-block'}),
            
            html.Div([
                html.Div("PPO Metrics", className='small-metric'),
                html.Div(id='ppo-loss', className='big-number'),
                html.Div(id='ppo-details', className='small-metric'),
            ], className='metric-card', style={'width': '15%', 'display': 'inline-block'}),
            
            html.Div([
                html.Div("Performance", className='small-metric'),
                html.Div(id='steps-per-sec', className='big-number'),
                html.Div(id='eta-time', className='small-metric'),
            ], className='metric-card', style={'width': '15%', 'display': 'inline-block'}),
        ], style={'display': 'flex', 'justify-content': 'space-between'})
    
    def _create_momentum_day_card(self):
        """Create momentum day information card"""
        return html.Div([
            html.H3("📅 Current Momentum Day", style={'color': COLORS['dark'], 'margin': '0 0 15px 0'}),
            html.Div(id='current-momentum-day'),
            html.Hr(),
            html.H4("🎯 Reset Points", style={'color': COLORS['dark'], 'margin': '15px 0 10px 0'}),
            html.Div(id='reset-points-info'),
        ], className='metric-card')
    
    def _create_curriculum_card(self):
        """Create curriculum learning progress card"""
        return html.Div([
            html.H3("📚 Curriculum Learning", style={'color': COLORS['dark'], 'margin': '0 0 15px 0'}),
            html.Div([
                html.Div("Progress", className='small-metric'),
                html.Div(id='curriculum-progress-bar', style={'margin': '10px 0'}),
                html.Div(id='curriculum-progress-text', className='small-metric'),
            ]),
            html.Hr(),
            html.Div([
                html.Div("Strategy", className='small-metric'),
                html.Div(id='curriculum-strategy', style={'font-weight': 'bold'}),
            ], style={'margin': '10px 0'}),
            html.Div([
                html.Div("Difficulty Level", className='small-metric'),
                html.Div(id='difficulty-level', style={'font-weight': 'bold'}),
            ]),
        ], className='metric-card')
    
    def _create_performance_chart(self):
        """Create performance over time chart"""
        return html.Div([
            html.H3("📈 Training Performance", style={'color': COLORS['dark'], 'margin': '0 0 10px 0'}),
            dcc.Graph(id='performance-chart', config={'displayModeBar': False}),
        ], className='metric-card')
    
    def _create_curriculum_progress_chart(self):
        """Create curriculum progress chart"""
        return html.Div([
            html.H3("🎓 Curriculum Progress", style={'color': COLORS['dark'], 'margin': '0 0 10px 0'}),
            dcc.Graph(id='curriculum-chart', config={'displayModeBar': False}),
        ], className='metric-card')
    
    def _create_momentum_days_overview(self):
        """Create momentum days overview"""
        return html.Div([
            html.H3("📊 Momentum Days Overview", style={'color': COLORS['dark'], 'margin': '0 0 10px 0'}),
            dcc.Graph(id='momentum-days-chart', config={'displayModeBar': False}),
        ], className='metric-card')
    
    def _create_reset_points_analysis(self):
        """Create reset points analysis"""
        return html.Div([
            html.H3("🎯 Reset Points Analysis", style={'color': COLORS['dark'], 'margin': '0 0 10px 0'}),
            dcc.Graph(id='reset-points-chart', config={'displayModeBar': False}),
        ], className='metric-card')
    
    def _create_reward_components_chart(self):
        """Create reward components breakdown chart"""
        return html.Div([
            html.H3("🏆 Reward Components Analysis", style={'color': COLORS['dark'], 'margin': '0 0 10px 0'}),
            dcc.Graph(id='reward-components-chart', config={'displayModeBar': False}),
        ], className='metric-card')
    
    def _register_callbacks(self):
        """Register all dashboard callbacks"""
        
        @self.app.callback(
            [Output('training-status', 'children'),
             Output('training-status', 'className'),
             Output('training-progress', 'children'),
             Output('total-episodes', 'children'),
             Output('episode-rate', 'children'),
             Output('mean-reward', 'children'),
             Output('reward-trend', 'children'),
             Output('learning-rate', 'children'),
             Output('lr-schedule', 'children'),
             Output('ppo-loss', 'children'),
             Output('ppo-details', 'children'),
             Output('steps-per-sec', 'children'),
             Output('eta-time', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_metrics(n):
            """Update top-level metrics"""
            try:
                data = self._get_latest_data()
                
                # Training status
                status = data.get('mode', 'Unknown')
                status_class = 'big-number status-running' if status == 'Training' else 'big-number status-stopped'
                
                # Progress bar
                progress = data.get('overall_progress', 0)
                progress_bar = html.Div([
                    html.Div(style={
                        'width': f'{progress}%',
                        'height': '8px',
                        'background': COLORS['success'],
                        'border-radius': '4px',
                        'transition': 'width 0.3s ease'
                    })
                ], style={
                    'width': '100%',
                    'height': '8px',
                    'background': COLORS['light'],
                    'border-radius': '4px',
                    'overflow': 'hidden'
                })
                
                # Episodes
                episodes = data.get('total_episodes', 0)
                episode_rate = f"{data.get('episodes_per_hour', 0):.1f}/hr"
                
                # Reward
                mean_reward = f"{data.get('mean_reward', 0):.3f}"
                reward_std = data.get('reward_std', 0)
                reward_trend = f"σ: {reward_std:.3f}" if reward_std else ""
                
                # Learning rate
                lr = f"{data.get('lr', 0):.2e}"
                lr_schedule = data.get('lr_schedule', 'Fixed')
                
                # PPO metrics
                policy_loss = f"{data.get('policy_loss', 0):.4f}"
                clip_frac = data.get('clip_fraction', 0)
                entropy = data.get('entropy', 0)
                ppo_details = f"Clip: {clip_frac:.3f} | H: {entropy:.3f}"
                
                # Performance
                steps_per_sec = f"{data.get('steps_per_second', 0):.1f}"
                eta = data.get('eta_hours', 0)
                eta_text = f"ETA: {eta:.1f}h" if eta > 0 else "ETA: --"
                
                return (
                    status, status_class, progress_bar,
                    f"{episodes:,}", episode_rate,
                    mean_reward, reward_trend,
                    lr, lr_schedule,
                    policy_loss, ppo_details,
                    steps_per_sec, eta_text
                )
                
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
                return ("Error", "big-number", "", "0", "", "0.000", "", "0", "", "0", "", "0", "")
        
        @self.app.callback(
            [Output('current-momentum-day', 'children'),
             Output('reset-points-info', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_momentum_day_info(n):
            """Update momentum day and reset points information"""
            try:
                data = self._get_latest_data()
                momentum_day = data.get('current_momentum_day', {})
                
                if not momentum_day:
                    return "No momentum day selected", "No reset points available"
                
                # Momentum day info
                date_str = momentum_day.get('date', 'Unknown')
                quality = momentum_day.get('activity_score', 0)
                move = momentum_day.get('max_intraday_move', 0)
                volume_mult = momentum_day.get('volume_multiplier', 0)
                
                quality_class = 'momentum-high' if quality > 0.7 else 'momentum-med' if quality > 0.4 else 'momentum-low'
                
                day_info = html.Div([
                    html.Div([
                        html.Span(className=f'momentum-indicator {quality_class}', 
                                style={'backgroundColor': COLORS['momentum_high'] if quality > 0.7 else 
                                      COLORS['momentum_med'] if quality > 0.4 else COLORS['momentum_low']}),
                        html.Span(f"{date_str}", style={'font-weight': 'bold', 'font-size': '1.2em'})
                    ], style={'margin': '5px 0'}),
                    html.Div(f"Quality Score: {quality:.3f}", style={'margin': '3px 0'}),
                    html.Div(f"Max Move: {move:.1%}", style={'margin': '3px 0'}),
                    html.Div(f"Volume: {volume_mult:.1f}x avg", style={'margin': '3px 0'}),
                ])
                
                # Reset points info
                reset_points = momentum_day.get('reset_points', [])
                used_points = data.get('used_reset_point_indices', set())
                
                if reset_points:
                    total_points = len(reset_points)
                    used_count = len(used_points)
                    
                    reset_info = html.Div([
                        html.Div(f"Total Points: {total_points}", style={'margin': '3px 0'}),
                        html.Div(f"Used: {used_count}/{total_points}", style={'margin': '3px 0'}),
                        html.Div([
                            html.Div(style={
                                'width': f'{(used_count/total_points)*100}%',
                                'height': '6px',
                                'background': COLORS['warning'],
                                'border-radius': '3px'
                            })
                        ], style={
                            'width': '100%',
                            'height': '6px',
                            'background': COLORS['light'],
                            'border-radius': '3px',
                            'margin': '5px 0'
                        })
                    ])
                else:
                    reset_info = "No reset points available"
                
                return day_info, reset_info
                
            except Exception as e:
                logger.error(f"Error updating momentum day info: {e}")
                return "Error loading momentum day", "Error loading reset points"
        
        @self.app.callback(
            [Output('curriculum-progress-bar', 'children'),
             Output('curriculum-progress-text', 'children'),
             Output('curriculum-strategy', 'children'),
             Output('difficulty-level', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_curriculum_info(n):
            """Update curriculum learning information"""
            try:
                data = self._get_latest_data()
                
                progress = data.get('curriculum_progress', 0)
                strategy = data.get('curriculum_strategy', 'Unknown')
                
                # Progress bar
                progress_bar = html.Div([
                    html.Div(style={
                        'width': f'{progress*100}%',
                        'height': '100%',
                        'background': 'linear-gradient(90deg, #3498db 0%, #2ecc71 50%, #e74c3c 100%)',
                        'border-radius': '5px',
                        'transition': 'width 0.5s ease'
                    })
                ], className='curriculum-progress')
                
                progress_text = f"{progress:.1%} - "
                if progress < 0.3:
                    progress_text += "Beginner (Easy episodes)"
                elif progress < 0.7:
                    progress_text += "Intermediate (Mixed difficulty)"
                else:
                    progress_text += "Advanced (Hard episodes)"
                
                # Difficulty level
                if progress < 0.2:
                    difficulty = "🟢 Easy"
                elif progress < 0.5:
                    difficulty = "🟡 Medium"
                elif progress < 0.8:
                    difficulty = "🟠 Hard"
                else:
                    difficulty = "🔴 Expert"
                
                return progress_bar, progress_text, strategy.title(), difficulty
                
            except Exception as e:
                logger.error(f"Error updating curriculum info: {e}")
                return "", "Error", "Unknown", "Unknown"
        
        # Additional callbacks for charts will be implemented...
        self._register_chart_callbacks()
    
    def _register_chart_callbacks(self):
        """Register callbacks for charts"""
        
        @self.app.callback(
            Output('performance-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_performance_chart(n):
            """Update performance chart"""
            try:
                history = self.state.get_training_history()
                if not history:
                    return self._empty_chart("No training data available")
                
                df = pd.DataFrame(history)
                
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Episode Rewards', 'Training Metrics'),
                    vertical_spacing=0.1,
                    row_heights=[0.7, 0.3]
                )
                
                # Episode rewards
                if 'mean_reward' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df['mean_reward'],
                            mode='lines',
                            name='Mean Reward',
                            line=dict(color=COLORS['primary'], width=2)
                        ),
                        row=1, col=1
                    )
                
                # Learning rate
                if 'lr' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df['lr'],
                            mode='lines',
                            name='Learning Rate',
                            line=dict(color=COLORS['warning'], width=1),
                            yaxis='y2'
                        ),
                        row=2, col=1
                    )
                
                fig.update_layout(
                    height=400,
                    showlegend=True,
                    margin=dict(l=40, r=40, t=60, b=40),
                    plot_bgcolor='white'
                )
                
                return fig
                
            except Exception as e:
                logger.error(f"Error updating performance chart: {e}")
                return self._empty_chart("Error loading performance data")
        
        @self.app.callback(
            Output('curriculum-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_curriculum_chart(n):
            """Update curriculum progress chart"""
            try:
                if not self.curriculum_history:
                    return self._empty_chart("No curriculum data available")
                
                fig = go.Figure()
                
                x_vals = list(range(len(self.curriculum_history)))
                y_vals = self.curriculum_history
                
                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='lines+markers',
                    name='Curriculum Progress',
                    line=dict(color=COLORS['info'], width=3),
                    marker=dict(size=6)
                ))
                
                fig.update_layout(
                    title="Curriculum Learning Progress Over Time",
                    xaxis_title="Training Updates",
                    yaxis_title="Progress (0-1)",
                    height=300,
                    margin=dict(l=40, r=40, t=60, b=40),
                    plot_bgcolor='white',
                    yaxis=dict(range=[0, 1])
                )
                
                return fig
                
            except Exception as e:
                logger.error(f"Error updating curriculum chart: {e}")
                return self._empty_chart("Error loading curriculum data")
    
    def _empty_chart(self, message: str = "No data available"):
        """Create an empty chart with a message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color=COLORS['dark'])
        )
        fig.update_layout(
            height=300,
            margin=dict(l=40, r=40, t=40, b=40),
            plot_bgcolor='white',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig
    
    def _get_latest_data(self) -> Dict[str, Any]:
        """Get the latest dashboard data"""
        try:
            # Process any queued updates
            while not self.update_queue.empty():
                try:
                    update_data = self.update_queue.get_nowait()
                    self._process_update(update_data)
                except queue.Empty:
                    break
                except Exception as e:
                    logger.error(f"Error processing update: {e}")
            
            return self.state.get_current_state()
            
        except Exception as e:
            logger.error(f"Error getting latest data: {e}")
            return {}
    
    def _process_update(self, update_data: Dict[str, Any]):
        """Process incoming update data"""
        try:
            update_type = update_data.get('type', 'unknown')
            
            if update_type == 'training_update':
                self.state.update_training_state(update_data.get('data', {}))
                
            elif update_type == 'momentum_day':
                momentum_day = update_data.get('data', {})
                self.momentum_days[momentum_day.get('date')] = momentum_day
                
            elif update_type == 'curriculum_progress':
                progress = update_data.get('data', {}).get('progress', 0)
                self.curriculum_history.append(progress)
                # Keep only last 1000 points
                if len(self.curriculum_history) > 1000:
                    self.curriculum_history = self.curriculum_history[-1000:]
                    
            elif update_type == 'reset_point_performance':
                self.reset_point_performance.update(update_data.get('data', {}))
                
        except Exception as e:
            logger.error(f"Error processing update {update_type}: {e}")
    
    # Public interface methods
    def start(self, debug: bool = False, open_browser: bool = True):
        """Start the dashboard server"""
        if self.is_running:
            logger.warning("Dashboard is already running")
            return
        
        try:
            def run_server():
                self.app.run_server(
                    debug=debug,
                    host='127.0.0.1',
                    port=self.port,
                    use_reloader=False,
                    dev_tools_silence_routes_logging=True
                )
            
            self.server_thread = threading.Thread(target=run_server, daemon=True)
            self.server_thread.start()
            self.is_running = True
            
            # Wait a moment for server to start
            time.sleep(2)
            
            dashboard_url = f"http://127.0.0.1:{self.port}"
            logger.info(f"🚀 Momentum Dashboard started at {dashboard_url}")
            
            if open_browser:
                try:
                    webbrowser.open(dashboard_url)
                    logger.info("🌐 Opened dashboard in browser")
                except Exception as e:
                    logger.warning(f"Could not open browser: {e}")
            
        except Exception as e:
            logger.error(f"Failed to start dashboard: {e}")
            self.is_running = False
    
    def stop(self):
        """Stop the dashboard server"""
        if not self.is_running:
            return
        
        self.is_running = False
        logger.info("🛑 Dashboard stopped")
    
    def update_training_state(self, data: Dict[str, Any]):
        """Update training state from external source"""
        try:
            self.update_queue.put({
                'type': 'training_update',
                'data': data,
                'timestamp': time.time()
            })
        except Exception as e:
            logger.error(f"Error queuing training update: {e}")
    
    def update_momentum_day(self, momentum_day: Dict[str, Any]):
        """Update current momentum day information"""
        try:
            self.update_queue.put({
                'type': 'momentum_day',
                'data': momentum_day,
                'timestamp': time.time()
            })
        except Exception as e:
            logger.error(f"Error queuing momentum day update: {e}")
    
    def update_curriculum_progress(self, progress: float, strategy: str = None):
        """Update curriculum learning progress"""
        try:
            self.update_queue.put({
                'type': 'curriculum_progress',
                'data': {'progress': progress, 'strategy': strategy},
                'timestamp': time.time()
            })
        except Exception as e:
            logger.error(f"Error queuing curriculum update: {e}")
    
    def is_dashboard_running(self) -> bool:
        """Check if dashboard is running"""
        return self.is_running