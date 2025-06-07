"""Dashboard Server - Modular implementation using panel components"""

import logging
import threading
import webbrowser
from typing import Dict, Optional, List
from datetime import datetime

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from .shared_state import dashboard_state

# Import all panel components
from .panels.market_info_panel import MarketInfoPanel
from .panels.portfolio_panel import PortfolioPanel
from .panels.performance_panel import PerformancePanel
from .panels.actions_panel import ActionsPanel
from .panels.episode_training_panel import EpisodeTrainingPanel
from .panels.ppo_metrics_panel import PPOMetricsPanel
from .panels.training_manager_panel import TrainingManagerPanel
from .panels.trades_table_panel import TradesTablePanel
from .panels.reward_components_panel import RewardComponentsPanel
from .panels.captum_attribution_panel import CaptumAttributionPanel
from .panels.price_chart_panel import PriceChartPanel

# Dark theme colors (GitHub-inspired)
DARK_THEME = {
    "bg_primary": "#0d1117",  # Main background
    "bg_secondary": "#161b22",  # Cards/panels
    "bg_tertiary": "#21262d",  # Inputs/tables
    "border": "#30363d",  # Borders
    "text_primary": "#f0f6fc",  # Main text
    "text_secondary": "#8b949e",  # Secondary text
    "text_muted": "#6e7681",  # Muted text
    "accent_blue": "#58a6ff",  # Links/accents
    "accent_green": "#56d364",  # Success/profit
    "accent_red": "#f85149",  # Error/loss
    "accent_orange": "#d29922",  # Warning
    "accent_purple": "#bc8cff",  # Special
}


class DashboardServer:
    """Modular dashboard server using panel components"""

    def __init__(self, port: int = 8053):
        self.port = port
        self.app = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize all panels
        self.market_info_panel = MarketInfoPanel(DARK_THEME)
        self.portfolio_panel = PortfolioPanel(DARK_THEME)
        self.performance_panel = PerformancePanel(DARK_THEME)
        self.actions_panel = ActionsPanel(DARK_THEME)
        self.episode_training_panel = EpisodeTrainingPanel(DARK_THEME)
        self.ppo_metrics_panel = PPOMetricsPanel(DARK_THEME)
        self.training_manager_panel = TrainingManagerPanel(DARK_THEME)
        self.trades_table_panel = TradesTablePanel(DARK_THEME)
        self.reward_components_panel = RewardComponentsPanel(DARK_THEME)
        self.captum_panel = CaptumAttributionPanel(DARK_THEME)
        self.price_chart_panel = PriceChartPanel(DARK_THEME)

    def create_app(self) -> dash.Dash:
        """Create and configure the Dash application"""
        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)

        # Configure app
        self.app.title = "FxAI Trading Dashboard"
        self.app.layout = self._create_layout()

        # Register callbacks
        self._register_callbacks()

        return self.app

    def _create_layout(self) -> html.Div:
        """Create the main dashboard layout using panel components"""
        return html.Div([
            # Header with info (like original)
            html.Div(
                [
                    html.Div(
                        [
                            html.H2(
                                "FxAI Trading Dashboard",
                                style={
                                    "margin": "0",
                                    "color": DARK_THEME["text_primary"],
                                    "fontSize": "18px",
                                },
                            ),
                            html.Div(
                                id="header-info",
                                style={
                                    "color": DARK_THEME["text_secondary"],
                                    "fontSize": "12px",
                                },
                            ),
                        ],
                        style={"flex": "1"},
                    ),
                    html.Div(
                        [
                            html.Span(
                                "Session Time: ",
                                style={"color": DARK_THEME["text_secondary"], "fontSize": "12px"},
                            ),
                            html.Span(
                                id="session-time",
                                style={"color": DARK_THEME["accent_blue"], "fontSize": "12px", "marginRight": "15px"},
                            ),
                            html.Span(
                                id="performance-metrics",
                                style={"color": DARK_THEME["text_secondary"], "fontSize": "11px"},
                            ),
                        ],
                        style={"display": "flex", "alignItems": "center"}
                    ),
                ],
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "center",
                    "padding": "8px 12px",
                    "backgroundColor": DARK_THEME["bg_secondary"],
                    "borderBottom": f"1px solid {DARK_THEME['border']}",
                    "marginBottom": "6px",
                },
            ),
            
            # Main grid layout using CSS Grid instead of Bootstrap
            html.Div([
                # Row 1: Core Trading Information (4 columns) - x x x x
                html.Div([
                    html.Div([self.market_info_panel.create_layout()], style={"width": "25%", "padding": "5px"}),
                    html.Div([self.portfolio_panel.create_layout()], style={"width": "25%", "padding": "5px"}),
                    html.Div([self.performance_panel.create_layout()], style={"width": "25%", "padding": "5px"}),
                    html.Div([self.actions_panel.create_layout()], style={"width": "25%", "padding": "5px"}),
                ], style={"display": "flex", "marginBottom": "10px"}),
                
                # Row 2: Training Information with Training Manager spanning right (x x x l)
                html.Div([
                    html.Div([self.episode_training_panel.create_layout()], style={"width": "25%", "padding": "5px"}),
                    html.Div([self.ppo_metrics_panel.create_layout()], style={"width": "25%", "padding": "5px"}),
                    html.Div([self.captum_panel.create_layout()], style={"width": "25%", "padding": "5px"}),
                    html.Div([self.training_manager_panel.create_layout()], style={"width": "25%", "padding": "5px"}),
                ], style={"display": "flex", "marginBottom": "10px"}),
                
                # Row 3: Data Tables (2 columns) - t t
                html.Div([
                    html.Div([self.trades_table_panel.create_layout()], style={"width": "50%", "padding": "5px"}),
                    html.Div([self.reward_components_panel.create_layout()], style={"width": "50%", "padding": "5px"}),
                ], style={"display": "flex", "marginBottom": "10px"}),
                
                # Row 4: Chart (Full width) - l (continuing from above)
                html.Div([
                    html.Div([self.price_chart_panel.create_layout()], style={"width": "100%", "padding": "5px"}),
                ], style={"display": "flex"}),
            ]),
            
            # Auto-refresh component
            dcc.Interval(
                id='dashboard-interval-v3',
                interval=1000,  # Update every second
                n_intervals=0
            ),
        ], style={
            "backgroundColor": DARK_THEME["bg_primary"],
            "minHeight": "100vh",
            "padding": "10px",
            "fontFamily": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
        })

    def _register_callbacks(self):
        """Register all dashboard callbacks"""
        
        @self.app.callback(
            [
                Output('header-info', 'children'),
                Output('session-time', 'children'),
                Output('performance-metrics', 'children'),
                Output('market-content', 'children'),
                Output('portfolio-content', 'children'),
                Output('performance-content', 'children'),
                Output('actions-content', 'children'),
                Output('episode-training-content', 'children'),
                Output('ppo-content', 'children'),
                Output('captum-content', 'children'),
                Output('training-manager-content', 'children'),
                Output('trades-content', 'children'),
                Output('reward-components-content', 'children'),
                Output('chart-content', 'children'),
            ],
            [Input('dashboard-interval-v3', 'n_intervals')],
            prevent_initial_call=False
        )
        def update_dashboard(n):
            """Update all dashboard panels"""
            try:
                # Get current state
                state = dashboard_state.get_state()
                
                # Calculate session time (from original)
                current_time = datetime.now()
                start_time = state.session_start_time
                
                # Ensure both are datetime objects, not pandas Timestamps
                if hasattr(current_time, 'to_pydatetime'):
                    current_time = current_time.to_pydatetime()
                if hasattr(start_time, 'to_pydatetime'):
                    start_time = start_time.to_pydatetime()
                
                session_duration = current_time - start_time
                hours, remainder = divmod(int(session_duration.total_seconds()), 3600)
                minutes, seconds = divmod(remainder, 60)
                session_time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

                # Header info with training manager and data lifecycle info (from original)
                lifecycle_info = ""
                if hasattr(state, 'data_lifecycle_stage') and state.data_lifecycle_stage and state.data_lifecycle_stage != "unknown":
                    lifecycle_info = f" | Stage: {state.data_lifecycle_stage}"
                
                momentum_day_info = ""
                if state.current_momentum_day_date:
                    momentum_day_info = f" | Day: {state.current_momentum_day_date} (Q: {state.current_momentum_day_quality:.2f})"

                training_mode_info = ""
                if hasattr(state, 'training_mode') and state.training_mode:
                    training_mode_info = f" | Mode: {state.training_mode}"

                header_info = (
                    f"Model: {state.model_name} | Symbol: {state.symbol}{training_mode_info}{lifecycle_info}{momentum_day_info}"
                )
                
                # Performance metrics for header
                steps_per_second = getattr(state, "steps_per_second", 0.0)
                episodes_per_hour = getattr(state, "episodes_per_hour", 0.0)
                updates_per_hour = getattr(state, "updates_per_hour", 0.0)
                
                performance_metrics_str = f"Steps/sec: {steps_per_second:.1f} | Eps/hour: {episodes_per_hour:.0f} | Updates/hour: {updates_per_hour:.0f}"
                
                # Update all panels using their create_content methods
                return [
                    header_info,
                    session_time_str,
                    performance_metrics_str,
                    self.market_info_panel.create_content(state),
                    self.portfolio_panel.create_content(state),
                    self.performance_panel.create_content(state),
                    self.actions_panel.create_content(state),
                    self.episode_training_panel.create_content(state),
                    self.ppo_metrics_panel.create_content(state),
                    self.captum_panel.create_content(state),
                    self.training_manager_panel.create_content(state),
                    self.trades_table_panel.create_content(state),
                    self.reward_components_panel.create_content(state),
                    self.price_chart_panel.create_content(state),
                ]
                
            except Exception as e:
                self.logger.error(f"Error updating dashboard: {e}")
                error_msg = html.Div([
                    html.P(f"Dashboard update error: {str(e)}", 
                           style={"color": DARK_THEME["accent_red"], "textAlign": "center"})
                ])
                # Return error message for all panels (now 14 total outputs)
                return [error_msg] * 14

    def start_server(self, debug: bool = False, open_browser: bool = True) -> None:
        """Start the dashboard server"""
        try:
            self.logger.info(f"Starting dashboard server on port {self.port}")
            
            if open_browser:
                # Open browser in a separate thread to avoid blocking
                threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{self.port}")).start()
            
            # Start the server
            self.app.run(
                host="0.0.0.0",
                port=self.port,
                debug=debug,
                use_reloader=False,  # Prevent issues with threading
            )
            
        except Exception as e:
            self.logger.error(f"Failed to start dashboard server: {e}")
            raise

    def stop_server(self) -> None:
        """Stop the dashboard server"""
        # Dash doesn't have a built-in stop method, so we'll use a workaround
        self.logger.info("Dashboard server shutdown requested")
        # In a production environment, you might use a process manager
        # or implement a more sophisticated shutdown mechanism


def start_dashboard(port: int = 8053, open_browser: bool = True) -> None:
    """Start the dashboard server
    
    Args:
        port: Port to run the server on
        open_browser: Whether to open browser automatically
    """
    server = DashboardServer(port=port)
    server.create_app()
    
    # Open browser if requested
    if open_browser:
        def open_browser_delayed():
            import time
            time.sleep(1.5)  # Give server time to start
            webbrowser.open(f"http://localhost:{port}")
        
        browser_thread = threading.Thread(target=open_browser_delayed, daemon=True)
        browser_thread.start()
    
    # Run server (blocking)
    server.start_server(debug=False, open_browser=False)  # We handle browser opening above