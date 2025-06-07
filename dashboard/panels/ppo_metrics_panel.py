"""PPO Metrics Panel"""

from dash import html, dcc
import plotly.graph_objs as go
from typing import Optional

class PPOMetricsPanel:
    def __init__(self, dark_theme: dict):
        self.DARK_THEME = dark_theme
    
    def create_layout(self) -> html.Div:
        return html.Div(
            [
                html.H4(
                    "PPO Metrics",
                    style={
                        "color": self.DARK_THEME["text_primary"],
                        "marginBottom": "4px",
                        "fontSize": "12px",
                        "fontWeight": "bold",
                    },
                ),
                html.Div(id="ppo-content"),
            ],
            style=self._card_style(),
        )
    
    def create_content(self, state) -> html.Div:
        # PPO metrics
        total_loss = getattr(state, "total_loss", 0.0)
        policy_loss = getattr(state, "policy_loss", 0.0)
        value_loss = getattr(state, "value_loss", 0.0)
        entropy = getattr(state, "entropy", 0.0)
        kl_divergence = getattr(state, "kl_divergence", 0.0)
        clip_fraction = getattr(state, "clip_fraction", 0.0)
        explained_variance = getattr(state, "explained_variance", 0.0)
        learning_rate = getattr(state, "learning_rate", 0.0)
        mean_episode_reward = getattr(state, "mean_episode_reward", 0.0)
        
        # Get sparkline data for all metrics
        total_loss_history = getattr(state, "total_loss_history", [])
        policy_loss_history = getattr(state, "policy_loss_history", [])
        value_loss_history = getattr(state, "value_loss_history", [])
        entropy_history = getattr(state, "entropy_history", [])
        kl_divergence_history = getattr(state, "kl_divergence_history", [])
        clip_fraction_history = getattr(state, "clip_fraction_history", [])
        explained_variance_history = getattr(state, "explained_variance_history", [])
        learning_rate_history = getattr(state, "learning_rate_history", [])
        mean_episode_reward_history = getattr(state, "mean_episode_reward_history", [])
        
        return html.Div([
            self._metric_with_sparkline("Total Loss", f"{total_loss:.4f}", total_loss_history, "total_loss"),
            self._metric_with_sparkline("Policy Loss", f"{policy_loss:.4f}", policy_loss_history, "policy_loss"),
            self._metric_with_sparkline("Value Loss", f"{value_loss:.4f}", value_loss_history, "value_loss"),
            self._metric_with_sparkline("Entropy", f"{entropy:.4f}", entropy_history, "entropy"),
            self._metric_with_sparkline("KL Divergence", f"{kl_divergence:.4f}", kl_divergence_history, "kl_divergence"),
            self._metric_with_sparkline("Clip Fraction", f"{clip_fraction:.2%}", clip_fraction_history, "clip_fraction"),
            self._metric_with_sparkline("Explained Var", f"{explained_variance:.4f}", explained_variance_history, "explained_variance"),
            self._metric_with_sparkline("Learning Rate", f"{learning_rate:.2e}", learning_rate_history, "learning_rate"),
            self._metric_with_sparkline("Mean Reward", f"{mean_episode_reward:.4f}", mean_episode_reward_history, "mean_reward"),
        ])
    
    def _metric_with_sparkline(self, label: str, value: str, history: list, metric_id: str) -> html.Div:
        """Create a metric row with sparkline chart"""
        # Parse value for guidance calculation
        try:
            if '%' in value:
                # For percentage values, convert to decimal (e.g., "25%" -> 0.25)
                numeric_value = float(value.replace('%', '')) / 100.0
            else:
                numeric_value = float(value)
        except (ValueError, TypeError):
            numeric_value = 0.0
        
        guidance = self._get_ppo_metric_guidance(label.lower().replace(" ", "_"), numeric_value)
        
        # Create sparkline if history exists and is a proper sequence
        if history and len(history) > 1:
            try:
                # Convert to list first to handle deque objects safely, then slice
                history_list = list(history)
                history_slice = history_list[-20:] if len(history_list) > 0 else []
                
                if history_slice and len(history_slice) > 0:
                    sparkline = dcc.Graph(
                        figure={
                            'data': [go.Scatter(
                                y=history_slice,  # Last 20 points
                                mode='lines',
                                line=dict(color=guidance['color'], width=1),
                                showlegend=False,
                                hovertemplate='%{y:.4f}<extra></extra>'
                            )],
                            'layout': go.Layout(
                                height=30,
                                margin=dict(l=0, r=0, t=0, b=0),
                                xaxis=dict(visible=False),
                                yaxis=dict(visible=False),
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                            )
                        },
                        config={'displayModeBar': False},
                        style={'height': '30px', 'width': '60px'}
                    )
                else:
                    sparkline = html.Div(style={'height': '30px', 'width': '60px'})
            except Exception:
                # If any error occurs with history processing, create empty sparkline
                sparkline = html.Div(style={'height': '30px', 'width': '60px'})
        else:
            sparkline = html.Div(style={'height': '30px', 'width': '60px'})
        
        return html.Div([
            html.Div([
                html.Span(guidance['emoji'], title=guidance['tooltip'], style={"fontSize": "12px", "marginRight": "6px"}),
                html.Span(label, style={"color": self.DARK_THEME["text_secondary"], "fontSize": "11px"}),
            ], style={"display": "flex", "alignItems": "center"}),
            html.Div([
                html.Span(value, style={"color": guidance['color'], "fontWeight": "bold", "fontSize": "11px", "marginRight": "8px"}),
                sparkline,
            ], style={"display": "flex", "alignItems": "center", "justifyContent": "flex-end"}),
        ], style={
            "display": "flex",
            "justifyContent": "space-between",
            "alignItems": "center",
            "marginBottom": "3px",
            "minHeight": "30px",
        })
    
    def _get_ppo_metric_guidance(self, metric_name: str, value: float) -> dict:
        """Get health status and guidance for PPO metrics"""
        guidance_map = {
            "policy_loss": {
                "good_range": (-0.1, 0.1),
                "warning_range": (-0.3, 0.3),
                "good_emoji": "✅",
                "warning_emoji": "⚠️",
                "bad_emoji": "❌",
                "tooltip": "Policy loss should be small and stable"
            },
            "value_loss": {
                "good_range": (0, 0.5),
                "warning_range": (0, 1.0),
                "good_emoji": "✅",
                "warning_emoji": "⚠️", 
                "bad_emoji": "❌",
                "tooltip": "Value loss should decrease over time"
            },
            "entropy": {
                "good_range": (0.1, 0.5),
                "warning_range": (0.05, 0.7),
                "good_emoji": "✅",
                "warning_emoji": "⚠️",
                "bad_emoji": "❌",
                "tooltip": "Entropy should be moderate for exploration"
            },
            "kl_divergence": {
                "good_range": (0, 0.01),
                "warning_range": (0, 0.05),
                "good_emoji": "✅",
                "warning_emoji": "⚠️",
                "bad_emoji": "❌",
                "tooltip": "KL divergence should be small (< 0.01)"
            },
            "clip_fraction": {
                "good_range": (0.1, 0.3),
                "warning_range": (0.05, 0.5),
                "good_emoji": "✅",
                "warning_emoji": "⚠️",
                "bad_emoji": "❌",
                "tooltip": "Clip fraction should be 10-30% for good updates"
            },
            "explained_variance": {
                "good_range": (0.7, 1.0),
                "warning_range": (0.5, 1.0),
                "good_emoji": "✅",
                "warning_emoji": "⚠️",
                "bad_emoji": "❌",
                "tooltip": "Explained variance should be high (> 0.7)"
            },
            "explained_var": {
                "good_range": (0.7, 1.0),
                "warning_range": (0.5, 1.0),
                "good_emoji": "✅",
                "warning_emoji": "⚠️",
                "bad_emoji": "❌",
                "tooltip": "Explained variance should be high (> 0.7)"
            },
            "learning_rate": {
                "good_range": (1e-5, 1e-3),
                "warning_range": (1e-6, 1e-2),
                "good_emoji": "📈",
                "warning_emoji": "📊",
                "bad_emoji": "📉",
                "tooltip": "Learning rate should be in range 1e-5 to 1e-3"
            },
            "total_loss": {
                "good_range": (0, 1.0),
                "warning_range": (0, 2.0),
                "good_emoji": "✅",
                "warning_emoji": "⚠️",
                "bad_emoji": "❌",
                "tooltip": "Total loss should decrease over time"
            },
            "mean_reward": {
                "good_range": (0.01, float('inf')),
                "warning_range": (-0.1, float('inf')),
                "good_emoji": "🎯",
                "warning_emoji": "📊",
                "bad_emoji": "📉",
                "tooltip": "Mean episode reward - higher is better"
            }
        }
        
        guidance = guidance_map.get(metric_name, {
            "good_range": (0, 1),
            "warning_range": (0, 2),
            "good_emoji": "ℹ️",
            "warning_emoji": "ℹ️",
            "bad_emoji": "ℹ️",
            "tooltip": "No guidance available"
        })
        
        good_min, good_max = guidance["good_range"]
        warn_min, warn_max = guidance["warning_range"]
        
        if good_min <= value <= good_max:
            return {
                "color": self.DARK_THEME["accent_green"],
                "emoji": guidance["good_emoji"],
                "tooltip": guidance["tooltip"]
            }
        elif warn_min <= value <= warn_max:
            return {
                "color": self.DARK_THEME["accent_orange"],
                "emoji": guidance["warning_emoji"],
                "tooltip": guidance["tooltip"]
            }
        else:
            return {
                "color": self.DARK_THEME["accent_red"],
                "emoji": guidance["bad_emoji"],
                "tooltip": guidance["tooltip"]
            }
    
    def _info_row(self, label: str, value: str, color: Optional[str] = None) -> html.Div:
        value_color = color or self.DARK_THEME["text_primary"]
        return html.Div(
            [
                html.Span(label, style={"color": self.DARK_THEME["text_secondary"], "fontSize": "12px"}),
                html.Span(value, style={"color": value_color, "fontWeight": "bold", "fontSize": "12px"}),
            ],
            style={
                "display": "flex",
                "justifyContent": "space-between", 
                "alignItems": "center",
                "marginBottom": "2px",
                "minHeight": "16px",
            },
        )
    
    def _card_style(self) -> dict:
        return {
            "backgroundColor": self.DARK_THEME["bg_secondary"],
            "border": f"1px solid {self.DARK_THEME['border']}",
            "borderRadius": "6px",
            "padding": "8px",
            "height": "100%",
            "overflow": "auto",
        }