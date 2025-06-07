"""Episode Panel"""

from dash import html
from typing import Optional

class EpisodePanel:
    def __init__(self, dark_theme: dict):
        self.DARK_THEME = dark_theme
    
    def create_layout(self) -> html.Div:
        return html.Div(
            [
                html.H4(
                    "Episode",
                    style={
                        "color": self.DARK_THEME["text_primary"],
                        "marginBottom": "4px",
                        "fontSize": "12px",
                        "fontWeight": "bold",
                    },
                ),
                html.Div(id="episode-content"),
            ],
            style=self._card_style(),
        )
    
    def create_content(self, state) -> html.Div:
        # Episode info
        effective_max_steps = getattr(state, "max_steps", 0)
        if effective_max_steps <= 0:
            effective_max_steps = getattr(state, "curriculum_episode_length", 256)
        
        current_step = getattr(state, "current_step", 0)
        episode_number = getattr(state, "episode_number", 0)
        
        progress = (current_step / effective_max_steps * 100) if effective_max_steps > 0 else 0
        
        # Display values
        episode_display = episode_number if episode_number > 0 else "-"
        step_display = (
            f"{current_step:,}/{effective_max_steps:,}" if effective_max_steps > 0 
            else f"{current_step:,}/∞"
        )
        progress_display = f"{progress:.1f}%" if effective_max_steps > 0 else "∞"
        
        # Combine episode number with progress percentage
        episode_with_progress = f"Episode {episode_display} ({progress_display})"
        
        # Calculate current episode reward (sum of episode components)
        episode_reward_total = sum(getattr(state, "episode_reward_components", {}).values())
        last_step_reward = getattr(state, "last_step_reward", 0.0)
        
        return html.Div([
            self._info_row("Episode", episode_with_progress),
            self._info_row("Step", step_display),
            self._info_row("Episode Reward", f"{episode_reward_total:.3f}"),
            self._info_row("Step Reward", f"{last_step_reward:.3f}"),
            # Progress bar
            html.Div([
                html.Div(
                    style={
                        "backgroundColor": self.DARK_THEME["bg_tertiary"],
                        "height": "8px",
                        "borderRadius": "4px",
                        "marginTop": "4px",
                        "marginBottom": "2px",
                    },
                    children=[
                        html.Div(
                            style={
                                "backgroundColor": self.DARK_THEME["accent_blue"],
                                "height": "100%",
                                "width": f"{progress}%",
                                "borderRadius": "4px",
                                "transition": "width 0.3s ease",
                            }
                        )
                    ],
                ),
                html.Div(
                    step_display,
                    style={
                        "color": self.DARK_THEME["text_primary"],
                        "fontSize": "11px",
                        "textAlign": "center",
                        "marginTop": "2px",
                    },
                ),
            ]),
        ])
    
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