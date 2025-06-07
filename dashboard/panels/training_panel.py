"""Training Panel"""

from dash import html
from typing import Optional

class TrainingPanel:
    def __init__(self, dark_theme: dict):
        self.DARK_THEME = dark_theme
    
    def create_layout(self) -> html.Div:
        return html.Div(
            [
                html.H4(
                    "Training",
                    style={
                        "color": self.DARK_THEME["text_primary"],
                        "marginBottom": "4px",
                        "fontSize": "12px",
                        "fontWeight": "bold",
                    },
                ),
                html.Div(id="training-content"),
            ],
            style=self._card_style(),
        )
    
    def create_content(self, state) -> html.Div:
        # Training progress
        training_max_episodes = getattr(state, "training_max_episodes", float('inf'))
        training_max_updates = getattr(state, "training_max_updates", float('inf'))
        
        total_episodes = getattr(state, "total_episodes", 0)
        updates = getattr(state, "updates", 0)
        global_steps = getattr(state, "global_steps", 0)
        
        # Format displays
        if training_max_episodes != float('inf'):
            episode_display = f"{total_episodes:,}/{training_max_episodes:,}"
        else:
            episode_display = f"{total_episodes:,}"
        
        if training_max_updates != float('inf'):
            update_display = f"{updates:,}/{training_max_updates:,}"
        else:
            update_display = f"{updates:,}"

        # Performance metrics
        steps_per_second = getattr(state, "steps_per_second", 0.0)
        episodes_per_hour = getattr(state, "episodes_per_hour", 0.0)
        
        # Calculate updates per hour
        updates_per_hour = getattr(state, "updates_per_hour", 0.0)
        if updates_per_hour == 0.0:
            updates_per_hour = getattr(state, "updates_per_second", 0.0) * 3600

        # Training state
        mode = getattr(state, "mode", "Unknown")
        stage = getattr(state, "stage", "Unknown")

        training_children = [
            self._info_row("Mode", mode, color=self.DARK_THEME["accent_purple"]),
            self._info_row("Stage", stage),
            self._info_row("Episodes", episode_display),
            self._info_row("Updates", update_display),
            self._info_row("Global Steps", f"{global_steps:,}"),
            self._info_row("Steps/sec", f"{steps_per_second:.1f}"),
            self._info_row("Eps/hour", f"{episodes_per_hour:.0f}"),
            self._info_row("Updates/hour", f"{updates_per_hour:.0f}"),
        ]

        # Add training completion progress bar (if training_max_updates is available)
        if training_max_updates != float('inf'):
            progress_section = self._create_progress_section(
                "Training Progress",
                updates,
                training_max_updates,
                self.DARK_THEME["accent_green"]
            )
            training_children.append(progress_section)

        return html.Div(training_children)
    
    def _create_progress_section(self, title: str, current: int, maximum: int, color: str) -> html.Div:
        """Create a progress section with bar and text"""
        progress = (current / maximum * 100) if maximum > 0 else 0
        
        return html.Div([
            html.Hr(style={"margin": "4px 0", "borderColor": self.DARK_THEME["border"]}),
            html.Div(
                title,
                style={
                    "color": self.DARK_THEME["text_secondary"],
                    "fontSize": "11px",
                    "marginBottom": "2px",
                },
            ),
            html.Div([
                html.Div(
                    style={
                        "backgroundColor": self.DARK_THEME["bg_tertiary"],
                        "height": "8px",
                        "borderRadius": "4px",
                        "marginBottom": "2px",
                    },
                    children=[
                        html.Div(
                            style={
                                "backgroundColor": color,
                                "height": "100%",
                                "width": f"{progress}%",
                                "borderRadius": "4px",
                                "transition": "width 0.3s ease",
                            }
                        )
                    ],
                ),
                html.Div(
                    f"{current:,}/{maximum:,} ({progress:.1f}%)",
                    style={
                        "color": self.DARK_THEME["text_primary"],
                        "fontSize": "10px",
                        "textAlign": "center",
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