"""Merged Episode & Training Panel with dual progress bars"""

from dash import html
from typing import Optional

class EpisodeTrainingPanel:
    def __init__(self, dark_theme: dict):
        self.DARK_THEME = dark_theme
    
    def create_layout(self) -> html.Div:
        return html.Div(
            [
                html.H4(
                    "Episode & Training",
                    style={
                        "color": self.DARK_THEME["text_primary"],
                        "marginBottom": "4px",
                        "fontSize": "12px",
                        "fontWeight": "bold",
                    },
                ),
                html.Div(id="episode-training-content"),
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
        
        episode_progress = (current_step / effective_max_steps * 100) if effective_max_steps > 0 else 0
        
        # Training info
        training_max_updates = getattr(state, "training_max_updates", float('inf'))
        updates = getattr(state, "updates", 0)
        total_episodes = getattr(state, "total_episodes", 0)
        
        training_progress = (updates / training_max_updates * 100) if training_max_updates != float('inf') else 0
        
        # Episode reward calculation
        episode_reward_total = sum(getattr(state, "episode_reward_components", {}).values())
        last_step_reward = getattr(state, "last_step_reward", 0.0)
        
        # Display values
        episode_display = episode_number if episode_number > 0 else "-"
        step_display = (
            f"{current_step:,}/{effective_max_steps:,}" if effective_max_steps > 0 
            else f"{current_step:,}/∞"
        )
        
        if training_max_updates != float('inf'):
            update_display = f"{updates:,}/{training_max_updates:,}"
        else:
            update_display = f"{updates:,}"
        
        # Training stage info
        mode = getattr(state, "mode", "Training")
        stage = getattr(state, "stage", "Active")
        stage_status = getattr(state, "stage_status", "")
        
        content_sections = [
            # Episode section
            self._info_row("Episode", f"#{episode_display}"),
            self._info_row("Step", step_display),
            self._info_row("Episode Reward", f"{episode_reward_total:.3f}"),
            self._info_row("Step Reward", f"{last_step_reward:.3f}"),
            
            # Episode progress bar
            self._create_progress_section(
                "Episode Progress",
                current_step,
                effective_max_steps,
                self.DARK_THEME["accent_blue"],
                show_percentage=True
            ),
            
            html.Hr(style={"margin": "6px 0", "borderColor": self.DARK_THEME["border"]}),
            
            # Training section
            self._info_row("Mode", mode, color=self.DARK_THEME["accent_purple"]),
            self._info_row("Stage", stage),
            self._info_row("Total Episodes", f"{total_episodes:,}"),
            self._info_row("Updates", update_display),
        ]
        
        # Add stage progress if available
        stage_progress = self._create_stage_progress_section(state)
        if stage_progress:
            content_sections.append(stage_progress)
        
        # Training progress bar (only if training_max_updates is available)
        if training_max_updates != float('inf'):
            training_progress = self._create_progress_section(
                "Training Progress",
                updates,
                training_max_updates,
                self.DARK_THEME["accent_green"],
                show_percentage=True
            )
            content_sections.append(training_progress)
        
        return html.Div(content_sections)
    
    def _create_progress_section(self, title: str, current: int, maximum: int, color: str, show_percentage: bool = False) -> html.Div:
        """Create a progress section with bar and text"""
        if maximum <= 0:
            progress = 0
            progress_text = f"{current:,}/∞"
        else:
            progress = (current / maximum * 100)
            if show_percentage:
                progress_text = f"{current:,}/{maximum:,} ({progress:.1f}%)"
            else:
                progress_text = f"{current:,}/{maximum:,}"
        
        return html.Div([
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
                                "width": f"{min(progress, 100)}%",
                                "borderRadius": "4px",
                                "transition": "width 0.3s ease",
                            }
                        )
                    ],
                ),
                html.Div(
                    progress_text,
                    style={
                        "color": self.DARK_THEME["text_primary"],
                        "fontSize": "10px",
                        "textAlign": "center",
                    },
                ),
            ]),
        ], style={"marginTop": "4px"})
    
    def _create_stage_progress_section(self, state) -> Optional[html.Div]:
        """Create progress section for current training stage"""
        stage = getattr(state, "stage", "").lower()
        stage_status = getattr(state, "stage_status", "")
        
        # Handle rollout collection progress
        if "rollout" in stage or getattr(state, "is_collecting_rollout", False):
            rollout_steps = getattr(state, "rollout_steps", 0)
            rollout_total = getattr(state, "rollout_total", 0)
            if rollout_total > 0:
                progress_pct = min(100, (rollout_steps / rollout_total) * 100)
                text = f"{rollout_steps:,}/{rollout_total:,} steps collected"
                return self._create_progress_bar_with_text(
                    "Rollout Collection", progress_pct, text, self.DARK_THEME["accent_orange"]
                )
            elif stage_status:
                return self._create_progress_bar_with_text(
                    "Rollout Collection", 0, stage_status, self.DARK_THEME["accent_orange"]
                )
        
        # Handle PPO update progress
        elif "update" in stage or "ppo" in stage or getattr(state, "is_updating", False):
            current_epoch = getattr(state, "current_epoch", 0)
            total_epochs = getattr(state, "total_epochs", 0)
            current_batch = getattr(state, "current_batch", 0)
            total_batches = getattr(state, "total_batches", 0)
            
            if total_epochs > 0:
                epoch_progress = min(100, (current_epoch / total_epochs) * 100)
                if total_batches > 0:
                    # Calculate global batch number
                    global_batch = (
                        (current_epoch - 1) * total_batches + current_batch
                        if current_epoch > 0
                        else current_batch
                    )
                    global_total_batches = total_epochs * total_batches
                    text = f"Epoch {current_epoch}/{total_epochs}, Batch {global_batch}/{global_total_batches}"
                else:
                    text = f"Epoch {current_epoch}/{total_epochs}"
                
                return self._create_progress_bar_with_text(
                    "PPO Update", epoch_progress, text, self.DARK_THEME["accent_blue"]
                )
            elif stage_status:
                return self._create_progress_bar_with_text(
                    "PPO Update", 0, stage_status, self.DARK_THEME["accent_blue"]
                )
        
        # Handle evaluation progress
        elif "evaluation" in stage or getattr(state, "is_evaluating", False):
            eval_progress = 0
            if hasattr(state, "eval_episode_current") and hasattr(state, "eval_episode_total"):
                eval_current = getattr(state, "eval_episode_current", 0)
                eval_total = getattr(state, "eval_episode_total", 0)
                if eval_total > 0:
                    eval_progress = min(100, (eval_current / eval_total) * 100)
                    text = f"Episode {eval_current}/{eval_total}"
                else:
                    text = stage_status or "Running evaluation..."
            else:
                text = stage_status or "Running evaluation..."
            
            return self._create_progress_bar_with_text(
                "Evaluation", eval_progress, text, self.DARK_THEME["accent_green"]
            )
        
        # Handle other stages with status
        elif stage_status and stage != "active":
            return self._create_progress_bar_with_text(
                f"{stage.title()} Stage", 0, stage_status, self.DARK_THEME["accent_purple"]
            )
        
        return None
    
    def _create_progress_bar_with_text(self, title: str, progress_pct: float, text: str, color: str) -> html.Div:
        """Create a progress bar with title and descriptive text"""
        return html.Div([
            html.Div(
                title,
                style={
                    "color": self.DARK_THEME["text_secondary"],
                    "fontSize": "11px",
                    "marginBottom": "2px",
                    "marginTop": "4px",
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
                        html.Div(style={
                            "backgroundColor": color,
                            "height": "100%",
                            "width": f"{min(progress_pct, 100)}%",
                            "borderRadius": "4px",
                            "transition": "width 0.3s ease",
                        })
                    ],
                )
            ]),
            html.Div(
                text,
                style={
                    "color": self.DARK_THEME["text_primary"],
                    "fontSize": "10px",
                    "textAlign": "center",
                },
            ),
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