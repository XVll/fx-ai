"""Actions Panel"""

from dash import html, dash_table
from typing import Optional

class ActionsPanel:
    def __init__(self, dark_theme: dict):
        self.DARK_THEME = dark_theme
    
    def create_layout(self) -> html.Div:
        return html.Div(
            [
                html.H4(
                    "Actions",
                    style={
                        "color": self.DARK_THEME["text_primary"],
                        "marginBottom": "4px",
                        "fontSize": "12px",
                        "fontWeight": "bold",
                    },
                ),
                html.Div(id="actions-content"),
            ],
            style=self._card_style(),
        )
    
    def create_content(self, state) -> html.Div:
        # Get action data from state - using correct variable names
        episode_actions = getattr(state, "episode_action_distribution", {})
        session_actions = getattr(state, "session_action_distribution", {})
        event_stream_actions = getattr(state, "action_distribution", {})
        
        # Initialize with default structure if empty
        default_actions = {"HOLD": 0, "BUY": 0, "SELL": 0}
        if not episode_actions:
            episode_actions = default_actions.copy()
        if not session_actions:
            session_actions = default_actions.copy()
        if not event_stream_actions:
            event_stream_actions = default_actions.copy()
        
        # Use event stream as session actions if session actions are empty
        if all(v == 0 for v in session_actions.values()) and sum(event_stream_actions.values()) > 0:
            session_actions = event_stream_actions.copy()

        # If episode actions are empty but event stream has data, show partial episode progress
        if (all(v == 0 for v in episode_actions.values()) and 
            sum(event_stream_actions.values()) > 0 and 
            getattr(state, "current_step", 0) > 0):
            episode_actions = event_stream_actions.copy()
        
        # Also check for current action being taken
        current_action = getattr(state, "last_action", None)
        if current_action and current_action in ["HOLD", "BUY", "SELL"]:
            # Show current action indicator
            pass  # We'll handle this in the display logic

        # Get current/last action for indicator
        current_action = getattr(state, "last_action", None)
        if not current_action:
            current_action = getattr(state, "current_action", None)
            
        action_data = []
        for action_type in ["HOLD", "BUY", "SELL"]:
            episode_count = episode_actions.get(action_type, 0)
            session_count = session_actions.get(action_type, 0)

            # Calculate percentages
            episode_total = sum(episode_actions.values()) or 1
            session_total = sum(session_actions.values()) or 1
            episode_pct = episode_count / episode_total * 100
            session_pct = session_count / session_total * 100

            # Add current action indicator
            action_display = action_type
            if current_action == action_type:
                action_display = f"▶ {action_type}"  # Indicator for current action

            action_data.append({
                "Action": action_display,
                "Episode": f"{episode_count}",
                "Episode %": f"{episode_pct:.1f}%",
                "Session": f"{session_count}",
                "Session %": f"{session_pct:.1f}%",
            })

        actions_table = dash_table.DataTable(
            data=action_data,
            columns=[
                {"name": "Action", "id": "Action"},
                {"name": "Episode", "id": "Episode", "type": "numeric"},
                {"name": "Ep %", "id": "Episode %"},
                {"name": "Session", "id": "Session", "type": "numeric"},
                {"name": "Sess %", "id": "Session %"},
            ],
            style_cell={
                "backgroundColor": self.DARK_THEME["bg_tertiary"],
                "color": self.DARK_THEME["text_primary"],
                "border": f"1px solid {self.DARK_THEME['border']}",
                "fontSize": "11px",
                "padding": "4px 6px",
                "textAlign": "left",
            },
            style_data_conditional=[
                {
                    "if": {
                        "column_id": "Action",
                        "filter_query": "{Action} = HOLD",
                    },
                    "color": self.DARK_THEME["accent_blue"],
                },
                {
                    "if": {"column_id": "Action", "filter_query": "{Action} = BUY"},
                    "color": self.DARK_THEME["accent_green"],
                },
                {
                    "if": {
                        "column_id": "Action",
                        "filter_query": "{Action} = SELL",
                    },
                    "color": self.DARK_THEME["accent_red"],
                },
            ],
            style_header={
                "backgroundColor": self.DARK_THEME["bg_secondary"],
                "color": self.DARK_THEME["text_secondary"],
                "fontWeight": "bold",
                "fontSize": "10px",
            },
            page_size=10,
        )

        return html.Div([actions_table])
    
    def _card_style(self) -> dict:
        return {
            "backgroundColor": self.DARK_THEME["bg_secondary"],
            "border": f"1px solid {self.DARK_THEME['border']}",
            "borderRadius": "6px",
            "padding": "8px",
            "height": "100%",
            "overflow": "auto",
        }