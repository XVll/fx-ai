"""Reward Components Panel"""

from dash import html, dash_table
import pandas as pd

class RewardComponentsPanel:
    def __init__(self, dark_theme: dict):
        self.DARK_THEME = dark_theme
    
    def create_layout(self) -> html.Div:
        return html.Div(
            [
                html.H4(
                    "Reward Components",
                    style={
                        "color": self.DARK_THEME["text_primary"],
                        "marginBottom": "4px",
                        "fontSize": "12px",
                        "fontWeight": "bold",
                    },
                ),
                html.Div(id="reward-components-content"),
            ],
            style=self._card_style(),
        )
    
    def create_content(self, state) -> html.Div:
        # Get reward components data
        episode_components = getattr(state, "episode_reward_components", {})
        session_components = getattr(state, "session_reward_components", {})
        episode_component_counts = getattr(state, "episode_reward_component_counts", {})
        session_component_counts = getattr(state, "session_reward_component_counts", {})
        
        # Define standard reward components (always show these)
        standard_components = [
            "pnl_reward",
            "position_penalty", 
            "action_efficiency",
            "risk_penalty",
            "holding_penalty",
            "momentum_reward",
            "volume_penalty",
            "spread_penalty",
            "drawdown_penalty",
            "volatility_penalty"
        ]
        
        # Get all components (standard + any extras from data)
        all_components = set(standard_components)
        all_components.update(episode_components.keys())
        all_components.update(session_components.keys())
        
        reward_data = []
        
        for component in sorted(all_components):
            episode_value = episode_components.get(component, 0.0)
            session_value = session_components.get(component, 0.0)
            episode_count = episode_component_counts.get(component, 0)
            session_count = session_component_counts.get(component, 0)
            
            # Calculate session percentage
            total_session_count = sum(session_component_counts.values()) if session_component_counts else 1
            session_pct = (session_count / total_session_count * 100) if total_session_count > 0 else 0
            
            reward_data.append({
                "Component": component.replace("_", " ").title(),
                "Episode": f"{episode_value:.3f}",
                "Session": f"{session_value:.3f}",
                "Session %": f"{session_pct:.1f}%",
                "Count": f"{session_count}",
            })
        
        # Add totals
        episode_total = sum(episode_components.values()) if episode_components else 0.0
        session_total = sum(session_components.values()) if session_components else 0.0
        total_count = sum(session_component_counts.values()) if session_component_counts else 0
        
        reward_data.append({
            "Component": "TOTAL",
            "Episode": f"{episode_total:.3f}",
            "Session": f"{session_total:.3f}",
            "Session %": "100.0%",
            "Count": f"{total_count}",
        })
        
        reward_table = dash_table.DataTable(
            data=reward_data,
            columns=[
                {"name": "Component", "id": "Component"},
                {"name": "Episode", "id": "Episode", "type": "numeric"},
                {"name": "Session", "id": "Session", "type": "numeric"},
                {"name": "Session %", "id": "Session %"},
                {"name": "Count", "id": "Count", "type": "numeric"},
            ],
            style_cell={
                "backgroundColor": self.DARK_THEME["bg_tertiary"],
                "color": self.DARK_THEME["text_primary"],
                "border": f"1px solid {self.DARK_THEME['border']}",
                "fontSize": "10px",
                "padding": "4px 6px",
                "textAlign": "left",
                "whiteSpace": "nowrap",
                "overflow": "hidden",
                "textOverflow": "ellipsis",
            },
            style_data_conditional=[
                # Positive values in green
                {
                    "if": {
                        "column_id": "Episode",
                        "filter_query": "{Episode} > 0"
                    },
                    "color": self.DARK_THEME["accent_green"],
                },
                {
                    "if": {
                        "column_id": "Session", 
                        "filter_query": "{Session} > 0"
                    },
                    "color": self.DARK_THEME["accent_green"],
                },
                # Negative values in red
                {
                    "if": {
                        "column_id": "Episode",
                        "filter_query": "{Episode} < 0"
                    },
                    "color": self.DARK_THEME["accent_red"],
                },
                {
                    "if": {
                        "column_id": "Session",
                        "filter_query": "{Session} < 0"
                    },
                    "color": self.DARK_THEME["accent_red"],
                },
                # Total row styling
                {
                    "if": {
                        "column_id": "Component",
                        "filter_query": "{Component} = TOTAL"
                    },
                    "backgroundColor": self.DARK_THEME["bg_secondary"],
                    "fontWeight": "bold",
                },
            ],
            style_header={
                "backgroundColor": self.DARK_THEME["bg_secondary"],
                "color": self.DARK_THEME["text_secondary"],
                "fontWeight": "bold",
                "fontSize": "9px",
            },
            sort_action="native",
        )
        
        return html.Div([reward_table])
    
    def _card_style(self) -> dict:
        return {
            "backgroundColor": self.DARK_THEME["bg_secondary"],
            "border": f"1px solid {self.DARK_THEME['border']}",
            "borderRadius": "6px",
            "padding": "8px",
            "height": "100%",
            "overflow": "auto",
        }