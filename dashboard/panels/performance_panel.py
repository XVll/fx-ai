"""Performance Metrics Panel"""

from dash import html, dash_table
from typing import Optional

class PerformancePanel:
    def __init__(self, dark_theme: dict):
        self.DARK_THEME = dark_theme
    
    def create_layout(self) -> html.Div:
        return html.Div(
            [
                html.H4(
                    "Performance Metrics",
                    style={
                        "color": self.DARK_THEME["text_primary"],
                        "marginBottom": "4px",
                        "fontSize": "12px",
                        "fontWeight": "bold",
                    },
                ),
                html.Div(id="performance-content"),
            ],
            style=self._card_style(),
        )
    
    def create_content(self, state) -> html.Div:
        # Performance Metrics info - Table format with Episode/Session columns (from original)
        # Calculate episode metrics
        episode_wins = getattr(state, "episode_winning_trades", 0)
        episode_losses = getattr(state, "episode_losing_trades", 0)
        episode_total = episode_wins + episode_losses

        # Calculate session metrics
        session_wins = getattr(state, "session_winning_trades", 0)
        session_losses = getattr(state, "session_losing_trades", 0)
        session_total = session_wins + session_losses

        # Calculate win/loss ratios
        episode_win_loss_ratio = (
            episode_wins / episode_losses
            if episode_losses > 0
            else float("inf")
            if episode_wins > 0
            else 0
        )
        session_win_loss_ratio = (
            session_wins / session_losses
            if session_losses > 0
            else float("inf")
            if session_wins > 0
            else 0
        )

        # Format win/loss ratios
        episode_wl_display = (
            f"{episode_win_loss_ratio:.2f}"
            if episode_win_loss_ratio != float("inf")
            else "∞"
            if episode_wins > 0
            else "0.00"
        )
        session_wl_display = (
            f"{session_win_loss_ratio:.2f}"
            if session_win_loss_ratio != float("inf")
            else "∞"
            if session_wins > 0
            else "0.00"
        )

        # Calculate episode and session win rates
        episode_win_rate = (
            (episode_wins / episode_total * 100) if episode_total > 0 else 0
        )
        session_win_rate = (
            (session_wins / session_total * 100) if session_total > 0 else 0
        )

        # Format profit factor (handle infinity case)
        profit_factor = getattr(state, "profit_factor", 0.0)
        profit_factor_display = (
            f"{profit_factor:.2f}"
            if profit_factor != float("inf")
            else "∞"
            if profit_factor > 0
            else "0.00"
        )

        # Performance data for table
        performance_data = [
            {
                "Metric": "Win Rate %",
                "Episode": f"{episode_win_rate:.1f}%",
                "Session": f"{session_win_rate:.1f}%",
            },
            {
                "Metric": "W/L Ratio",
                "Episode": episode_wl_display,
                "Session": session_wl_display,
            },
            {
                "Metric": "Profit Factor",
                "Episode": "N/A",
                "Session": profit_factor_display,
            },
        ]

        return dash_table.DataTable(
            data=performance_data,
            columns=[
                {"name": "Metric", "id": "Metric"},
                {"name": "Episode", "id": "Episode"},
                {"name": "Session", "id": "Session"},
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
                # Highlight good win rates
                {
                    "if": {
                        "column_id": "Episode",
                        "filter_query": '{Metric} = "Win Rate %" && {Episode} > 60',
                    },
                    "color": self.DARK_THEME["accent_green"],
                },
                {
                    "if": {
                        "column_id": "Session",
                        "filter_query": '{Metric} = "Win Rate %" && {Session} > 60',
                    },
                    "color": self.DARK_THEME["accent_green"],
                },
            ],
            style_header={
                "backgroundColor": self.DARK_THEME["bg_secondary"],
                "color": self.DARK_THEME["text_secondary"],
                "fontWeight": "bold",
                "fontSize": "10px",
            },
        )
    
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