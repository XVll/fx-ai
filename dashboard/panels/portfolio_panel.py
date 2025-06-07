"""Portfolio & Position Panel"""

from dash import html
from typing import Optional

class PortfolioPanel:
    def __init__(self, dark_theme: dict):
        self.DARK_THEME = dark_theme
    
    def create_layout(self) -> html.Div:
        return html.Div(
            [
                html.H4(
                    "Portfolio & Position",
                    style={
                        "color": self.DARK_THEME["text_primary"],
                        "marginBottom": "4px",
                        "fontSize": "12px",
                        "fontWeight": "bold",
                    },
                ),
                html.Div(id="portfolio-content"),
            ],
            style=self._card_style(),
        )
    
    def create_content(self, state) -> html.Div:
        # Combined Portfolio & Position info - matching original structure
        
        # Position section
        if state.position_side != "FLAT":
            position_pnl = getattr(state, "position_pnl_dollar", 0.0)
            pnl_color = (
                self.DARK_THEME["accent_green"]
                if position_pnl >= 0
                else self.DARK_THEME["accent_red"]
            )
            position_section = [
                self._info_row("Side", state.position_side, color=self.DARK_THEME["accent_blue"]),
                self._info_row("Qty", f"{state.position_qty:,}"),
                self._info_row("Entry", f"${state.avg_entry_price:.3f}"),
                self._info_row("P&L $", f"${position_pnl:.2f}", color=pnl_color),
                self._info_row("P&L %", f"{state.position_pnl_percent:.1f}%", color=pnl_color),
                self._info_row(
                    "Hold Time",
                    f"{getattr(state, 'position_hold_time_seconds', 0) // 60:.0f}m"
                    if getattr(state, "position_hold_time_seconds", 0) > 0
                    else "0m",
                ),
            ]
        else:
            position_section = [
                self._info_row("Side", "FLAT", color=self.DARK_THEME["text_muted"]),
                self._info_row("Qty", "0"),
                self._info_row("Entry", "-"),
                self._info_row("P&L $", "$0.00"),
                self._info_row("P&L %", "0.0%"),
                self._info_row("Hold Time", "0m"),
            ]

        # Portfolio section
        session_pnl_color = (
            self.DARK_THEME["accent_green"]
            if state.session_pnl >= 0
            else self.DARK_THEME["accent_red"]
        )
        portfolio_section = [
            html.Hr(style={"margin": "4px 0", "borderColor": self.DARK_THEME["border"]}),
            self._info_row("Equity", f"${state.total_equity:.0f}"),
            self._info_row("Cash", f"${state.cash_balance:.0f}"),
            self._info_row("Session P&L", f"${state.session_pnl:.2f}", color=session_pnl_color),
            self._info_row("Realized", f"${state.realized_pnl:.2f}"),
        ]

        return html.Div(position_section + portfolio_section)
    
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