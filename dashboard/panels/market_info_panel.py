"""Market Info Panel"""

from dash import html
from typing import Optional

class MarketInfoPanel:
    def __init__(self, dark_theme: dict):
        self.DARK_THEME = dark_theme
    
    def create_layout(self) -> html.Div:
        return html.Div(
            [
                html.H4(
                    "Market Info",
                    style={
                        "color": self.DARK_THEME["text_primary"],
                        "marginBottom": "4px",
                        "fontSize": "12px",
                        "fontWeight": "bold",
                    },
                ),
                html.Div(id="market-content"),
            ],
            style=self._card_style(),
        )
    
    def create_content(self, state) -> html.Div:
        # Market data - using actual fields from original dashboard
        current_price = getattr(state, "current_price", 0.0)
        bid_price = getattr(state, "bid_price", 0.0)
        ask_price = getattr(state, "ask_price", 0.0)
        volume = getattr(state, "volume", 0)
        
        # Calculate spread
        spread = ask_price - bid_price
        
        # Market session and time
        ny_time = getattr(state, "ny_time", "00:00:00")
        trading_hours = getattr(state, "trading_hours", "MARKET")
        
        return html.Div([
            self._info_row("Time", ny_time),
            self._info_row("Session", trading_hours, color=self.DARK_THEME["accent_orange"]),
            self._info_row("Price", f"${current_price:.2f}", color=self.DARK_THEME["accent_blue"]),
            self._info_row("Bid", f"${bid_price:.2f}"),
            self._info_row("Ask", f"${ask_price:.2f}"),
            self._info_row("Spread", f"${spread:.3f}"),
            self._info_row("Volume", f"{volume:,}"),
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