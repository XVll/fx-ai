"""Attribution Quality Panel"""

from dash import html
from typing import Optional

class AttributionQualityPanel:
    def __init__(self, dark_theme: dict):
        self.DARK_THEME = dark_theme
    
    def create_layout(self) -> html.Div:
        return html.Div(
            [
                html.H4(
                    "Attribution Quality",
                    style={
                        "color": self.DARK_THEME["text_primary"],
                        "marginBottom": "4px",
                        "fontSize": "12px",
                        "fontWeight": "bold",
                    },
                ),
                html.Div(id="attribution-quality-content"),
            ],
            style=self._card_style(),
        )
    
    def create_content(self, state) -> html.Div:
        # Get attribution quality metrics
        attribution_enabled = getattr(state, "attribution_enabled", False)
        
        if not attribution_enabled:
            return html.Div([
                html.P("Attribution disabled", 
                       style={"color": self.DARK_THEME["text_muted"], 
                              "textAlign": "center", 
                              "margin": "20px 0"})
            ])
        
        # Attribution metrics
        attribution_confidence = getattr(state, "attribution_confidence", 0.0)
        attribution_coverage = getattr(state, "attribution_coverage", 0.0)
        attribution_stability = getattr(state, "attribution_stability", 0.0)
        attribution_method = getattr(state, "attribution_method", "N/A")
        last_attribution_time = getattr(state, "last_attribution_time", "N/A")
        
        # Branch-specific metrics
        hf_attribution_strength = getattr(state, "hf_attribution_strength", 0.0)
        mf_attribution_strength = getattr(state, "mf_attribution_strength", 0.0)
        lf_attribution_strength = getattr(state, "lf_attribution_strength", 0.0)
        portfolio_attribution_strength = getattr(state, "portfolio_attribution_strength", 0.0)
        
        # Color coding for quality metrics
        confidence_color = self._get_quality_color(attribution_confidence)
        coverage_color = self._get_quality_color(attribution_coverage)
        stability_color = self._get_quality_color(attribution_stability)
        
        return html.Div([
            self._info_row("Method", attribution_method),
            self._info_row("Last Update", last_attribution_time),
            html.Hr(style={"margin": "4px 0", "borderColor": self.DARK_THEME["border"]}),
            self._info_row("Confidence", f"{attribution_confidence:.2%}", color=confidence_color),
            self._info_row("Coverage", f"{attribution_coverage:.2%}", color=coverage_color),
            self._info_row("Stability", f"{attribution_stability:.2%}", color=stability_color),
            html.Hr(style={"margin": "4px 0", "borderColor": self.DARK_THEME["border"]}),
            html.Div([
                html.Span("Branch Strength", 
                         style={"color": self.DARK_THEME["text_secondary"], 
                                "fontSize": "11px", 
                                "fontWeight": "bold"})
            ]),
            self._info_row("HF", f"{hf_attribution_strength:.3f}", color=self.DARK_THEME["accent_blue"]),
            self._info_row("MF", f"{mf_attribution_strength:.3f}", color=self.DARK_THEME["accent_green"]),
            self._info_row("LF", f"{lf_attribution_strength:.3f}", color=self.DARK_THEME["accent_orange"]),
            self._info_row("Portfolio", f"{portfolio_attribution_strength:.3f}", color=self.DARK_THEME["accent_purple"]),
        ])
    
    def _get_quality_color(self, value: float) -> str:
        """Get color based on quality metric value (0-1 scale)"""
        if value >= 0.8:
            return self.DARK_THEME["accent_green"]
        elif value >= 0.6:
            return self.DARK_THEME["accent_orange"]
        else:
            return self.DARK_THEME["accent_red"]
    
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