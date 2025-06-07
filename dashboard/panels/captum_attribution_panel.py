"""Comprehensive Captum Attribution Analysis Panel"""

from dash import html, dcc, dash_table
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Optional, Dict, Any, List
from datetime import datetime
import base64
import io

class CaptumAttributionPanel:
    def __init__(self, dark_theme: dict):
        self.DARK_THEME = dark_theme
    
    def create_layout(self) -> html.Div:
        return html.Div(
            [
                html.H4(
                    "Captum Attribution Analysis",
                    style={
                        "color": self.DARK_THEME["text_primary"],
                        "marginBottom": "4px",
                        "fontSize": "12px",
                        "fontWeight": "bold",
                    },
                ),
                html.Div(id="captum-content"),
            ],
            style=self._card_style(),
        )
    
    def create_content(self, state) -> html.Div:
        # Get Captum data from state
        captum_results = getattr(state, "captum_results", {})
        attribution_summary = getattr(state, "attribution_summary", {})
        
        # Use captum_results if available, otherwise fall back to attribution_summary
        results = captum_results if captum_results else attribution_summary
        
        if not results:
            return self._create_empty_state()
        
        # Extract key metrics
        analysis_count = results.get("analysis_count", 0)
        
        # Get feature and branch data
        branch_importance = results.get("branch_importance", {})
        top_features = results.get("top_features", {})
        
        # Attribution summary data
        feature_importance = results.get("feature_importance_scores", {})
        dead_features_count = results.get("dead_features_count", 0)
        total_features = results.get("total_features", 0)
        
        content_sections = []
        
        # Analysis Status Section - simplified
        content_sections.append(self._create_status_section(analysis_count))
        
        # Feature Statistics Section
        if total_features > 0:
            content_sections.append(self._create_feature_stats_section(
                total_features, dead_features_count
            ))
        
        # Branch Importance Visualization
        if branch_importance:
            content_sections.append(self._create_branch_importance_section(
                branch_importance
            ))
        
        # Top Features Analysis
        if top_features or feature_importance:
            content_sections.append(self._create_top_features_section(
                top_features, feature_importance
            ))
        
        return html.Div(content_sections)
    
    def _create_empty_state(self) -> html.Div:
        return html.Div([
            html.Div([
                html.I(className="fas fa-brain", style={
                    "fontSize": "24px", 
                    "color": self.DARK_THEME["text_muted"],
                    "marginBottom": "8px"
                }),
                html.P("No attribution analysis available", style={
                    "color": self.DARK_THEME["text_muted"],
                    "fontSize": "12px",
                    "margin": "0"
                }),
                html.P("Analysis will appear after first PPO update", style={
                    "color": self.DARK_THEME["text_secondary"],
                    "fontSize": "10px",
                    "margin": "4px 0 0 0"
                })
            ], style={
                "textAlign": "center",
                "padding": "20px 0"
            })
        ])
    
    def _create_status_section(self, analysis_count: int) -> html.Div:
        """Create analysis status section"""
        status_color = self.DARK_THEME["accent_green"] if analysis_count > 0 else self.DARK_THEME["text_muted"]
        
        return html.Div([
            html.Div([
                html.Span("🧠", style={"fontSize": "12px", "marginRight": "6px"}),
                html.Span("Attribution Status", style={
                    "color": self.DARK_THEME["text_secondary"], 
                    "fontSize": "11px",
                    "fontWeight": "bold"
                })
            ]),
            html.Div([
                self._info_row("Analysis Count", f"#{analysis_count}", color=status_color),
            ], style={"marginTop": "4px"})
        ])
    
    def _create_feature_stats_section(self, total_features: int, 
                                    dead_features: int) -> html.Div:
        """Create feature statistics section"""
        active_features = total_features - dead_features
        active_pct = (active_features / total_features * 100) if total_features > 0 else 0
        
        return html.Div([
            html.Hr(style={"margin": "8px 0 4px 0", "borderColor": self.DARK_THEME["border"]}),
            html.Div([
                html.Span("📊", style={"fontSize": "12px", "marginRight": "6px"}),
                html.Span("Feature Statistics", style={
                    "color": self.DARK_THEME["text_secondary"], 
                    "fontSize": "11px",
                    "fontWeight": "bold"
                })
            ]),
            html.Div([
                self._info_row("Total Features", f"{total_features:,}"),
                self._info_row("Active Features", f"{active_features:,} ({active_pct:.1f}%)", 
                              color=self.DARK_THEME["accent_green"]),
                self._info_row("Dead Features", f"{dead_features:,}", 
                              color=self.DARK_THEME["accent_red"] if dead_features > 0 else self.DARK_THEME["accent_green"]),
            ], style={"marginTop": "4px"})
        ])
    
    def _create_branch_importance_section(self, branch_importance: Dict[str, float]) -> html.Div:
        """Create branch importance visualization"""
        if not branch_importance:
            return html.Div()
        
        # Sort branches by importance
        sorted_branches = sorted(branch_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Create bar chart
        branches = [branch.replace("_", " ").title() for branch, _ in sorted_branches]
        importances = [importance for _, importance in sorted_branches]
        
        # Color mapping for branches
        branch_colors = {
            "Hf": self.DARK_THEME["accent_blue"],      # High-frequency
            "Mf": self.DARK_THEME["accent_green"],     # Medium-frequency
            "Lf": self.DARK_THEME["accent_orange"],    # Low-frequency
            "Portfolio": self.DARK_THEME["accent_purple"], # Portfolio
            "Fusion": self.DARK_THEME["accent_red"],   # Fusion layer
        }
        
        colors = [branch_colors.get(branch, self.DARK_THEME["text_primary"]) for branch in branches]
        
        fig = go.Figure(data=[
            go.Bar(
                x=branches,
                y=importances,
                marker=dict(color=colors),
                text=[f"{imp:.3f}" for imp in importances],
                textposition='auto',
                textfont=dict(size=10, color=self.DARK_THEME["text_primary"])
            )
        ])
        
        fig.update_layout(
            height=120,
            margin=dict(l=0, r=0, t=20, b=30),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=self.DARK_THEME["text_primary"], size=10),
            title=dict(
                text="Branch Importance",
                font=dict(size=11, color=self.DARK_THEME["text_secondary"]),
                x=0.02, y=0.95
            ),
            xaxis=dict(
                showgrid=False,
                showline=False,
                tickfont=dict(size=9),
                color=self.DARK_THEME["text_secondary"]
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor=self.DARK_THEME["border"],
                showline=False,
                tickfont=dict(size=9),
                color=self.DARK_THEME["text_secondary"]
            )
        )
        
        return html.Div([
            html.Hr(style={"margin": "8px 0 4px 0", "borderColor": self.DARK_THEME["border"]}),
            dcc.Graph(
                figure=fig,
                config={'displayModeBar': False},
                style={'height': '120px'}
            )
        ])
    
    def _create_top_features_section(self, top_features: Dict[str, List], 
                                   feature_importance: Dict[str, float]) -> html.Div:
        """Create top features analysis section"""
        if not top_features and not feature_importance:
            return html.Div()
        
        # Prepare data for table
        table_data = []
        
        if top_features:
            # Handle branch-based top features - limit to 5 total
            all_features = []
            for branch, features in top_features.items():
                if isinstance(features, list) and features:
                    for feature in features:
                        if isinstance(feature, dict):
                            name = feature.get("name", "Unknown")
                            importance = feature.get("importance", 0.0)
                        elif isinstance(feature, (list, tuple)) and len(feature) >= 2:
                            name = str(feature[0])
                            importance = float(feature[1])
                        else:
                            name = str(feature)
                            importance = 0.0
                        
                        all_features.append({
                            "branch": branch.upper(),
                            "name": name,
                            "importance": importance
                        })
            
            # Sort all features by importance and take top 5
            all_features.sort(key=lambda x: x["importance"], reverse=True)
            for i, feature in enumerate(all_features[:5]):
                table_data.append({
                    "Branch": feature["branch"],
                    "Feature": feature["name"][:25] + "..." if len(feature["name"]) > 25 else feature["name"],
                    "Importance": f"{feature['importance']:.4f}",
                    "Rank": i + 1
                })
                
        elif feature_importance:
            # Handle flat feature importance dictionary - limit to 5
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            for i, (feature_name, importance) in enumerate(sorted_features[:5]):
                table_data.append({
                    "Branch": "ALL",
                    "Feature": feature_name[:25] + "..." if len(feature_name) > 25 else feature_name,
                    "Importance": f"{importance:.4f}",
                    "Rank": i + 1
                })
        
        if not table_data:
            return html.Div()
        
        return html.Div([
            html.Hr(style={"margin": "8px 0 4px 0", "borderColor": self.DARK_THEME["border"]}),
            html.Div([
                html.Span("⭐", style={"fontSize": "12px", "marginRight": "6px"}),
                html.Span("Top Features", style={
                    "color": self.DARK_THEME["text_secondary"], 
                    "fontSize": "11px",
                    "fontWeight": "bold"
                })
            ]),
            dash_table.DataTable(
                data=table_data,
                columns=[
                    {"name": "Branch", "id": "Branch"},
                    {"name": "Feature", "id": "Feature"},
                    {"name": "Importance", "id": "Importance"},
                    {"name": "#", "id": "Rank"},
                ],
                style_cell={
                    "backgroundColor": self.DARK_THEME["bg_tertiary"],
                    "color": self.DARK_THEME["text_primary"],
                    "border": f"1px solid {self.DARK_THEME['border']}",
                    "fontSize": "9px",
                    "padding": "2px 4px",
                    "textAlign": "left",
                    "whiteSpace": "nowrap",
                    "overflow": "hidden",
                    "textOverflow": "ellipsis",
                },
                style_data_conditional=[
                    # Color code by importance
                    {
                        "if": {"column_id": "Importance", "filter_query": "{Importance} > 0.1"},
                        "color": self.DARK_THEME["accent_green"],
                    },
                    {
                        "if": {"column_id": "Importance", "filter_query": "{Importance} > 0.05 && {Importance} <= 0.1"},
                        "color": self.DARK_THEME["accent_orange"],
                    },
                    # Color code branches
                    {
                        "if": {"column_id": "Branch", "filter_query": "{Branch} = HF"},
                        "color": self.DARK_THEME["accent_blue"],
                    },
                    {
                        "if": {"column_id": "Branch", "filter_query": "{Branch} = MF"},
                        "color": self.DARK_THEME["accent_green"],
                    },
                    {
                        "if": {"column_id": "Branch", "filter_query": "{Branch} = LF"},
                        "color": self.DARK_THEME["accent_orange"],
                    },
                    {
                        "if": {"column_id": "Branch", "filter_query": "{Branch} = PORTFOLIO"},
                        "color": self.DARK_THEME["accent_purple"],
                    },
                ],
                style_header={
                    "backgroundColor": self.DARK_THEME["bg_secondary"],
                    "color": self.DARK_THEME["text_secondary"],
                    "fontWeight": "bold",
                    "fontSize": "8px",
                },
                style_table={"maxHeight": "120px", "overflowY": "auto"},
            )
        ], style={"marginTop": "4px"})
    
    
    def _info_row(self, label: str, value: str, color: Optional[str] = None) -> html.Div:
        """Create info row with label and value"""
        value_color = color or self.DARK_THEME["text_primary"]
        return html.Div(
            [
                html.Span(label, style={
                    "color": self.DARK_THEME["text_secondary"], 
                    "fontSize": "10px"
                }),
                html.Span(value, style={
                    "color": value_color, 
                    "fontWeight": "bold", 
                    "fontSize": "10px"
                }),
            ],
            style={
                "display": "flex",
                "justifyContent": "space-between", 
                "alignItems": "center",
                "marginBottom": "1px",
                "minHeight": "14px",
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