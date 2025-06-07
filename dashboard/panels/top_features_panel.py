"""Top Features Panel"""

from dash import html, dash_table
import pandas as pd

class TopFeaturesPanel:
    def __init__(self, dark_theme: dict):
        self.DARK_THEME = dark_theme
    
    def create_layout(self) -> html.Div:
        return html.Div(
            [
                html.H4(
                    "Top Features",
                    style={
                        "color": self.DARK_THEME["text_primary"],
                        "marginBottom": "4px",
                        "fontSize": "12px",
                        "fontWeight": "bold",
                    },
                ),
                html.Div(id="features-content"),
            ],
            style=self._card_style(),
        )
    
    def create_content(self, state) -> html.Div:
        # Get feature attribution data
        attribution_data = getattr(state, "attribution_data", {})
        
        if not attribution_data:
            return html.Div([
                html.P("No attribution data available", 
                       style={"color": self.DARK_THEME["text_muted"], 
                              "textAlign": "center", 
                              "margin": "20px 0"})
            ])
        
        # Create feature data by branch
        all_features = []
        
        for branch_name, features in attribution_data.items():
            if isinstance(features, dict):
                for feature_name, importance in features.items():
                    all_features.append({
                        "Branch": branch_name,
                        "Feature": feature_name,
                        "Importance": abs(importance),
                        "Value": importance,
                    })
        
        if not all_features:
            return html.Div([
                html.P("No feature data available", 
                       style={"color": self.DARK_THEME["text_muted"], 
                              "textAlign": "center", 
                              "margin": "20px 0"})
            ])
        
        # Sort by importance and take top 15
        df = pd.DataFrame(all_features)
        df = df.nlargest(15, 'Importance')
        
        # Format for display
        df['Importance Display'] = df['Importance'].apply(lambda x: f"{x:.4f}")
        df['Value Display'] = df['Value'].apply(lambda x: f"{x:.4f}")
        
        table_data = df[['Branch', 'Feature', 'Importance Display', 'Value Display']].to_dict('records')
        
        features_table = dash_table.DataTable(
            data=table_data,
            columns=[
                {"name": "Branch", "id": "Branch"},
                {"name": "Feature", "id": "Feature"},
                {"name": "Importance", "id": "Importance Display", "type": "numeric"},
                {"name": "Value", "id": "Value Display", "type": "numeric"},
            ],
            style_cell={
                "backgroundColor": self.DARK_THEME["bg_tertiary"],
                "color": self.DARK_THEME["text_primary"],
                "border": f"1px solid {self.DARK_THEME['border']}",
                "fontSize": "9px",
                "padding": "3px 5px",
                "textAlign": "left",
                "whiteSpace": "nowrap",
                "overflow": "hidden",
                "textOverflow": "ellipsis",
                "maxWidth": "100px",
            },
            style_data_conditional=[
                # Branch colors
                {
                    "if": {"column_id": "Branch", "filter_query": "{Branch} = hf"},
                    "color": self.DARK_THEME["accent_blue"],
                },
                {
                    "if": {"column_id": "Branch", "filter_query": "{Branch} = mf"},
                    "color": self.DARK_THEME["accent_green"],
                },
                {
                    "if": {"column_id": "Branch", "filter_query": "{Branch} = lf"},
                    "color": self.DARK_THEME["accent_orange"],
                },
                {
                    "if": {"column_id": "Branch", "filter_query": "{Branch} = portfolio"},
                    "color": self.DARK_THEME["accent_purple"],
                },
                # Value colors
                {
                    "if": {
                        "column_id": "Value Display",
                        "filter_query": "{Value Display} > 0"
                    },
                    "color": self.DARK_THEME["accent_green"],
                },
                {
                    "if": {
                        "column_id": "Value Display",
                        "filter_query": "{Value Display} < 0"
                    },
                    "color": self.DARK_THEME["accent_red"],
                },
            ],
            style_header={
                "backgroundColor": self.DARK_THEME["bg_secondary"],
                "color": self.DARK_THEME["text_secondary"],
                "fontWeight": "bold",
                "fontSize": "8px",
            },
            page_size=12,
            sort_action="native",
        )
        
        return html.Div([features_table])
    
    def _card_style(self) -> dict:
        return {
            "backgroundColor": self.DARK_THEME["bg_secondary"],
            "border": f"1px solid {self.DARK_THEME['border']}",
            "borderRadius": "6px",
            "padding": "8px",
            "height": "100%",
            "overflow": "auto",
        }