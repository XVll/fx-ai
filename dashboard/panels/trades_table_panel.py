"""Trades Table Panel"""

from dash import html, dash_table
import pandas as pd
from enum import Enum

class TradesTablePanel:
    def __init__(self, dark_theme: dict):
        self.DARK_THEME = dark_theme
    
    def create_layout(self) -> html.Div:
        return html.Div(
            [
                html.H4(
                    "Recent Trades",
                    style={
                        "color": self.DARK_THEME["text_primary"],
                        "marginBottom": "4px",
                        "fontSize": "12px",
                        "fontWeight": "bold",
                    },
                ),
                html.Div(id="trades-content"),
            ],
            style=self._card_style(),
        )
    
    def _sanitize_data(self, data):
        """Convert enum values and other non-serializable objects to strings"""
        if isinstance(data, dict):
            return {k: self._sanitize_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_data(item) for item in data]
        elif isinstance(data, Enum):
            return str(data.value) if hasattr(data, 'value') else str(data)
        else:
            return data
    
    def create_content(self, state) -> html.Div:
        # Get episode-only trades data
        episode_trades = getattr(state, "episode_trades", [])
        
        # Fallback to recent_trades but filter by current episode
        if not episode_trades:
            all_trades = getattr(state, "recent_trades", [])
            current_episode = getattr(state, "episode_number", 0)
            
            # Filter trades for current episode only
            episode_trades = []
            for trade in all_trades:
                trade_episode = trade.get("episode", 0) if isinstance(trade, dict) else 0
                if trade_episode == current_episode:
                    episode_trades.append(trade)
        
        if not episode_trades:
            return html.Div([
                html.P("No trades this episode", 
                       style={"color": self.DARK_THEME["text_muted"], 
                              "textAlign": "center", 
                              "margin": "20px 0"})
            ])
        
        # Sanitize trades data to ensure JSON serialization
        trades_data = self._sanitize_data(episode_trades)
        
        # Create DataFrame and format
        df = pd.DataFrame(trades_data)
        
        # Format columns for display
        if not df.empty:
            # Time column
            if 'timestamp' in df.columns:
                df['Time'] = pd.to_datetime(df['timestamp']).dt.strftime('%H:%M:%S')
            elif 'time' in df.columns:
                df['Time'] = pd.to_datetime(df['time']).dt.strftime('%H:%M:%S')
            
            # Entry price (buy/open price)
            if 'entry_price' in df.columns:
                df['Entry'] = df['entry_price'].apply(lambda x: f"${x:.3f}")
            elif 'open_price' in df.columns:
                df['Entry'] = df['open_price'].apply(lambda x: f"${x:.3f}")
            elif 'price' in df.columns:
                # Fallback to price if entry_price not available
                df['Entry'] = df['price'].apply(lambda x: f"${x:.3f}")
            
            # Exit price (sell/close price)
            if 'exit_price' in df.columns:
                df['Exit'] = df['exit_price'].apply(lambda x: f"${x:.3f}")
            elif 'close_price' in df.columns:
                df['Exit'] = df['close_price'].apply(lambda x: f"${x:.3f}")
            elif 'fill_price' in df.columns:
                df['Exit'] = df['fill_price'].apply(lambda x: f"${x:.3f}")
            
            # Quantity
            if 'quantity' in df.columns:
                df['Qty'] = df['quantity'].apply(lambda x: f"{x:,}")
            elif 'qty' in df.columns:
                df['Qty'] = df['qty'].apply(lambda x: f"{x:,}")
            elif 'size' in df.columns:
                df['Qty'] = df['size'].apply(lambda x: f"{x:,}")
            
            # P&L
            if 'pnl' in df.columns:
                df['PnL'] = df['pnl'].apply(lambda x: f"${x:.2f}")
            elif 'profit_loss' in df.columns:
                df['PnL'] = df['profit_loss'].apply(lambda x: f"${x:.2f}")
            elif 'realized_pnl' in df.columns:
                df['PnL'] = df['realized_pnl'].apply(lambda x: f"${x:.2f}")
            
            # Select display columns in the requested order
            requested_columns = ['Time', 'Entry', 'Exit', 'Qty', 'PnL']
            display_columns = []
            column_configs = []
            
            for col in requested_columns:
                if col in df.columns:
                    display_columns.append(col)
                    column_configs.append({"name": col, "id": col})
            
            # If no columns found, create empty table with proper structure
            if not display_columns:
                table_data = []
                column_configs = [
                    {"name": "Time", "id": "Time"},
                    {"name": "Entry", "id": "Entry"},
                    {"name": "Exit", "id": "Exit"},
                    {"name": "Qty", "id": "Qty"},
                    {"name": "PnL", "id": "PnL"},
                ]
            else:
                table_data = df[display_columns].to_dict('records')
        else:
            table_data = []
            column_configs = [
                {"name": "Time", "id": "Time"},
                {"name": "Entry", "id": "Entry"},
                {"name": "Exit", "id": "Exit"},
                {"name": "Qty", "id": "Qty"},
                {"name": "PnL", "id": "PnL"},
            ]
        
        trades_table = dash_table.DataTable(
            data=table_data,
            columns=column_configs,
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
                "maxWidth": "80px",
            },
            style_data_conditional=[
                # Positive P&L in green
                {
                    "if": {"column_id": "PnL", "filter_query": "{PnL} > $0"},
                    "color": self.DARK_THEME["accent_green"],
                },
                # Negative P&L in red
                {
                    "if": {"column_id": "PnL", "filter_query": "{PnL} contains $-"},
                    "color": self.DARK_THEME["accent_red"],
                },
                # Zero P&L in muted color
                {
                    "if": {"column_id": "PnL", "filter_query": "{PnL} = $0.00"},
                    "color": self.DARK_THEME["text_muted"],
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
        
        return html.Div([trades_table])
    
    def _card_style(self) -> dict:
        return {
            "backgroundColor": self.DARK_THEME["bg_secondary"],
            "border": f"1px solid {self.DARK_THEME['border']}",
            "borderRadius": "6px",
            "padding": "8px",
            "height": "100%",
            "overflow": "auto",
        }