"""Price Chart Panel - Exact port from old dashboard candlestick implementation"""

from dash import html, dcc
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Dict, Any

class PriceChartPanel:
    def __init__(self, dark_theme: dict):
        self.DARK_THEME = dark_theme
    
    def create_layout(self) -> html.Div:
        return html.Div(
            [
                html.H4(
                    "Price & Volume Chart",
                    style={
                        "color": self.DARK_THEME["text_primary"],
                        "marginBottom": "4px",
                        "fontSize": "12px",
                        "fontWeight": "bold",
                    },
                ),
                html.Div(id="chart-content"),
            ],
            style=self._card_style(),
        )
    
    def create_content(self, state) -> html.Div:
        """Create candlestick chart content exactly like old dashboard"""
        fig = self._create_price_chart(state)
        
        return html.Div([
            dcc.Graph(
                figure=fig,
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['pan2d', 'select2d', 'lasso2d', 'autoScale2d']
                },
                style={'height': '500px'}
            )
        ])
    
    def _create_price_chart(self, state) -> go.Figure:
        """Create candlestick chart with 1m bars using Plotly - EXACT PORT FROM OLD DASHBOARD"""
        # Get 1m candle data from state
        candle_data = getattr(state, "candle_data_1m", [])
        trades_data = list(state.recent_trades) if state.recent_trades else []

        # Create subplots with candlestick and volume
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=(None, None),
        )

        # Initialize candles to ensure it's always defined
        candles = candle_data

        if not candle_data:
            # Return empty figure with message
            fig.add_annotation(
                text="No market data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14, color=self.DARK_THEME["text_muted"]),
            )
        else:
            # Get episode and reset information to determine chart window
            episode_start_time = getattr(state, "episode_start_time", None)
            current_timestamp = getattr(state, "current_timestamp", None)

            # Determine chart window based on episode state
            if episode_start_time and current_timestamp:
                try:
                    # Parse timestamps
                    episode_start = pd.to_datetime(episode_start_time)
                    current_time = pd.to_datetime(current_timestamp)

                    # Remove timezone info if present for consistent handling
                    if episode_start.tz is not None:
                        episode_start = episode_start.tz_localize(None)
                    if current_time.tz is not None:
                        current_time = current_time.tz_localize(None)

                    # Calculate time windows:
                    # - Show 1 hour before episode start (reset point)
                    # - Show up to 30 minutes after episode start
                    window_start = episode_start - pd.Timedelta(hours=1)
                    window_end = episode_start + pd.Timedelta(minutes=30)

                    # Extend window if current time is beyond the 30-minute mark
                    if current_time > window_end:
                        window_end = current_time + pd.Timedelta(
                            minutes=5
                        )  # Small buffer

                    # Filter candle data to the focused window
                    candle_df = pd.DataFrame(candle_data)
                    candle_df["timestamp"] = pd.to_datetime(
                        candle_df["timestamp"]
                    ).dt.tz_localize(None)

                    # Filter to window
                    mask = (candle_df["timestamp"] >= window_start) & (
                        candle_df["timestamp"] <= window_end
                    )
                    filtered_candles = candle_df[mask]
                    candles = (
                        filtered_candles.to_dict("records")
                        if not filtered_candles.empty
                        else candle_data
                    )

                except Exception:
                    # Fallback to all data if timestamp parsing fails - candles already set above
                    pass

            if candles:
                # Convert to dataframe for easier handling
                df = pd.DataFrame(candles)

                # Parse timestamps and ensure timezone-naive
                df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

                # Add candlestick trace
                fig.add_trace(
                    go.Candlestick(
                        x=df["timestamp"],
                        open=df["open"],
                        high=df["high"],
                        low=df["low"],
                        close=df["close"],
                        name="Price",
                        increasing_line_color=self.DARK_THEME["accent_green"],
                        decreasing_line_color=self.DARK_THEME["accent_red"],
                        increasing_fillcolor=self.DARK_THEME["accent_green"],
                        decreasing_fillcolor=self.DARK_THEME["accent_red"],
                    ),
                    row=1,
                    col=1,
                )

                # Add volume bars
                colors = [
                    self.DARK_THEME["accent_green"]
                    if close >= open
                    else self.DARK_THEME["accent_red"]
                    for open, close in zip(df["open"], df["close"])
                ]

                fig.add_trace(
                    go.Bar(
                        x=df["timestamp"],
                        y=df["volume"],
                        name="Volume",
                        marker_color=colors,
                        opacity=0.5,
                    ),
                    row=2,
                    col=1,
                )

                # Add horizontal line for current price
                current_price = getattr(state, "current_price", 0)
                if current_price > 0:
                    fig.add_hline(
                        y=current_price,
                        line_dash="dash",
                        line_color=self.DARK_THEME["accent_blue"],
                        line_width=1,
                        annotation_text=f"${current_price:.2f}",
                        annotation_position="right",
                        annotation_font_size=10,
                        annotation_font_color=self.DARK_THEME["accent_blue"],
                        row=1,
                        col=1,
                    )

                # Add vertical line for episode start (reset point)
                episode_start_time = getattr(state, "episode_start_time", None)
                if episode_start_time:
                    try:
                        # Convert to pandas Timestamp and ensure timezone-naive
                        if isinstance(episode_start_time, str):
                            episode_start_ts = pd.to_datetime(
                                episode_start_time
                            ).tz_localize(None)
                        else:
                            episode_start_ts = pd.to_datetime(episode_start_time)
                            if (
                                hasattr(episode_start_ts, "tz")
                                and episode_start_ts.tz is not None
                            ):
                                episode_start_ts = episode_start_ts.tz_localize(None)

                        # Check if episode start is within chart range
                        if (
                            episode_start_ts >= df["timestamp"].min()
                            and episode_start_ts <= df["timestamp"].max()
                        ):
                            # Add vertical line spanning both price and volume subplots
                            fig.add_shape(
                                type="line",
                                x0=episode_start_ts,
                                x1=episode_start_ts,
                                y0=0,
                                y1=1,
                                yref="paper",
                                line=dict(
                                    color=self.DARK_THEME["accent_purple"],
                                    width=3,
                                    dash="solid",
                                ),
                            )
                            # Add annotation for episode start
                            price_mid = (df["high"].max() + df["low"].min()) / 2
                            fig.add_annotation(
                                x=episode_start_ts,
                                y=price_mid,
                                yref="y",
                                text="Reset",
                                showarrow=True,
                                arrowhead=2,
                                arrowsize=1,
                                arrowcolor=self.DARK_THEME["accent_purple"],
                                font=dict(size=10, color=self.DARK_THEME["accent_purple"]),
                                bgcolor="rgba(0,0,0,0.7)",
                                bordercolor=self.DARK_THEME["accent_purple"],
                                borderwidth=1,
                                xshift=-20,
                            )
                    except Exception:
                        # Skip episode start line if timestamp conversion fails
                        pass

                # Add vertical line for current trading time
                current_timestamp = getattr(state, "current_timestamp", None)
                if current_timestamp:
                    try:
                        # Convert to pandas Timestamp and ensure timezone-naive
                        if isinstance(current_timestamp, str):
                            current_ts = pd.to_datetime(current_timestamp).tz_localize(
                                None
                            )
                        else:
                            current_ts = pd.to_datetime(current_timestamp)
                            if hasattr(current_ts, "tz") and current_ts.tz is not None:
                                current_ts = current_ts.tz_localize(None)

                        # Check if current timestamp is within chart range
                        if (
                            current_ts >= df["timestamp"].min()
                            and current_ts <= df["timestamp"].max()
                        ):
                            # Add vertical line spanning both price and volume subplots
                            fig.add_shape(
                                type="line",
                                x0=current_ts,
                                x1=current_ts,
                                y0=0,
                                y1=1,
                                yref="paper",
                                line=dict(
                                    color=self.DARK_THEME["accent_orange"],
                                    width=2,
                                    dash="dash",
                                ),
                            )
                            # Add annotation for current time at top of price chart
                            price_max = df["high"].max()
                            fig.add_annotation(
                                x=current_ts,
                                y=price_max,
                                yref="y",
                                text=f"Now: {getattr(state, 'ny_time', current_ts.strftime('%H:%M'))}",
                                showarrow=False,
                                font=dict(size=10, color=self.DARK_THEME["accent_orange"]),
                                bgcolor="rgba(0,0,0,0.7)",
                                bordercolor=self.DARK_THEME["accent_orange"],
                                borderwidth=1,
                                yshift=10,
                            )
                    except Exception:
                        # Skip vertical line if timestamp conversion fails
                        pass

                # Add reset point markers (show all for now - strategy filtering in agent)
                reset_points_data = getattr(state, "reset_points_data", [])
                if reset_points_data:
                    # Note: Strategy-based filtering is now handled in the PPO agent
                    # Dashboard shows all reset points for visibility

                    for reset_point in reset_points_data:
                        # Parse reset point timestamp
                        reset_time = reset_point.get("timestamp")
                        reset_price = reset_point.get("price", 0)
                        activity_score = reset_point.get("activity_score", 0)
                        roc_score = reset_point.get("roc_score", 0)
                        combined_score = reset_point.get("combined_score", 0)

                        if reset_time and reset_price > 0:
                            try:
                                reset_dt = pd.to_datetime(reset_time)
                                # Convert UTC reset points to ET for proper chart display
                                if reset_dt.tz is not None:
                                    # Convert from UTC to ET, then remove timezone for chart compatibility
                                    reset_dt = reset_dt.tz_convert(
                                        "America/New_York"
                                    ).tz_localize(None)
                                else:
                                    # If no timezone, assume UTC and convert to ET
                                    reset_dt = (
                                        reset_dt.tz_localize("UTC")
                                        .tz_convert("America/New_York")
                                        .tz_localize(None)
                                    )
                            except:
                                continue

                            # Show reset points within full trading session (4 AM to 8 PM ET)
                            chart_date = df["timestamp"].iloc[0].date()
                            session_start = pd.Timestamp(
                                f"{chart_date} 04:00:00"
                            ).tz_localize(None)
                            session_end = pd.Timestamp(
                                f"{chart_date} 20:00:00"
                            ).tz_localize(None)

                            if session_start <= reset_dt <= session_end:
                                # Color based on combined score - rank-based system
                                combined_score = reset_point.get("combined_score", 0.5)
                                if combined_score >= 0.8:
                                    marker_color = self.DARK_THEME[
                                        "accent_purple"
                                    ]  # Very high quality
                                    marker_size = 10
                                elif combined_score >= 0.6:
                                    marker_color = self.DARK_THEME[
                                        "accent_blue"
                                    ]  # High quality
                                    marker_size = 8
                                elif combined_score >= 0.4:
                                    marker_color = self.DARK_THEME[
                                        "accent_orange"
                                    ]  # Medium quality
                                    marker_size = 7
                                else:
                                    marker_color = self.DARK_THEME[
                                        "text_muted"
                                    ]  # Low quality
                                    marker_size = 6

                                # Place reset points at bottom of volume chart for better visibility
                                # Get the volume range to position markers consistently
                                volume_max = (
                                    df["volume"].max()
                                    if "volume" in df.columns and not df["volume"].empty
                                    else 1000
                                )
                                marker_y_volume = (
                                    -volume_max * 0.1
                                )  # 10% below the volume chart

                                fig.add_trace(
                                    go.Scatter(
                                        x=[reset_dt],
                                        y=[marker_y_volume],
                                        mode="markers",
                                        marker=dict(
                                            size=marker_size,
                                            color=marker_color,
                                            symbol="diamond",
                                            line=dict(
                                                width=1,
                                                color=self.DARK_THEME["text_primary"],
                                            ),
                                        ),
                                        name="Reset Point",
                                        showlegend=False,
                                        hovertemplate=f"Reset Point<br>Time: {reset_dt.strftime('%H:%M')}<br>Price: ${reset_price:.3f}<br>Activity: {activity_score:.3f}<br>ROC: {roc_score:.3f}<br>Combined: {combined_score:.3f}<extra></extra>",
                                    ),
                                    row=2,
                                    col=1,  # Place on volume chart (row 2)
                                )

                # Add execution markers (not completed trades)
                executions_data = (
                    list(state.recent_executions) if hasattr(state, 'recent_executions') and state.recent_executions else []
                )
                if executions_data:
                    for execution in executions_data[-20:]:  # Last 20 executions
                        # Use raw timestamp for chart plotting
                        exec_time = execution.get(
                            "timestamp_raw", execution.get("timestamp")
                        )
                        exec_price = execution.get("fill_price", 0)

                        if exec_time and exec_price > 0:
                            # Parse execution timestamp and ensure timezone-naive
                            try:
                                exec_dt = pd.to_datetime(exec_time)
                                # Remove timezone if present
                                if exec_dt.tz is not None:
                                    exec_dt = exec_dt.tz_localize(None)
                            except:
                                continue

                            # Only show executions within the chart time range
                            if (
                                exec_dt >= df["timestamp"].min()
                                and exec_dt <= df["timestamp"].max()
                            ):
                                is_buy = execution.get("side") == "BUY"
                                marker_color = (
                                    self.DARK_THEME["accent_green"]
                                    if is_buy
                                    else self.DARK_THEME["accent_red"]
                                )
                                marker_symbol = (
                                    "triangle-up" if is_buy else "triangle-down"
                                )

                                fig.add_trace(
                                    go.Scatter(
                                        x=[exec_dt],
                                        y=[exec_price],
                                        mode="markers",
                                        marker=dict(
                                            size=12,
                                            color=marker_color,
                                            symbol=marker_symbol,
                                            line=dict(
                                                width=1,
                                                color=self.DARK_THEME["text_primary"],
                                            ),
                                        ),
                                        name=execution.get("side", "Execution"),
                                        showlegend=False,
                                        hovertemplate=f"{execution.get('side', 'Execution')}<br>Price: ${exec_price:.3f}<br>Qty: {execution.get('quantity', 0)}<extra></extra>",
                                    ),
                                    row=1,
                                    col=1,
                                )

        # Update layout
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor=self.DARK_THEME["bg_tertiary"],
            paper_bgcolor=self.DARK_THEME["bg_secondary"],
            font_color=self.DARK_THEME["text_primary"],
            xaxis=dict(
                gridcolor=self.DARK_THEME["border"],
                rangeslider=dict(visible=False),
                # Force show all data without zoom
                autorange=True,
                fixedrange=True,  # Disable zoom/pan to see all data
            ),
            yaxis=dict(gridcolor=self.DARK_THEME["border"], title="Price"),
            xaxis2=dict(gridcolor=self.DARK_THEME["border"], fixedrange=True),
            yaxis2=dict(gridcolor=self.DARK_THEME["border"], title="Volume"),
            margin=dict(l=60, r=40, t=20, b=40),
            showlegend=False,
            hovermode="x unified",
            height=500,
        )

        # Update x-axis to show time nicely in NY time
        fig.update_xaxes(
            tickformat="%H:%M",
            tickmode="auto",
            nticks=16,  # Show more ticks for full day
            type="date",
        )

        # Set x-axis range based on filtered candle data for focused view
        if candles and len(candles) > 0:
            # Use the filtered candles data range for focused view
            df_filtered = pd.DataFrame(candles)
            df_filtered["timestamp"] = pd.to_datetime(
                df_filtered["timestamp"]
            ).dt.tz_localize(None)

            # Set range based on actual filtered data with small buffer
            start_time = df_filtered["timestamp"].min() - pd.Timedelta(minutes=5)
            end_time = df_filtered["timestamp"].max() + pd.Timedelta(minutes=5)

            fig.update_xaxes(range=[start_time, end_time])

        return fig
    
    def _card_style(self) -> dict:
        return {
            "backgroundColor": self.DARK_THEME["bg_secondary"],
            "border": f"1px solid {self.DARK_THEME['border']}",
            "borderRadius": "6px",
            "padding": "8px",
            "height": "100%",
            "overflow": "auto",
        }