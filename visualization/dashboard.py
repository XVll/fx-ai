# visualization/dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional


class TradingDashboard:
    """
    Custom dashboard for visualizing trading metrics using Streamlit.
    Can connect to W&B for real-time training monitoring.
    """

    def __init__(self, wandb_project: str = None, wandb_entity: str = None):
        """
        Initialize the dashboard with optional W&B connection.

        Args:
            wandb_project: W&B project name (optional)
            wandb_entity: W&B entity/username (optional)
        """
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.api = None

        # Initialize Streamlit app
        st.set_page_config(
            page_title="AI Trading Dashboard",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Try to connect to W&B
        if wandb_project:
            self._connect_to_wandb()

    def _connect_to_wandb(self):
        """Connect to W&B API."""
        try:
            self.api = wandb.Api()
            st.sidebar.success(f"Connected to W&B!")
        except Exception as e:
            st.sidebar.error(f"Failed to connect to W&B: {str(e)}")
            self.api = None

    def _load_runs(self) -> List[Any]:
        """Load runs from W&B project."""
        if not self.api:
            return []

        try:
            # Create path for W&B project
            path = f"{self.wandb_entity}/{self.wandb_project}" if self.wandb_entity else self.wandb_project
            runs = list(self.api.runs(path=path))
            return runs
        except Exception as e:
            st.sidebar.error(f"Failed to load runs: {str(e)}")
            return []

    def _load_run_data(self, run: Any, metrics: List[str]) -> pd.DataFrame:
        """Load metric data from a W&B run."""
        try:
            history = run.history(keys=metrics)
            return history
        except Exception as e:
            st.error(f"Failed to load run data: {str(e)}")
            return pd.DataFrame()

    def _create_trade_analysis(self, trades_df: pd.DataFrame):
        """Create trade analysis visualizations."""
        if trades_df.empty:
            st.warning("No trade data available.")
            return

        # Trade statistics
        st.subheader("Trade Statistics")

        # Calculate statistics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['realized_pnl'] > 0])
        losing_trades = len(trades_df[trades_df['realized_pnl'] <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        total_pnl = trades_df['realized_pnl'].sum()
        avg_win = trades_df[trades_df['realized_pnl'] > 0]['realized_pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['realized_pnl'] <= 0]['realized_pnl'].mean() if losing_trades > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

        # Create metrics layout
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Trades", total_trades)
        col1.metric("Win Rate", f"{win_rate:.2%}")
        col1.metric("Total P&L", f"${total_pnl:.2f}")

        col2.metric("Winning Trades", winning_trades)
        col2.metric("Losing Trades", losing_trades)
        col2.metric("Profit Factor", f"{profit_factor:.2f}")

        col3.metric("Avg Win", f"${avg_win:.2f}")
        col3.metric("Avg Loss", f"${avg_loss:.2f}")
        col3.metric("W/L Ratio", f"{abs(avg_win / avg_loss):.2f}" if avg_loss != 0 else "∞")

        # Create visualizations
        st.subheader("Trade Analysis")

        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs([
            "P&L Distribution", "Cumulative P&L", "Win/Loss by Time", "Trade Duration"
        ])

        with tab1:
            # P&L Distribution
            fig = px.histogram(
                trades_df, x="realized_pnl",
                color_discrete_sequence=['lightgreen' if x > 0 else 'salmon' for x in trades_df['realized_pnl']],
                title="P&L Distribution",
                labels={"realized_pnl": "P&L ($)"}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            # Cumulative P&L
            trades_df['cumulative_pnl'] = trades_df['realized_pnl'].cumsum()
            fig = px.line(
                trades_df, y="cumulative_pnl",
                title="Cumulative P&L",
                labels={"index": "Trade #", "cumulative_pnl": "Cumulative P&L ($)"}
            )
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            # Win/Loss by Time
            if 'close_time' in trades_df.columns:
                # Convert close_time to datetime if it's not already
                if not pd.api.types.is_datetime64_any_dtype(trades_df['close_time']):
                    trades_df['close_time'] = pd.to_datetime(trades_df['close_time'])

                # Extract hour
                trades_df['hour'] = trades_df['close_time'].dt.hour

                # Group by hour
                hourly_stats = trades_df.groupby('hour').agg({
                    'realized_pnl': ['count', 'mean', 'sum'],
                    'realized_pnl': lambda x: (x > 0).mean()  # Win rate
                }).reset_index()
                hourly_stats.columns = ['hour', 'trade_count', 'avg_pnl', 'total_pnl', 'win_rate']

                fig = make_subplots(specs=[[{"secondary_y": True}]])

                # Add trade count bars
                fig.add_trace(
                    go.Bar(x=hourly_stats['hour'], y=hourly_stats['trade_count'],
                           name="Trade Count", marker_color="lightblue"),
                    secondary_y=False
                )

                # Add win rate line
                fig.add_trace(
                    go.Scatter(x=hourly_stats['hour'], y=hourly_stats['win_rate'],
                               name="Win Rate", marker_color="green", mode="lines+markers"),
                    secondary_y=True
                )

                fig.update_layout(
                    title_text="Win Rate by Hour",
                    xaxis_title="Hour of Day",
                )

                fig.update_yaxes(title_text="Trade Count", secondary_y=False)
                fig.update_yaxes(title_text="Win Rate", secondary_y=True)

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No time data available for time-based analysis.")

        with tab4:
            # Trade Duration Analysis
            if 'open_time' in trades_df.columns and 'close_time' in trades_df.columns:
                # Calculate duration in seconds
                if not pd.api.types.is_datetime64_any_dtype(trades_df['open_time']):
                    trades_df['open_time'] = pd.to_datetime(trades_df['open_time'])
                if not pd.api.types.is_datetime64_any_dtype(trades_df['close_time']):
                    trades_df['close_time'] = pd.to_datetime(trades_df['close_time'])

                trades_df['duration_seconds'] = (trades_df['close_time'] - trades_df['open_time']).dt.total_seconds()

                # Create bins for duration
                duration_bins = [0, 1, 5, 10, 30, 60, 300, float('inf')]
                duration_labels = ['<1s', '1-5s', '5-10s', '10-30s', '30-60s', '1-5m', '>5m']
                trades_df['duration_group'] = pd.cut(trades_df['duration_seconds'], bins=duration_bins,
                                                     labels=duration_labels)

                # Group by duration
                duration_stats = trades_df.groupby('duration_group').agg({
                    'realized_pnl': ['count', 'mean', 'sum'],
                    'realized_pnl': lambda x: (x > 0).mean()  # Win rate
                }).reset_index()
                duration_stats.columns = ['duration_group', 'trade_count', 'avg_pnl', 'total_pnl', 'win_rate']

                fig = make_subplots(specs=[[{"secondary_y": True}]])

                # Add trade count bars
                fig.add_trace(
                    go.Bar(x=duration_stats['duration_group'], y=duration_stats['trade_count'],
                           name="Trade Count", marker_color="lightblue"),
                    secondary_y=False
                )

                # Add win rate line
                fig.add_trace(
                    go.Scatter(x=duration_stats['duration_group'], y=duration_stats['win_rate'],
                               name="Win Rate", marker_color="green", mode="lines+markers"),
                    secondary_y=True
                )

                fig.update_layout(
                    title_text="Win Rate by Trade Duration",
                    xaxis_title="Trade Duration",
                )

                fig.update_yaxes(title_text="Trade Count", secondary_y=False)
                fig.update_yaxes(title_text="Win Rate", secondary_y=True)

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No time data available for duration-based analysis.")

    def _create_training_visualizations(self, metrics_df: pd.DataFrame):
        """Create training metrics visualizations."""
        if metrics_df.empty:
            st.warning("No training metrics available.")
            return

        # Display metric evolution
        st.subheader("Training Metrics")

        # Select metrics to display
        available_metrics = [col for col in metrics_df.columns if col not in ['_step', '_runtime', '_timestamp']]
        if not available_metrics:
            st.warning("No metrics found in data.")
            return

        default_metrics = [metric for metric in [
            'episode/reward', 'update/loss', 'trades/accuracy', 'trades/profit_factor'
        ] if metric in available_metrics]

        selected_metrics = st.multiselect(
            "Select metrics to display",
            options=available_metrics,
            default=default_metrics[:2]  # Select first two metrics by default
        )

        if not selected_metrics:
            st.warning("Please select at least one metric to display.")
            return

        # Create line chart
        fig = px.line(
            metrics_df, x='_step', y=selected_metrics,
            title="Training Metrics",
            labels={"_step": "Step", "value": "Value", "variable": "Metric"}
        )

        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # Create visualizations for specific metric types
        reward_metrics = [m for m in metrics_df.columns if 'reward' in m.lower()]
        loss_metrics = [m for m in metrics_df.columns if 'loss' in m.lower()]

        tabs = st.tabs(["Rewards", "Losses", "Trade Metrics"])

        with tabs[0]:
            if reward_metrics:
                fig = px.line(
                    metrics_df, x='_step', y=reward_metrics,
                    title="Reward Metrics",
                    labels={"_step": "Step", "value": "Value", "variable": "Metric"}
                )
                fig.update_layout(hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No reward metrics found.")

        with tabs[1]:
            if loss_metrics:
                fig = px.line(
                    metrics_df, x='_step', y=loss_metrics,
                    title="Loss Components",
                    labels={"_step": "Step", "value": "Value", "variable": "Metric"}
                )
                fig.update_layout(hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No loss metrics found.")

        with tabs[2]:
            trade_metrics = [m for m in metrics_df.columns if 'trades/' in m]
            if trade_metrics:
                fig = px.line(
                    metrics_df, x='_step', y=trade_metrics,
                    title="Trading Performance",
                    labels={"_step": "Step", "value": "Value", "variable": "Metric"}
                )
                fig.update_layout(hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No trade metrics found.")

    def _create_model_visualizations(self, metrics_df: pd.DataFrame):
        """Create model-specific visualizations."""
        # Extract gradient norms if available
        grad_cols = [col for col in metrics_df.columns if 'gradients/' in col]

        if grad_cols:
            st.subheader("Model Gradients")

            # Select which gradients to display
            selected_grads = st.multiselect(
                "Select gradients to display",
                options=grad_cols,
                default=grad_cols[:3]  # Show first 3 by default
            )

            if selected_grads:
                fig = px.line(
                    metrics_df, x='_step', y=selected_grads,
                    title="Gradient Norms",
                    labels={"_step": "Step", "value": "Norm", "variable": "Parameter"}
                )
                fig.update_layout(hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

    def run(self):
        """Run the dashboard application."""
        st.title("AI Trading Dashboard")

        # Sidebar
        st.sidebar.title("Settings")

        # W&B connection
        st.sidebar.header("Weights & Biases")
        if not self.wandb_project:
            wandb_project = st.sidebar.text_input("Project Name")
            wandb_entity = st.sidebar.text_input("Entity (optional)")

            if st.sidebar.button("Connect"):
                self.wandb_project = wandb_project
                self.wandb_entity = wandb_entity
                self._connect_to_wandb()

        # Load runs if connected to W&B
        runs = []
        if self.api and self.wandb_project:
            runs = self._load_runs()

            # Run selection
            if runs:
                st.sidebar.header("Run Selection")

                # Filter options
                filter_options = st.sidebar.expander("Filter Options", expanded=False)
                with filter_options:
                    # Filter by status
                    status_options = list(set(run.state for run in runs))
                    selected_status = st.multiselect(
                        "Status", options=status_options, default=status_options
                    )

                    # Filter by tags (if any runs have tags)
                    all_tags = set()
                    for run in runs:
                        if hasattr(run, 'tags') and run.tags:
                            all_tags.update(run.tags)

                    selected_tags = st.multiselect(
                        "Tags", options=list(all_tags) if all_tags else []
                    )

                    # Filter by time
                    time_options = ["All time", "Last 24 hours", "Last 7 days", "Last 30 days"]
                    time_filter = st.selectbox("Time period", options=time_options)

                # Apply filters
                filtered_runs = runs

                # Filter by status
                if selected_status:
                    filtered_runs = [run for run in filtered_runs if run.state in selected_status]

                # Filter by tags
                if selected_tags:
                    filtered_runs = [run for run in filtered_runs if
                                     hasattr(run, 'tags') and all(tag in run.tags for tag in selected_tags)]

                # Filter by time
                if time_filter != "All time":
                    now = datetime.now()
                    if time_filter == "Last 24 hours":
                        cutoff = now - timedelta(days=1)
                    elif time_filter == "Last 7 days":
                        cutoff = now - timedelta(days=7)
                    else:  # Last 30 days
                        cutoff = now - timedelta(days=30)

                    filtered_runs = [run for run in filtered_runs if
                                     datetime.strptime(run.created_at, "%Y-%m-%dT%H:%M:%S") > cutoff]

                # Sort runs
                sort_options = ["Created (newest first)", "Created (oldest first)", "Name (A-Z)", "Name (Z-A)"]
                sort_by = st.sidebar.selectbox("Sort by", options=sort_options)

                if sort_by == "Created (newest first)":
                    filtered_runs.sort(key=lambda x: x.created_at, reverse=True)
                elif sort_by == "Created (oldest first)":
                    filtered_runs.sort(key=lambda x: x.created_at)
                elif sort_by == "Name (A-Z)":
                    filtered_runs.sort(key=lambda x: x.name)
                elif sort_by == "Name (Z-A)":
                    filtered_runs.sort(key=lambda x: x.name, reverse=True)

                # Run display names
                run_options = {f"{run.name} ({run.state})": run for run in filtered_runs}

                if run_options:
                    selected_run_name = st.sidebar.selectbox(
                        "Select Run", options=list(run_options.keys())
                    )

                    selected_run = run_options[selected_run_name]

                    # Display run details
                    st.sidebar.subheader("Run Details")
                    st.sidebar.write(f"ID: {selected_run.id}")
                    st.sidebar.write(f"Created: {selected_run.created_at}")
                    st.sidebar.write(f"Status: {selected_run.state}")

                    if hasattr(selected_run, 'tags') and selected_run.tags:
                        st.sidebar.write(f"Tags: {', '.join(selected_run.tags)}")

                    # Main content - Run details
                    st.header(f"Run: {selected_run.name}")

                    # Tabs for different visualizations
                    tab1, tab2, tab3 = st.tabs([
                        "Training Metrics", "Trade Analysis", "Model Analytics"
                    ])

                    with tab1:
                        # Load metrics
                        metrics = [
                            'episode/reward', 'episode/length',
                            'update/actor_loss', 'update/critic_loss', 'update/entropy',
                            'trades/accuracy', 'trades/profit_factor', 'trades/avg_win', 'trades/avg_loss'
                        ]
                        metrics_df = self._load_run_data(selected_run, metrics)

                        if not metrics_df.empty:
                            self._create_training_visualizations(metrics_df)
                        else:
                            st.warning("No metrics data available for this run.")

                    with tab2:
                        # Try to load trade data
                        if hasattr(selected_run, 'summary') and 'trades_table' in selected_run.summary:
                            try:
                                trades_table = selected_run.summary['trades_table']
                                trades_df = pd.DataFrame(trades_table.data, columns=trades_table.columns)
                                self._create_trade_analysis(trades_df)
                            except Exception as e:
                                st.error(f"Failed to load trade data: {str(e)}")
                                st.warning("No trade data available for this run.")
                        else:
                            st.warning("No trade data available for this run.")

                    with tab3:
                        # Load gradient data
                        grad_metrics = [col for col in metrics_df.columns if 'gradients/' in col]
                        if grad_metrics:
                            self._create_model_visualizations(metrics_df)
                        else:
                            st.warning("No model analytics data available for this run.")
                else:
                    st.warning("No runs found matching the selected filters.")
            else:
                st.warning("No runs found in the selected project.")
        else:
            # Display sample data when not connected to W&B
            st.info("Connect to Weights & Biases to view live training data.")

            # Load sample data
            sample_data_path = os.path.join(os.path.dirname(__file__), "sample_data.json")
            if os.path.exists(sample_data_path):
                with open(sample_data_path, "r") as f:
                    sample_data = json.load(f)

                # Convert to DataFrames
                metrics_df = pd.DataFrame(sample_data["metrics"])
                trades_df = pd.DataFrame(sample_data["trades"])

                # Create tabs for sample data
                tab1, tab2 = st.tabs(["Sample Training Metrics", "Sample Trade Analysis"])

                with tab1:
                    self._create_training_visualizations(metrics_df)

                with tab2:
                    self._create_trade_analysis(trades_df)
            else:
                st.write("No sample data available. Connect to W&B to view real training data.")


if __name__ == "__main__":
    # Load W&B project from environment variables or use defaults
    project = os.environ.get("WANDB_PROJECT", "ai-trading")
    entity = os.environ.get("WANDB_ENTITY", None)

    # Create and run dashboard
    dashboard = TradingDashboard(wandb_project=project, wandb_entity=entity)
    dashboard.run()