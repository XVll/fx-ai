# utils/wandb_reports.py
import wandb
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json
from typing import List, Dict, Any, Optional, Union


class ReportGenerator:
    """
    Generate comprehensive reports from W&B runs.
    """

    def __init__(
            self,
            project: str,
            entity: Optional[str] = None,
            output_dir: str = "reports"
    ):
        """
        Initialize the report generator.

        Args:
            project: W&B project name
            entity: W&B entity/username (optional)
            output_dir: Directory to save reports
        """
        self.project = project
        self.entity = entity
        self.output_dir = output_dir
        self.api = wandb.Api()

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def _get_runs(
            self,
            filters: Optional[Dict[str, Any]] = None,
            order: str = "-created_at",
            limit: int = 20
    ) -> List[Any]:
        """
        Get runs from W&B project with filters.

        Args:
            filters: Dictionary of filters to apply
            order: Order to sort runs by
            limit: Maximum number of runs to return

        Returns:
            List of W&B runs
        """
        # Create path
        path = f"{self.entity}/{self.project}" if self.entity else self.project

        # Create filter string
        filter_str = ""
        if filters:
            filter_parts = []
            for k, v in filters.items():
                if isinstance(v, str):
                    filter_parts.append(f"{k}=\"{v}\"")
                elif isinstance(v, list):
                    values_str = ', '.join([f'"{val}"' for val in v])
                    filter_parts.append(f"{k} IN [{values_str}]")
                else:
                    filter_parts.append(f"{k}={v}")

            filter_str = " AND ".join(filter_parts)

        # Get runs
        runs = list(self.api.runs(
            path=path,
            filters=filter_str,
            order=order,
            per_page=limit
        ))

        return runs

    def generate_performance_report(
            self,
            run_ids: Optional[List[str]] = None,
            filters: Optional[Dict[str, Any]] = None,
            limit: int = 5,
            metrics: Optional[List[str]] = None,
            output_format: str = "html",
            title: Optional[str] = None
    ) -> str:
        """
        Generate a performance comparison report for selected runs.

        Args:
            run_ids: List of specific run IDs to include (optional)
            filters: Filters to apply when selecting runs (if run_ids not provided)
            limit: Maximum number of runs to include (if selecting by filters)
            metrics: List of metrics to include (optional, will auto-detect if None)
            output_format: Output format (html, md, or json)
            title: Custom report title (optional)

        Returns:
            Path to the generated report file
        """
        # Get runs
        if run_ids:
            # Get specific runs by ID
            path = f"{self.entity}/{self.project}" if self.entity else self.project
            runs = [self.api.run(f"{path}/{run_id}") for run_id in run_ids]
        else:
            # Get runs by filters
            runs = self._get_runs(filters=filters, limit=limit)

        if not runs:
            raise ValueError("No runs found matching the criteria")

        # Create report timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create report title
        if not title:
            title = f"Performance Report - {self.project} - {timestamp}"

        # Auto-detect metrics if not provided
        if not metrics:
            # Get common metrics from all runs
            all_metrics = set()
            for run in runs:
                run_metrics = list(run.summary.keys())
                all_metrics.update(run_metrics)

            # Filter out non-numeric metrics and internal metrics
            metrics = []
            for metric in all_metrics:
                if metric.startswith("_"):
                    continue

                try:
                    # Check if metric is numeric in at least one run
                    for run in runs:
                        if metric in run.summary:
                            value = run.summary[metric]
                            if isinstance(value, (int, float)):
                                metrics.append(metric)
                                break
                except Exception:
                    # Skip metrics that cause errors
                    continue

            # Prioritize some common metrics
            priority_patterns = [
                "reward", "pnl", "accuracy", "profit_factor",
                "win_rate", "loss", "episode"
            ]

            # Sort metrics with priority ones first
            metrics.sort(key=lambda x: (
                not any(pattern in x.lower() for pattern in priority_patterns),
                x
            ))

            # Limit to top 15 metrics
            metrics = metrics[:15]

        # Create report data
        report_data = {
            "title": title,
            "timestamp": timestamp,
            "project": self.project,
            "entity": self.entity,
            "runs": [],
            "metrics": {}
        }

        # Extract run data
        for run in runs:
            run_data = {
                "id": run.id,
                "name": run.name,
                "created_at": run.created_at,
                "state": run.state,
                "config": self._extract_config(run)
            }

            # Extract metrics
            run_metrics = {}
            for metric in metrics:
                if metric in run.summary:
                    value = run.summary[metric]
                    if isinstance(value, (int, float)):
                        run_metrics[metric] = value
                    elif hasattr(value, "item") and callable(getattr(value, "item")):
                        try:
                            run_metrics[metric] = value.item()
                        except Exception:
                            pass

            run_data["metrics"] = run_metrics
            report_data["runs"].append(run_data)

            # Update metrics data
            for metric, value in run_metrics.items():
                if metric not in report_data["metrics"]:
                    report_data["metrics"][metric] = []

                report_data["metrics"][metric].append({
                    "run_id": run.id,
                    "run_name": run.name,
                    "value": value
                })

        # Generate plots
        plots = self._generate_metric_plots(report_data["metrics"])
        report_data["plots"] = plots

        # Generate report file
        if output_format == "json":
            # JSON report
            report_path = os.path.join(self.output_dir, f"report_{timestamp}.json")
            with open(report_path, "w") as f:
                json.dump(report_data, f, indent=2)
        elif output_format == "md":
            # Markdown report
            report_path = os.path.join(self.output_dir, f"report_{timestamp}.md")
            markdown = self._generate_markdown_report(report_data)
            with open(report_path, "w") as f:
                f.write(markdown)
        else:
            # HTML report (default)
            report_path = os.path.join(self.output_dir, f"report_{timestamp}.html")
            html = self._generate_html_report(report_data)
            with open(report_path, "w") as f:
                f.write(html)

        return report_path

    def generate_trade_analysis_report(
            self,
            run_id: str,
            output_format: str = "html"
    ) -> str:
        """
        Generate a detailed trade analysis report for a single run.

        Args:
            run_id: W&B run ID
            output_format: Output format (html, md, or json)

        Returns:
            Path to the generated report file
        """
        # Get run
        path = f"{self.entity}/{self.project}" if self.entity else self.project
        run = self.api.run(f"{path}/{run_id}")

        # Create report timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create report title
        title = f"Trade Analysis - {run.name} - {timestamp}"

        # Create report data
        report_data = {
            "title": title,
            "timestamp": timestamp,
            "project": self.project,
            "entity": self.entity,
            "run": {
                "id": run.id,
                "name": run.name,
                "created_at": run.created_at,
                "state": run.state,
                "config": self._extract_config(run)
            },
            "trade_metrics": {},
            "trades": []
        }

        # Extract trade metrics
        trade_metric_patterns = [
            "trades/", "win_rate", "profit_factor", "pnl", "accuracy"
        ]

        for key, value in run.summary.items():
            if any(pattern in key.lower() for pattern in trade_metric_patterns):
                if isinstance(value, (int, float)):
                    report_data["trade_metrics"][key] = value

        # Try to extract trade data
        try:
            if "trades_table" in run.summary:
                trades_table = run.summary["trades_table"]
                trades_data = []

                for i, row in enumerate(trades_table.data):
                    trade = {}
                    for j, col in enumerate(trades_table.columns):
                        trade[col] = row[j]
                    trades_data.append(trade)

                report_data["trades"] = trades_data
        except Exception as e:
            print(f"Warning: Failed to extract trade data: {str(e)}")

        # Generate report file
        if output_format == "json":
            # JSON report
            report_path = os.path.join(self.output_dir, f"trade_report_{run_id}_{timestamp}.json")
            with open(report_path, "w") as f:
                json.dump(report_data, f, indent=2)
        elif output_format == "md":
            # Markdown report
            report_path = os.path.join(self.output_dir, f"trade_report_{run_id}_{timestamp}.md")
            markdown = self._generate_trade_markdown_report(report_data)
            with open(report_path, "w") as f:
                f.write(markdown)
        else:
            # HTML report (default)
            report_path = os.path.join(self.output_dir, f"trade_report_{run_id}_{timestamp}.html")
            html = self._generate_trade_html_report(report_data)
            with open(report_path, "w") as f:
                f.write(html)

        return report_path

    def _extract_config(self, run: Any) -> Dict[str, Any]:
        """Extract key configuration from a run."""
        config = {}

        try:
            # Extract important config values
            for key, value in run.config.items():
                # Skip internal values
                if key.startswith("_"):
                    continue

                # Extract value
                if hasattr(value, "value") and value.value is not None:
                    config[key] = value.value
                else:
                    config[key] = value
        except Exception as e:
            print(f"Warning: Failed to extract config for run {run.id}: {str(e)}")

        return config

    def _generate_metric_plots(self, metrics_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, str]:
        """Generate plots for metrics data and return as base64 images."""
        plots = {}

        # Create plots for each metric
        for metric, data in metrics_data.items():
            try:
                # Create comparison bar chart
                plt.figure(figsize=(10, 6))

                # Sort data by value (ascending or descending based on metric)
                # For metrics where higher is better (reward, accuracy, etc.), sort descending
                # For metrics where lower is better (loss, etc.), sort ascending
                if any(term in metric.lower() for term in ["loss", "error"]):
                    data.sort(key=lambda x: x["value"])
                else:
                    data.sort(key=lambda x: x["value"], reverse=True)

                # Extract values and labels
                values = [item["value"] for item in data]
                labels = [item["run_name"] for item in data]

                # Shorten long labels
                labels = [label[:20] + "..." if len(label) > 20 else label for label in labels]

                # Create bar chart
                bars = plt.bar(labels, values)

                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width() / 2.,
                        height,
                        f"{height:.4f}",
                        ha="center", va="bottom", rotation=0
                    )

                plt.title(f"{metric} Comparison")
                plt.ylabel(metric)
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()

                # Save to memory
                import io
                from base64 import b64encode

                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)

                # Convert to base64
                img_str = b64encode(buf.read()).decode("ascii")
                plots[metric] = img_str

                plt.close()
            except Exception as e:
                print(f"Warning: Failed to generate plot for {metric}: {str(e)}")

        return plots

    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML report from data."""
        # Create HTML template
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{report_data["title"]}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .metric-card {{
            margin-bottom: 30px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .metric-card h3 {{
            margin-top: 0;
        }}
        .plot-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .run-card {{
            margin-bottom: 30px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .config-table {{
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>{report_data["title"]}</h1>
    <p>
        <strong>Project:</strong> {report_data["project"]}
        {f'<br><strong>Entity:</strong> {report_data["entity"]}' if report_data["entity"] else ''}
        <br><strong>Generated:</strong> {report_data["timestamp"]}
    </p>

    <h2>Metrics Comparison</h2>
"""

        # Add plots
        for metric, plot_data in report_data["plots"].items():
            html += f"""
    <div class="metric-card">
        <h3>{metric}</h3>
        <div class="plot-container">
            <img src="data:image/png;base64,{plot_data}" alt="{metric} chart">
        </div>
    </div>
"""

        # Create metrics table
        html += """
    <h2>Metrics Summary</h2>
    <table>
        <tr>
            <th>Run</th>
"""

        # Add metric columns
        for metric in report_data["metrics"].keys():
            html += f"""            <th>{metric}</th>
"""

        html += """        </tr>
"""

        # Add run rows
        for run in report_data["runs"]:
            html += f"""        <tr>
            <td><strong>{run["name"]}</strong> ({run["id"]})</td>
"""

            # Add metric values
            for metric in report_data["metrics"].keys():
                value = run["metrics"].get(metric, "N/A")
                if isinstance(value, (int, float)):
                    html += f"""            <td>{value:.4f}</td>
"""
                else:
                    html += f"""            <td>{value}</td>
"""

            html += """        </tr>
"""

        html += """    </table>

    <h2>Run Details</h2>
"""

        # Add run details
        for run in report_data["runs"]:
            html += f"""
    <div class="run-card">
        <h3>{run["name"]}</h3>
        <p>
            <strong>ID:</strong> {run["id"]}<br>
            <strong>Created:</strong> {run["created_at"]}<br>
            <strong>State:</strong> {run["state"]}
        </p>

        <h4>Configuration</h4>
        <table class="config-table">
            <tr>
                <th>Parameter</th>
                <th>Value</th>
            </tr>
"""

            # Add config values
            for key, value in run["config"].items():
                html += f"""            <tr>
                <td>{key}</td>
                <td>{value}</td>
            </tr>
"""

            html += """        </table>
    </div>
"""

        # Close HTML
        html += """
</body>
</html>
"""

        return html

    def _generate_markdown_report(self, report_data: Dict[str, Any]) -> str:
        """Generate Markdown report from data."""
        # Create Markdown template
        md = f"""# {report_data["title"]}

**Project:** {report_data["project"]}  
{f'**Entity:** {report_data["entity"]}  ' if report_data["entity"] else ''}
**Generated:** {report_data["timestamp"]}

## Metrics Comparison

"""

        # Metrics summary table
        md += "| Run |"

        # Add metric columns
        for metric in report_data["metrics"].keys():
            md += f" {metric} |"

        md += "\n|---|"

        # Add separator row
        for _ in report_data["metrics"].keys():
            md += "---|"

        md += "\n"

        # Add run rows
        for run in report_data["runs"]:
            md += f"| **{run['name']}** ({run['id']}) |"

            # Add metric values
            for metric in report_data["metrics"].keys():
                value = run["metrics"].get(metric, "N/A")
                if isinstance(value, (int, float)):
                    md += f" {value:.4f} |"
                else:
                    md += f" {value} |"

            md += "\n"

        md += "\n## Run Details\n"

        # Add run details
        for run in report_data["runs"]:
            md += f"""
### {run["name"]}

**ID:** {run["id"]}  
**Created:** {run["created_at"]}  
**State:** {run["state"]}

#### Configuration

| Parameter | Value |
|---|---|
"""

            # Add config values
            for key, value in run["config"].items():
                md += f"| {key} | {value} |\n"

            md += "\n"

        return md

    def _generate_trade_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate trade analysis HTML report from data."""
        run = report_data["run"]

        # Create HTML template
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{report_data["title"]}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .metric-card {{
            margin-bottom: 30px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .positive {{
            color: green;
        }}
        .negative {{
            color: red;
        }}
        .summary-stats {{
            display: flex;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }}
        .stat-card {{
            flex: 1;
            min-width: 200px;
            margin: 10px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 4px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            margin: 10px 0;
        }}
        .stat-label {{
            font-size: 14px;
            color: #666;
        }}
    </style>
</head>
<body>
    <h1>{report_data["title"]}</h1>
    <p>
        <strong>Project:</strong> {report_data["project"]}
        {f'<br><strong>Entity:</strong> {report_data["entity"]}' if report_data["entity"] else ''}
        <br><strong>Run:</strong> {run["name"]} ({run["id"]})
        <br><strong>Generated:</strong> {report_data["timestamp"]}
    </p>

    <h2>Trade Performance Summary</h2>

    <div class="summary-stats">
"""

        # Add summary statistics
        trade_metrics = report_data["trade_metrics"]

        # Define key metrics to display
        key_metrics = [
            {"key": "trades/total", "label": "Total Trades", "format": "int"},
            {"key": "trades/accuracy", "label": "Win Rate", "format": "percent"},
            {"key": "trades/profit_factor", "label": "Profit Factor", "format": "float"},
            {"key": "trades/avg_win", "label": "Avg. Win", "format": "dollar"},
            {"key": "trades/avg_loss", "label": "Avg. Loss", "format": "dollar"},
            {"key": "total_pnl", "label": "Total P&L", "format": "dollar"}
        ]

        for metric in key_metrics:
            value = None

            # Look for the metric in different variations
            for k in trade_metrics.keys():
                if metric["key"] in k or k in metric["key"]:
                    value = trade_metrics[k]
                    break

            if value is None:
                continue

            # Format value
            if metric["format"] == "percent":
                formatted_value = f"{value:.2%}"
                css_class = "positive" if value > 0.5 else "negative"
            elif metric["format"] == "dollar":
                formatted_value = f"${value:.2f}"
                css_class = "positive" if value > 0 else "negative"
            elif metric["format"] == "int":
                formatted_value = f"{int(value)}"
                css_class = ""
            else:  # float
                formatted_value = f"{value:.2f}"
                css_class = "positive" if value > 1 else "negative"

            html += f"""
        <div class="stat-card">
            <div class="stat-label">{metric["label"]}</div>
            <div class="stat-value {css_class}">{formatted_value}</div>
        </div>
"""

        html += """
    </div>

    <h2>Trade Details</h2>
"""

        # Add trade details if available
        trades = report_data["trades"]
        if trades:
            html += """
    <table>
        <tr>
"""

            # Add column headers
            columns = list(trades[0].keys())
            for col in columns:
                html += f"""            <th>{col}</th>
"""

            html += """        </tr>
"""

            # Add trade rows
            for trade in trades:
                html += """        <tr>
"""

                for col in columns:
                    value = trade.get(col, "")

                    # Format value
                    if col == "realized_pnl":
                        css_class = "positive" if value > 0 else "negative"
                        html += f"""            <td class="{css_class}">${value:.2f}</td>
"""
                    else:
                        html += f"""            <td>{value}</td>
"""

                html += """        </tr>
"""

            html += """    </table>
"""
        else:
            html += """
    <p>No trade details available for this run.</p>
"""

        # Add run configuration
        html += """
    <h2>Run Configuration</h2>
    <table>
        <tr>
            <th>Parameter</th>
            <th>Value</th>
        </tr>
"""

        # Add config values
        for key, value in run["config"].items():
            html += f"""        <tr>
            <td>{key}</td>
            <td>{value}</td>
        </tr>
"""

        html += """    </table>
</body>
</html>
"""

        return html

    def _generate_trade_markdown_report(self, report_data: Dict[str, Any]) -> str:
        """Generate trade analysis Markdown report from data."""
        run = report_data["run"]

        # Create Markdown template
        md = f"""# {report_data["title"]}

**Project:** {report_data["project"]}  
{f'**Entity:** {report_data["entity"]}  ' if report_data["entity"] else ''}
**Run:** {run["name"]} ({run["id"]})  
**Generated:** {report_data["timestamp"]}

## Trade Performance Summary

"""

        # Add summary statistics
        trade_metrics = report_data["trade_metrics"]

        # Define key metrics to display
        key_metrics = [
            {"key": "trades/total", "label": "Total Trades", "format": "int"},
            {"key": "trades/accuracy", "label": "Win Rate", "format": "percent"},
            {"key": "trades/profit_factor", "label": "Profit Factor", "format": "float"},
            {"key": "trades/avg_win", "label": "Avg. Win", "format": "dollar"},
            {"key": "trades/avg_loss", "label": "Avg. Loss", "format": "dollar"},
            {"key": "total_pnl", "label": "Total P&L", "format": "dollar"}
        ]

        md += "| Metric | Value |\n|---|---|\n"

        for metric in key_metrics:
            value = None

            # Look for the metric in different variations
            for k in trade_metrics.keys():
                if metric["key"] in k or k in metric["key"]:
                    value = trade_metrics[k]
                    break

            if value is None:
                continue

            # Format value
            if metric["format"] == "percent":
                formatted_value = f"{value:.2%}"
            elif metric["format"] == "dollar":
                formatted_value = f"${value:.2f}"
            elif metric["format"] == "int":
                formatted_value = f"{int(value)}"
            else:  # float
                formatted_value = f"{value:.2f}"

            md += f"| {metric['label']} | {formatted_value} |\n"

        md += "\n## Trade Details\n\n"

        # Add trade details if available
        trades = report_data["trades"]
        if trades:
            # Add column headers
            columns = list(trades[0].keys())
            md += "| " + " | ".join(columns) + " |\n"
            md += "|" + "---|" * len(columns) + "\n"

            # Add trade rows
            for trade in trades:
                row = []
                for col in columns:
                    value = trade.get(col, "")

                    # Format value
                    if col == "realized_pnl":
                        row.append(f"${value:.2f}")
                    else:
                        row.append(str(value))

                md += "| " + " | ".join(row) + " |\n"
        else:
            md += "No trade details available for this run.\n"

        md += "\n## Run Configuration\n\n"

        # Add config values
        md += "| Parameter | Value |\n|---|---|\n"
        for key, value in run["config"].items():
            md += f"| {key} | {value} |\n"

        return md


def main():
    parser = argparse.ArgumentParser(description="Generate W&B reports")
    parser.add_argument("--project", type=str, required=True, help="W&B project name")
    parser.add_argument("--entity", type=str, help="W&B entity/username (optional)")
    parser.add_argument("--output-dir", type=str, default="reports", help="Output directory")
    parser.add_argument("--report-type", type=str, choices=["performance", "trade"], default="performance",
                        help="Type of report to generate")
    parser.add_argument("--run-ids", type=str, nargs="*", help="Specific run IDs to include")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of runs to include")
    parser.add_argument("--format", type=str, choices=["html", "md", "json"], default="html",
                        help="Output format")

    # Add arguments for trade report
    parser.add_argument("--run-id", type=str, help="Run ID for trade analysis report")

    args = parser.parse_args()

    # Create report generator
    generator = ReportGenerator(
        project=args.project,
        entity=args.entity,
        output_dir=args.output_dir
    )

    if args.report_type == "performance":
        # Generate performance report
        try:
            report_path = generator.generate_performance_report(
                run_ids=args.run_ids,
                limit=args.limit,
                output_format=args.format
            )
            print(f"Performance report generated: {report_path}")
        except Exception as e:
            print(f"Error generating performance report: {str(e)}")
            return 1
    else:
        # Generate trade analysis report
        if not args.run_id and not args.run_ids:
            print("Error: run-id or run-ids required for trade analysis report")
            return 1

        try:
            run_id = args.run_id or args.run_ids[0]
            report_path = generator.generate_trade_analysis_report(
                run_id=run_id,
                output_format=args.format
            )
            print(f"Trade analysis report generated: {report_path}")
        except Exception as e:
            print(f"Error generating trade analysis report: {str(e)}")
            return 1

    return 0


if __name__ == "__main__":
    main()