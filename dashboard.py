# dashboard.py
import os
import argparse
import streamlit.web.bootstrap as bootstrap
from visualization.dashboard import TradingDashboard


def parse_args():
    parser = argparse.ArgumentParser(description="Launch the AI Trading Dashboard")
    parser.add_argument("--project", type=str, default=os.environ.get("WANDB_PROJECT", "fx-ai"),
                        help="W&B project name")
    parser.add_argument("--entity", type=str, default=os.environ.get("WANDB_ENTITY", None),
                        help="W&B entity/username")
    parser.add_argument("--port", type=int, default=8501,
                        help="Port to run the dashboard on")
    return parser.parse_args()


def main():
    args = parse_args()

    # Set environment variables for W&B
    os.environ["WANDB_PROJECT"] = args.project
    if args.entity:
        os.environ["WANDB_ENTITY"] = args.entity

    # Path to the dashboard script
    dashboard_path = os.path.join(os.path.dirname(__file__), "visualization", "dashboard.py")

    # Launch Streamlit app
    bootstrap.run(dashboard_path, "", args=[], flag_options={
        "server.port": args.port,
        "server.headless": True,
        "browser.serverAddress": "localhost"
    })


if __name__ == "__main__":
    main()