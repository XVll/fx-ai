#!/usr/bin/env python3
"""
Easy launcher for Trading AI training and testing.
This script provides a simple interface for running continuous training
or other operations via the command line or IDE.

Example usage:
- Continue training from previous best model:
  python run.py train --continue-training --symbol MLGO

- Quick test new architecture:
  python run.py train --quick-test --symbol MLGO

- Run hyperparameter sweep:
  python run.py sweep

- Test model performance:
  python run.py backtest --symbol MLGO
"""

import os
import sys
import argparse
import subprocess
import logging
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("run")


def main():
    """Parse arguments and run the appropriate command."""
    parser = argparse.ArgumentParser(description="Trading AI Runner")

    # Top-level commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--continue-training", action="store_true",
                             help="Continue training from best model")
    train_parser.add_argument("--symbol", type=str, default="MLGO",
                             help="Symbol to train on")
    train_parser.add_argument("--start-date", type=str,
                             help="Start date (YYYY-MM-DD)")
    train_parser.add_argument("--end-date", type=str,
                             help="End date (YYYY-MM-DD)")
    train_parser.add_argument("--days", type=int, default=1,
                             help="Number of trading days to use")
    train_parser.add_argument("--quick-test", action="store_true",
                             help="Use quick test configuration")
    train_parser.add_argument("--models-dir", type=str, default="./best_models",
                             help="Directory for best models")

    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Backtest a model")
    backtest_parser.add_argument("--symbol", type=str, required=True,
                                help="Symbol to backtest on")
    backtest_parser.add_argument("--model-path", type=str,
                                help="Path to model file")
    backtest_parser.add_argument("--start-date", type=str, required=True,
                                help="Start date (YYYY-MM-DD)")
    backtest_parser.add_argument("--end-date", type=str, required=True,
                                help="End date (YYYY-MM-DD)")

    # Sweep command
    sweep_parser = subparsers.add_parser("sweep", help="Run hyperparameter sweep")
    sweep_parser.add_argument("--count", type=int, default=20,
                             help="Number of sweep runs")
    sweep_parser.add_argument("--project", type=str, default="fx-ai",
                             help="W&B project name")

    # Parse arguments
    args = parser.parse_args()

    # Exit if no command specified
    if not args.command:
        parser.print_help()
        return

    # Determine which script to use
    script_path = "main.py" if os.path.exists("main.py") else "main.py"

    # Run the appropriate command
    if args.command == "train":
        run_training(args, script_path)
    elif args.command == "backtest":
        run_backtest(args, script_path)
    elif args.command == "sweep":
        run_sweep(args)
    else:
        parser.print_help()


def run_training(args, script_path):
    """Run training with the specified arguments."""
    cmd = [sys.executable, script_path]

    # Add config override if specified
    if args.quick_test:
        cmd.extend(["--config", "quick_test"])

    # Symbol override
    if args.symbol:
        cmd.extend(["--symbol", args.symbol])

    # Continue training flag
    if args.continue_training:
        cmd.append("--continue")

    # Date overrides (if needed, we'll need to add these to main.py argparse)
    # For now, we'll use environment variables or config files
    
    # Run the command
    logger.info(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)


def run_backtest(args, script_path):
    """Run backtest with the specified arguments."""
    # For backtesting, we'll need a separate backtest script or mode
    # For now, let's create a simple error message
    logger.error("Backtest functionality needs to be implemented separately from main.py")
    logger.info("Please create a backtest.py script that uses the Pydantic config system")
    return


def run_sweep(args):
    """Run hyperparameter sweep with the specified arguments."""
    sweep_script = Path(__file__).parent / "sweep.py"
    cmd = [sys.executable, str(sweep_script), f"--project={args.project}", f"--count={args.count}"]

    # Run the command
    logger.info(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
