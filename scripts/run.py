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

    # Determine which script to use (main.py or main_continuous.py)
    script_path = "main_continuous.py" if os.path.exists("main_continuous.py") else "main.py"

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
    cmd = ["python", script_path]

    # Add Hydra args
    # Basic configs
    if args.quick_test:
        cmd.append("quick_test=true")

    # Continuous training
    if args.continue_training:
        cmd.append("training=continuous")
        cmd.append("training.enabled=true")
        cmd.append("training.load_best_model=true")
        cmd.append(f"training.best_models_dir={args.models_dir}")

    # Symbol and dates
    if args.symbol:
        cmd.append(f"data.symbol={args.symbol}")

    if args.start_date:
        cmd.append(f"data.start_date={args.start_date}")

    if args.end_date:
        cmd.append(f"data.end_date={args.end_date}")

    # Run the command
    logger.info(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)


def run_backtest(args, script_path):
    """Run backtest with the specified arguments."""
    cmd = ["python", script_path]

    # Add standard Hydra overrides
    cmd.append("env.training_mode=backtesting")
    cmd.append(f"data.symbol={args.symbol}")
    cmd.append(f"data.start_date={args.start_date}")
    cmd.append(f"data.end_date={args.end_date}")

    if args.model_path:
        cmd.append(f"eval.model_path={args.model_path}")

    # Run the command
    logger.info(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)


def run_sweep(args):
    """Run hyperparameter sweep with the specified arguments."""
    cmd = ["python", "run_sweep.py", f"--project={args.project}", f"--count={args.count}"]

    # Run the command
    logger.info(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()