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
    train_parser.add_argument("--quick-test", action="store_true",
                             help="Use quick test configuration")
    train_parser.add_argument("--momentum", action="store_true",
                             help="Use momentum training configuration")
    train_parser.add_argument("--config", type=str,
                             help="Configuration override file")
    train_parser.add_argument("--experiment", type=str,
                             help="Experiment name")
    train_parser.add_argument("--device", type=str, choices=["cuda", "cpu", "mps"],
                             help="Device to use for training")

    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Backtest a model")
    backtest_parser.add_argument("--symbol", type=str, required=True,
                                help="Symbol to backtest on")
    backtest_parser.add_argument("--model-path", type=str,
                                help="Path to model file")
    backtest_parser.add_argument("--date", type=str,
                                help="Date to backtest (YYYY-MM-DD)")

    # Sweep command
    sweep_parser = subparsers.add_parser("sweep", help="Run hyperparameter sweep")
    sweep_parser.add_argument("--count", type=int, default=20,
                             help="Number of sweep runs")
    sweep_parser.add_argument("--config", type=str, default="default.yaml",
                             help="Sweep configuration file")
    
    # Scan command for momentum days
    scan_parser = subparsers.add_parser("scan", help="Scan for momentum days")
    scan_parser.add_argument("--symbol", type=str, required=True,
                            help="Symbol to scan")
    scan_parser.add_argument("--min-quality", type=float, default=0.5,
                            help="Minimum quality score")

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
    elif args.command == "scan":
        run_scan(args)
    else:
        parser.print_help()


def run_training(args, script_path):
    """Run training with the specified arguments."""
    # Determine which script to use - prefer main.py for momentum training
    if args.momentum or args.config == "momentum_training":
        cmd = [sys.executable, "main.py"]
        
        # Add momentum training config
        cmd.extend(["--config", "momentum_training"])
        
        # Symbol override
        if args.symbol:
            cmd.extend(["--symbol", args.symbol])
            
        # Continue training flag
        if args.continue_training:
            cmd.append("--continue")
            
        # Device override
        if args.device:
            cmd.extend(["--device", args.device])
            
    else:
        # Use legacy scripts/run.py approach
        cmd = [sys.executable, script_path]

        # Add config override if specified
        if args.quick_test:
            cmd.extend(["--config", "quick_test"])
        elif args.config:
            cmd.extend(["--config", args.config])

        # Symbol override
        if args.symbol:
            cmd.extend(["--symbol", args.symbol])

        # Continue training flag
        if args.continue_training:
            cmd.append("--continue")
            
        # Experiment name
        if args.experiment:
            cmd.extend(["--experiment", args.experiment])
            
        # Device override
        if args.device:
            cmd.extend(["--device", args.device])
    
    # Run the command
    logger.info(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        logger.error(f"Training failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def run_backtest(args, script_path):
    """Run backtest with the specified arguments."""
    logger.info("Backtest functionality is coming soon")
    logger.info("The new architecture uses momentum-based episode selection")
    logger.info("Backtesting will evaluate performance on specific momentum days")
    return


def run_sweep(args):
    """Run hyperparameter sweep with the specified arguments."""
    sweep_script = Path(__file__).parent / "sweep.py"
    cmd = [sys.executable, str(sweep_script), "--config", args.config, "--count", str(args.count)]

    # Run the command
    logger.info(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        logger.error(f"Sweep failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def run_scan(args):
    """Scan for momentum days."""
    scan_script = Path(__file__).parent / "scan_momentum_days.py"
    cmd = [sys.executable, str(scan_script), "--symbol", args.symbol, "--min-quality", str(args.min_quality)]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        logger.error(f"Scan failed with exit code {result.returncode}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()

