# run_sweep.py
import argparse
import subprocess
import wandb
import yaml
import os


def main():
    parser = argparse.ArgumentParser(description="Run a W&B hyperparameter sweep")
    parser.add_argument("--config", type=str, default="sweep_config.yaml", help="Path to sweep configuration file")
    parser.add_argument("--name", type=str, help="Sweep name (optional)")
    parser.add_argument("--project", type=str, help="W&B project name (overrides config)")
    parser.add_argument("--entity", type=str, help="W&B entity/username (overrides config)")
    parser.add_argument("--count", type=int, default=20, help="Number of runs to execute")
    args = parser.parse_args()

    # Load sweep configuration
    with open(args.config, "r") as f:
        sweep_config = yaml.safe_load(f)

    # Override with command line args if provided
    if args.project:
        sweep_config["project"] = args.project
    if args.name:
        sweep_config["name"] = args.name

    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project=sweep_config.get("project"))
    print(f"Created sweep with ID: {sweep_id}")

    # Launch sweep agent
    command = ["wandb", "agent", sweep_id]

    # Add entity if specified
    if args.entity:
        command.extend(["--entity", args.entity])

    # Add count parameter
    command.extend(["--count", str(args.count)])

    # Start the agent process
    print(f"Starting sweep agent with command: {' '.join(command)}")
    subprocess.run(command)


if __name__ == "__main__":
    main()