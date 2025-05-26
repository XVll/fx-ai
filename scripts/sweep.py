import argparse
import subprocess
import sys
import json
from pathlib import Path

import wandb
import yaml
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="default.yaml", help="Sweep config file name")
    parser.add_argument("--project", type=str, default="fx-ai", help="WandB project name")
    parser.add_argument("--count", type=int, default=10, help="Number of runs")
    args = parser.parse_args()

    # Construct the path to the sweep config file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "config", "sweep", args.config)
    
    # Load the sweep config from YAML
    try:
        with open(config_path, 'r') as f:
            sweep_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"ERROR: {config_path} not found.")
        print("Available sweep configs should be in config/sweep/ directory.")
        return
    except yaml.YAMLError as e:
        print(f"ERROR: Could not parse {config_path}: {e}")
        return

    # Update sweep config to use our wrapper script
    sweep_config['program'] = str(Path(__file__).parent / "sweep_wrapper.py")
    
    # Ensure the command structure uses args_no_hyphens for clean parameter passing
    sweep_config['command'] = [
        "${env}",             # Passes environment variables (e.g., CUDA_VISIBLE_DEVICES)
        "${interpreter}",     # The Python interpreter
        "${program}",         # Our sweep_wrapper.py script
        "${args_no_hyphens}"  # W&B will substitute parameters here as key=value
    ]
    
    # Parameters should be in dot notation (e.g., "training.learning_rate")
    if 'parameters' in sweep_config and sweep_config['parameters'] is not None:
        # Parameters are already in the correct format
        print(f"Found {len(sweep_config['parameters'])} parameters to sweep")
    else:
        print("WARNING: No 'parameters' section found in sweep config or it is empty.")

    # Determine entity for the sweep URL and agent command
    try:
        api = wandb.Api()
        entity = os.environ.get('WANDB_ENTITY') or api.default_entity
        if not entity: # Fallback if API default_entity is also None
            # Attempt to parse from a previous run or use a known default.yaml
            # For this example, assuming 'onur03-fx' if not found.
            # You might need to set WANDB_ENTITY environment variable
            # or login to wandb CLI for api.default_entity to work reliably.
            print("Warning: WANDB_ENTITY not found. Trying to use a default.yaml or parsed value.")
            # A common pattern is that the project implies the entity.
            # If args.project is "username/projectname", then entity is "username"
            if '/' in args.project:
                entity = args.project.split('/')[0]
            else: # Fallback, replace with your actual entity if this fails
                entity = "onur03-fx"
                print(f"Falling back to entity: {entity}. Please set WANDB_ENTITY if this is incorrect.")

    except wandb.errors.CommError as e:
        print(f"Error communicating with W&B API to get default.yaml entity: {e}")
        print("Please ensure you are logged in to W&B ('wandb login'). Using 'onur03-fx' as a fallback entity.")
        entity = "onur03-fx" # Fallback entity

    # Create the sweep
    print(f"Creating sweep with config: {sweep_config}")
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=args.project,
        entity=entity # Explicitly provide entity
    )
    print(f"Created sweep with ID: {sweep_id}")
    print(f"Sweep URL: https://wandb.ai/{entity}/{args.project}/sweeps/{sweep_id}")

    # Run the agent
    agent_command_path = f"{entity}/{args.project}/{sweep_id}"
    print(f"Starting wandb agent for sweep: {agent_command_path} with count {args.count}")
    try:
        subprocess.run(["wandb", "agent", agent_command_path, "--count", str(args.count)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Wandb agent failed with error: {e}")
    except KeyboardInterrupt:
        print("\nWandb agent interrupted by user.")
    finally:
        print("Sweep agent process finished.")
        # No wandb.finish() here, as this script only creates the sweep and launches the agent.
        # The agent and its child processes (main.py runs) manage their own W&B lifecycles.

if __name__ == "__main__":
    main()