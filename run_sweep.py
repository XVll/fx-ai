import argparse
import subprocess
import sys

import wandb
import yaml
import os

def main():
    sys.argv.extend(["sweep_config_file=sweep_config.yaml"])
    sys.argv.extend(["project=fx-ai"])
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="fx-ai", help="WandB project name")
    parser.add_argument("--count", type=int, default=10, help="Number of runs")
    # We will always use '++' for Hydra overrides, so --use-plus is removed.
    args = parser.parse_args()

    # Load the sweep config from YAML
    try:
        with open('sweep_config.yaml', 'r') as f:
            sweep_config = yaml.safe_load(f)
    except FileNotFoundError:
        print("ERROR: sweep_config.yaml not found. Please ensure it's in the same directory.")
        return
    except yaml.YAMLError as e:
        print(f"ERROR: Could not parse sweep_config.yaml: {e}")
        return

    # Modify parameter keys to include '++' for Hydra override
    # This ensures that when wandb agent substitutes them with ${args_no_hyphens},
    # they will be in the format ++param=value.
    if 'parameters' in sweep_config and sweep_config['parameters'] is not None:
        new_parameters = {}
        for key, value in sweep_config['parameters'].items():
            if not key.startswith("++"): # Avoid double-prefixing if already done
                new_parameters[f"++{key}"] = value
            else:
                new_parameters[key] = value
        sweep_config['parameters'] = new_parameters
    else:
        print("WARNING: No 'parameters' section found in sweep_config.yaml or it is empty.")
        # Allow proceeding if the sweep has no tunable parameters (unlikely for a sweep)

    # Ensure the command structure uses args_no_hyphens
    # This tells wandb to format arguments as key=value.
    # Since our keys are now "++param", it will become "++param=value".
    sweep_config['command'] = [
        "${env}",        # Passes environment variables (e.g., CUDA_VISIBLE_DEVICES)
        "python",        # The Python interpreter
        "${program}",    # Uses 'program' from sweep_config (e.g., main.py)
        "${args_no_hyphens}"  # Wandb will substitute parameters here
    ]
    # If your main.py needs other fixed Hydra args, you can add them to the list:
    # e.g., "hydra.run.dir=." , "hydra.output_subdir=null"

    # Determine entity for the sweep URL and agent command
    try:
        api = wandb.Api()
        entity = os.environ.get('WANDB_ENTITY') or api.default_entity
        if not entity: # Fallback if API default_entity is also None
            # Attempt to parse from a previous run or use a known default
            # For this example, assuming 'onur03-fx' if not found.
            # You might need to set WANDB_ENTITY environment variable
            # or login to wandb CLI for api.default_entity to work reliably.
            print("Warning: WANDB_ENTITY not found. Trying to use a default or parsed value.")
            # A common pattern is that the project implies the entity.
            # If args.project is "username/projectname", then entity is "username"
            if '/' in args.project:
                entity = args.project.split('/')[0]
            else: # Fallback, replace with your actual entity if this fails
                entity = "onur03-fx"
                print(f"Falling back to entity: {entity}. Please set WANDB_ENTITY if this is incorrect.")

    except wandb.errors.CommError as e:
        print(f"Error communicating with W&B API to get default entity: {e}")
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