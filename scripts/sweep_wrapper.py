#!/usr/bin/env python3
"""
Wrapper script for W&B sweeps that converts sweep parameters to Pydantic config overrides.
This script is called by W&B agent with parameters and creates a temporary config file.
"""

import os
import sys
import yaml
import json
import tempfile
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.loader import load_config
from config.schemas import Config


def parse_wandb_args():
    """Parse arguments passed by W&B sweep agent."""
    # W&B passes parameters as command line arguments in the format: param=value
    params = {}
    for arg in sys.argv[1:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            # Convert string values to appropriate types
            try:
                # Try to parse as JSON first (handles lists, dicts, bools, numbers)
                value = json.loads(value)
            except json.JSONDecodeError:
                # Keep as string if not valid JSON
                pass
            
            # Handle nested parameters (e.g., training.learning_rate)
            keys = key.split('.')
            current = params
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
    
    return params


def create_override_config(params):
    """Create a temporary override config file for the sweep run."""
    # Create a unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = os.environ.get('WANDB_RUN_ID', 'unknown')
    filename = f"sweep_{run_id}_{timestamp}.yaml"
    filepath = Path("config/overrides") / filename
    
    # Ensure directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Write parameters to YAML file
    with open(filepath, 'w') as f:
        yaml.dump(params, f, default_flow_style=False)
    
    return filepath


def main():
    """Main entry point for sweep wrapper."""
    # Parse W&B parameters
    params = parse_wandb_args()
    print(f"Received sweep parameters: {params}")
    
    # Create override config
    override_path = create_override_config(params)
    print(f"Created override config: {override_path}")
    
    try:
        # Import and run main training function
        from main import train, load_config
        
        # Load config with overrides
        config = load_config(override_path.stem)
        
        # Run training
        train(config)
        
    finally:
        # Clean up temporary config file
        if override_path.exists():
            override_path.unlink()
            print(f"Cleaned up override config: {override_path}")


if __name__ == "__main__":
    main()