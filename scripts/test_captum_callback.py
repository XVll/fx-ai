#!/usr/bin/env python3
"""Test Captum callback creation."""

import sys
import os
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

from config.loader import load_config
from agent.callbacks import create_callback_manager

# Load config with quick override  
config = load_config('quick')

# Convert to dict like main.py does
config_dict = {
    "wandb": config.wandb.__dict__ if hasattr(config.wandb, "__dict__") else config.wandb,
    "dashboard": config.dashboard.__dict__ if hasattr(config.dashboard, "__dict__") else config.dashboard,
    "optuna_trial": getattr(config, "optuna_trial", None),
    "callbacks": getattr(config, "callbacks", []),
    "captum": config.captum.model_dump()
    if hasattr(config, "captum") and config.captum and hasattr(config.captum, "model_dump")
    else getattr(config, "captum", None),
}

print(f"Config dict captum: {config_dict.get('captum')}")
print(f"Captum enabled: {config_dict.get('captum', {}).get('enabled')}")

# Create callback manager
callback_manager = create_callback_manager(config_dict)

# Check callbacks
print(f"\nCreated callbacks:")
for cb in callback_manager.callbacks:
    print(f"  - {cb.__class__.__name__} (enabled={cb.enabled})")

# Check specifically for Captum
captum_found = False
for cb in callback_manager.callbacks:
    if cb.__class__.__name__ == "CaptumCallback":
        captum_found = True
        print(f"\nCaptum callback found!")
        print(f"  analyze_every_n_episodes: {cb.analyze_every_n_episodes}")
        print(f"  analyze_every_n_updates: {cb.analyze_every_n_updates}")
        break

if not captum_found:
    print("\nNo Captum callback found!")