"""
Configuration loader with validation and usage tracking
"""
import os
from pathlib import Path
from typing import Optional, Set, Any
import yaml
import logging
from datetime import datetime

from .schemas import Config


class ConfigLoader:
    """Handles config loading, validation, and usage tracking"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._used_paths: Set[str] = set()
        self._config: Optional[Config] = None
        
    def load(self, override_file: Optional[str] = None) -> Config:
        """Load configuration with optional overrides"""
        # Start with defaults
        config = Config()
        
        # Apply overrides if specified
        if override_file:
            override_path = Path(override_file)
            
            # Check in multiple locations
            if not override_path.exists():
                # Try in config/overrides directory
                override_path = Path("config/overrides") / override_file
                if not override_path.exists():
                    # Try with .yaml extension
                    override_path = Path("config/overrides") / f"{override_file}.yaml"
            
            if override_path.exists():
                self.logger.info(f"Loading config overrides from: {override_path}")
                with open(override_path) as f:
                    overrides = yaml.safe_load(f)
                
                # Validate and apply overrides
                try:
                    # Convert the config to dict, apply overrides, then recreate
                    config_dict = config.model_dump()
                    self._deep_update(config_dict, overrides)
                    config = Config(**config_dict)
                    self.logger.info("Config overrides applied successfully")
                except Exception as e:
                    self.logger.error(f"Failed to apply config overrides: {e}")
                    raise ValueError(f"Invalid config overrides: {e}")
            else:
                raise FileNotFoundError(f"Config override file not found: {override_file}")
        
        # Validate final config
        self.logger.debug(f"Config type before validation: {type(config)}")
        self.logger.debug(f"Config.model type: {type(config.model) if hasattr(config, 'model') else 'No model attr'}")
        self._validate_config(config)
        
        # Store for usage tracking
        self._config = config
        
        # Save the complete config for reproducibility
        self._save_used_config(config)
        
        return config
    
    def _validate_config(self, config: Config):
        """Perform additional validation beyond Pydantic"""
        # Check action space consistency
        try:
            if hasattr(config.model, 'action_dim'):
                action_types, position_sizes = config.model.action_dim
                expected_actions = action_types * position_sizes
                self.logger.info(f"Action space: {action_types} types × {position_sizes} sizes = {expected_actions} total actions")
        except Exception as e:
            self.logger.warning(f"Could not validate action space: {e}")
        
        # Validate reward components
        reward_v2_dict = config.env.reward_v2.model_dump()
        enabled_components = []
        for name, value in reward_v2_dict.items():
            if isinstance(value, dict) and value.get('enabled', True):
                enabled_components.append(name)
            elif name not in ['scale_factor', 'clip_range']:  # Skip non-component fields
                # For simple component configs like RewardComponentConfig
                if hasattr(config.env.reward_v2, name):
                    comp = getattr(config.env.reward_v2, name)
                    if hasattr(comp, 'enabled') and comp.enabled:
                        enabled_components.append(name)
        self.logger.info(f"Enabled reward components: {', '.join(enabled_components)}")
        
        # Check for deprecated configs
        if hasattr(config.env, 'reward'):
            self.logger.warning("Deprecated 'reward' config found - use 'reward_v2' instead")
    
    def _save_used_config(self, config: Config):
        """Save the complete config used for this run"""
        # Create output directory
        output_dir = Path("outputs/configs")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"config_{config.experiment_name}_{timestamp}.yaml"
        filepath = output_dir / filename
        
        # Save config
        with open(filepath, 'w') as f:
            yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)
        
        self.logger.info(f"Saved used config to: {filepath}")
    
    def track_usage(self, path: str):
        """Track config path usage for detecting unused configs"""
        self._used_paths.add(path)
    
    def get_unused_configs(self) -> Set[str]:
        """Get list of config paths that were never accessed"""
        if not self._config:
            return set()
        
        # Get all possible paths from config
        all_paths = self._get_all_paths(self._config.model_dump())
        
        # Find unused
        unused = all_paths - self._used_paths
        return unused
    
    def _get_all_paths(self, d: dict, prefix: str = "") -> Set[str]:
        """Recursively get all config paths"""
        paths = set()
        
        for key, value in d.items():
            current_path = f"{prefix}.{key}" if prefix else key
            paths.add(current_path)
            
            if isinstance(value, dict):
                paths.update(self._get_all_paths(value, current_path))
        
        return paths
    
    def _deep_update(self, base_dict: dict, update_dict: dict):
        """Deep update base_dict with update_dict"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value


# Global config instance
_config_loader = ConfigLoader()


def load_config(override_file: Optional[str] = None) -> Config:
    """Load configuration with optional overrides
    
    Args:
        override_file: Path to override YAML file or name in config/overrides/
        
    Returns:
        Validated Config object
    """
    return _config_loader.load(override_file)


def get_config_value(path: str, default: Any = None) -> Any:
    """Get config value with usage tracking
    
    Args:
        path: Dot-separated path (e.g., 'model.d_model')
        default: Default value if path not found
        
    Returns:
        Config value
    """
    _config_loader.track_usage(path)
    
    # Navigate config
    parts = path.split('.')
    value = _config_loader._config
    
    try:
        for part in parts:
            value = getattr(value, part)
        return value
    except (AttributeError, TypeError):
        if default is not None:
            return default
        raise KeyError(f"Config path not found: {path}")


def check_unused_configs():
    """Log warning about unused configuration parameters"""
    unused = _config_loader.get_unused_configs()
    if unused:
        logger = logging.getLogger(__name__)
        logger.warning(f"Unused config parameters detected: {sorted(unused)}")
        logger.warning("Consider removing these from your config to reduce complexity")