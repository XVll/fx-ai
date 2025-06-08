"""
Configuration management interfaces.

These interfaces enable flexible, hierarchical configuration
with validation, composition, and runtime updates.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, Protocol, runtime_checkable, Type
from pathlib import Path
import json

from ..types.common import RunMode, Configurable


@runtime_checkable
class IConfigSchema(Protocol):
    """Interface for configuration schemas.
    
    Design principles:
    - Define valid configuration structure
    - Enable validation
    - Support defaults and documentation
    """
    
    def get_schema(self) -> dict[str, Any]:
        """Get JSON schema definition.
        
        Returns:
            JSON schema dict
            
        Design notes:
        - Use standard JSON Schema format
        - Include descriptions
        - Define constraints
        """
        ...
    
    def get_defaults(self) -> dict[str, Any]:
        """Get default values.
        
        Returns:
            Default configuration
        """
        ...
    
    def validate(
        self,
        config: dict[str, Any]
    ) -> tuple[bool, list[str]]:
        """Validate configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        ...


class IConfigProvider(Configurable):
    """Interface for configuration providers.
    
    Design principles:
    - Load configuration from various sources
    - Support composition and overrides
    - Enable hot reloading
    """
    
    @abstractmethod
    def load(
        self,
        source: Union[str, Path, dict]
    ) -> dict[str, Any]:
        """Load configuration from source.
        
        Args:
            source: File path, URL, or dict
            
        Returns:
            Configuration dict
            
        Design notes:
        - Support multiple formats (YAML, JSON, TOML)
        - Handle includes/imports
        - Resolve variables
        """
        ...
    
    @abstractmethod
    def merge(
        self,
        configs: list[dict[str, Any]],
        strategy: str = "deep"
    ) -> dict[str, Any]:
        """Merge multiple configurations.
        
        Args:
            configs: List of configs (priority order)
            strategy: Merge strategy
            
        Returns:
            Merged configuration
            
        Design notes:
        - Later configs override earlier
        - Deep merge for nested dicts
        - Handle special keys (_delete, _append)
        """
        ...
    
    @abstractmethod
    def resolve_references(
        self,
        config: dict[str, Any],
        context: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """Resolve configuration references.
        
        Args:
            config: Configuration with references
            context: Resolution context
            
        Returns:
            Resolved configuration
            
        Design notes:
        - Support ${var} syntax
        - Handle environment variables
        - Enable cross-references
        """
        ...


class IConfigRegistry(Protocol):
    """Central registry for configurations.
    
    Design principles:
    - Organize configurations by type
    - Enable discovery and reuse
    - Support versioning
    """
    
    def register(
        self,
        name: str,
        config: dict[str, Any],
        category: str,
        version: str = "1.0"
    ) -> None:
        """Register configuration.
        
        Args:
            name: Config name
            config: Configuration dict
            category: Category (model, env, etc.)
            version: Config version
        """
        ...
    
    def get(
        self,
        name: str,
        category: Optional[str] = None,
        version: Optional[str] = None
    ) -> dict[str, Any]:
        """Get configuration.
        
        Args:
            name: Config name
            category: Filter by category
            version: Specific version
            
        Returns:
            Configuration dict
        """
        ...
    
    def list_configs(
        self,
        category: Optional[str] = None
    ) -> list[dict[str, str]]:
        """List available configurations.
        
        Args:
            category: Filter by category
            
        Returns:
            List of config metadata
        """
        ...


class IModeConfig(IConfigSchema):
    """Interface for mode-specific configuration.
    
    Design principles:
    - Each mode has its own config schema
    - Share common elements
    - Enable mode-specific validation
    """
    
    @abstractmethod
    def get_mode_type(self) -> RunMode:
        """Get associated mode type.
        
        Returns:
            Mode identifier
        """
        ...
    
    @abstractmethod
    def get_required_components(self) -> list[str]:
        """Get required component types.
        
        Returns:
            List of component names
            
        Design notes:
        - Used for dependency checking
        - Examples: "agent", "environment", "data_manager"
        """
        ...
    
    @abstractmethod
    def create_component_configs(
        self,
        base_config: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        """Create component configurations.
        
        Args:
            base_config: Mode configuration
            
        Returns:
            Dict mapping component to config
            
        Design notes:
        - Extract component-specific settings
        - Apply mode-specific defaults
        """
        ...


class IConfigBuilder(Protocol):
    """Interface for building configurations programmatically.
    
    Design principles:
    - Fluent API for config creation
    - Type safety where possible
    - Validation at build time
    """
    
    def set_mode(
        self,
        mode: RunMode
    ) -> 'IConfigBuilder':
        """Set execution mode.
        
        Args:
            mode: Run mode
            
        Returns:
            Self for chaining
        """
        ...
    
    def set_component(
        self,
        component: str,
        config: dict[str, Any]
    ) -> 'IConfigBuilder':
        """Set component configuration.
        
        Args:
            component: Component name
            config: Component config
            
        Returns:
            Self for chaining
        """
        ...
    
    def add_override(
        self,
        path: str,
        value: Any
    ) -> 'IConfigBuilder':
        """Add configuration override.
        
        Args:
            path: Dot-separated path
            value: Override value
            
        Returns:
            Self for chaining
        """
        ...
    
    def build(
        self,
        validate: bool = True
    ) -> dict[str, Any]:
        """Build final configuration.
        
        Args:
            validate: Whether to validate
            
        Returns:
            Complete configuration
        """
        ...


class IConfigValidator(Protocol):
    """Interface for configuration validation.
    
    Design principles:
    - Comprehensive validation
    - Clear error messages
    - Support custom rules
    """
    
    def add_schema(
        self,
        component: str,
        schema: IConfigSchema
    ) -> None:
        """Add component schema.
        
        Args:
            component: Component name
            schema: Schema instance
        """
        ...
    
    def add_rule(
        self,
        name: str,
        rule: Callable[[dict[str, Any]], tuple[bool, str]]
    ) -> None:
        """Add custom validation rule.
        
        Args:
            name: Rule name
            rule: Validation function
        """
        ...
    
    def validate(
        self,
        config: dict[str, Any]
    ) -> list[str]:
        """Validate complete configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of error messages
        """
        ...


class IConfigManager(Protocol):
    """High-level configuration management interface.
    
    Design principles:
    - Central configuration authority
    - Support multiple sources
    - Enable runtime updates
    """
    
    def initialize(
        self,
        config_dir: Path,
        env_prefix: str = "TRADING_"
    ) -> None:
        """Initialize configuration system.
        
        Args:
            config_dir: Configuration directory
            env_prefix: Environment variable prefix
        """
        ...
    
    def load_config(
        self,
        name: str,
        overrides: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """Load named configuration.
        
        Args:
            name: Configuration name
            overrides: Runtime overrides
            
        Returns:
            Complete configuration
        """
        ...
    
    def watch_config(
        self,
        name: str,
        callback: Callable[[dict[str, Any]], None]
    ) -> None:
        """Watch configuration for changes.
        
        Args:
            name: Configuration name
            callback: Change callback
        """
        ...
    
    def export_config(
        self,
        config: dict[str, Any],
        path: Path,
        format: str = "yaml"
    ) -> None:
        """Export configuration to file.
        
        Args:
            config: Configuration to export
            path: Output path
            format: Output format
        """
        ...
