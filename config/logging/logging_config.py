"""
Logging configuration for system-wide logging settings.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class LoggingConfig:
    """Logging configuration"""

    level: str = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR
    show_time: bool = True
    show_path: bool = True
    compact_errors: bool = True
    console_enabled: bool = True                      # Console logging
    console_format: str = "simple"                    # Console format
    file_enabled: bool = True                         # File logging
    # NOTE: Paths now managed by PathManager - use PathManager.global_logs_dir instead
    log_interval: int = 10                            # Metric log interval