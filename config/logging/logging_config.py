"""
Logging configuration for system-wide logging settings.
"""

from typing import Literal, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class LoggingConfig:
    """Logging configuration"""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"  # Logging level
    console_enabled: bool = True                      # Console logging
    console_format: str = "simple"                    # Console format
    file_enabled: bool = True                         # File logging
    log_dir: str = "logs"                             # Log directory
    log_interval: int = 10                            # Metric log interval