"""
Logging configuration for system-wide logging settings.
"""

from typing import Literal, Dict, List, Optional
from pydantic import BaseModel, Field


class LoggingConfig(BaseModel):
    """Logging configuration"""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field("INFO")
    console_enabled: bool = Field(True, description="Console logging")
    console_format: str = Field("simple", description="Console format")
    file_enabled: bool = Field(True, description="File logging")
    log_dir: str = Field("logs", description="Log directory")
    log_interval: int = Field(10, description="Metric log interval")
