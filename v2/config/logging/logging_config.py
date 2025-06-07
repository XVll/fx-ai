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

    # Component logging
    log_rewards: bool = Field(True, description="Log rewards")
    log_actions: bool = Field(True, description="Log actions")
    log_features: bool = Field(False, description="Log features")
    log_portfolio: bool = Field(True, description="Log portfolio")


class WandbConfig(BaseModel):
    """Weights & Biases configuration"""

    enabled: bool = Field(True, description="Enable W&B")
    project: str = Field("fx-ai-momentum", description="Project name")
    entity: Optional[str] = Field(None, description="W&B entity")
    name: Optional[str] = Field(None, description="Run name")
    tags: List[str] = Field(default_factory=list, description="Tags")
    notes: Optional[str] = Field(None, description="Run notes")

    log_frequency: Dict[str, int] = Field(
        default_factory=lambda: {
            "training": 1,
            "episode": 1,
            "rollout": 1,
            "evaluation": 1,
        }
    )

    save_code: bool = Field(True, description="Save code")
    save_model: bool = Field(True, description="Save model")