# utils/logger.py - CLEAN: Simple Rich logging setup

import logging
import sys
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

# Install rich traceback handler for better error displays
install(show_locals=True)

# Global console instance
console = Console()


def setup_rich_logging(level: int = logging.INFO, show_time: bool = True, show_path: bool = False):
    """
    Set up Rich logging for clean console output.

    Args:
        level: Logging level (logging.DEBUG, INFO, WARNING, ERROR)
        show_time: Whether to show timestamps
        show_path: Whether to show file paths
    """
    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Create Rich handler with clean formatting
    rich_handler = RichHandler(
        console=console,
        show_time=show_time,
        show_path=show_path,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        markup=True
    )

    # Configure root logger
    root_logger.setLevel(level)
    root_logger.addHandler(rich_handler)

    # Set clean format
    rich_handler.setFormatter(logging.Formatter("%(message)s"))

    return root_logger


def get_logger(name: str = None) -> logging.Logger:
    """Get a logger instance with the specified name"""
    return logging.getLogger(name) if name else logging.getLogger()


# Initialize default rich logging on import
setup_rich_logging(level=logging.INFO)