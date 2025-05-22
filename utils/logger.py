# utils/logger.py - Simplified Rich logger setup compatible with dashboard
import logging
import sys
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

# Install rich traceback handler
install(show_locals=True)

# Global console instance for the application
console = Console()


def setup_rich_logging(level: int = logging.INFO, show_time: bool = True, show_path: bool = False):
    """
    Set up Rich logging as the default Python logging handler.
    This will be temporarily replaced when dashboard starts.

    Args:
        level: Logging level
        show_time: Whether to show timestamps
        show_path: Whether to show file paths
    """
    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Create Rich handler
    rich_handler = RichHandler(
        console=console,
        show_time=show_time,
        show_path=show_path,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        markup=True
    )

    # Set up root logger
    root_logger.setLevel(level)
    root_logger.addHandler(rich_handler)

    # Set format
    rich_handler.setFormatter(logging.Formatter("%(message)s"))

    return root_logger


def get_logger(name: str = None) -> logging.Logger:
    """Get a logger instance"""
    return logging.getLogger(name) if name else logging.getLogger()


# Setup default rich logging
setup_rich_logging()