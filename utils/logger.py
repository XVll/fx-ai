# utils/logger.py - CLEAN: Simple Rich logging setup

import logging
import sys
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

# Global console instance
console = Console()


def setup_rich_logging(level: int = logging.INFO, show_time: bool = True, show_path: bool = False, compact_errors: bool = True):
    """
    Set up Rich logging for clean console output.

    Args:
        level: Logging level (logging.DEBUG, INFO, WARNING, ERROR)
        show_time: Whether to show timestamps
        show_path: Whether to show file paths
        compact_errors: Whether to show compact error tracebacks
    """
    # Install rich traceback handler with compact settings
    if compact_errors:
        install(

            show_locals=False,  # Don't show local variables
            width=100,          # Limit width
            extra_lines=3,      # Show only 3 lines of context
            word_wrap=True,     # Wrap long lines
            suppress=[          # Suppress these modules in traceback
                "gymnasium", "torch", "numpy", "pandas", 
                "wandb", "hydra", "dash", "plotly", "werkzeug"
            ]
        )
    else:
        install(show_locals=True)
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Create Rich handler with clean formatting
    rich_handler = RichHandler(
        console=console,
        show_time=show_time,
        show_path=show_path,
        rich_tracebacks=False,
        tracebacks_show_locals=not compact_errors,
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