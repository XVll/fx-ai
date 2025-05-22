# utils/logger.py - Simplified Rich logger setup
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

    Args:
        level: Logging level
        show_time: Whether to show timestamps
        show_path: Whether to show file paths
    """
    # Clear any existing handlers
    logging.getLogger().handlers.clear()

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
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[rich_handler]
    )

    return logging.getLogger()


# Setup default rich logging
setup_rich_logging()