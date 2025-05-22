# utils/centralized_logger.py
import logging
import sys
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Deque
from dataclasses import dataclass
from enum import Enum
import time

from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text
from rich.highlighter import ReprHighlighter


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogEntry:
    timestamp: datetime
    level: LogLevel
    module: str
    message: str
    extra_data: Optional[Dict[str, Any]] = None


class DashboardLogHandler(logging.Handler):
    """Custom log handler that feeds logs to the dashboard display"""

    def __init__(self, log_manager):
        super().__init__()
        self.log_manager = log_manager
        self.highlighter = ReprHighlighter()

    def emit(self, record):
        try:
            # Format the log message
            message = self.format(record)

            # Determine log level
            level_map = {
                logging.DEBUG: LogLevel.DEBUG,
                logging.INFO: LogLevel.INFO,
                logging.WARNING: LogLevel.WARNING,
                logging.ERROR: LogLevel.ERROR,
                logging.CRITICAL: LogLevel.CRITICAL,
            }
            level = level_map.get(record.levelno, LogLevel.INFO)

            # Create log entry
            log_entry = LogEntry(
                timestamp=datetime.fromtimestamp(record.created),
                level=level,
                module=record.name,
                message=message,
                extra_data=getattr(record, 'extra_data', None)
            )

            # Add to dashboard
            self.log_manager.add_log_entry(log_entry)

        except Exception:
            self.handleError(record)


class CentralizedLogger:
    """
    Centralized logging manager with Rich console integration and dashboard support.

    Features:
    - Rich console formatting
    - File logging
    - Dashboard integration
    - Thread-safe log collection
    - Multiple output targets
    """

    def __init__(
            self,
            app_name: str = "fx-ai",
            log_file: Optional[str] = None,
            max_dashboard_logs: int = 1000,
            console: Optional[Console] = None
    ):
        self.app_name = app_name
        self.console = console or Console()
        self.max_dashboard_logs = max_dashboard_logs

        # Thread-safe log storage for dashboard
        self._log_entries: Deque[LogEntry] = deque(maxlen=max_dashboard_logs)
        self._lock = threading.RLock()

        # Dashboard integration
        self.dashboard_handler: Optional[DashboardLogHandler] = None

        # Set up logging
        self._setup_logging(log_file)

        # Root logger reference
        self.root_logger = logging.getLogger()

    def _setup_logging(self, log_file: Optional[str]):
        """Set up logging configuration"""
        # Clear existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Set log level
        root_logger.setLevel(logging.INFO)

        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )

        simple_formatter = logging.Formatter('%(message)s')

        # File handler if specified
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(detailed_formatter)
            root_logger.addHandler(file_handler)

        # Console handler for non-dashboard output
        console_handler = RichHandler(
            console=self.console,
            show_time=False,
            show_path=False,
            markup=True,
            rich_tracebacks=True
        )
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        root_logger.addHandler(console_handler)

        # Dashboard handler
        self.dashboard_handler = DashboardLogHandler(self)
        self.dashboard_handler.setLevel(logging.INFO)
        self.dashboard_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(self.dashboard_handler)

    def add_log_entry(self, entry: LogEntry):
        """Add a log entry to the dashboard display (thread-safe)"""
        with self._lock:
            self._log_entries.append(entry)

    def get_recent_logs(self, count: Optional[int] = None) -> List[LogEntry]:
        """Get recent log entries for dashboard display"""
        with self._lock:
            if count is None:
                return list(self._log_entries)
            else:
                return list(self._log_entries)[-count:]

    def get_formatted_logs_for_display(self, count: int = 50) -> List[Text]:
        """Get formatted log entries for Rich display"""
        logs = self.get_recent_logs(count)
        formatted_logs = []

        for log_entry in logs:
            # Color coding by level
            level_colors = {
                LogLevel.DEBUG: "dim white",
                LogLevel.INFO: "white",
                LogLevel.WARNING: "yellow",
                LogLevel.ERROR: "red",
                LogLevel.CRITICAL: "bold red"
            }

            level_color = level_colors.get(log_entry.level, "white")
            time_str = log_entry.timestamp.strftime("%H:%M:%S")

            # Create formatted text
            text = Text()
            text.append(f"{time_str} ", style="dim cyan")
            text.append(f"{log_entry.level.value:>8}", style=level_color)
            text.append(" | ", style="dim white")
            text.append(f"{log_entry.module:>15}", style="dim blue")
            text.append(" | ", style="dim white")
            text.append(log_entry.message,
                        style=level_color if log_entry.level in [LogLevel.ERROR, LogLevel.CRITICAL] else "white")

            formatted_logs.append(text)

        return formatted_logs

    def clear_dashboard_logs(self):
        """Clear dashboard log history"""
        with self._lock:
            self._log_entries.clear()

    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance for a specific module"""
        return logging.getLogger(name)

    def info(self, message: str, module: str = "app", **kwargs):
        """Log info message"""
        logger = self.get_logger(module)
        logger.info(message, extra={'extra_data': kwargs})

    def warning(self, message: str, module: str = "app", **kwargs):
        """Log warning message"""
        logger = self.get_logger(module)
        logger.warning(message, extra={'extra_data': kwargs})

    def error(self, message: str, module: str = "app", **kwargs):
        """Log error message"""
        logger = self.get_logger(module)
        logger.error(message, extra={'extra_data': kwargs})

    def debug(self, message: str, module: str = "app", **kwargs):
        """Log debug message"""
        logger = self.get_logger(module)
        logger.debug(message, extra={'extra_data': kwargs})

    def critical(self, message: str, module: str = "app", **kwargs):
        """Log critical message"""
        logger = self.get_logger(module)
        logger.critical(message, extra={'extra_data': kwargs})


# Global logger instance
_global_logger: Optional[CentralizedLogger] = None


def get_logger() -> CentralizedLogger:
    """Get the global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = CentralizedLogger()
    return _global_logger


def initialize_logger(
        app_name: str = "fx-ai",
        log_file: Optional[str] = None,
        max_dashboard_logs: int = 1000,
        console: Optional[Console] = None
) -> CentralizedLogger:
    """Initialize the global logger"""
    global _global_logger
    _global_logger = CentralizedLogger(
        app_name=app_name,
        log_file=log_file,
        max_dashboard_logs=max_dashboard_logs,
        console=console
    )
    return _global_logger


def log_info(message: str, module: str = "app", **kwargs):
    """Convenience function for info logging"""
    get_logger().info(message, module, **kwargs)


def log_warning(message: str, module: str = "app", **kwargs):
    """Convenience function for warning logging"""
    get_logger().warning(message, module, **kwargs)


def log_error(message: str, module: str = "app", **kwargs):
    """Convenience function for error logging"""
    get_logger().error(message, module, **kwargs)


def log_debug(message: str, module: str = "app", **kwargs):
    """Convenience function for debug logging"""
    get_logger().debug(message, module, **kwargs)


def log_critical(message: str, module: str = "app", **kwargs):
    """Convenience function for critical logging"""
    get_logger().critical(message, module, **kwargs)