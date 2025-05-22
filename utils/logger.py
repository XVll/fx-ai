# utils/logger.py - Fixed for Windows Unicode support
import logging
import sys
import threading
import platform
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
    Centralized logging manager with Rich console integration and Windows Unicode support.

    Features:
    - Rich console formatting
    - File logging with UTF-8 encoding
    - Dashboard integration
    - Thread-safe log collection
    - Windows Unicode compatibility
    """

    def __init__(
            self,
            app_name: str = "fx-ai",
            log_file: Optional[str] = None,
            max_dashboard_logs: int = 1000,
            console: Optional[Console] = None
    ):
        self.app_name = app_name
        self.max_dashboard_logs = max_dashboard_logs

        # Detect Windows and set up console appropriately
        self.is_windows = platform.system() == "Windows"

        # Set up console with proper encoding for Windows
        if console is None:
            # Force UTF-8 encoding on Windows
            if self.is_windows:
                # Try to enable UTF-8 mode on Windows
                try:
                    import os
                    os.environ['PYTHONIOENCODING'] = 'utf-8'
                    # Create console with explicit UTF-8 file if needed
                    import io
                    if hasattr(sys.stdout, 'buffer'):
                        stdout_utf8 = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
                        self.console = Console(file=stdout_utf8, force_terminal=True)
                    else:
                        self.console = Console(force_terminal=True)
                except Exception:
                    # Fallback to regular console
                    self.console = Console(force_terminal=True)
            else:
                self.console = Console()
        else:
            self.console = console

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
        """Set up logging configuration with Windows Unicode support"""
        # Clear existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Set log level
        root_logger.setLevel(logging.INFO)

        # Create formatters without emoji for Windows compatibility
        if self.is_windows:
            # Use text symbols instead of emoji on Windows
            detailed_formatter = logging.Formatter(
                '%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
                datefmt='%H:%M:%S'
            )
        else:
            # Use regular format on Unix systems
            detailed_formatter = logging.Formatter(
                '%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
                datefmt='%H:%M:%S'
            )

        simple_formatter = logging.Formatter('%(message)s')

        # File handler with explicit UTF-8 encoding if specified
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # Force UTF-8 encoding for file handler
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(detailed_formatter)
            root_logger.addHandler(file_handler)

        # Console handler for non-dashboard output
        try:
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
        except Exception as e:
            # Fallback to standard console handler if Rich fails
            fallback_handler = logging.StreamHandler(sys.stdout)
            fallback_handler.setLevel(logging.INFO)
            fallback_handler.setFormatter(simple_formatter)
            root_logger.addHandler(fallback_handler)

        # Dashboard handler
        self.dashboard_handler = DashboardLogHandler(self)
        self.dashboard_handler.setLevel(logging.INFO)
        self.dashboard_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(self.dashboard_handler)

    def _clean_message_for_windows(self, message: str) -> str:
        """Clean message for Windows compatibility by replacing emoji with text"""
        if not self.is_windows:
            return message

        # Replace common emoji with text equivalents
        emoji_replacements = {
            '🚀': '[START]',
            '📁': '[DIR]',
            '📄': '[FILE]',
            '💾': '[SAVE]',
            '📂': '[LOAD]',
            '✅': '[OK]',
            '🔄': '[REFRESH]',
            '📊': '[CHART]',
            '🏗️': '[BUILD]',
            '🤖': '[AI]',
            '🎲': '[DICE]',
            '🔍': '[SEARCH]',
            '📈': '[UP]',
            '📉': '[DOWN]',
            '🏁': '[FINISH]',
            '⚙️': '[CONFIG]',
            '🧠': '[BRAIN]',
            '🧹': '[CLEAN]',
            '👋': '[WAVE]',
            '🎉': '[PARTY]',
            '⚠️': '[WARN]',
            '🛑': '[STOP]',
            '💀': '[DEAD]',
            '👁️': '[EYE]',
            '📅': '[DATE]',
            '🚫': '[NO]',
            '⏰': '[TIME]',
            '🏃': '[RUN]',
            '⏹️': '[PAUSE]',
            '🟢': '[GREEN]',
            '🔴': '[RED]',
            '🟡': '[YELLOW]',
            '💼': '[CASE]',
            '🏆': '[TROPHY]',
        }

        cleaned_message = message
        for emoji, replacement in emoji_replacements.items():
            cleaned_message = cleaned_message.replace(emoji, replacement)

        return cleaned_message

    def add_log_entry(self, entry: LogEntry):
        """Add a log entry to the dashboard display (thread-safe)"""
        with self._lock:
            # Clean the message for Windows
            entry.message = self._clean_message_for_windows(entry.message)
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
        cleaned_message = self._clean_message_for_windows(message)
        logger = self.get_logger(module)
        logger.info(cleaned_message, extra={'extra_data': kwargs})

    def warning(self, message: str, module: str = "app", **kwargs):
        """Log warning message"""
        cleaned_message = self._clean_message_for_windows(message)
        logger = self.get_logger(module)
        logger.warning(cleaned_message, extra={'extra_data': kwargs})

    def error(self, message: str, module: str = "app", **kwargs):
        """Log error message"""
        cleaned_message = self._clean_message_for_windows(message)
        logger = self.get_logger(module)
        logger.error(cleaned_message, extra={'extra_data': kwargs})

    def debug(self, message: str, module: str = "app", **kwargs):
        """Log debug message"""
        cleaned_message = self._clean_message_for_windows(message)
        logger = self.get_logger(module)
        logger.debug(cleaned_message, extra={'extra_data': kwargs})

    def critical(self, message: str, module: str = "app", **kwargs):
        """Log critical message"""
        cleaned_message = self._clean_message_for_windows(message)
        logger = self.get_logger(module)
        logger.critical(cleaned_message, extra={'extra_data': kwargs})


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