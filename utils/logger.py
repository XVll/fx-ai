# utils/logger.py - Fixed: No dashboard conflicts, clean console logging
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


class CentralizedLogger:
    """
    Centralized logging manager with Rich console integration and Windows Unicode support.

    Features:
    - Rich console formatting (left side of terminal)
    - File logging with UTF-8 encoding
    - Thread-safe log collection
    - Windows Unicode compatibility
    - No dashboard conflicts
    """

    def __init__(
            self,
            app_name: str = "fx-ai",
            log_file: Optional[str] = None,
            max_logs: int = 1000,
            console: Optional[Console] = None
    ):
        self.app_name = app_name
        self.max_logs = max_logs

        # Detect Windows and set up console appropriately
        self.is_windows = platform.system() == "Windows"

        # Set up main console for logging (left side of terminal)
        if console is None:
            # Force UTF-8 encoding on Windows
            if self.is_windows:
                try:
                    import os
                    os.environ['PYTHONIOENCODING'] = 'utf-8'
                    # Create console with explicit UTF-8 file if needed
                    import io
                    if hasattr(sys.stdout, 'buffer'):
                        stdout_utf8 = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
                        self.console = Console(
                            file=stdout_utf8,
                            force_terminal=True,
                            width=80  # Leave space for dashboard on right
                        )
                    else:
                        self.console = Console(
                            force_terminal=True,
                            width=80  # Leave space for dashboard on right
                        )
                except Exception:
                    # Fallback to regular console
                    self.console = Console(
                        force_terminal=True,
                        width=80  # Leave space for dashboard on right
                    )
            else:
                self.console = Console(width=80)  # Leave space for dashboard on right
        else:
            self.console = console

        # Thread-safe log storage (minimal, just for tracking)
        self._log_entries: Deque[LogEntry] = deque(maxlen=max_logs)
        self._lock = threading.RLock()

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
                '%(asctime)s | %(name)-15s | %(levelname)-8s | %(message)s',
                datefmt='%H:%M:%S'
            )
        else:
            # Use regular format on Unix systems
            detailed_formatter = logging.Formatter(
                '%(asctime)s | %(name)-15s | %(levelname)-8s | %(message)s',
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

        # Console handler for left side logging
        try:
            console_handler = RichHandler(
                console=self.console,
                show_time=True,
                show_path=False,
                markup=True,
                rich_tracebacks=True,
                tracebacks_width=80,
                tracebacks_show_locals=False
            )
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(simple_formatter)
            root_logger.addHandler(console_handler)
        except Exception as e:
            # Fallback to standard console handler if Rich fails
            fallback_handler = logging.StreamHandler(sys.stdout)
            fallback_handler.setLevel(logging.INFO)
            fallback_handler.setFormatter(detailed_formatter)
            root_logger.addHandler(fallback_handler)

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
        """Add a log entry to internal storage (thread-safe)"""
        with self._lock:
            # Clean the message for Windows
            entry.message = self._clean_message_for_windows(entry.message)
            self._log_entries.append(entry)

    def get_recent_logs(self, count: Optional[int] = None) -> List[LogEntry]:
        """Get recent log entries"""
        with self._lock:
            if count is None:
                return list(self._log_entries)
            else:
                return list(self._log_entries)[-count:]

    def clear_logs(self):
        """Clear log history"""
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

        # Store in internal history
        self.add_log_entry(LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            module=module,
            message=cleaned_message,
            extra_data=kwargs
        ))

    def warning(self, message: str, module: str = "app", **kwargs):
        """Log warning message"""
        cleaned_message = self._clean_message_for_windows(message)
        logger = self.get_logger(module)
        logger.warning(cleaned_message, extra={'extra_data': kwargs})

        # Store in internal history
        self.add_log_entry(LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.WARNING,
            module=module,
            message=cleaned_message,
            extra_data=kwargs
        ))

    def error(self, message: str, module: str = "app", **kwargs):
        """Log error message"""
        cleaned_message = self._clean_message_for_windows(message)
        logger = self.get_logger(module)
        logger.error(cleaned_message, extra={'extra_data': kwargs})

        # Store in internal history
        self.add_log_entry(LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.ERROR,
            module=module,
            message=cleaned_message,
            extra_data=kwargs
        ))

    def debug(self, message: str, module: str = "app", **kwargs):
        """Log debug message"""
        cleaned_message = self._clean_message_for_windows(message)
        logger = self.get_logger(module)
        logger.debug(cleaned_message, extra={'extra_data': kwargs})

        # Store in internal history
        self.add_log_entry(LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.DEBUG,
            module=module,
            message=cleaned_message,
            extra_data=kwargs
        ))

    def critical(self, message: str, module: str = "app", **kwargs):
        """Log critical message"""
        cleaned_message = self._clean_message_for_windows(message)
        logger = self.get_logger(module)
        logger.critical(cleaned_message, extra={'extra_data': kwargs})

        # Store in internal history
        self.add_log_entry(LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.CRITICAL,
            module=module,
            message=cleaned_message,
            extra_data=kwargs
        ))


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
        max_logs: int = 1000,
        console: Optional[Console] = None
) -> CentralizedLogger:
    """Initialize the global logger"""
    global _global_logger
    _global_logger = CentralizedLogger(
        app_name=app_name,
        log_file=log_file,
        max_logs=max_logs,
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