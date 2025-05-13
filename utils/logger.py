# utils/logger.py
import logging
import os
import sys
from datetime import datetime
import json
import traceback
from typing import Dict, Any, Optional, Union
import torch


class EnhancedLogger:
    """
    Enhanced logging system with support for different log levels,
    structured logging, exception tracking, and W&B integration.
    """

    def __init__(
            self,
            name: str,
            log_dir: str = "logs",
            console_level: int = logging.INFO,
            file_level: int = logging.DEBUG,
            wandb_level: int = logging.INFO,
            log_to_wandb: bool = False
    ):
        """
        Initialize the enhanced logger.

        Args:
            name: Logger name
            log_dir: Directory to store log files
            console_level: Logging level for console output
            file_level: Logging level for file output
            wandb_level: Logging level for W&B
            log_to_wandb: Whether to log to W&B
        """
        self.name = name
        self.log_dir = log_dir
        self.log_to_wandb = log_to_wandb
        self.wandb_level = wandb_level

        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  # Capture all logs
        self.logger.propagate = False  # Don't propagate to parent

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )

        # formatter = logging.Formatter(
        #     '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        # )
        #

        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Create file handler
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Keep track of initialized status
        self.initialized = True

        # Track exceptions
        self.exception_count = 0

        self.debug(f"Logger initialized: {name}")

    def debug(self, msg: str, *args, **kwargs):
        """Log a debug message."""
        self.logger.debug(msg, *args, **kwargs)
        self._log_to_wandb("debug", msg, kwargs.get("extra"))

    def info(self, msg: str, *args, **kwargs):
        """Log an info message."""
        self.logger.info(msg, *args, **kwargs)
        self._log_to_wandb("info", msg, kwargs.get("extra"))

    def warning(self, msg: str, *args, **kwargs):
        """Log a warning message."""
        self.logger.warning(msg, *args, **kwargs)
        self._log_to_wandb("warning", msg, kwargs.get("extra"))

    def error(self, msg: str, *args, **kwargs):
        """Log an error message."""
        self.logger.error(msg, *args, **kwargs)
        self._log_to_wandb("error", msg, kwargs.get("extra"))

    def critical(self, msg: str, *args, **kwargs):
        """Log a critical message."""
        self.logger.critical(msg, *args, **kwargs)
        self._log_to_wandb("critical", msg, kwargs.get("extra"))

    def exception(self, msg: str, *args, exc_info=True, **kwargs):
        """Log an exception with traceback."""
        self.exception_count += 1
        self.logger.exception(msg, *args, exc_info=exc_info, **kwargs)

        # Get exception details
        exc_type, exc_value, exc_tb = sys.exc_info()
        tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))

        # Log to W&B if enabled
        if self.log_to_wandb:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({
                        "exception/count": self.exception_count,
                        "exception/type": str(exc_type),
                        "exception/message": str(exc_value),
                        "exception/traceback": tb_str
                    })
            except ImportError:
                self.warning("wandb not installed, skipping exception logging to W&B")

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None, prefix: str = ""):
        """
        Log structured metrics for analysis.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
            prefix: Optional prefix for metric names
        """
        # Log to standard logger
        metrics_str = json.dumps(self._prepare_metrics_for_json(metrics))
        self.logger.info(f"Metrics: {metrics_str}")

        # Log to W&B if enabled
        if self.log_to_wandb:
            self._log_dict_to_wandb(metrics, step, prefix)

    def log_model_metrics(self, model: torch.nn.Module, step: Optional[int] = None, log_weights: bool = False):
        """
        Log model-specific metrics like parameter norms, gradients, etc.

        Args:
            model: PyTorch model
            step: Optional step number
            log_weights: Whether to log raw weight values (can be large)
        """
        metrics = {}

        # Parameter stats
        for name, param in model.named_parameters():
            if param.requires_grad:
                metrics[f"model/param_norm/{name}"] = torch.norm(param).item()

                if param.grad is not None:
                    metrics[f"model/grad_norm/{name}"] = torch.norm(param.grad).item()

                    # Log weight values if requested
                    if log_weights:
                        # For large tensors, this can be expensive
                        try:
                            if param.numel() < 1000:  # Only log small tensors
                                metrics[f"model/weights/{name}"] = param.detach().cpu().numpy()
                        except Exception as e:
                            self.warning(f"Failed to log weights for {name}: {str(e)}")

        # Log metrics
        self.log_metrics(metrics, step)

    def log_system_info(self):
        """Log system and environment information."""
        import platform
        import psutil

        metrics = {
            "system/os": platform.platform(),
            "system/python": platform.python_version(),
            "system/cpu_count": os.cpu_count(),
            "system/memory_total": psutil.virtual_memory().total / (1024 ** 3),  # GB
            "system/memory_available": psutil.virtual_memory().available / (1024 ** 3),  # GB
            "system/disk_usage": psutil.disk_usage('/').percent
        }

        # GPU info if available
        if torch.cuda.is_available():
            metrics.update({
                "system/gpu_count": torch.cuda.device_count(),
                "system/gpu_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "N/A",
                "system/cuda_version": torch.version.cuda
            })

            # Add memory info for each GPU
            for i in range(torch.cuda.device_count()):
                metrics[f"system/gpu{i}_memory_allocated"] = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
                metrics[f"system/gpu{i}_memory_reserved"] = torch.cuda.memory_reserved(i) / (1024 ** 3)  # GB

        self.info("System info:")
        for k, v in metrics.items():
            self.info(f"  {k}: {v}")

        # Log to W&B if enabled
        if self.log_to_wandb:
            self._log_dict_to_wandb(metrics)

    def enable_wandb(self):
        """Enable logging to W&B."""
        self.log_to_wandb = True
        self.info("W&B logging enabled")

    def disable_wandb(self):
        """Disable logging to W&B."""
        self.log_to_wandb = False
        self.info("W&B logging disabled")

    def _log_to_wandb(self, level: str, msg: str, extra: Optional[Dict[str, Any]] = None):
        """Log message to W&B if enabled."""
        if not self.log_to_wandb:
            return

        # Check log level
        level_map = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL
        }

        if level_map.get(level, 0) < self.wandb_level:
            return

        try:
            import wandb
            if wandb.run is not None:
                # Log message
                log_data = {f"log/{level}_count": 1}

                # Include extra data if provided
                if extra:
                    for k, v in extra.items():
                        if isinstance(v, (int, float, bool, str)):
                            log_data[f"log/extra/{k}"] = v

                wandb.log(log_data)
        except ImportError:
            # Only warn once about missing wandb
            if not hasattr(self, "_warned_wandb"):
                self.warning("wandb not installed, skipping logging to W&B")
                self._warned_wandb = True

    def _log_dict_to_wandb(self, metrics: Dict[str, Any], step: Optional[int] = None, prefix: str = ""):
        """Log dictionary of metrics to W&B."""
        if not self.log_to_wandb:
            return

        try:
            import wandb
            if wandb.run is not None:
                # Process metrics for W&B
                wandb_metrics = {}
                for k, v in metrics.items():
                    key = f"{prefix}/{k}" if prefix else k
                    if isinstance(v, (int, float, bool, str)):
                        wandb_metrics[key] = v
                    elif isinstance(v, torch.Tensor):
                        try:
                            # Convert tensor to scalar if possible
                            if v.numel() == 1:
                                wandb_metrics[key] = v.item()
                            else:
                                wandb_metrics[key] = wandb.Histogram(v.detach().cpu().numpy())
                        except Exception as e:
                            self.warning(f"Failed to log tensor for {key}: {str(e)}")
                    else:
                        # Try to convert to something W&B can handle
                        try:
                            wandb_metrics[key] = v
                        except Exception as e:
                            self.warning(f"Failed to log metric for {key}: {str(e)}")

                # Log to W&B
                if step is not None:
                    wandb.log(wandb_metrics, step=step)
                else:
                    wandb.log(wandb_metrics)
        except ImportError:
            if not hasattr(self, "_warned_wandb"):
                self.warning("wandb not installed, skipping metrics logging to W&B")
                self._warned_wandb = True

    def _prepare_metrics_for_json(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare metrics dictionary for JSON serialization."""
        result = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float, bool, str)):
                result[k] = v
            elif isinstance(v, torch.Tensor):
                try:
                    if v.numel() == 1:
                        result[k] = v.item()
                    else:
                        result[k] = f"Tensor(shape={tuple(v.shape)})"
                except Exception:
                    result[k] = "Tensor(error)"
            elif hasattr(v, "__dict__"):
                result[k] = "Object"
            else:
                try:
                    # Try to convert to string
                    result[k] = str(v)
                except Exception:
                    result[k] = "Unprintable"

        return result