# metrics/transmitters/wandb_transmitter.py - FIXED: Ensure proper W&B transmission

import logging
from typing import Dict, Optional, Any, List
from collections import defaultdict

from metrics.core import MetricTransmitter, MetricValue

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class WandBTransmitter(MetricTransmitter):
    """Transmit metrics to Weights & Biases with guaranteed transmission"""

    def __init__(self,
                 project_name: str,
                 entity: Optional[str] = None,
                 run_name: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None,
                 tags: Optional[List[str]] = None,
                 group: Optional[str] = None,
                 job_type: str = "training",
                 save_code: bool = True,
                 log_frequency: Dict[str, int] = None,
                 prefix_categories: bool = True):
        """
        Initialize W&B transmitter with enhanced reliability.
        """
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is required for WandBTransmitter")

        self.logger = logging.getLogger(__name__)
        self.project_name = project_name
        self.entity = entity
        self.run_name = run_name
        self.prefix_categories = prefix_categories
        self._finished = False  # Track if wandb.finish() has been called

        # Default log frequencies (in steps) - more aggressive for initial testing
        self.log_frequency = {
            "model": 1,
            "training": 1,
            "trading": 1,
            "execution": 1,
            "environment": 1,
            "system": 5,  # Reduced from 10 for better visibility
            **(log_frequency or {})
        }

        # Track last transmission times for frequency control
        self._last_transmitted: Dict[str, int] = defaultdict(int)
        self._total_metrics_sent = 0
        self._transmission_errors = 0
        self._last_logged_step = -1  # Track the last step logged to wandb

        # Initialize W&B run with enhanced error handling
        try:
            # Check if W&B is already initialized
            if wandb.run is not None:
                self.logger.info("Using existing W&B run")
                self.run = wandb.run
            else:
                self.logger.info(f"Initializing new W&B run: {project_name}")

                # Initialize with explicit settings
                self.run = wandb.init(
                    project=project_name,
                    entity=entity,
                    name=run_name,
                    config=config or {},
                    tags=tags or [],
                    group=group,
                    job_type=job_type,
                    save_code=save_code,
                    resume="allow"  # Allow resume if accidentally restarted
                )

            # Verify the run is properly initialized
            if self.run is None:
                raise RuntimeError("W&B run failed to initialize")

            self.logger.info(f"✅ W&B transmitter initialized successfully")
            self.logger.info(f"   🏷️  Run: {self.run.name} ({self.run.id})")
            self.logger.info(f"   🔗 URL: {self.run.url}")

            # Send a test metric to verify connection
            self._send_test_metric()

        except Exception as e:
            self.logger.error(f"Failed to initialize W&B: {e}")
            self.run = None
            raise

    def _send_test_metric(self):
        """Send a test metric to verify W&B connection."""
        try:
            # Don't specify step for initialization metrics to avoid warnings
            wandb.log({"system/test_metric": 1.0, "system/initialization_time": wandb.run.start_time})
            self.logger.info("✅ W&B connection test successful")
        except Exception as e:
            self.logger.error(f"W&B connection test failed: {e}")

    def transmit(self, metrics: Dict[str, MetricValue], step: Optional[int] = None):
        """Transmit metrics to W&B with enhanced error handling."""
        if self._finished:
            # Silently skip transmission if W&B has been finished
            return
            
        if not self.run:
            self.logger.warning("W&B run not available, skipping transmission")
            return

        # Group metrics by category for frequency control
        categorized_metrics = defaultdict(dict)
        total_metrics_this_batch = 0

        for metric_name, metric_value in metrics.items():
            try:
                # Extract category from metric name
                category = metric_name.split('.')[0] if '.' in metric_name else 'unknown'

                # Check frequency (but be less aggressive for debugging)
                if step is not None and step > 10:  # Skip frequency check for first 10 steps
                    frequency = self.log_frequency.get(category, 1)
                    last_step = self._last_transmitted[category]
                    if step - last_step < frequency:
                        continue

                # Prepare metric name for W&B
                display_name = self._format_metric_name(metric_name)

                # Convert value to appropriate type
                value = self._convert_value(metric_value.value)
                if value is not None:
                    categorized_metrics[category][display_name] = value
                    total_metrics_this_batch += 1

            except Exception as e:
                self.logger.debug(f"Error processing metric {metric_name}: {e}")

        # Transmit metrics if we have any
        if categorized_metrics:
            try:
                # Flatten all metrics for W&B
                wandb_metrics = {}
                categories_logged = []

                for category, metrics_dict in categorized_metrics.items():
                    wandb_metrics.update(metrics_dict)
                    categories_logged.append(category)
                    self._last_transmitted[category] = step or 0

                # Add step information
                if step is not None:
                    wandb_metrics["global_step"] = step

                # Log to W&B with error handling
                # Always use monotonically increasing global step for W&B
                # Use a global step counter instead of the local step to avoid backward jumps
                if step is not None and step > 0:
                    # Use a monotonically increasing global step based on _total_metrics_sent
                    # This ensures W&B always sees increasing steps even if episodes reset
                    global_step = max(self._last_logged_step + 1, step)
                    wandb.log(wandb_metrics, step=global_step)
                    self._last_logged_step = global_step
                    # Debug logging for step tracking
                    if global_step % 100 == 0:
                        self.logger.debug(f"W&B transmitted metrics at global step {global_step} (episode step {step})")
                elif step is None or step == 0:
                    wandb.log(wandb_metrics)
                else:
                    # Skip logging for negative steps
                    self.logger.debug(f"Skipping W&B log for invalid step {step}")
                    return

                self._total_metrics_sent += total_metrics_this_batch

                # Log successful transmission (but not too frequently)
                if self._total_metrics_sent % 50 == 0 or step is None or step < 5:
                    self.logger.debug(f"📊 W&B: Sent {total_metrics_this_batch} metrics "
                                      f"(categories: {', '.join(categories_logged)}) "
                                      f"[total: {self._total_metrics_sent}]")

            except Exception as e:
                self._transmission_errors += 1
                self.logger.error(f"Failed to transmit metrics to W&B: {e}")

                # Log error summary every 10 errors
                if self._transmission_errors % 10 == 1:
                    self.logger.error(f"W&B transmission errors: {self._transmission_errors}")

    def _format_metric_name(self, metric_name: str) -> str:
        """Format metric name for W&B display with improved hierarchy."""
        if not self.prefix_categories:
            # Remove category prefix
            parts = metric_name.split('.')
            if len(parts) > 2:
                return '.'.join(parts[2:])  # Remove category.collector prefix
            return metric_name

        # Keep hierarchical name but make it W&B friendly
        # Convert "category.collector.metric" to "category/metric"
        parts = metric_name.split('.')
        if len(parts) >= 3:
            category = parts[0]
            metric = '_'.join(parts[2:])  # Join remaining parts
            return f"{category}/{metric}"
        elif len(parts) == 2:
            return f"{parts[0]}/{parts[1]}"
        else:
            return metric_name

    def _convert_value(self, value: Any) -> Optional[Any]:
        """Convert value to W&B compatible type with better handling."""
        if value is None:
            return None

        # Handle numeric types
        if isinstance(value, (int, float)):
            if not (value != value):  # Not NaN
                # Ensure finite values
                if abs(value) < 1e10:  # Reasonable range for W&B
                    return float(value)
            return None

        # Handle boolean
        if isinstance(value, bool):
            return int(value)

        # Handle strings (for categorical data)
        if isinstance(value, str):
            return value

        # Handle numpy types
        try:
            import numpy as np
            if isinstance(value, np.number):
                return float(value)
        except ImportError:
            pass

        # Try to convert to float
        try:
            float_val = float(value)
            if not (float_val != float_val) and abs(float_val) < 1e10:  # Not NaN and reasonable
                return float_val
        except (ValueError, TypeError):
            pass

        # Fallback to string
        try:
            return str(value)
        except:
            return None

    def log_artifact(self, file_path: str, name: str, artifact_type: str = "model"):
        """Log an artifact to W&B with error handling."""
        if not self.run:
            return

        try:
            artifact = wandb.Artifact(name, type=artifact_type)
            artifact.add_file(file_path)
            self.run.log_artifact(artifact)
            self.logger.info(f"📦 W&B artifact logged: {name}")
        except Exception as e:
            self.logger.error(f"Failed to log artifact {name}: {e}")

    def log_table(self, table_data: List[List], columns: List[str], name: str):
        """Log a table to W&B with error handling."""
        if not self.run:
            return

        try:
            table = wandb.Table(data=table_data, columns=columns)
            wandb.log({name: table})
            self.logger.info(f"📋 W&B table logged: {name}")
        except Exception as e:
            self.logger.error(f"Failed to log table {name}: {e}")

    def log_plot(self, data: Dict[str, Any], name: str):
        """Log a plot to W&B with error handling."""
        if not self.run:
            return

        try:
            wandb.log({name: data})
            self.logger.info(f"📈 W&B plot logged: {name}")
        except Exception as e:
            self.logger.error(f"Failed to log plot {name}: {e}")

    def watch_model(self, model, log: str = "all", log_freq: int = 100):
        """Watch a model for parameter and gradient tracking."""
        if not self.run:
            return

        try:
            wandb.watch(model, log=log, log_freq=log_freq)
            self.logger.info("👁️  W&B model watching enabled")
        except Exception as e:
            self.logger.error(f"Failed to watch model: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get transmission statistics."""
        return {
            "total_metrics_sent": self._total_metrics_sent,
            "transmission_errors": self._transmission_errors,
            "run_id": self.run.id if self.run else None,
            "run_url": self.run.get_url() if self.run else None
        }

    def finish(self):
        """Finish the W&B run with stats logging."""
        if self._finished:
            return  # Already finished
            
        self._finished = True  # Mark as finished to prevent further transmissions
        
        if self.run:
            try:
                # Log final stats
                stats = self.get_stats()
                self.logger.info(f"📊 W&B session stats: {stats['total_metrics_sent']} metrics sent, "
                                 f"{stats['transmission_errors']} errors")

                wandb.finish()
                self.logger.info("✅ W&B run finished successfully")
            except Exception as e:
                self.logger.error(f"Error finishing W&B run: {e}")

    def close(self):
        """Close the transmitter."""
        self.finish()


class WandBConfig:
    """Configuration for W&B transmitter with enhanced defaults."""

    def __init__(self,
                 project_name: str,
                 entity: Optional[str] = None,
                 run_name: Optional[str] = None,
                 tags: Optional[List[str]] = None,
                 group: Optional[str] = None,
                 job_type: str = "training",
                 save_code: bool = True,
                 log_model: bool = True,
                 watch_model: bool = True,
                 log_frequency: Optional[Dict[str, int]] = None):
        self.project_name = project_name
        self.entity = entity
        self.run_name = run_name
        self.tags = tags or []
        self.group = group
        self.job_type = job_type
        self.save_code = save_code
        self.log_model = log_model
        self.watch_model = watch_model

        # Enhanced default frequencies for better visibility
        self.log_frequency = {
            "model": 1,
            "training": 1,
            "trading": 1,
            "execution": 1,
            "environment": 1,
            "system": 5,
            **(log_frequency or {})
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "project_name": self.project_name,
            "entity": self.entity,
            "run_name": self.run_name,
            "tags": self.tags,
            "group": self.group,
            "job_type": self.job_type,
            "save_code": self.save_code,
            "log_model": self.log_model,
            "watch_model": self.watch_model,
            "log_frequency": self.log_frequency
        }