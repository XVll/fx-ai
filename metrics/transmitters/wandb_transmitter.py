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
    """Transmit metrics to Weights & Biases"""

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
        Initialize W&B transmitter

        Args:
            project_name: W&B project name
            entity: W&B entity (username/team)
            run_name: Optional run name
            config: Configuration to log
            tags: Tags for the run
            group: Group for the run
            job_type: Type of job (training, evaluation, etc.)
            save_code: Whether to save code
            log_frequency: Frequency for different metric types
            prefix_categories: Whether to prefix metrics with category names
        """
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is required for WandBTransmitter")

        self.logger = logging.getLogger(__name__)
        self.project_name = project_name
        self.entity = entity
        self.run_name = run_name
        self.prefix_categories = prefix_categories

        # Default log frequencies (in steps)
        self.log_frequency = {
                                 "model": 1,
                                 "training": 1,
                                 "trading": 1,
                                 "execution": 1,
                                 "environment": 1,
                                 "system": 10,
                                 **(log_frequency or {})
        }

        # Track last transmission times for frequency control
        self._last_transmitted: Dict[str, int] = defaultdict(int)

        # Initialize W&B run
        try:
            if not wandb.run:
                self.run = wandb.init(
                    project=project_name,
                    entity=entity,
                    name=run_name,
                    config=config,
                    tags=tags,
                    group=group,
                    job_type=job_type,
                    save_code=save_code,
                    reinit=True
                )
            else:
                self.run = wandb.run

            self.logger.info(f"W&B transmitter initialized: {self.run.name} ({self.run.id})")

        except Exception as e:
            self.logger.error(f"Failed to initialize W&B: {e}")
            raise

    def transmit(self, metrics: Dict[str, MetricValue], step: Optional[int] = None):
        """Transmit metrics to W&B"""
        if not self.run:
            self.logger.warning("W&B run not available, skipping transmission")
            return

        # Group metrics by category for frequency control
        categorized_metrics = defaultdict(dict)

        for metric_name, metric_value in metrics.items():
            try:
                # Extract category from metric name
                category = metric_name.split('.')[0] if '.' in metric_name else 'unknown'

                # Check frequency
                if step is not None:
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

            except Exception as e:
                self.logger.debug(f"Error processing metric {metric_name}: {e}")

        # Transmit metrics
        if categorized_metrics:
            try:
                # Flatten all metrics for W&B
                wandb_metrics = {}
                for category, metrics_dict in categorized_metrics.items():
                    wandb_metrics.update(metrics_dict)
                    self._last_transmitted[category] = step or 0

                # Log to W&B
                wandb.log(wandb_metrics, step=step)

                self.logger.debug(f"Transmitted {len(wandb_metrics)} metrics to W&B")

            except Exception as e:
                self.logger.error(f"Failed to transmit metrics to W&B: {e}")

    def _format_metric_name(self, metric_name: str) -> str:
        """Format metric name for W&B display"""
        if not self.prefix_categories:
            # Remove category prefix
            parts = metric_name.split('.')
            if len(parts) > 2:
                return '.'.join(parts[2:])  # Remove category.collector prefix
            return metric_name

        # Keep full hierarchical name but make it readable
        return metric_name.replace('_', ' ').title()

    def _convert_value(self, value: Any) -> Optional[Any]:
        """Convert value to W&B compatible type"""
        if value is None:
            return None

        # Handle numeric types
        if isinstance(value, (int, float)):
            if not (value != value):  # Not NaN
                return float(value)
            return None

        # Handle boolean
        if isinstance(value, bool):
            return int(value)

        # Handle strings (for categorical data)
        if isinstance(value, str):
            return value

        # Try to convert to float
        try:
            return float(value)
        except (ValueError, TypeError):
            return str(value)

    def log_artifact(self, file_path: str, name: str, artifact_type: str = "model"):
        """Log an artifact to W&B"""
        if not self.run:
            return

        try:
            artifact = wandb.Artifact(name, type=artifact_type)
            artifact.add_file(file_path)
            self.run.log_artifact(artifact)
            self.logger.info(f"Logged artifact: {name}")
        except Exception as e:
            self.logger.error(f"Failed to log artifact {name}: {e}")

    def log_table(self, table_data: List[List], columns: List[str], name: str):
        """Log a table to W&B"""
        if not self.run:
            return

        try:
            table = wandb.Table(data=table_data, columns=columns)
            wandb.log({name: table})
            self.logger.info(f"Logged table: {name}")
        except Exception as e:
            self.logger.error(f"Failed to log table {name}: {e}")

    def log_plot(self, data: Dict[str, Any], name: str):
        """Log a plot to W&B"""
        if not self.run:
            return

        try:
            wandb.log({name: data})
            self.logger.info(f"Logged plot: {name}")
        except Exception as e:
            self.logger.error(f"Failed to log plot {name}: {e}")

    def watch_model(self, model, log: str = "all", log_freq: int = 100):
        """Watch a model for parameter and gradient tracking"""
        if not self.run:
            return

        try:
            wandb.watch(model, log=log, log_freq=log_freq)
            self.logger.info("Started watching model")
        except Exception as e:
            self.logger.error(f"Failed to watch model: {e}")

    def finish(self):
        """Finish the W&B run"""
        if self.run:
            try:
                wandb.finish()
                self.logger.info("W&B run finished")
            except Exception as e:
                self.logger.error(f"Error finishing W&B run: {e}")

    def close(self):
        """Close the transmitter"""
        self.finish()


class WandBConfig:
    """Configuration for W&B transmitter"""

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
        self.log_frequency = log_frequency or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
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