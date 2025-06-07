"""Integration module for Optuna with the FxAI training system."""

from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml
import numpy as np
import optuna

from config.loader import load_config
from utils.logger import get_logger

logger = get_logger(__name__)


class OptunaConfigManager:
    """Manages configuration merging for Optuna trials."""

    @staticmethod
    def merge_configs(
        base_config: Dict[str, Any], optuna_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge Optuna parameters into base configuration."""
        config = base_config.copy()

        # Apply Optuna parameters
        for param_path, value in optuna_params.items():
            # Split nested parameter path
            keys = param_path.split(".")

            # Navigate to the target location
            target = config
            for key in keys[:-1]:
                if key not in target:
                    target[key] = {}
                target = target[key]

            # Set the value
            target[keys[-1]] = value
            logger.debug(f"Set {param_path} = {value}")

        return config

    @staticmethod
    def create_trial_config(
        base_config_name: str,
        optuna_params: Dict[str, Any],
        trial_id: int,
        output_dir: Optional[str] = None,
    ) -> str:
        """Create configuration file for Optuna trial."""
        # Load base configuration
        base_config = load_config(base_config_name)

        # Merge with Optuna parameters
        trial_config = OptunaConfigManager.merge_configs(base_config, optuna_params)

        # Add trial metadata
        trial_config["optuna_trial_id"] = trial_id
        trial_config["experiment_name"] = f"optuna_trial_{trial_id}"

        # Save to file
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = Path("optuna_configs")

        output_path.mkdir(exist_ok=True)
        config_file = output_path / f"trial_{trial_id}_config.yaml"

        with open(config_file, "w") as f:
            yaml.dump(trial_config, f, default_flow_style=False)

        logger.info(f"Created trial config: {config_file}")
        return str(config_file)


class OptunaMetricsCollector:
    """Collects and formats metrics for Optuna optimization."""

    def __init__(self):
        self.metrics_history = []
        self.best_metrics = {}

    def update(self, metrics: Dict[str, float], step: int):
        """Update metrics from training step."""
        self.metrics_history.append({"step": step, **metrics})

        # Update best metrics
        for key, value in metrics.items():
            if key not in self.best_metrics or value > self.best_metrics[key]:
                self.best_metrics[key] = value

    def get_final_metrics(self) -> Dict[str, float]:
        """Get final metrics for Optuna trial."""
        if not self.metrics_history:
            return {}

        # Get last metrics
        last_metrics = self.metrics_history[-1].copy()
        last_metrics.pop("step", None)

        # Add best metrics
        for key, value in self.best_metrics.items():
            last_metrics[f"best_{key}"] = value

        # Add aggregated metrics
        if len(self.metrics_history) > 10:
            # Last 10 episodes average
            recent_rewards = [
                m.get("episode_reward", 0) for m in self.metrics_history[-10:]
            ]
            last_metrics["mean_reward"] = sum(recent_rewards) / len(recent_rewards)

            # Stability metric
            if len(recent_rewards) > 1:
                last_metrics["reward_stability"] = 1.0 / (1.0 + np.std(recent_rewards))

        return last_metrics

    def get_intermediate_values(self, metric_name: str) -> List[float]:
        """Get intermediate values for pruning."""
        return [m.get(metric_name, float("-inf")) for m in self.metrics_history]


class OptunaPruningCallback:
    """Callback for Optuna pruning during training."""

    def __init__(self, trial, metric_name: str = "mean_reward"):
        self.trial = trial
        self.metric_name = metric_name
        self.step = 0

    def __call__(self, metrics: Dict[str, float]):
        """Check if trial should be pruned."""
        if self.metric_name in metrics:
            value = metrics[self.metric_name]
            self.trial.report(value, self.step)

            if self.trial.should_prune():
                logger.info(f"Trial {self.trial.number} pruned at step {self.step}")
                raise optuna.TrialPruned()

            self.step += 1


def create_optuna_training_wrapper(objective_func):
    """Create a wrapper function for Optuna optimization."""

    def wrapper(trial):
        try:
            # Get suggested parameters
            params = {}
            for param_name in trial.study.user_attrs.get("param_names", []):
                param_config = trial.study.user_attrs.get(f"param_{param_name}", {})

                if param_config["type"] == "float":
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config["low"],
                        param_config["high"],
                        log=param_config.get("log", False),
                    )
                elif param_config["type"] == "int":
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config["low"],
                        param_config["high"],
                        log=param_config.get("log", False),
                    )
                elif param_config["type"] == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config["choices"]
                    )

            # Run objective function
            return objective_func(trial, params)

        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return float("-inf")

    return wrapper


# Convenience functions for common optimization scenarios
def optimize_training_parameters(
    n_trials: int = 50,
    study_name: str = "training_params_optimization",
    base_config: str = "momentum_training",
):
    """Quick optimization of training parameters."""
    import optuna

    def objective(trial):
        params = {
            "training.learning_rate": trial.suggest_float(
                "training.learning_rate", 1e-5, 1e-3, log=True
            ),
            "training.batch_size": trial.suggest_categorical(
                "training.batch_size", [32, 64, 128]
            ),
            "training.n_epochs": trial.suggest_int("training.n_epochs", 4, 12),
            "training.gamma": trial.suggest_float("training.gamma", 0.95, 0.999),
            "training.gae_lambda": trial.suggest_float(
                "training.gae_lambda", 0.9, 0.98
            ),
            "training.entropy_coef": trial.suggest_float(
                "training.entropy_coef", 1e-4, 1e-1, log=True
            ),
        }

        # Create config
        config_path = OptunaConfigManager.create_trial_config(
            base_config, params, trial.number
        )

        # Train and return metric
        # This would be replaced with actual training call
        return trial.number  # Placeholder

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage="sqlite:///optuna_studies.db",
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=n_trials)

    return study
