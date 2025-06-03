"""Advanced Optuna hyperparameter optimization system for FxAI."""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import optuna
import pandas as pd
import torch
import yaml
from optuna import Study, Trial
from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler, GridSampler, NSGAIISampler, QMCSampler
from optuna.pruners import (
    MedianPruner,
    PercentilePruner,
    SuccessiveHalvingPruner,
    HyperbandPruner,
    ThresholdPruner,
    PatientPruner,
)
from optuna.visualization import (
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_slice,
    plot_contour,
    plot_edf,
)
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
import wandb

from config.optuna.optuna_config import (
    OptunaStudySpec,
    StudyConfig,
    ParameterConfig,
    SamplerConfig,
    PrunerConfig,
    SamplerType,
    PrunerType,
    DistributionType,
)
from config.loader import load_config
from main import train_agent
from utils.logger import get_logger

console = Console()
logger = get_logger(__name__)


class OptunaOptimizer:
    """Advanced Optuna hyperparameter optimization system."""
    
    def __init__(self, spec_path: Optional[str] = None):
        """Initialize optimizer with specification."""
        self.spec = self._load_spec(spec_path)
        self.results_dir = Path(self.spec.results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Set up logging
        optuna.logging.set_verbosity(getattr(optuna.logging, self.spec.log_level))
        
        # Track active studies
        self.studies: Dict[str, Study] = {}
        self.trial_results: Dict[str, List[Dict]] = {}
        
    def _load_spec(self, spec_path: Optional[str]) -> OptunaStudySpec:
        """Load study specification from file or create default."""
        if spec_path and Path(spec_path).exists():
            with open(spec_path, "r") as f:
                data = yaml.safe_load(f)
            return OptunaStudySpec(**data)
        else:
            # Create default specification
            return self._create_default_spec()
    
    def _create_default_spec(self) -> OptunaStudySpec:
        """Create default optimization specification."""
        return OptunaStudySpec(
            name="default_optimization",
            description="Default hyperparameter optimization for FxAI",
            studies=[
                StudyConfig(
                    study_name="fx_ai_optimization",
                    direction="maximize",
                    metric_name="mean_reward",
                    parameters=[
                        # Training parameters
                        ParameterConfig(
                            name="training.learning_rate",
                            type=DistributionType.FLOAT_LOG,
                            low=1e-5,
                            high=1e-3,
                        ),
                        ParameterConfig(
                            name="training.batch_size",
                            type=DistributionType.CATEGORICAL,
                            choices=[32, 64, 128, 256],
                        ),
                        ParameterConfig(
                            name="training.n_epochs",
                            type=DistributionType.INT,
                            low=4,
                            high=16,
                        ),
                        ParameterConfig(
                            name="training.gamma",
                            type=DistributionType.FLOAT,
                            low=0.95,
                            high=0.999,
                        ),
                        ParameterConfig(
                            name="training.gae_lambda",
                            type=DistributionType.FLOAT,
                            low=0.9,
                            high=0.98,
                        ),
                        ParameterConfig(
                            name="training.clip_epsilon",
                            type=DistributionType.FLOAT,
                            low=0.1,
                            high=0.3,
                        ),
                        ParameterConfig(
                            name="training.entropy_coef",
                            type=DistributionType.FLOAT_LOG,
                            low=1e-4,
                            high=1e-1,
                        ),
                        ParameterConfig(
                            name="training.value_coef",
                            type=DistributionType.FLOAT,
                            low=0.25,
                            high=1.0,
                        ),
                        
                        # Model parameters
                        ParameterConfig(
                            name="model.d_model",
                            type=DistributionType.CATEGORICAL,
                            choices=[64, 128, 256],
                        ),
                        ParameterConfig(
                            name="model.n_layers",
                            type=DistributionType.INT,
                            low=4,
                            high=12,
                        ),
                        ParameterConfig(
                            name="model.dropout",
                            type=DistributionType.FLOAT,
                            low=0.0,
                            high=0.3,
                        ),
                        
                        # Reward parameters
                        ParameterConfig(
                            name="env.reward.pnl_coefficient",
                            type=DistributionType.FLOAT,
                            low=50.0,
                            high=300.0,
                        ),
                        ParameterConfig(
                            name="env.reward.holding_penalty_coefficient",
                            type=DistributionType.FLOAT,
                            low=0.5,
                            high=5.0,
                        ),
                        ParameterConfig(
                            name="env.reward.drawdown_penalty_coefficient",
                            type=DistributionType.FLOAT,
                            low=1.0,
                            high=20.0,
                        ),
                        ParameterConfig(
                            name="env.reward.profit_closing_bonus_coefficient",
                            type=DistributionType.FLOAT,
                            low=50.0,
                            high=200.0,
                        ),
                        ParameterConfig(
                            name="env.reward.clean_trade_coefficient",
                            type=DistributionType.FLOAT,
                            low=10.0,
                            high=50.0,
                        ),
                    ],
                    n_trials=100,
                    episodes_per_trial=2000,
                    training_config={
                        "mode": "train",
                        "experiment_name": "optuna_optimization",
                        "training": {
                            "total_updates": 100,  # Shorter for optimization
                            "checkpoint_interval": 20,
                            "eval_frequency": 20,
                            "eval_episodes": 5,
                        },
                        "wandb": {
                            "enabled": True,
                            "project": "fx-ai-optuna",
                        },
                    },
                )
            ],
        )
    
    def _create_sampler(self, config: SamplerConfig) -> optuna.samplers.BaseSampler:
        """Create Optuna sampler from configuration."""
        common_kwargs = {
            "seed": config.seed,
        }
        
        if config.type == SamplerType.TPE:
            return TPESampler(
                n_startup_trials=config.n_startup_trials,
                n_ei_candidates=config.n_ei_candidates,
                multivariate=config.multivariate,
                warn_independent_sampling=config.warn_independent_sampling,
                consider_prior=config.consider_prior,
                prior_weight=config.prior_weight,
                consider_magic_clip=config.consider_magic_clip,
                consider_endpoints=config.consider_endpoints,
                **common_kwargs,
            )
        elif config.type == SamplerType.CMA_ES:
            return CmaEsSampler(
                n_startup_trials=config.n_startup_trials,
                warn_independent_sampling=config.warn_independent_sampling,
                **common_kwargs,
            )
        elif config.type == SamplerType.RANDOM:
            return RandomSampler(**common_kwargs)
        elif config.type == SamplerType.GRID:
            return GridSampler()
        elif config.type == SamplerType.NSGA2:
            return NSGAIISampler(seed=config.seed)
        elif config.type == SamplerType.QMC:
            return QMCSampler(
                warn_independent_sampling=config.warn_independent_sampling,
                **common_kwargs,
            )
        else:
            raise ValueError(f"Unknown sampler type: {config.type}")
    
    def _create_pruner(self, config: Optional[PrunerConfig]) -> Optional[optuna.pruners.BasePruner]:
        """Create Optuna pruner from configuration."""
        if not config:
            return None
            
        if config.type == PrunerType.MEDIAN:
            return MedianPruner(
                n_startup_trials=config.n_startup_trials,
                n_warmup_steps=config.n_warmup_steps,
                interval_steps=config.interval_steps,
                n_min_trials=config.n_min_trials,
            )
        elif config.type == PrunerType.PERCENTILE:
            return PercentilePruner(
                percentile=config.percentile,
                n_startup_trials=config.n_startup_trials,
                n_warmup_steps=config.n_warmup_steps,
                interval_steps=config.interval_steps,
                n_min_trials=config.n_min_trials,
            )
        elif config.type == PrunerType.SUCCESSIVE_HALVING:
            return SuccessiveHalvingPruner(
                min_resource=config.min_resource,
                reduction_factor=config.reduction_factor,
                n_min_trials=config.n_min_trials,
            )
        elif config.type == PrunerType.HYPERBAND:
            return HyperbandPruner(
                min_resource=config.min_resource,
                max_resource=config.max_resource,
                reduction_factor=config.reduction_factor,
                n_min_trials=config.n_min_trials,
            )
        elif config.type == PrunerType.THRESHOLD:
            return ThresholdPruner(
                lower=config.lower,
                upper=config.upper,
                n_warmup_steps=config.n_warmup_steps,
                interval_steps=config.interval_steps,
            )
        elif config.type == PrunerType.PATIENT:
            return PatientPruner(
                patience=config.patience,
                min_delta=config.min_delta,
            )
        else:
            raise ValueError(f"Unknown pruner type: {config.type}")
    
    def _suggest_parameter(self, trial: Trial, param: ParameterConfig) -> Any:
        """Suggest parameter value based on distribution type."""
        if param.type == DistributionType.FLOAT:
            return trial.suggest_float(param.name, param.low, param.high, step=param.step)
        elif param.type == DistributionType.FLOAT_LOG:
            return trial.suggest_float(param.name, param.low, param.high, log=True)
        elif param.type == DistributionType.INT:
            return trial.suggest_int(param.name, param.low, param.high, step=param.step or 1)
        elif param.type == DistributionType.INT_LOG:
            return trial.suggest_int(param.name, param.low, param.high, log=True)
        elif param.type == DistributionType.CATEGORICAL:
            return trial.suggest_categorical(param.name, param.choices)
        else:
            raise ValueError(f"Unknown distribution type: {param.type}")
    
    def _create_objective(self, study_config: StudyConfig):
        """Create objective function for study."""
        def objective(trial: Trial) -> float:
            # Suggest parameters
            params = {}
            for param_config in study_config.parameters:
                value = self._suggest_parameter(trial, param_config)
                params[param_config.name] = value
            
            # Log trial start
            console.print(f"\n[bold blue]Trial {trial.number}[/bold blue]")
            console.print(Panel(
                f"Parameters:\n" + "\n".join([f"  {k}: {v}" for k, v in params.items()]),
                title="Trial Configuration"
            ))
            
            # Create configuration with suggested parameters
            config = self._create_trial_config(study_config, params, trial.number)
            
            # Train agent with configuration
            try:
                metrics = self._train_with_config(config, trial, study_config)
                
                # Get optimization metric
                metric_value = metrics.get(study_config.metric_name, float('-inf'))
                
                # Save trial results
                self._save_trial_results(study_config.study_name, trial, params, metrics)
                
                console.print(
                    f"[bold green]Trial {trial.number} completed: "
                    f"{study_config.metric_name} = {metric_value:.4f}[/bold green]"
                )
                
                return metric_value
                
            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {e}")
                if study_config.catch_exceptions:
                    return float('-inf') if study_config.direction == "maximize" else float('inf')
                raise
        
        return objective
    
    def _create_trial_config(
        self, 
        study_config: StudyConfig, 
        params: Dict[str, Any], 
        trial_number: int
    ) -> Dict[str, Any]:
        """Create configuration for trial with suggested parameters."""
        # Start with base configuration
        config = study_config.training_config.copy()
        
        # Add trial metadata
        config["experiment_name"] = f"{study_config.study_name}_trial_{trial_number}"
        config["optuna_trial"] = trial_number
        
        # Apply suggested parameters (handle nested parameters)
        for param_name, value in params.items():
            keys = param_name.split(".")
            target = config
            for key in keys[:-1]:
                if key not in target:
                    target[key] = {}
                target = target[key]
            target[keys[-1]] = value
        
        # Set training episodes
        config.setdefault("training", {})["total_updates"] = (
            study_config.episodes_per_trial // 
            config.get("training", {}).get("rollout_steps", 2048)
        )
        
        return config
    
    def _train_with_config(
        self, 
        config: Dict[str, Any], 
        trial: Trial,
        study_config: StudyConfig
    ) -> Dict[str, float]:
        """Train agent with configuration and return metrics."""
        import subprocess
        import tempfile
        
        # Save config to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name
        
        try:
            # Prepare command
            cmd = [
                sys.executable,
                "main.py",
                "--config", config["config"] if "config" in config else "momentum_training",
                "--no-dashboard",
            ]
            
            # Add overrides for all parameters
            for key, value in self._flatten_dict(config).items():
                if key not in ["config", "experiment_name", "optuna_trial"]:
                    cmd.extend([f"++{key}={value}"])
            
            # Add trial tracking
            cmd.extend([
                f"++experiment_name={config['experiment_name']}",
                f"++optuna_trial={trial.number}",
                "++wandb.tags=[optuna]",
            ])
            
            logger.info(f"Running command: {' '.join(cmd)}")
            
            # Run training subprocess
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=os.getcwd(),
            )
            
            if result.returncode != 0:
                logger.error(f"Training failed: {result.stderr}")
                raise RuntimeError(f"Training failed with code {result.returncode}")
            
            # Parse metrics from output
            metrics = self._parse_metrics_from_output(result.stdout)
            
            # Report intermediate values for pruning
            eval_steps = []
            eval_rewards = []
            for line in result.stdout.split('\n'):
                if "eval_mean_reward" in line:
                    try:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if "eval_mean_reward" in part and i + 1 < len(parts):
                                reward = float(parts[i + 1])
                                eval_rewards.append(reward)
                                eval_steps.append(len(eval_rewards))
                    except:
                        pass
            
            # Report to Optuna for pruning
            for step, reward in zip(eval_steps, eval_rewards):
                trial.report(reward, step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return metrics
            
        finally:
            # Clean up config file
            if os.path.exists(config_path):
                os.unlink(config_path)
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '') -> Dict[str, Any]:
        """Flatten nested dictionary with dot notation."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _parse_metrics_from_output(self, output: str) -> Dict[str, float]:
        """Parse metrics from training output."""
        metrics = {}
        
        # Look for final metrics in output
        lines = output.split('\n')
        for line in lines:
            # Parse W&B summary metrics
            if "wandb:" in line and "Summary" in line:
                # Start parsing summary section
                in_summary = True
                continue
            
            # Common metric patterns
            metric_patterns = [
                ("mean_reward", r"mean_reward[:\s]+(-?\d+\.?\d*)"),
                ("best_reward", r"best_reward[:\s]+(-?\d+\.?\d*)"),
                ("final_reward", r"final_reward[:\s]+(-?\d+\.?\d*)"),
                ("total_pnl", r"total_pnl[:\s]+(-?\d+\.?\d*)"),
                ("win_rate", r"win_rate[:\s]+(\d+\.?\d*)"),
                ("sharpe_ratio", r"sharpe_ratio[:\s]+(-?\d+\.?\d*)"),
            ]
            
            for metric_name, pattern in metric_patterns:
                import re
                match = re.search(pattern, line)
                if match:
                    metrics[metric_name] = float(match.group(1))
        
        # If no metrics found, try to get from final evaluation
        if "mean_reward" not in metrics:
            # Look for evaluation results
            for line in reversed(lines):
                if "eval_mean_reward" in line:
                    try:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if "eval_mean_reward" in part and i + 1 < len(parts):
                                metrics["mean_reward"] = float(parts[i + 1])
                                break
                    except:
                        pass
                    break
        
        # Default to negative infinity if no metrics found
        if not metrics:
            metrics["mean_reward"] = float('-inf')
        
        return metrics
    
    def _save_trial_results(
        self,
        study_name: str,
        trial: Trial,
        params: Dict[str, Any],
        metrics: Dict[str, float]
    ):
        """Save trial results for analysis."""
        if study_name not in self.trial_results:
            self.trial_results[study_name] = []
        
        result = {
            "trial_number": trial.number,
            "params": params,
            "metrics": metrics,
            "datetime": datetime.now().isoformat(),
            "duration": trial.datetime_complete - trial.datetime_start if trial.datetime_complete else None,
        }
        
        self.trial_results[study_name].append(result)
        
        # Save to file
        results_file = self.results_dir / f"{study_name}_results.json"
        with open(results_file, "w") as f:
            json.dump(self.trial_results[study_name], f, indent=2, default=str)
    
    def run_study(self, study_config: StudyConfig) -> Study:
        """Run optimization study."""
        console.print(f"\n[bold cyan]Starting study: {study_config.study_name}[/bold cyan]")
        
        # Create study
        study = optuna.create_study(
            study_name=study_config.study_name,
            direction=study_config.direction,
            sampler=self._create_sampler(study_config.sampler),
            pruner=self._create_pruner(study_config.pruner),
            storage=study_config.storage,
            load_if_exists=study_config.load_if_exists,
        )
        
        # Store study
        self.studies[study_config.study_name] = study
        
        # Create objective
        objective = self._create_objective(study_config)
        
        # Run optimization
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Optimizing {study_config.study_name}",
                total=study_config.n_trials
            )
            
            def callback(study: Study, trial: Trial):
                progress.update(task, advance=1)
                self._print_best_params(study)
            
            study.optimize(
                objective,
                n_trials=study_config.n_trials,
                timeout=study_config.timeout,
                n_jobs=study_config.n_jobs,
                catch=(Exception,) if study_config.catch_exceptions else (),
                callbacks=[callback],
            )
        
        # Save final results
        self._save_study_results(study, study_config)
        
        return study
    
    def _print_best_params(self, study: Study):
        """Print current best parameters."""
        if study.best_trial:
            console.print(
                f"\n[bold yellow]Current best value: {study.best_value:.4f}[/bold yellow]"
            )
    
    def _save_study_results(self, study: Study, study_config: StudyConfig):
        """Save study results and visualizations."""
        study_dir = self.results_dir / study_config.study_name
        study_dir.mkdir(exist_ok=True)
        
        # Save best parameters
        best_params_file = study_dir / "best_params.json"
        with open(best_params_file, "w") as f:
            json.dump(
                {
                    "best_value": study.best_value,
                    "best_params": study.best_params,
                    "best_trial": study.best_trial.number,
                    "n_trials": len(study.trials),
                },
                f,
                indent=2,
            )
        
        console.print(f"\n[bold green]Best parameters saved to {best_params_file}[/bold green]")
        
        # Save visualizations
        if self.spec.save_study_plots:
            self._save_visualizations(study, study_dir)
    
    def _save_visualizations(self, study: Study, study_dir: Path):
        """Save study visualizations."""
        console.print("\n[bold]Saving visualizations...[/bold]")
        
        # Optimization history
        fig = plot_optimization_history(study)
        fig.write_html(study_dir / "optimization_history.html")
        
        # Parallel coordinate plot
        fig = plot_parallel_coordinate(study)
        fig.write_html(study_dir / "parallel_coordinate.html")
        
        # Parameter importances
        try:
            fig = plot_param_importances(study)
            fig.write_html(study_dir / "param_importances.html")
        except:
            logger.warning("Could not compute parameter importances")
        
        # Slice plot
        fig = plot_slice(study)
        fig.write_html(study_dir / "slice_plot.html")
        
        # Contour plot for 2D relationships
        if len(study.best_params) >= 2:
            param_names = list(study.best_params.keys())[:2]
            fig = plot_contour(study, params=param_names)
            fig.write_html(study_dir / "contour_plot.html")
        
        # EDF plot
        fig = plot_edf([study])
        fig.write_html(study_dir / "edf_plot.html")
        
        console.print(f"[bold green]Visualizations saved to {study_dir}[/bold green]")
    
    def run_all_studies(self):
        """Run all studies in specification."""
        console.print(
            Panel(
                f"[bold]Optuna Hyperparameter Optimization[/bold]\n"
                f"Specification: {self.spec.name}\n"
                f"Studies: {len(self.spec.studies)}",
                title="Optimization Session"
            )
        )
        
        for study_config in self.spec.studies:
            study = self.run_study(study_config)
            
            # Print summary
            self._print_study_summary(study, study_config)
    
    def _print_study_summary(self, study: Study, study_config: StudyConfig):
        """Print study summary."""
        table = Table(title=f"Study Summary: {study_config.study_name}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Best value", f"{study.best_value:.4f}")
        table.add_row("Best trial", str(study.best_trial.number))
        table.add_row("Total trials", str(len(study.trials)))
        table.add_row("Completed trials", str(len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])))
        table.add_row("Pruned trials", str(len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])))
        table.add_row("Failed trials", str(len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])))
        
        console.print(table)
        
        # Print best parameters
        console.print("\n[bold]Best parameters:[/bold]")
        for param, value in study.best_params.items():
            console.print(f"  {param}: {value}")
    
    def show_best_config(self, study_name: str):
        """Show best configuration for a study."""
        if study_name not in self.studies:
            # Try to load from storage
            study = optuna.load_study(
                study_name=study_name,
                storage=self.spec.studies[0].storage,  # Use first study's storage
            )
        else:
            study = self.studies[study_name]
        
        # Find matching study config
        study_config = next(
            (s for s in self.spec.studies if s.study_name == study_name),
            None
        )
        
        if not study_config:
            console.print(f"[red]Study config not found for {study_name}[/red]")
            return
        
        # Create best configuration
        best_config = self._create_trial_config(
            study_config,
            study.best_params,
            study.best_trial.number
        )
        
        # Save to file
        best_config_file = self.results_dir / f"{study_name}_best_config.yaml"
        with open(best_config_file, "w") as f:
            yaml.dump(best_config, f, default_flow_style=False)
        
        console.print(f"\n[bold green]Best configuration saved to {best_config_file}[/bold green]")
        console.print("\n[bold]Best configuration:[/bold]")
        console.print(yaml.dump(best_config, default_flow_style=False))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Optuna hyperparameter optimization for FxAI")
    parser.add_argument(
        "--spec",
        type=str,
        help="Path to optimization specification YAML file",
    )
    parser.add_argument(
        "--study",
        type=str,
        help="Run specific study by name",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs",
    )
    parser.add_argument(
        "--show-best",
        type=str,
        help="Show best configuration for study",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch Optuna dashboard",
    )
    
    args = parser.parse_args()
    
    # Handle dashboard
    if args.dashboard:
        console.print("[bold]Launching Optuna dashboard...[/bold]")
        os.system("optuna-dashboard sqlite:///optuna_studies.db")
        return
    
    # Create optimizer
    optimizer = OptunaOptimizer(args.spec)
    
    # Handle show best
    if args.show_best:
        optimizer.show_best_config(args.show_best)
        return
    
    # Override n_jobs if specified
    if args.n_jobs > 1:
        for study in optimizer.spec.studies:
            study.n_jobs = args.n_jobs
    
    # Run studies
    if args.study:
        # Run specific study
        study_config = next(
            (s for s in optimizer.spec.studies if s.study_name == args.study),
            None
        )
        if study_config:
            optimizer.run_study(study_config)
        else:
            console.print(f"[red]Study '{args.study}' not found[/red]")
    else:
        # Run all studies
        optimizer.run_all_studies()


if __name__ == "__main__":
    main()