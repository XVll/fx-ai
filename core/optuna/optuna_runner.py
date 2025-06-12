"""Refactored Optuna hyperparameter optimization system.

This module provides a clean integration between Optuna and the new callback system,
using the evaluation system for proper metric collection.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import optuna
import yaml
from optuna import Study, Trial
from optuna.samplers import (
    TPESampler,
    CmaEsSampler,
    RandomSampler,
    GridSampler,
    NSGAIISampler,
    QMCSampler,
)
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
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.panel import Panel

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
from core.optuna.optuna_training import run_optuna_trial
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)
console = Console()


class OptunaRunner:
    """Optuna hyperparameter optimization runner with new callback integration."""
    
    def __init__(self, spec_path: Optional[str] = None):
        """Initialize Optuna runner.
        
        Args:
            spec_path: Path to optuna study specification YAML file
        """
        self.spec = self._load_spec(spec_path)
        self.results_dir = Path(self.spec.results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Set Optuna verbosity
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # Track studies and results
        self.studies: Dict[str, Study] = {}
        self.trial_results: Dict[str, List[Dict]] = {}
        
    def _load_spec(self, spec_path: Optional[str]) -> OptunaStudySpec:
        """Load study specification from YAML file."""
        if spec_path and Path(spec_path).exists():
            with open(spec_path, "r") as f:
                data = yaml.safe_load(f)
            return OptunaStudySpec(**data)
        else:
            return self._create_default_spec()
            
    def _create_default_spec(self) -> OptunaStudySpec:
        """Create default optimization specification."""
        return OptunaStudySpec(
            name="default_optimization",
            description="Default hyperparameter optimization",
            studies=[
                StudyConfig(
                    study_name="fx_ai_optimization",
                    direction="maximize",
                    metric_name="mean_reward",
                    parameters=[
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
                    ],
                    n_trials=50,
                )
            ],
        )
        
    def _create_sampler(self, config: SamplerConfig) -> optuna.samplers.BaseSampler:
        """Create Optuna sampler from configuration."""
        common_kwargs = {"seed": config.seed}
        
        sampler_map = {
            SamplerType.TPE: lambda: TPESampler(
                n_startup_trials=config.n_startup_trials,
                n_ei_candidates=config.n_ei_candidates,
                multivariate=config.multivariate,
                warn_independent_sampling=config.warn_independent_sampling,
                consider_prior=config.consider_prior,
                prior_weight=config.prior_weight,
                consider_magic_clip=config.consider_magic_clip,
                consider_endpoints=config.consider_endpoints,
                **common_kwargs,
            ),
            SamplerType.CMA_ES: lambda: CmaEsSampler(
                n_startup_trials=config.n_startup_trials,
                warn_independent_sampling=config.warn_independent_sampling,
                **common_kwargs,
            ),
            SamplerType.RANDOM: lambda: RandomSampler(**common_kwargs),
            SamplerType.GRID: lambda: GridSampler(),
            SamplerType.NSGA2: lambda: NSGAIISampler(seed=config.seed),
            SamplerType.QMC: lambda: QMCSampler(
                warn_independent_sampling=config.warn_independent_sampling,
                **common_kwargs,
            ),
        }
        
        if config.type not in sampler_map:
            raise ValueError(f"Unknown sampler type: {config.type}")
            
        return sampler_map[config.type]()
        
    def _create_pruner(
        self, config: Optional[PrunerConfig]
    ) -> Optional[optuna.pruners.BasePruner]:
        """Create Optuna pruner from configuration."""
        if not config:
            return None
            
        pruner_map = {
            PrunerType.MEDIAN: lambda: MedianPruner(
                n_startup_trials=config.n_startup_trials,
                n_warmup_steps=config.n_warmup_steps,
                interval_steps=config.interval_steps,
                n_min_trials=config.n_min_trials,
            ),
            PrunerType.PERCENTILE: lambda: PercentilePruner(
                percentile=config.percentile,
                n_startup_trials=config.n_startup_trials,
                n_warmup_steps=config.n_warmup_steps,
                interval_steps=config.interval_steps,
                n_min_trials=config.n_min_trials,
            ),
            PrunerType.SUCCESSIVE_HALVING: lambda: SuccessiveHalvingPruner(
                min_resource=config.min_resource,
                reduction_factor=config.reduction_factor,
                n_min_trials=config.n_min_trials,
            ),
            PrunerType.HYPERBAND: lambda: HyperbandPruner(
                min_resource=config.min_resource,
                max_resource=config.max_resource,
                reduction_factor=config.reduction_factor,
                n_min_trials=config.n_min_trials,
            ),
            PrunerType.THRESHOLD: lambda: ThresholdPruner(
                lower=config.lower,
                upper=config.upper,
                n_warmup_steps=config.n_warmup_steps,
                interval_steps=config.interval_steps,
            ),
            PrunerType.PATIENT: lambda: PatientPruner(
                patience=config.patience,
                min_delta=config.min_delta,
            ),
        }
        
        if config.type not in pruner_map:
            raise ValueError(f"Unknown pruner type: {config.type}")
            
        return pruner_map[config.type]()
        
    def _suggest_parameter(self, trial: Trial, param: ParameterConfig) -> Any:
        """Suggest parameter value based on distribution type."""
        if param.type == DistributionType.FLOAT:
            return trial.suggest_float(
                param.name, param.low, param.high, step=param.step
            )
        elif param.type == DistributionType.FLOAT_LOG:
            return trial.suggest_float(param.name, param.low, param.high, log=True)
        elif param.type == DistributionType.INT:
            return trial.suggest_int(
                param.name, param.low, param.high, step=param.step or 1
            )
        elif param.type == DistributionType.INT_LOG:
            return trial.suggest_int(param.name, param.low, param.high, log=True)
        elif param.type == DistributionType.CATEGORICAL:
            return trial.suggest_categorical(param.name, param.choices)
        else:
            raise ValueError(f"Unknown distribution type: {param.type}")
            
    def _apply_parameters_to_config(
        self, config: DictConfig, params: Dict[str, Any]
    ) -> None:
        """Apply suggested parameters to configuration using dot notation."""
        for param_name, value in params.items():
            if param_name == "trial_id":
                continue
                
            # Split the parameter name by dots
            keys = param_name.split(".")
            
            # Navigate to the nested location
            current = config
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
                
            # Set the value
            current[keys[-1]] = value
            
    def _run_training_with_callbacks(
        self, config: DictConfig, trial: Trial, study_config: StudyConfig
    ) -> float:
        """Run training with optuna callback integration.
        
        Args:
            config: Hydra configuration with parameters applied
            trial: Optuna trial object
            study_config: Study configuration
            
        Returns:
            Metric value for optimization
        """
        # Run training using the optuna training function
        try:
            metric_value = run_optuna_trial(
                config=config,
                trial=trial,
                metric_name=study_config.metric_name,
            )
            return metric_value
        except optuna.TrialPruned:
            # This is expected - trial was pruned
            raise
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            raise
            
    def _create_objective(self, study_config: StudyConfig):
        """Create objective function for the study."""
        
        def objective(trial: Trial) -> float:
            console.print(f"\n[bold blue]Trial {trial.number}[/bold blue] starting...")
            
            # Suggest parameters
            params = {}
            for param_config in study_config.parameters:
                value = self._suggest_parameter(trial, param_config)
                params[param_config.name] = value
                
            # Add trial metadata
            params["trial_id"] = trial.number
            
            # Log parameters
            param_str = ", ".join(
                [f"{k}={v}" for k, v in params.items() if k != "trial_id"]
            )
            console.print(f"Parameters: {param_str}")
            
            # Load base configuration
            base_config = study_config.base_config or "config"
            
            # Initialize Hydra and compose config
            with initialize_config_dir(config_dir=str(Path("config").absolute()), version_base=None):
                # Compose base config
                cfg = compose(config_name=base_config)
                
                # Apply suggested parameters
                self._apply_parameters_to_config(cfg, params)
                
                # Apply trial overrides if specified
                if study_config.trial_overrides:
                    for key, value in study_config.trial_overrides.items():
                        OmegaConf.update(cfg, key, value)
                        
                # Ensure evaluation is enabled for optuna
                if "evaluation" not in cfg:
                    cfg.evaluation = {}
                cfg.evaluation.enabled = True
                cfg.evaluation.interval = study_config.evaluation_interval or 50000
                cfg.evaluation.num_episodes = study_config.evaluation_episodes or 10
                
                # Run training with callbacks
                try:
                    metric_value = self._run_training_with_callbacks(cfg, trial, study_config)
                    
                    # Save trial results
                    self._save_trial_results(
                        study_config.study_name,
                        trial,
                        params,
                        {study_config.metric_name: metric_value},
                    )
                    
                    console.print(
                        f"✅ Trial {trial.number}: {study_config.metric_name} = {metric_value:.4f}"
                    )
                    return metric_value
                    
                except optuna.TrialPruned:
                    console.print(f"✂️ Trial {trial.number}: Pruned")
                    raise
                except Exception as e:
                    logger.error(f"Trial {trial.number} failed: {e}")
                    console.print(f"❌ Trial {trial.number}: Failed - {str(e)}")
                    
                    if study_config.catch_exceptions:
                        return (
                            float("-inf")
                            if study_config.direction == "maximize"
                            else float("inf")
                        )
                    raise
                    
        return objective
        
        
    def _save_trial_results(
        self,
        study_name: str,
        trial: Trial,
        params: Dict[str, Any],
        metrics: Dict[str, float],
    ):
        """Save trial results for analysis."""
        if study_name not in self.trial_results:
            self.trial_results[study_name] = []
            
        result = {
            "trial_number": trial.number,
            "params": params,
            "metrics": metrics,
            "datetime": datetime.now().isoformat(),
            "state": trial.state.name,
        }
        
        self.trial_results[study_name].append(result)
        
        # Save to file
        results_file = self.results_dir / f"{study_name}_results.json"
        with open(results_file, "w") as f:
            json.dump(self.trial_results[study_name], f, indent=2, default=str)
            
    def run_study(self, study_config: StudyConfig) -> Study:
        """Run optimization study.
        
        Args:
            study_config: Study configuration
            
        Returns:
            Completed Optuna study
        """
        console.print(
            Panel(
                f"[bold cyan]Starting study: {study_config.study_name}[/bold cyan]\n"
                f"Direction: {study_config.direction}\n"
                f"Metric: {study_config.metric_name}\n"
                f"Trials: {study_config.n_trials}",
                title="Optuna Study",
            )
        )
        
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
        
        # Run optimization with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Optimizing {study_config.study_name}", total=study_config.n_trials
            )
            
            def callback(study: Study, trial: Trial):
                progress.update(task, advance=1)
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    self._print_best_value(study)
                    
            study.optimize(
                objective,
                n_trials=study_config.n_trials,
                n_jobs=study_config.n_jobs,
                timeout=study_config.timeout,
                callbacks=[callback],
                show_progress_bar=False,  # We use our own progress bar
            )
            
        # Print final results
        self._print_study_summary(study)
        
        return study
        
    def _print_best_value(self, study: Study):
        """Print current best value."""
        if study.best_trial:
            console.print(
                f"[green]Best value: {study.best_value:.4f} "
                f"(Trial {study.best_trial.number})[/green]"
            )
            
    def _print_study_summary(self, study: Study):
        """Print study summary table."""
        table = Table(title=f"Study Summary: {study.study_name}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total trials", str(len(study.trials)))
        table.add_row("Completed trials", str(len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))))
        table.add_row("Pruned trials", str(len(study.get_trials(states=[optuna.trial.TrialState.PRUNED]))))
        table.add_row("Failed trials", str(len(study.get_trials(states=[optuna.trial.TrialState.FAIL]))))
        
        if study.best_trial:
            table.add_row("Best value", f"{study.best_value:.4f}")
            table.add_row("Best trial", str(study.best_trial.number))
            
            # Add best parameters
            table.add_row("", "")  # Empty row
            table.add_row("[bold]Best Parameters[/bold]", "")
            for key, value in study.best_params.items():
                if isinstance(value, float):
                    table.add_row(f"  {key}", f"{value:.6f}")
                else:
                    table.add_row(f"  {key}", str(value))
                    
        console.print(table)
        
    def run_all_studies(self) -> Dict[str, Study]:
        """Run all studies in the specification.
        
        Returns:
            Dictionary of study name to Study object
        """
        console.print(
            Panel(
                f"[bold]Running Optuna Optimization[/bold]\n"
                f"Specification: {self.spec.name}\n"
                f"Studies: {len(self.spec.studies)}",
                title="Optuna Runner",
            )
        )
        
        for i, study_config in enumerate(self.spec.studies, 1):
            console.print(f"\n[bold]Study {i}/{len(self.spec.studies)}[/bold]")
            self.run_study(study_config)
            
        # Generate visualizations if requested
        if self.spec.generate_plots:
            self._generate_visualizations()
            
        return self.studies
        
    def _generate_visualizations(self):
        """Generate visualization plots for all studies."""
        console.print("\n[bold cyan]Generating visualizations...[/bold cyan]")
        
        plots_dir = self.results_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        for study_name, study in self.studies.items():
            if len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE])) == 0:
                continue
                
            # Optimization history
            fig = plot_optimization_history(study)
            fig.write_html(plots_dir / f"{study_name}_optimization_history.html")
            
            # Parameter importance
            try:
                fig = plot_param_importances(study)
                fig.write_html(plots_dir / f"{study_name}_param_importances.html")
            except:
                pass  # May fail with some samplers
                
            # Parallel coordinates
            fig = plot_parallel_coordinate(study)
            fig.write_html(plots_dir / f"{study_name}_parallel_coordinate.html")
            
            # Contour plots
            fig = plot_contour(study)
            fig.write_html(plots_dir / f"{study_name}_contour.html")
            
        console.print(f"[green]Visualizations saved to {plots_dir}[/green]")


def main():
    """Main entry point for Optuna runner."""
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter optimization")
    parser.add_argument(
        "--spec",
        type=str,
        help="Path to study specification YAML file",
    )
    parser.add_argument(
        "--study",
        type=str,
        help="Run only a specific study by name",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch Optuna dashboard after optimization",
    )
    
    args = parser.parse_args()
    
    # Create runner
    runner = OptunaRunner(spec_path=args.spec)
    
    # Run studies
    if args.study:
        # Find specific study
        study_config = next(
            (s for s in runner.spec.studies if s.study_name == args.study), None
        )
        if not study_config:
            console.print(f"[red]Study '{args.study}' not found in specification[/red]")
            return
        runner.run_study(study_config)
    else:
        # Run all studies
        runner.run_all_studies()
        
    # Launch dashboard if requested
    if args.dashboard:
        console.print("\n[cyan]Launching Optuna dashboard...[/cyan]")
        import subprocess
        subprocess.run(["optuna-dashboard", "sqlite:///optuna_studies.db"])


if __name__ == "__main__":
    main()