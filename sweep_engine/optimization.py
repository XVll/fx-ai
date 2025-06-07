"""Advanced Optuna hyperparameter optimization system for FxAI.

This is the main Optuna runner with full optimization features:
- Multi-study support
- Advanced sampling and pruning
- Visualization and analysis
- Progress tracking and results management

Usage:
    python sweep_engine/optimization.py --spec optuna-1-foundation
    
Or via poe commands:
    poetry run poe optuna-foundation
    poetry run poe optuna-reward  
    poetry run poe optuna-finetune
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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

# Add parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

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
from utils.logger import get_logger

console = Console()
logger = get_logger(__name__)


class OptunaOptimizer:
    """Standard Optuna hyperparameter optimization system."""

    def __init__(self, spec_path: Optional[str] = None):
        """Initialize optimizer with specification."""
        self.spec = self._load_spec(spec_path)
        self.results_dir = Path(self.spec.results_dir)
        self.results_dir.mkdir(exist_ok=True)

        # Set up logging - reduce Optuna verbosity
        optuna.logging.set_verbosity(optuna.logging.WARNING)

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
                    ],
                    n_trials=100,
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

    def _create_pruner(
        self, config: Optional[PrunerConfig]
    ) -> Optional[optuna.pruners.BasePruner]:
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


    def _apply_overrides_to_config(self, config, overrides: Dict[str, Any]):
        """Apply override dictionary to config object."""
        for key, value in overrides.items():
            if isinstance(value, dict):
                # Nested override
                if hasattr(config, key):
                    self._apply_overrides_to_config(getattr(config, key), value)
            else:
                # Direct value
                if hasattr(config, key):
                    setattr(config, key, value)

    def _create_objective(self, study_config: StudyConfig):
        """Create objective function for study - STANDARD OPTUNA APPROACH."""

        def objective(trial: Trial) -> float:
            # Import training function directly (standard way)
            from main import train

            console.print(f"[blue]Trial {trial.number}[/blue]: Starting...")

            # Suggest parameters 
            params = {}
            for param_config in study_config.parameters:
                value = self._suggest_parameter(trial, param_config)
                params[param_config.name] = value

            # Add trial metadata
            params["trial_id"] = trial.number

            # Use centralized config system
            from config.config import Config
            base_config_name = study_config.base_config or "optuna"
            config = Config.load(base_config_name)
            
            # Apply suggested parameters using centralized approach
            for param_name, value in params.items():
                if param_name != "trial_id":  # Skip trial metadata
                    # Use the helper from integration.py
                    from sweep_engine.integration import _set_nested_attr
                    _set_nested_attr(config, param_name, value)

            # Log parameters
            param_str = ", ".join(
                [f"{k.split('.')[-1]}={v}" for k, v in params.items() if k != "trial_id"]
            )
            console.print(f"[blue]Trial {trial.number}[/blue]: {param_str}")
            # Don't override dashboard/wandb settings - let config files control them
            # config.dashboard.enabled = False  # Disable for speed
            # config.wandb.enabled = False  # Disable for speed
            # config.optuna_trial = trial  # Removed - Config schema doesn't have this field

            # Apply trial overrides if specified
            if (
                hasattr(study_config, "trial_overrides")
                and study_config.trial_overrides
            ):
                self._apply_overrides_to_config(config, study_config.trial_overrides)

            try:
                # Call training function directly (STANDARD OPTUNA WAY!)
                training_stats = train(config)

                # Extract optimization metric
                if training_stats and not training_stats.get("interrupted", False):
                    # Debug: log what we got back
                    console.print(f"[cyan]Trial {trial.number} training_stats keys: {list(training_stats.keys())}[/cyan]")
                    
                    # Look for the specified metric
                    metric_value = training_stats.get(
                        study_config.metric_name, float("-inf")
                    )
                    console.print(f"[cyan]Trial {trial.number} {study_config.metric_name}: {metric_value}[/cyan]")

                    # Fallback to common metric names
                    if metric_value == float("-inf"):
                        for fallback in [
                            "final_eval_reward",
                            "best_reward",
                            "episode_reward",
                        ]:
                            if fallback in training_stats:
                                metric_value = training_stats[fallback]
                                console.print(f"[cyan]Using fallback {fallback}: {metric_value}[/cyan]")
                                break
                else:
                    metric_value = float("-inf")
                    console.print(f"[red]Trial {trial.number} interrupted or no stats[/red]")

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
                # Re-raise pruning (this is normal and expected)
                console.print(f"✂️ Trial {trial.number}: pruned")
                raise
            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {e}")
                console.print(f"❌ Trial {trial.number}: failed")
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
        }

        self.trial_results[study_name].append(result)

        # Save to file
        results_file = self.results_dir / f"{study_name}_results.json"
        with open(results_file, "w") as f:
            json.dump(self.trial_results[study_name], f, indent=2, default=str)

    def run_study(self, study_config: StudyConfig) -> Study:
        """Run optimization study."""
        console.print(
            f"\n[bold cyan]Starting study: {study_config.study_name}[/bold cyan]"
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
                f"Optimizing {study_config.study_name}", total=study_config.n_trials
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

        console.print(
            f"\n[bold green]Best parameters saved to {best_params_file}[/bold green]"
        )

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
                title="Optimization Session",
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
        table.add_row(
            "Completed trials",
            str(
                len(
                    [
                        t
                        for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE
                    ]
                )
            ),
        )
        table.add_row(
            "Pruned trials",
            str(
                len(
                    [
                        t
                        for t in study.trials
                        if t.state == optuna.trial.TrialState.PRUNED
                    ]
                )
            ),
        )
        table.add_row(
            "Failed trials",
            str(
                len(
                    [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
                )
            ),
        )

        console.print(table)

        # Print best parameters
        console.print("\n[bold]Best parameters:[/bold]")
        for param, value in study.best_params.items():
            console.print(f"  {param}: {value}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter optimization for FxAI"
    )
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

    args = parser.parse_args()

    # Create optimizer
    optimizer = OptunaOptimizer(args.spec)

    # Override n_jobs if specified
    if args.n_jobs > 1:
        for study in optimizer.spec.studies:
            study.n_jobs = args.n_jobs

    # Run studies
    if args.study:
        # Run specific study
        study_config = next(
            (s for s in optimizer.spec.studies if s.study_name == args.study), None
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
