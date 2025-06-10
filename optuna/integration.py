"""Simple integration - Config.load() handles everything."""

from config.config import Config


def run_optuna_study(config: Config):
    """Run optuna study using config.optuna settings."""
    if not config.optuna:
        raise ValueError("No optuna configuration found in config")
    
    import optuna
    from main import train
    
    study_config = config.optuna
    
    def objective(trial):
        # Load base config if specified, otherwise use current config as base
        if study_config.base_config:
            trial_config = Config.load(study_config.base_config)
        else:
            trial_config = Config.load()  # Fresh copy
        
        # Apply suggested parameters
        for param_config in study_config.parameters:
            if param_config.type.value in ["float", "float_log"]:
                value = trial.suggest_float(
                    param_config.name,
                    param_config.low,
                    param_config.high,
                    log=(param_config.type.value == "float_log")
                )
            elif param_config.type.value in ["int", "int_log"]:
                value = trial.suggest_int(
                    param_config.name,
                    param_config.low,
                    param_config.high,
                    log=(param_config.type.value == "int_log")
                )
            elif param_config.type.value == "categorical":
                value = trial.suggest_categorical(param_config.name, param_config.choices)
            
            # Set the parameter using dot notation
            _set_nested_attr(trial_config, param_config.name, value)
        
        # Apply trial overrides if specified
        if study_config.trial_overrides:
            _apply_dict_to_config(trial_config, study_config.trial_overrides)
        
        # Set trial-specific experiment name
        trial_config.experiment_name = f"optuna_trial_{trial.number}"
        
        # Run training
        result = train(trial_config)
        return result.get(study_config.metric_name, 0.0)
    
    # Create sampler and pruner from config
    sampler = _create_sampler(study_config.sampler)
    pruner = _create_pruner(study_config.pruner) if study_config.pruner else None
    
    # Create and run study
    study = optuna.create_study(
        study_name=study_config.study_name,
        direction=study_config.direction,
        sampler=sampler,
        pruner=pruner,
        storage=study_config.storage,
        load_if_exists=study_config.load_if_exists
    )
    
    study.optimize(objective, n_trials=study_config.n_trials, timeout=study_config.timeout)
    return study


def _create_sampler(sampler_config):
    """Create optuna sampler from config."""
    import optuna
    
    sampler_type = sampler_config.type.value
    
    if sampler_type == "TPESampler":
        return optuna.samplers.TPESampler(
            n_startup_trials=sampler_config.n_startup_trials,
            n_ei_candidates=sampler_config.n_ei_candidates,
            seed=sampler_config.seed,
            multivariate=sampler_config.multivariate,
            warn_independent_sampling=sampler_config.warn_independent_sampling
        )
    elif sampler_type == "RandomSampler":
        return optuna.samplers.RandomSampler(seed=sampler_config.seed)
    elif sampler_type == "CmaEsSampler":
        return optuna.samplers.CmaEsSampler(seed=sampler_config.seed)
    elif sampler_type == "GridSampler":
        return optuna.samplers.GridSampler()
    else:
        return optuna.samplers.TPESampler()  # Default


def _create_pruner(pruner_config):
    """Create optuna pruner from config."""
    import optuna
    
    pruner_type = pruner_config.type.value
    
    if pruner_type == "MedianPruner":
        return optuna.pruners.MedianPruner(
            n_startup_trials=pruner_config.n_startup_trials,
            n_warmup_steps=pruner_config.n_warmup_steps,
            interval_steps=pruner_config.interval_steps
        )
    elif pruner_type == "PercentilePruner":
        return optuna.pruners.PercentilePruner(
            percentile=pruner_config.percentile,
            n_startup_trials=pruner_config.n_startup_trials,
            n_warmup_steps=pruner_config.n_warmup_steps,
            interval_steps=pruner_config.interval_steps
        )
    else:
        return None  # No pruning


def _set_nested_attr(obj, name: str, value):
    """Set nested attribute using dot notation."""
    parts = name.split('.')
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def _apply_dict_to_config(config: Config, overrides: dict):
    """Apply dictionary overrides to config object."""
    for key, value in overrides.items():
        if isinstance(value, dict):
            # Handle nested dictionaries
            nested_obj = getattr(config, key)
            _apply_dict_to_config(nested_obj, value)
        else:
            setattr(config, key, value)