#!/usr/bin/env python3
"""
Transfer best parameters from completed phases to next phase configurations.
Usage:
    python scripts/transfer_best_params.py --from-phase 1 --to-phase 2
    python scripts/transfer_best_params.py --from-phase 2 --to-phase 3
"""

import argparse
import yaml
from pathlib import Path
from typing import Dict, Any
import optuna


def load_best_parameters(study_name: str) -> Dict[str, Any]:
    """Load best parameters from completed Optuna study."""
    try:
        study = optuna.load_study(
            study_name=study_name, storage="sqlite:///optuna_studies.db"
        )

        best_trial = study.best_trial
        best_params = best_trial.params
        best_value = best_trial.value

        print(f"✅ Loaded best parameters from study '{study_name}'")
        print(f"   Best value: {best_value:.4f}")
        print(f"   Best parameters: {len(best_params)} parameters")

        return best_params

    except Exception as e:
        print(f"❌ Error loading study '{study_name}': {e}")
        return {}


def update_config_with_params(
    config_path: Path, best_params: Dict[str, Any], param_mapping: Dict[str, str]
) -> None:
    """Update configuration file with best parameters using new base config + trial_overrides system."""

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    updated_count = 0

    # NEW: Update parameters in trial_overrides (not training_config)
    for optuna_param, config_path_str in param_mapping.items():
        if optuna_param in best_params:
            value = best_params[optuna_param]

            # Navigate nested config structure within trial_overrides
            config_parts = config_path_str.split(".")
            current = config["studies"][0]["trial_overrides"]

            # Navigate to parent
            for part in config_parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            # Set value
            current[config_parts[-1]] = value
            updated_count += 1

            print(f"   Updated {config_path_str} = {value}")

    # Write updated config
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=2)

    print(f"✅ Updated {updated_count} parameters in {config_path.name}")


def transfer_phase1_to_phase2():
    """Transfer foundation parameters to reward phase."""

    print("🔄 Transferring Phase 1 (Foundation) → Phase 2 (Reward)")

    # Load best foundation parameters
    best_params = load_best_parameters("fx_ai_foundation")
    if not best_params:
        print("❌ No best parameters found. Make sure Phase 1 completed successfully.")
        return

    # Parameter mapping: optuna_param_name -> config_path
    param_mapping = {
        "training.learning_rate": "training.learning_rate",
        "training.batch_size": "training.batch_size",
        "training.entropy_coef": "training.entropy_coef",
        "training.gamma": "training.gamma",
        "model.d_model": "model.d_model",
        "model.n_layers": "model.n_layers",
        "model.dropout": "model.dropout",
        # Note: pnl_coefficient will be re-optimized in phase 2
    }

    config_path = Path("config/optuna/phase2_reward.yaml")
    update_config_with_params(config_path, best_params, param_mapping)


def transfer_phase2_to_phase3():
    """Transfer foundation + reward parameters to fine-tune phase."""

    print("🔄 Transferring Phase 2 (Reward) → Phase 3 (Fine-tune)")

    # Load best foundation parameters
    foundation_params = load_best_parameters("fx_ai_foundation")
    if not foundation_params:
        print("❌ No foundation parameters found.")
        return

    # Load best reward parameters
    reward_params = load_best_parameters("fx_ai_reward")
    if not reward_params:
        print("❌ No reward parameters found.")
        return

    # Combine parameters
    all_best_params = {**foundation_params, **reward_params}

    # Parameter mapping for fine-tune phase
    param_mapping = {
        # Foundation parameters (fixed)
        "training.learning_rate": "training.learning_rate",
        "training.batch_size": "training.batch_size",
        "training.entropy_coef": "training.entropy_coef",
        "training.gamma": "training.gamma",
        "model.d_model": "model.d_model",
        "model.n_layers": "model.n_layers",
        "model.dropout": "model.dropout",
        # Reward parameters (fixed)
        "env.reward.pnl_coefficient": "env.reward.pnl_coefficient",
        "env.reward.holding_penalty_coefficient": "env.reward.holding_penalty_coefficient",
        "env.reward.drawdown_penalty_coefficient": "env.reward.drawdown_penalty_coefficient",
        "env.reward.profit_closing_bonus_coefficient": "env.reward.profit_closing_bonus_coefficient",
        "env.reward.base_multiplier": "env.reward.base_multiplier",
        "env.reward.bankruptcy_penalty_coefficient": "env.reward.bankruptcy_penalty_coefficient",
        "env.reward.profit_giveback_penalty_coefficient": "env.reward.profit_giveback_penalty_coefficient",
        "env.reward.max_drawdown_penalty_coefficient": "env.reward.max_drawdown_penalty_coefficient",
        "env.reward.activity_bonus_per_trade": "env.reward.activity_bonus_per_trade",
        "env.reward.hold_penalty_per_step": "env.reward.hold_penalty_per_step",
        "env.reward.max_holding_time_steps": "env.reward.max_holding_time_steps",
    }

    config_path = Path("config/optuna/phase3_finetune.yaml")
    update_config_with_params(config_path, all_best_params, param_mapping)

    print("✅ Phase 3 uses base config inheritance - no refinement ranges needed")


def show_phase_status():
    """Show status of all 3 phases."""

    print("📊 3-Phase Optimization Status")
    print("=" * 50)

    phases = [
        ("Phase 1 (Foundation)", "fx_ai_foundation"),
        ("Phase 2 (Reward)", "fx_ai_reward"),
        ("Phase 3 (Fine-tune)", "fx_ai_finetune"),
    ]

    for phase_name, study_name in phases:
        try:
            study = optuna.load_study(
                study_name=study_name, storage="sqlite:///optuna_studies.db"
            )
            n_trials = len(study.trials)

            if n_trials > 0:
                best_value = study.best_value
                completed_trials = len(
                    [
                        t
                        for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE
                    ]
                )
                print(
                    f"{phase_name}: {completed_trials}/{n_trials} trials, best={best_value:.4f}"
                )
            else:
                print(f"{phase_name}: Not started")

        except Exception:
            print(f"{phase_name}: Not found")


def main():
    parser = argparse.ArgumentParser(
        description="Transfer best parameters between Optuna phases"
    )
    parser.add_argument(
        "--from-phase", type=int, choices=[1, 2], help="Source phase (1 or 2)"
    )
    parser.add_argument(
        "--to-phase", type=int, choices=[2, 3], help="Target phase (2 or 3)"
    )
    parser.add_argument(
        "--status", action="store_true", help="Show status of all phases"
    )

    args = parser.parse_args()

    if args.status:
        show_phase_status()
        return

    if args.from_phase is None or args.to_phase is None:
        parser.print_help()
        return

    if args.from_phase == 1 and args.to_phase == 2:
        transfer_phase1_to_phase2()
    elif args.from_phase == 2 and args.to_phase == 3:
        transfer_phase2_to_phase3()
    else:
        print(
            "❌ Invalid phase combination. Use --from-phase 1 --to-phase 2 or --from-phase 2 --to-phase 3"
        )


if __name__ == "__main__":
    main()
