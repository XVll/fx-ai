def cleanup_resources():
    try:
        if "trainer" in current_components and "model_manager" in current_components:
            trainer = current_components["trainer"]
            model_manager = current_components["model_manager"]
            if hasattr(trainer, "model"):
                try:
                    interrupted_path = (
                        Path(model_manager.base_dir) / "interrupted_model.pt"
                    )
                    saved_path = model_manager.save_checkpoint(
                        trainer.model,
                        trainer.optimizer,
                        trainer.global_step_counter,
                        trainer.global_episode_counter,
                        trainer.global_update_counter,
                        {"interrupted": True},
                        str(interrupted_path),
                    )
                    if saved_path:
                        logging.info(f"Saved interrupted model to: {saved_path}")
                    else:
                        logging.warning("Failed to save interrupted model")
                except Exception as e:
                    logging.error(f"Error saving interrupted model: {e}")

        logging.info("Resource cleanup completed")

    except Exception as e:
        console.print(f"[bold red]Error during cleanup: {e}[/bold red]")





def get_feature_names_from_config():
    """Get feature names for each branch based on the feature system"""
    return {
        "hf": [
            "price_velocity",
            "price_acceleration",
            "tape_imbalance",
            "tape_aggression_ratio",
            "spread_compression",
            "quote_velocity",
            "quote_imbalance",
            "volume_velocity",
            "volume_acceleration",
        ],
        "mf": [
            "1m_position_in_current_candle",
            "5m_position_in_current_candle",
            "1m_body_size_relative",
            "5m_body_size_relative",
            "distance_to_ema9_1m",
            "distance_to_ema20_1m",
            "distance_to_ema9_5m",
            "distance_to_ema20_5m",
            "ema_interaction_pattern",
            "ema_crossover_dynamics",
            "ema_trend_alignment",
            "swing_high_distance_1m",
            "swing_low_distance_1m",
            "swing_high_distance_5m",
            "swing_low_distance_5m",
            "price_velocity_1m",
            "price_velocity_5m",
            "volume_velocity_1m",
            "volume_velocity_5m",
            "price_acceleration_1m",
            "price_acceleration_5m",
            "volume_acceleration_1m",
            "volume_acceleration_5m",
            "distance_to_vwap",
            "vwap_slope",
            "price_vwap_divergence",
            "vwap_interaction_dynamics",
            "vwap_breakout_quality",
            "vwap_mean_reversion_tendency",
            "relative_volume",
            "volume_surge",
            "cumulative_volume_delta",
            "volume_momentum",
            "professional_ema_system",
            "professional_vwap_analysis",
            "professional_momentum_quality",
            "professional_volatility_regime",
            "trend_acceleration",
            "volume_pattern_evolution",
            "momentum_quality",
            "pattern_maturation",
            "mf_trend_consistency",
            "mf_volume_price_divergence",
            "mf_momentum_persistence",
            "volatility_adjusted_momentum",
            "regime_relative_volume",
            "1m_position_in_previous_candle",
            "5m_position_in_previous_candle",
            "1m_upper_wick_relative",
            "1m_lower_wick_relative",
            "5m_upper_wick_relative",
            "5m_lower_wick_relative",
        ],
        "lf": [
            "daily_range_position",
            "prev_day_range_position",
            "price_change_from_prev_close",
            "support_distance",
            "resistance_distance",
            "whole_dollar_proximity",
            "half_dollar_proximity",
            "market_session_type",
            "time_of_day_sin",
            "time_of_day_cos",
            "halt_state",
            "time_since_halt",
            "distance_to_luld_up",
            "distance_to_luld_down",
            "luld_band_width",
            "session_progress",
            "market_stress",
            "session_volume_profile",
            "adaptive_support_resistance",
            "hf_momentum_summary",
            "hf_volume_dynamics",
            "hf_microstructure_quality",
        ],
        "portfolio": [
            "position_side",
            "position_size",
            "unrealized_pnl",
            "realized_pnl",
            "total_pnl",
        ],
    }



def create_model_components(
    config: Config, device: torch.device, output_dir: str, log: logging.Logger
):
    """Create model and training components with proper config passing"""
    # Model Manager with cache directory structure
    model_manager = ModelManager(
        base_dir="cache/model/checkpoint",
        best_models_dir="cache/model/best",
        model_prefix="model",
        max_best_models=config.training.keep_best_n_models,
    )

    # Model dimensions are known from config, no need to reset env here

    # Create model with ModelConfig
    model = MultiBranchTransformer(model_config=config.model, device=device, logger=log)
    logging.info("✅ Model created successfully")

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)

    return model, optimizer, model_manager




def train(config: Config):
    """Main training function with proper config passing"""
    try:
        # Data components
        data_manager = create_data_components(config, logger)
        env = create_env_components(config, data_manager, logger)

        # For momentum-based training, don't pre-setup environment session
        # The PPO agent will handle day selection and environment setup
        logger.info("🎯 Momentum-based training enabled - PPO agent will manage day selection")

        # Model components (create first to pass to callbacks)
        model, optimizer, model_manager = create_model_components(
            config, device, str(output_dir), logger
        )
        # Callback components
        callback_manager = create_callback_manager(config, logger, model)

        if callback_manager:
            env.callback_manager = callback_manager

        # Load best model if continuing
        loaded_metadata = {}
        if config.training.continue_training:
            best_model_info = model_manager.find_best_model()
            if best_model_info:
                logger.info(f"📂 Loading best model: {best_model_info['path']}")
                model, training_state = model_manager.load_model(
                    model, optimizer, best_model_info["path"]
                )
                loaded_metadata = training_state.get("metadata", {})
                logger.info(
                    f"✅ Model loaded: step={training_state.get('global_step', 0)}"
                )
            else:
                logger.info("🆕 No previous model found. Starting fresh.")
        else:
            # Starting fresh training - don't save initial model with 0 reward
            # Let the continuous training system save models after actual training progress
            logger.info("🆕 Starting fresh training - initial model will be saved after training progress")

        # Create callbacks
        callbacks = create_training_callbacks(
            config, model_manager, str(output_dir), loaded_metadata
        )
        
        # Callbacks created

        # Create trainer with simplified config-based constructor
        trainer = PPOTrainer(
            env=env,
            model=model,
            callback_manager=callback_manager,
            config=config,  # Pass full config - trainer will extract needed parameters
            device=device,
            output_dir=str(output_dir),
            callbacks=callbacks,
        )
        current_components["trainer"] = trainer
        

        # Set primary asset - this will be determined by adaptive data
        adaptive_symbols = get_adaptive_symbols(config)
        primary_symbol = adaptive_symbols[0] if adaptive_symbols else "adaptive"
        env.primary_asset = primary_symbol


        try:
            # Use new TrainingManager system
            training_stats = trainer.train_with_manager()

            return training_stats

        except KeyboardInterrupt:
            logger.warning("⚠️ Training interrupted by user")
            return {"interrupted": True}

    except Exception as e:
        logger.error(f"Critical error during training: {e}", exc_info=True)
        raise

