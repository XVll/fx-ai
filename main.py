import sys
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import typed config directly
from config.config import Config, EnvConfig, DataConfig, SimulationConfig

from data.data_manager import DataManager
from data.provider.data_bento.databento_file_provider import DabentoFileProvider
from envs.trading_env import TradingEnv
from envs.trading_simulator import TradingSimulator
from models.transformer import MultiBranchTransformer
from agent.ppo_agent import PPOTrainer
from agent.callbacks import ModelCheckpointCallback, TensorboardCallback, EarlyStoppingCallback
from agent.wandb_callback import WandbCallback

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def ensure_data_exists(data_dir: str, symbol: str, start_date: str, end_date: str) -> bool:
    """
    Check if data exists, if not, create dummy data for testing purposes

    Args:
        data_dir: Directory where data should be stored
        symbol: Trading symbol to generate data for
        start_date: Start date for data generation
        end_date: End date for data generation

    Returns:
        bool: True if data is available or was created successfully
    """
    if not os.path.exists(data_dir):
        log.warning(f"Data directory '{data_dir}' does not exist.")
        log.info("Creating directory and generating dummy data for testing...")

        os.makedirs(data_dir, exist_ok=True)

        # Create date range
        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
        except:
            # Default to single day if parsing fails
            start_dt = datetime(2025, 3, 27, 9, 30)
            end_dt = datetime(2025, 3, 27, 16, 0)

        # For 1-minute bars
        minutes = pd.date_range(start=start_dt, end=end_dt, freq='1min')

        # Create a simple price series (random walk with momentum)
        np.random.seed(42)  # For reproducibility
        price = 10.0  # Starting price
        prices = [price]
        momentum = 0

        # Generate random walk with some momentum
        for _ in range(1, len(minutes)):
            # Random price change with momentum component
            noise = np.random.normal(0.001, 0.01)
            momentum = 0.8 * momentum + 0.2 * noise
            change = momentum
            price = max(price + change * price, 0.1)  # Ensure price stays positive
            prices.append(price)

        # Create 1-minute bars dataframe
        bars_1m = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + np.random.uniform(0, 0.005)) for p in prices],
            'low': [p * (1 - np.random.uniform(0, 0.004)) for p in prices],
            'close': [p * (1 + np.random.normal(0, 0.002)) for p in prices],
            'volume': np.random.randint(100, 10000, size=len(minutes))
        }, index=minutes)

        # Create 5-minute bars by resampling
        bars_5m = bars_1m.resample('5min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })

        # Save dataframes
        log.info("Saving dummy data files...")
        bars_1m.to_csv(os.path.join(data_dir, 'bars_1m.csv'))
        bars_5m.to_csv(os.path.join(data_dir, 'bars_5m.csv'))

        # Also create dummy trades and quotes data - minimal for testing
        trades = []
        for minute, price in zip(minutes, prices):
            # Create a few trades around each minute
            for _ in range(np.random.randint(0, 5)):  # 0-4 trades per minute
                second_offset = np.random.randint(0, 60)
                trade_time = minute + timedelta(seconds=second_offset)
                trade_price = price * (1 + np.random.normal(0, 0.001))

                trades.append({
                    'timestamp': trade_time,
                    'price': trade_price,
                    'size': np.random.randint(10, 500)
                })

        if trades:
            trades_df = pd.DataFrame(trades)
            trades_df.set_index('timestamp', inplace=True)
            trades_df.to_csv(os.path.join(data_dir, 'trades.csv'))

        log.info(f"Created dummy data in {data_dir}")

        return True
    else:
        # Check if key files exist
        required_files = ['bars_1m.csv', 'bars_5m.csv']
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(data_dir, f))]

        if missing_files:
            log.warning(f"Missing data files: {missing_files}")
            return False

        return True


@hydra.main(version_base="1.2", config_path="config", config_name="config")
def run_training(cfg: Config):

    # Get output directory
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    # 1. Initialize data provider and manager
    log.info("Initializing data provider")
    data_dir = cfg.data.data_dir
    log.info(f"Data directory: {data_dir}")

    # Ensure data directory and files exist
    data_ok = ensure_data_exists(data_dir, cfg.data.symbol, cfg.data.start_date, cfg.data.end_date)
    if not data_ok:
        log.error("Data preparation failed. Cannot continue.")
        return {'error': 'Data preparation failed'}

    provider = DabentoFileProvider(data_dir)
    data_manager = DataManager(provider, logger=log)

    # 2. Set up simulator
    log.info("Setting up simulator")
    simulator = TradingSimulator(data_manager, cfg.simulation, logger=log)

    # 3. Initialize simulator with data
    symbol = cfg.data.symbol
    start_date = cfg.data.start_date
    end_date = cfg.data.end_date
    timeframes = cfg.data.timeframes

    log.info(f"Loading data for {symbol} from {start_date} to {end_date}")
    log.info(f"Using timeframes: {timeframes}")

    success = simulator.initialize_for_symbol(
        symbol,
        mode='backtesting',
        start_time=start_date,
        end_time=end_date,
        timeframes=timeframes
    )

    if not success:
        log.error(f"Failed to initialize simulator for {symbol}")
        return {'error': 'Failed to initialize simulator'}

    # 4. Create environment
    log.info("Creating trading environment")
    # Convert DictConfig to our strongly-typed EnvConfig
    env_config = OmegaConf.to_object(cfg.env)
    env = TradingEnv(simulator, env_config, logger=log)

    # 5. Select device
    if cfg.training.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            log.info("Using CUDA device")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            log.info("Using MPS (Apple Silicon) device")
        else:
            device = torch.device("cpu")
            log.info("Using CPU device")
    else:
        device = torch.device(cfg.training.device)
        log.info(f"Using specified device: {device}")

    # 6. Create model
    log.info("Creating multi-branch transformer model")
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    log.info(f"Model config: {model_config}")

    try:
        model = MultiBranchTransformer(**model_config, device=device)
        log.info(f"Model created successfully")
    except Exception as e:
        log.error(f"Error creating model: {e}")
        raise

    # 7. Set up training callbacks
    callbacks = [
        ModelCheckpointCallback(
            model_dir,
            save_freq=cfg.callbacks.save_freq,
            prefix=cfg.data.symbol
        ),
        TensorboardCallback(log_freq=cfg.callbacks.log_freq)
    ]

    # Add early stopping if enabled
    if hasattr(cfg.callbacks, "early_stopping") and cfg.callbacks.early_stopping.enabled:
        callbacks.append(
            EarlyStoppingCallback(
                patience=cfg.callbacks.early_stopping.patience,
                min_delta=cfg.callbacks.early_stopping.min_delta
            )
        )

    # Add W&B callback if enabled
    if cfg.wandb.enabled:
        try:
            # Prepare combined config for W&B
            flat_config = OmegaConf.to_container(cfg, resolve=True)

            wandb_callback = WandbCallback(
                project_name=cfg.wandb.project_name,
                entity=cfg.wandb.entity,
                log_freq=cfg.wandb.log_frequency.steps,
                config=flat_config,
                log_model=cfg.wandb.log_model,
                log_code=cfg.wandb.log_code
            )
            callbacks.append(wandb_callback)
            log.info("W&B callback added successfully")
        except Exception as e:
            log.error(f"Failed to initialize W&B: {e}")
            log.info("Continuing without W&B...")

    # 8. Create trainer
    log.info("Setting up PPO trainer")

    # Extract training parameters
    training_params = {
        "lr": cfg.training.lr,
        "gamma": cfg.training.gamma,
        "gae_lambda": cfg.training.gae_lambda,
        "clip_eps": cfg.training.clip_eps,
        "critic_coef": cfg.training.critic_coef,
        "entropy_coef": cfg.training.entropy_coef,
        "max_grad_norm": cfg.training.max_grad_norm,
        "n_epochs": cfg.training.n_epochs,
        "batch_size": cfg.training.batch_size,
        "buffer_size": cfg.training.buffer_size,
        "n_episodes_per_update": cfg.training.n_episodes_per_update,
    }

    log.info(f"Training parameters: {training_params}")

    trainer = PPOTrainer(
        env=env,
        model=model,
        model_config=model_config,
        device=device,
        output_dir=output_dir,
        logger=log,
        callbacks=callbacks,
        **training_params
    )

    # 9. Train the model
    log.info(f"Starting training for {cfg.training.total_updates} updates")
    training_stats = trainer.train(cfg.training.total_updates)

    # 10. Log final statistics
    log.info("Training completed")
    log.info(f"Total episodes: {training_stats['total_episodes']}")
    log.info(f"Total steps: {training_stats['total_steps']}")
    log.info(f"Best mean reward: {training_stats['best_mean_reward']}")
    log.info(f"Best model saved to: {training_stats['best_model_path']}")

    return training_stats


if __name__ == "__main__":
    # Set environment variable for detailed error messages
    os.environ["HYDRA_FULL_ERROR"] = "1"

    # For quick testing, add default arg
    if len(sys.argv) == 1 and "quick_test" not in sys.argv:
        sys.argv.extend(["quick_test=true"])
    run_training()