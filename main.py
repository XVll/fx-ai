# run_training.py
import logging
import torch
import os
from datetime import datetime

from data.data_manager import DataManager
from data.provider.data_bento.databento_file_provider import DabentoFileProvider
from simulation.simulator import Simulator
from envs.trading_env import TradingEnv
from models.transformer import MultiBranchTransformer
from agent.ppo_agent import PPOTrainer
from agent.callbacks import ModelCheckpointCallback, TensorboardCallback

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_training():
    # Create output directories
    output_dir = "./output"
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    # 1. Initialize data provider and manager
    logger.info("Initializing data provider")
    data_dir = "./dnb/mlgo"  # Your data directory
    provider = DabentoFileProvider(data_dir)
    data_manager = DataManager(provider, logger=logger)

    # 2. Set up simulator
    logger.info("Setting up simulator")
    sim_config = {
        'market_config': {'slippage_factor': 0.001},
        'execution_config': {'commission_per_share': 0.0},
        'portfolio_config': {'initial_cash': 100000.0}
    }
    simulator = Simulator(data_manager, sim_config, logger=logger)

    # 3. Initialize simulator with data
    symbol = "MLGO"  # Your trading symbol
    start_date = "2025-03-27"  # Adjust to your data range
    end_date = "2025-03-27"

    logger.info(f"Loading data for {symbol} from {start_date} to {end_date}")
    simulator.initialize_for_symbol(
        symbol, mode='backtesting',
        start_time=start_date, end_time=end_date,
        timeframes=["1m", "5m"]
    )

    # 4. Create environment
    logger.info("Creating trading environment")
    env_config = {
        'random_reset': True,  # Random episodes for training diversity
        'state_dim': 1000,  # Dimension of state space
        'max_steps': 500  # Maximum steps per episode
    }
    env = TradingEnv(simulator, env_config, logger=logger)

    # 5. Define model configuration
    model_config = {
        # Feature dimensions
        'hf_seq_len': 60,
        'hf_feat_dim': 20,
        'mf_seq_len': 30,
        'mf_feat_dim': 15,
        'lf_seq_len': 30,
        'lf_feat_dim': 10,
        'static_feat_dim': 15,

        # Model dimensions
        'd_model': 64,
        'd_fused': 256,

        # Transformer params
        'hf_layers': 2,
        'mf_layers': 2,
        'lf_layers': 2,
        'hf_heads': 4,
        'mf_heads': 4,
        'lf_heads': 4,

        # Output params
        'action_dim': 1,
        'continuous_action': True,

        # Other params
        'dropout': 0.1
    }

    # 6. Create model
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Creating multi-branch transformer model on {device}")
    model = MultiBranchTransformer(**model_config, device=device)

    # 7. Set up training callbacks
    callbacks = [
        ModelCheckpointCallback(model_dir, save_freq=5, prefix=symbol),
        TensorboardCallback(log_freq=100)
    ]

    # 8. Create trainer
    logger.info("Setting up PPO trainer")
    trainer = PPOTrainer(
        env=env,
        model=model,
        model_config=model_config,
        # Training params
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        batch_size=64,
        # Other params
        device=device,
        output_dir=output_dir,
        logger=logger,
        callbacks=callbacks
    )

    # 9. Train the model
    total_updates = 1  # Start with a small number for testing
    logger.info(f"Starting training for {total_updates} updates")
    training_stats = trainer.train(total_updates)

    # 10. Log final statistics
    logger.info("Training completed")
    logger.info(f"Total episodes: {training_stats['total_episodes']}")
    logger.info(f"Total steps: {training_stats['total_steps']}")
    logger.info(f"Best mean reward: {training_stats['best_mean_reward']}")
    logger.info(f"Best model saved to: {training_stats['best_model_path']}")


if __name__ == "__main__":
    run_training()