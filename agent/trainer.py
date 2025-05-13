# train.py
import os
import argparse
import logging
import numpy as np
import torch
from datetime import datetime

from data.data_manager import DataManager
from data.provider.data_bento.databento_file_provider import DabentoFileProvider
from feature.feature_extractor import FeatureExtractor
from simulation.trading_simulator import Simulator
from envs.trading_env import TradingEnv
from models.transformer import MultiBranchTransformer
from agent.ppo_agent import PPOTrainer
from agent.callbacks import ModelCheckpointCallback, TensorboardCallback, EarlyStoppingCallback


def parse_args():
    parser = argparse.ArgumentParser(description="Train AI Trading Agent")

    # Data and environment params
    parser.add_argument("--data-dir", type=str, default="./data", help="Path to data directory")
    parser.add_argument("--symbol", type=str, default="MLGO", help="Trading symbol")
    parser.add_argument("--start-date", type=str, default="2025-01-01", help="Start date for training")
    parser.add_argument("--end-date", type=str, default="2025-03-31", help="End date for training")

    # Model params
    parser.add_argument("--d-model", type=int, default=64, help="Model dimension")
    parser.add_argument("--d-fused", type=int, default=256, help="Fused model dimension")
    parser.add_argument("--hf-layers", type=int, default=2, help="Number of HF transformer layers")
    parser.add_argument("--mf-layers", type=int, default=2, help="Number of MF transformer layers")
    parser.add_argument("--lf-layers", type=int, default=2, help="Number of LF transformer layers")

    # Training params
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--total-updates", type=int, default=100, help="Total number of updates")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")

    # Output and logging
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--save-freq", type=int, default=5, help="Model save frequency (updates)")

    # Hardware
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda/cpu)")

    return parser.parse_args()


def setup_logging(log_dir, log_level):
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")

    # Configure root logger
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def main():
    # Parse command line arguments
    args = parse_args()

    # Set up logging
    log_dir = os.path.join(args.output_dir, "logs")
    logger = setup_logging(log_dir, args.log_level)

    # Create output directories
    model_dir = os.path.join(args.output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    logger.info("Starting AI Trading training")
    logger.info(f"Using device: {args.device}")

    # 1. Initialize data provider and manager
    logger.info("Initializing data provider")
    provider = DabentoFileProvider(args.data_dir)
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
    logger.info(f"Loading data for {args.symbol} from {args.start_date} to {args.end_date}")
    simulator.initialize_for_symbol(
        args.symbol, mode='backtesting',
        start_time=args.start_date, end_time=args.end_date,
        timeframes=["1m", "5m"]
    )

    # 4. Create environment
    logger.info("Creating trading environment")
    env_config = {
        'random_reset': True,  # Random episodes for training diversity
        'state_dim': 1000,  # Dimension of state space, should match model input
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
        'd_model': args.d_model,
        'd_fused': args.d_fused,

        # Transformer params
        'hf_layers': args.hf_layers,
        'mf_layers': args.mf_layers,
        'lf_layers': args.lf_layers,
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
    logger.info("Creating multi-branch transformer model")
    model = MultiBranchTransformer(**model_config, device=args.device)

    # 7. Set up training callbacks
    callbacks = [
        ModelCheckpointCallback(model_dir, save_freq=args.save_freq, prefix=args.symbol),
        TensorboardCallback(log_freq=100),
        EarlyStoppingCallback(patience=10)
    ]

    # 8. Create trainer
    logger.info("Setting up PPO trainer")
    trainer = PPOTrainer(
        env=env,
        model=model,
        model_config=model_config,
        # Training params
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        batch_size=args.batch_size,
        # Other params
        device=args.device,
        output_dir=args.output_dir,
        logger=logger,
        callbacks=callbacks
    )

    # 9. Train the model
    logger.info(f"Starting training for {args.total_updates} updates")
    training_stats = trainer.train(args.total_updates)

    # 10. Log final statistics
    logger.info("Training completed")
    logger.info(f"Total episodes: {training_stats['total_episodes']}")
    logger.info(f"Total steps: {training_stats['total_steps']}")
    logger.info(f"Best mean reward: {training_stats['best_mean_reward']}")
    logger.info(f"Best model saved to: {training_stats['best_model_path']}")

    return 0


if __name__ == "__main__":
    main()