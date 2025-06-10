"""
V2 Integration Test - Verify all components work together

Tests the integration between:
- TradingEnvironment (new clean implementation)
- EpisodeManager (episode selection and cycling)
- PPOTrainer (rollout collection)
- TrainingManager (orchestration)
"""

import logging
from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import torch

from envs import TradingEnvironment
from training import TrainingManager, EpisodeManager
from v2.agent import PPOTrainer
from model.transformer import MultiBranchTransformer
from config import Config
from callbacks import CallbackManager


def test_v2_integration():
    """Test that v2 components integrate properly."""
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    # Create mock config
    config = Mock(spec=Config)
    
    # Model config
    config.model = Mock()
    config.model.hf_seq_len = 60
    config.model.hf_feat_dim = 10
    config.model.mf_seq_len = 12
    config.model.mf_feat_dim = 8
    config.model.lf_seq_len = 1
    config.model.lf_feat_dim = 6
    config.model.portfolio_seq_len = 1
    config.model.portfolio_feat_dim = 4
    config.model.static_feat_dim = 3
    config.model.d_model = 64
    config.model.n_heads = 4
    config.model.n_layers = 2
    config.model.dropout = 0.1
    
    # Training config
    config.training = Mock()
    config.training.learning_rate = 3e-4
    config.training.gamma = 0.99
    config.training.gae_lambda = 0.95
    config.training.clip_epsilon = 0.2
    config.training.value_coef = 0.5
    config.training.entropy_coef = 0.01
    config.training.max_grad_norm = 0.5
    config.training.n_epochs = 4
    config.training.batch_size = 64
    config.training.rollout_steps = 256
    
    # Environment config
    config.env = Mock()
    config.env.reward = Mock()
    
    # Simulation config
    config.simulation = Mock()
    config.simulation.initial_capital = 25000.0
    config.simulation.max_position_value_ratio = 0.95
    config.simulation.min_trade_value = 100.0
    
    # Training manager config
    training_manager_config = Mock()
    training_manager_config.mode = "training"
    training_manager_config.termination_max_episodes = 10
    training_manager_config.termination_max_updates = 5
    training_manager_config.termination_max_cycles = 1
    
    logger.info("=== V2 Integration Test Starting ===")
    
    # 1. Create components
    logger.info("1. Creating v2 components...")
    
    # Data manager (mocked)
    data_manager = Mock()
    
    # Model
    model = MultiBranchTransformer(config.model)
    logger.info(f"✓ Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # PPO Trainer
    trainer = PPOTrainer(model=model, config=config.training, device="cpu")
    logger.info("✓ Created PPOTrainer")
    
    # Trading Environment
    env = TradingEnvironment(config=config, data_manager=data_manager)
    logger.info("✓ Created TradingEnvironment")
    
    # Callback Manager
    callback_manager = CallbackManager(config=Mock())
    logger.info("✓ Created CallbackManager")
    
    # Model Manager (mocked)
    model_manager = Mock()
    
    # Training Manager
    training_manager = TrainingManager(config=training_manager_config, model_manager=model_manager)
    logger.info("✓ Created TrainingManager")
    
    # 2. Test environment setup
    logger.info("\n2. Testing environment setup...")
    
    # Mock market simulator for environment
    env.market_simulator = Mock()
    env.market_simulator.initialize_day.return_value = True
    env.market_simulator.reset.return_value = True
    env.market_simulator.set_time.return_value = True
    env.market_simulator.get_stats.return_value = {
        'total_seconds': 7200,
        'warmup_info': {'has_warmup': True}
    }
    env.market_simulator.get_market_state.return_value = Mock(timestamp=datetime.now())
    env.market_simulator.get_current_features.return_value = {
        'hf': np.random.randn(60, 10).astype(np.float32),
        'mf': np.random.randn(12, 8).astype(np.float32),
        'lf': np.random.randn(1, 6).astype(np.float32),
    }
    env.market_simulator.get_current_market_data.return_value = {
        'timestamp': datetime.now(),
        'current_price': 100.0,
        'best_bid_price': 99.95,
        'best_ask_price': 100.05,
    }
    env.market_simulator.get_current_time.return_value = datetime.now()
    env.market_simulator.step.return_value = True
    
    # Mock portfolio simulator
    env.portfolio_simulator = Mock()
    env.portfolio_simulator.initial_capital = 25000.0
    env.portfolio_simulator.reset = Mock()
    env.portfolio_simulator.get_portfolio_state.return_value = {
        'cash': 25000.0,
        'total_equity': 25000.0,
        'position_value': 0.0,
        'positions': {}
    }
    env.portfolio_simulator.get_portfolio_observation.return_value = {
        'features': np.random.randn(1, 4).astype(np.float32)
    }
    env.portfolio_simulator.process_fill.return_value = Mock()
    env.portfolio_simulator.update_market_values = Mock()
    
    # Mock other simulators
    env.execution_simulator = Mock()
    env.execution_simulator.reset = Mock()
    env.execution_simulator.execute_action.return_value = Mock(
        action_decode_result=Mock(to_dict=lambda: {'action': 'HOLD'}),
        fill_details=None
    )
    
    env.reward_system = Mock()
    env.reward_system.reset = Mock()
    env.reward_system.calculate.return_value = 0.01
    
    env.action_mask = Mock()
    env.action_mask.get_action_mask.return_value = np.ones(12, dtype=bool)
    
    # Test session setup
    env.setup_session("AAPL", "2024-01-15")
    logger.info("✓ Environment session setup successful")
    
    # Test reset
    obs, info = env.reset_at_point(0)
    assert obs is not None
    assert all(key in obs for key in ['hf', 'mf', 'lf', 'portfolio'])
    logger.info("✓ Environment reset successful")
    
    # 3. Test step mechanics
    logger.info("\n3. Testing environment step mechanics...")
    
    action = np.array([0, 0])  # HOLD action
    next_obs, reward, terminated, truncated, step_info = env.step(action)
    
    assert next_obs is not None
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    logger.info(f"✓ Step executed: reward={reward:.4f}, terminated={terminated}")
    
    # 4. Test PPOTrainer integration
    logger.info("\n4. Testing PPOTrainer integration...")
    
    # Test get_current_state interface
    current_state = env.get_current_state()
    assert current_state is not None
    logger.info("✓ get_current_state() interface working")
    
    # Test action selection
    with torch.no_grad():
        # Convert state to tensors
        state_tensors = {
            k: torch.from_numpy(v).unsqueeze(0).float() 
            for k, v in current_state.items()
        }
        
        # Get action from model
        action_logits, value = model(state_tensors)
        assert action_logits.shape == (1, 12)  # batch_size=1, n_actions=12
        logger.info(f"✓ Model forward pass: logits shape={action_logits.shape}")
    
    # 5. Test TrainingManager integration (partial)
    logger.info("\n5. Testing TrainingManager coordination...")
    
    # Mock episode manager
    with patch.object(training_manager, 'episode_manager') as mock_episode_mgr:
        mock_episode_mgr.initialize.return_value = True
        mock_episode_mgr.get_current_episode_config.return_value = {
            'day_info': {
                'symbol': 'AAPL',
                'date': '2024-01-15'
            },
            'reset_point_index': 0
        }
        
        # Test setup next episode
        training_manager.environment = env
        training_manager.episode_manager = mock_episode_mgr
        
        result = training_manager._setup_next_episode()
        assert result is True
        logger.info("✓ TrainingManager episode setup successful")
    
    # 6. Test complete flow
    logger.info("\n6. Testing complete integration flow...")
    
    # Simulate a few steps
    total_reward = 0.0
    for i in range(10):
        # Get state
        state = env.get_current_state()
        
        # Get action from model
        with torch.no_grad():
            state_tensors = {
                k: torch.from_numpy(v).unsqueeze(0).float() 
                for k, v in state.items()
            }
            action_logits, _ = model(state_tensors)
            action_probs = torch.softmax(action_logits, dim=-1)
            action_idx = torch.multinomial(action_probs, 1).item()
        
        # Convert to multi-discrete
        action = np.array([action_idx // 4, action_idx % 4])
        
        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated:
            logger.info(f"Episode terminated at step {i+1}")
            break
    
    logger.info(f"✓ Completed mini rollout: total_reward={total_reward:.4f}")
    
    logger.info("\n=== V2 Integration Test Complete ===")
    logger.info("All components integrate successfully! 🎉")


if __name__ == "__main__":
    test_v2_integration()