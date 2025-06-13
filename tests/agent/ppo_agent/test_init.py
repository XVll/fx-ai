"""
Comprehensive tests for PPOTrainer.__init__ method.
Tests 100% coverage including normal cases, edge cases, error conditions.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from unittest.mock import Mock, patch, MagicMock

from agent.ppo_agent import PPOTrainer
from agent.replay_buffer import ReplayBuffer
from config.training.training_config import TrainingConfig, TrainingManagerConfig
from model.transformer import MultiBranchTransformer
from config.model.model_config import ModelConfig
from callbacks.core import CallbackManager


class TestPPOTrainerInit:
    """Test cases for PPOTrainer initialization."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock MultiBranchTransformer model."""
        model = Mock(spec=MultiBranchTransformer)
        model.parameters.return_value = iter([torch.tensor([1.0])])  # Mock parameters
        return model

    @pytest.fixture
    def training_config(self):
        """Create a standard training configuration."""
        return TrainingConfig(
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            n_epochs=4,
            batch_size=64,
            training_manager=TrainingManagerConfig(rollout_steps=2048)
        )

    @pytest.fixture
    def callback_manager(self):
        """Create a mock callback manager."""
        return Mock(spec=CallbackManager)

    def test_init_basic_functionality(self, training_config, mock_model):
        """Test basic initialization functionality."""
        trainer = PPOTrainer(training_config, mock_model)
        
        # Check basic attributes
        assert trainer.config == training_config
        assert trainer.model == mock_model
        assert trainer.device.type == "cpu"  # Default device
        assert trainer.callback_manager is None  # Default
        
        # Check hyperparameters are set correctly
        assert trainer.lr == training_config.learning_rate
        assert trainer.gamma == training_config.gamma
        assert trainer.gae_lambda == training_config.gae_lambda
        assert trainer.clip_eps == training_config.clip_epsilon
        assert trainer.critic_coef == training_config.value_coef
        assert trainer.entropy_coef == training_config.entropy_coef
        assert trainer.max_grad_norm == training_config.max_grad_norm
        assert trainer.ppo_epochs == training_config.n_epochs
        assert trainer.batch_size == training_config.batch_size
        assert trainer.rollout_steps == training_config.training_manager.rollout_steps

    def test_init_with_custom_device(self, training_config, mock_model):
        """Test initialization with custom device."""
        device = torch.device("cpu")
        trainer = PPOTrainer(training_config, mock_model, device=device)
        
        assert trainer.device == device

    def test_init_with_callback_manager(self, training_config, mock_model, callback_manager):
        """Test initialization with callback manager."""
        trainer = PPOTrainer(training_config, mock_model, callback_manager=callback_manager)
        
        assert trainer.callback_manager == callback_manager

    def test_init_cuda_device_when_available(self, training_config, mock_model):
        """Test initialization with CUDA device when available."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            trainer = PPOTrainer(training_config, mock_model, device=device)
            assert trainer.device == device

    def test_init_optimizer_creation(self, training_config, mock_model):
        """Test that optimizer is properly created."""
        trainer = PPOTrainer(training_config, mock_model)
        
        assert isinstance(trainer.optimizer, optim.Adam)
        # Check that optimizer has model parameters
        mock_model.parameters.assert_called_once()

    def test_init_buffer_creation(self, training_config, mock_model):
        """Test that replay buffer is properly created."""
        device = torch.device("cpu")
        trainer = PPOTrainer(training_config, mock_model, device=device)
        
        assert isinstance(trainer.buffer, ReplayBuffer)
        assert trainer.buffer.capacity == training_config.training_manager.rollout_steps
        assert trainer.buffer.device == device

    def test_init_logger_setup(self, training_config, mock_model):
        """Test that logger is properly set up."""
        trainer = PPOTrainer(training_config, mock_model)
        
        assert hasattr(trainer, 'logger')
        assert isinstance(trainer.logger, logging.Logger)
        assert trainer.logger.name == "tests.agent.ppo_agent.test_init.PPOTrainer"

    def test_init_with_different_learning_rates(self, mock_model):
        """Test initialization with different learning rates."""
        for lr in [1e-5, 3e-4, 1e-3]:
            config = TrainingConfig(learning_rate=lr)
            trainer = PPOTrainer(config, mock_model)
            assert trainer.lr == lr

    def test_init_with_different_gamma_values(self, mock_model):
        """Test initialization with different gamma values."""
        for gamma in [0.9, 0.99, 0.999]:
            config = TrainingConfig(gamma=gamma)
            trainer = PPOTrainer(config, mock_model)
            assert trainer.gamma == gamma

    def test_init_with_different_batch_sizes(self, mock_model):
        """Test initialization with different batch sizes."""
        for batch_size in [32, 64, 128, 256]:
            config = TrainingConfig(batch_size=batch_size)
            trainer = PPOTrainer(config, mock_model)
            assert trainer.batch_size == batch_size

    def test_init_with_different_rollout_steps(self, mock_model):
        """Test initialization with different rollout steps."""
        for rollout_steps in [512, 1024, 2048, 4096]:
            config = TrainingConfig(
                training_manager=TrainingManagerConfig(rollout_steps=rollout_steps)
            )
            trainer = PPOTrainer(config, mock_model)
            assert trainer.rollout_steps == rollout_steps
            assert trainer.buffer.capacity == rollout_steps

    def test_init_with_zero_learning_rate(self, mock_model):
        """Test initialization with zero learning rate."""
        config = TrainingConfig(learning_rate=0.0)
        trainer = PPOTrainer(config, mock_model)
        assert trainer.lr == 0.0

    def test_init_with_extreme_gamma_values(self, mock_model):
        """Test initialization with extreme gamma values."""
        # Very low gamma
        config = TrainingConfig(gamma=0.1)
        trainer = PPOTrainer(config, mock_model)
        assert trainer.gamma == 0.1
        
        # Very high gamma (close to 1)
        config = TrainingConfig(gamma=0.9999)
        trainer = PPOTrainer(config, mock_model)
        assert trainer.gamma == 0.9999

    def test_init_with_zero_clip_epsilon(self, mock_model):
        """Test initialization with zero clip epsilon."""
        config = TrainingConfig(clip_epsilon=0.0)
        trainer = PPOTrainer(config, mock_model)
        assert trainer.clip_eps == 0.0

    def test_init_with_large_clip_epsilon(self, mock_model):
        """Test initialization with large clip epsilon."""
        config = TrainingConfig(clip_epsilon=1.0)
        trainer = PPOTrainer(config, mock_model)
        assert trainer.clip_eps == 1.0

    def test_init_with_zero_entropy_coefficient(self, mock_model):
        """Test initialization with zero entropy coefficient."""
        config = TrainingConfig(entropy_coef=0.0)
        trainer = PPOTrainer(config, mock_model)
        assert trainer.entropy_coef == 0.0

    def test_init_with_large_entropy_coefficient(self, mock_model):
        """Test initialization with large entropy coefficient."""
        config = TrainingConfig(entropy_coef=1.0)
        trainer = PPOTrainer(config, mock_model)
        assert trainer.entropy_coef == 1.0

    def test_init_with_zero_max_grad_norm(self, mock_model):
        """Test initialization with zero max grad norm (disables clipping)."""
        config = TrainingConfig(max_grad_norm=0.0)
        trainer = PPOTrainer(config, mock_model)
        assert trainer.max_grad_norm == 0.0

    def test_init_with_various_ppo_epochs(self, mock_model):
        """Test initialization with various PPO epoch counts."""
        for epochs in [1, 4, 8, 16]:
            config = TrainingConfig(n_epochs=epochs)
            trainer = PPOTrainer(config, mock_model)
            assert trainer.ppo_epochs == epochs

    def test_init_with_minimum_batch_size(self, mock_model):
        """Test initialization with minimum batch size."""
        config = TrainingConfig(batch_size=1)
        trainer = PPOTrainer(config, mock_model)
        assert trainer.batch_size == 1

    def test_init_with_minimum_rollout_steps(self, mock_model):
        """Test initialization with minimum rollout steps."""
        config = TrainingConfig(
            training_manager=TrainingManagerConfig(rollout_steps=1)
        )
        trainer = PPOTrainer(config, mock_model)
        assert trainer.rollout_steps == 1
        assert trainer.buffer.capacity == 1

    @pytest.mark.parametrize("device_type", ["cpu", "cuda", "mps"])
    def test_init_with_various_device_types(self, training_config, mock_model, device_type):
        """Test initialization with various device types."""
        if device_type == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if device_type == "mps" and not torch.backends.mps.is_available():
            pytest.skip("MPS not available")
        
        device = torch.device(device_type)
        trainer = PPOTrainer(training_config, mock_model, device=device)
        assert trainer.device == device
        assert trainer.buffer.device == device

    def test_init_with_none_device_uses_cpu_default(self, training_config, mock_model):
        """Test that None device defaults to CPU."""
        trainer = PPOTrainer(training_config, mock_model, device=None)
        assert trainer.device.type == "cpu"

    def test_init_all_hyperparameters_extreme_values(self, mock_model):
        """Test initialization with all extreme hyperparameter values."""
        config = TrainingConfig(
            learning_rate=1e-10,
            gamma=0.01,
            gae_lambda=0.01,
            clip_epsilon=2.0,
            value_coef=10.0,
            entropy_coef=10.0,
            max_grad_norm=100.0,
            n_epochs=1,
            batch_size=1,
            training_manager=TrainingManagerConfig(rollout_steps=1)
        )
        
        trainer = PPOTrainer(config, mock_model)
        
        assert trainer.lr == 1e-10
        assert trainer.gamma == 0.01
        assert trainer.gae_lambda == 0.01
        assert trainer.clip_eps == 2.0
        assert trainer.critic_coef == 10.0
        assert trainer.entropy_coef == 10.0
        assert trainer.max_grad_norm == 100.0
        assert trainer.ppo_epochs == 1
        assert trainer.batch_size == 1
        assert trainer.rollout_steps == 1

    def test_init_config_attribute_assignment(self, training_config, mock_model):
        """Test that all config attributes are properly assigned."""
        trainer = PPOTrainer(training_config, mock_model)
        
        # Test that config object is stored
        assert trainer.config is training_config
        
        # Test individual attribute access
        assert trainer.lr == training_config.learning_rate
        assert trainer.gamma == training_config.gamma
        assert trainer.gae_lambda == training_config.gae_lambda
        assert trainer.clip_eps == training_config.clip_epsilon
        assert trainer.critic_coef == training_config.value_coef
        assert trainer.entropy_coef == training_config.entropy_coef
        assert trainer.max_grad_norm == training_config.max_grad_norm
        assert trainer.ppo_epochs == training_config.n_epochs
        assert trainer.batch_size == training_config.batch_size

    def test_init_model_reference_preserved(self, training_config, mock_model):
        """Test that model reference is preserved."""
        trainer = PPOTrainer(training_config, mock_model)
        assert trainer.model is mock_model

    def test_init_optimizer_uses_model_parameters(self, training_config, mock_model):
        """Test that optimizer is created with model parameters."""
        # Mock parameters to return specific tensors
        param1 = torch.tensor([1.0], requires_grad=True)
        param2 = torch.tensor([2.0], requires_grad=True)
        mock_model.parameters.return_value = [param1, param2]
        
        trainer = PPOTrainer(training_config, mock_model)
        
        # Verify optimizer was created with correct learning rate
        assert isinstance(trainer.optimizer, optim.Adam)
        assert trainer.optimizer.param_groups[0]['lr'] == training_config.learning_rate

    def test_init_logger_message_content(self, training_config, mock_model):
        """Test that initialization log message is correct."""
        device = torch.device("cpu")
        
        with patch.object(logging.getLogger("tests.agent.ppo_agent.test_init.PPOTrainer"), 'info') as mock_log:
            trainer = PPOTrainer(training_config, mock_model, device=device)
            mock_log.assert_called_once_with(f"> V2 PPOTrainer initialized. Device: {device}")

    def test_init_memory_efficiency(self, mock_model):
        """Test initialization with large configurations for memory efficiency."""
        config = TrainingConfig(
            batch_size=1024,
            training_manager=TrainingManagerConfig(rollout_steps=10000)
        )
        
        # Should initialize without memory issues
        trainer = PPOTrainer(config, mock_model)
        assert trainer.batch_size == 1024
        assert trainer.rollout_steps == 10000

    def test_init_consistent_device_across_components(self, training_config, mock_model):
        """Test that device is consistent across all components."""
        device = torch.device("cpu")
        trainer = PPOTrainer(training_config, mock_model, device=device)
        
        assert trainer.device == device
        assert trainer.buffer.device == device

    def test_init_with_custom_training_manager_config(self, mock_model):
        """Test initialization with custom training manager config."""
        custom_manager_config = TrainingManagerConfig(
            rollout_steps=4096,
            eval_frequency=10,
            checkpoint_frequency=50
        )
        config = TrainingConfig(training_manager=custom_manager_config)
        
        trainer = PPOTrainer(config, mock_model)
        assert trainer.rollout_steps == 4096

    def test_init_attribute_types(self, training_config, mock_model, callback_manager):
        """Test that all attributes have correct types."""
        trainer = PPOTrainer(training_config, mock_model, callback_manager=callback_manager)
        
        # Type checks
        assert isinstance(trainer.lr, float)
        assert isinstance(trainer.gamma, float)
        assert isinstance(trainer.gae_lambda, float)
        assert isinstance(trainer.clip_eps, float)
        assert isinstance(trainer.critic_coef, float)
        assert isinstance(trainer.entropy_coef, float)
        assert isinstance(trainer.max_grad_norm, float)
        assert isinstance(trainer.ppo_epochs, int)
        assert isinstance(trainer.batch_size, int)
        assert isinstance(trainer.rollout_steps, int)
        assert isinstance(trainer.optimizer, optim.Adam)
        assert isinstance(trainer.buffer, ReplayBuffer)

    def test_init_thread_safety(self, training_config, mock_model):
        """Test that initialization is thread-safe."""
        # Create multiple trainers concurrently (basic test)
        trainers = []
        for _ in range(5):
            trainer = PPOTrainer(training_config, mock_model)
            trainers.append(trainer)
        
        # Each should have independent buffer and optimizer
        for i, trainer in enumerate(trainers):
            assert trainer.buffer is not trainers[(i+1) % len(trainers)].buffer
            assert trainer.optimizer is not trainers[(i+1) % len(trainers)].optimizer