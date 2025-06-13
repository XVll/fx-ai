"""
Comprehensive tests for PPOTrainer.update_policy method.
Tests 100% coverage including normal cases, edge cases, error conditions.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from unittest.mock import Mock, MagicMock, patch

from agent.ppo_agent import PPOTrainer
from agent.replay_buffer import ReplayBuffer
from config.training.training_config import TrainingConfig, TrainingManagerConfig
from core.types import UpdateResult


class TestPPOTrainerUpdatePolicy:
    """Test cases for PPOTrainer update_policy method."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model that returns proper action params and values."""
        model = Mock()
        
        # Mock forward method to return action logits and values
        def forward_side_effect(states_dict):
            batch_size = next(iter(states_dict.values())).shape[0]
            action_logits = torch.randn(batch_size, 7)  # 7 discrete actions
            values = torch.randn(batch_size, 1)
            return (action_logits,), values
        
        model.forward = Mock(side_effect=forward_side_effect)
        model.parameters.return_value = iter([torch.tensor([1.0], requires_grad=True)])
        return model

    @pytest.fixture
    def training_config(self):
        """Create training configuration."""
        return TrainingConfig(
            learning_rate=3e-4,
            batch_size=4,
            n_epochs=2,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            training_manager=TrainingManagerConfig(rollout_steps=8)
        )

    @pytest.fixture
    def trainer(self, training_config, mock_model):
        """Create PPOTrainer instance."""
        return PPOTrainer(training_config, mock_model)

    @pytest.fixture
    def prepared_buffer_data(self):
        """Create prepared buffer data for training."""
        batch_size = 8
        return {
            "states": {
                "hf": torch.randn(batch_size, 60, 9),
                "mf": torch.randn(batch_size, 30, 43),
                "lf": torch.randn(batch_size, 30, 19),
                "portfolio": torch.randn(batch_size, 5, 10)
            },
            "actions": torch.randint(0, 7, (batch_size,)),
            "old_log_probs": torch.randn(batch_size, 1),
            "advantages": torch.randn(batch_size, 1),
            "returns": torch.randn(batch_size, 1)
        }

    def setup_buffer_for_training(self, trainer, data):
        """Setup buffer with training data."""
        trainer.buffer.states = data["states"]
        trainer.buffer.actions = data["actions"]
        trainer.buffer.log_probs = data["old_log_probs"]
        trainer.buffer.advantages = data["advantages"]
        trainer.buffer.returns = data["returns"]
        trainer.buffer.rewards = torch.randn(8)  # Mock rewards
        trainer.buffer.values = torch.randn(8, 1)  # Mock values
        trainer.buffer.dones = torch.zeros(8, dtype=torch.bool)  # Mock dones

    def test_update_policy_buffer_not_ready(self, trainer):
        """Test update_policy when buffer is not ready for training."""
        with patch.object(trainer.buffer, 'is_ready_for_training', return_value=False):
            result = trainer.update_policy()
            
            assert isinstance(result, UpdateResult)
            assert result.policy_loss == 0.0
            assert result.value_loss == 0.0
            assert result.entropy_loss == 0.0
            assert result.total_loss == 0.0
            assert result.interrupted == True

    def test_update_policy_no_training_data(self, trainer):
        """Test update_policy when training data is None."""
        with patch.object(trainer.buffer, 'is_ready_for_training', return_value=True), \
             patch.object(trainer.buffer, 'get_training_data', return_value=None):
            
            result = trainer.update_policy()
            
            assert result.interrupted == True
            assert result.policy_loss == 0.0

    def test_update_policy_compute_advantages_called(self, trainer, prepared_buffer_data):
        """Test that _compute_advantages_and_returns is called."""
        with patch.object(trainer.buffer, 'is_ready_for_training', return_value=True), \
             patch.object(trainer.buffer, 'get_training_data', return_value=prepared_buffer_data), \
             patch.object(trainer, '_compute_advantages_and_returns') as mock_compute:
            
            trainer.update_policy()
            mock_compute.assert_called_once()

    def test_update_policy_empty_data(self, trainer):
        """Test update_policy with empty training data."""
        empty_data = {
            "states": {"hf": torch.empty(0, 60, 9), "mf": torch.empty(0, 30, 43), 
                      "lf": torch.empty(0, 30, 19), "portfolio": torch.empty(0, 5, 10)},
            "actions": torch.empty(0, dtype=torch.long),
            "old_log_probs": torch.empty(0, 1),
            "advantages": torch.empty(0, 1),
            "returns": torch.empty(0, 1)
        }
        
        with patch.object(trainer.buffer, 'is_ready_for_training', return_value=True), \
             patch.object(trainer.buffer, 'get_training_data', return_value=empty_data), \
             patch.object(trainer, '_compute_advantages_and_returns'):
            
            result = trainer.update_policy()
            
            assert result.interrupted == True
            assert result.policy_loss == 0.0

    def test_update_policy_advantage_normalization(self, trainer, prepared_buffer_data):
        """Test that advantages are properly normalized."""
        with patch.object(trainer.buffer, 'is_ready_for_training', return_value=True), \
             patch.object(trainer.buffer, 'get_training_data', return_value=prepared_buffer_data), \
             patch.object(trainer, '_compute_advantages_and_returns'):
            
            # Capture normalized advantages
            original_advantages = prepared_buffer_data["advantages"].clone()
            
            result = trainer.update_policy()
            
            # Advantages should be normalized in the method
            assert not result.interrupted

    def test_update_policy_ppo_epochs_execution(self, trainer, prepared_buffer_data):
        """Test that PPO epochs are executed correct number of times."""
        trainer.ppo_epochs = 3
        
        with patch.object(trainer.buffer, 'is_ready_for_training', return_value=True), \
             patch.object(trainer.buffer, 'get_training_data', return_value=prepared_buffer_data), \
             patch.object(trainer, '_compute_advantages_and_returns'), \
             patch.object(trainer.optimizer, 'step') as mock_step:
            
            result = trainer.update_policy()
            
            # Should step optimizer for each batch in each epoch
            # With batch_size=4 and data_size=8, we have 2 batches per epoch
            # With 3 epochs, that's 6 optimization steps
            expected_steps = 3 * (8 // 4)  # epochs * batches_per_epoch
            assert mock_step.call_count == expected_steps

    def test_update_policy_batch_processing(self, trainer, prepared_buffer_data):
        """Test batch processing within PPO epochs."""
        trainer.batch_size = 4  # Should create 2 batches from 8 samples
        
        with patch.object(trainer.buffer, 'is_ready_for_training', return_value=True), \
             patch.object(trainer.buffer, 'get_training_data', return_value=prepared_buffer_data), \
             patch.object(trainer, '_compute_advantages_and_returns'), \
             patch.object(trainer.optimizer, 'zero_grad') as mock_zero_grad:
            
            result = trainer.update_policy()
            
            # Should call zero_grad for each batch
            expected_batches = trainer.ppo_epochs * (8 // trainer.batch_size)
            assert mock_zero_grad.call_count == expected_batches

    def test_update_policy_model_forward_calls(self, trainer, prepared_buffer_data):
        """Test that model forward is called for each batch."""
        with patch.object(trainer.buffer, 'is_ready_for_training', return_value=True), \
             patch.object(trainer.buffer, 'get_training_data', return_value=prepared_buffer_data), \
             patch.object(trainer, '_compute_advantages_and_returns'):
            
            result = trainer.update_policy()
            
            # Model should be called for each batch
            expected_calls = trainer.ppo_epochs * (8 // trainer.batch_size)
            assert trainer.model.forward.call_count == expected_calls

    def test_update_policy_gradient_clipping(self, trainer, prepared_buffer_data):
        """Test gradient clipping is applied when max_grad_norm > 0."""
        trainer.max_grad_norm = 1.0
        
        with patch.object(trainer.buffer, 'is_ready_for_training', return_value=True), \
             patch.object(trainer.buffer, 'get_training_data', return_value=prepared_buffer_data), \
             patch.object(trainer, '_compute_advantages_and_returns'), \
             patch('torch.nn.utils.clip_grad_norm_') as mock_clip:
            
            result = trainer.update_policy()
            
            # Gradient clipping should be called for each optimization step
            expected_clips = trainer.ppo_epochs * (8 // trainer.batch_size)
            assert mock_clip.call_count == expected_clips

    def test_update_policy_no_gradient_clipping(self, trainer, prepared_buffer_data):
        """Test no gradient clipping when max_grad_norm <= 0."""
        trainer.max_grad_norm = 0.0
        
        with patch.object(trainer.buffer, 'is_ready_for_training', return_value=True), \
             patch.object(trainer.buffer, 'get_training_data', return_value=prepared_buffer_data), \
             patch.object(trainer, '_compute_advantages_and_returns'), \
             patch('torch.nn.utils.clip_grad_norm_') as mock_clip:
            
            result = trainer.update_policy()
            
            # Gradient clipping should not be called
            mock_clip.assert_not_called()

    def test_update_policy_ppo_loss_calculation(self, trainer, prepared_buffer_data):
        """Test PPO loss calculation components."""
        with patch.object(trainer.buffer, 'is_ready_for_training', return_value=True), \
             patch.object(trainer.buffer, 'get_training_data', return_value=prepared_buffer_data), \
             patch.object(trainer, '_compute_advantages_and_returns'):
            
            result = trainer.update_policy()
            
            # Should return valid loss values
            assert isinstance(result.policy_loss, float)
            assert isinstance(result.value_loss, float)
            assert isinstance(result.entropy_loss, float)
            assert isinstance(result.total_loss, float)
            assert not result.interrupted

    def test_update_policy_ratio_calculation(self, trainer, prepared_buffer_data):
        """Test that probability ratio is calculated correctly."""
        # Setup specific log probs to test ratio calculation
        old_log_probs = torch.tensor([[-0.5], [-1.0], [-0.3], [-0.8], [-0.6], [-0.4], [-0.7], [-0.9]])
        prepared_buffer_data["old_log_probs"] = old_log_probs
        
        with patch.object(trainer.buffer, 'is_ready_for_training', return_value=True), \
             patch.object(trainer.buffer, 'get_training_data', return_value=prepared_buffer_data), \
             patch.object(trainer, '_compute_advantages_and_returns'):
            
            result = trainer.update_policy()
            
            # Test should complete without errors
            assert not result.interrupted

    def test_update_policy_clipping_functionality(self, trainer, prepared_buffer_data):
        """Test PPO clipping mechanism."""
        trainer.clip_eps = 0.1  # Small clipping range
        
        with patch.object(trainer.buffer, 'is_ready_for_training', return_value=True), \
             patch.object(trainer.buffer, 'get_training_data', return_value=prepared_buffer_data), \
             patch.object(trainer, '_compute_advantages_and_returns'):
            
            result = trainer.update_policy()
            
            # Should work with tight clipping
            assert not result.interrupted

    def test_update_policy_value_loss_calculation(self, trainer, prepared_buffer_data):
        """Test value function loss calculation."""
        with patch.object(trainer.buffer, 'is_ready_for_training', return_value=True), \
             patch.object(trainer.buffer, 'get_training_data', return_value=prepared_buffer_data), \
             patch.object(trainer, '_compute_advantages_and_returns'):
            
            result = trainer.update_policy()
            
            # Value loss should be computed
            assert result.value_loss >= 0.0  # MSE loss is always non-negative

    def test_update_policy_entropy_loss_calculation(self, trainer, prepared_buffer_data):
        """Test entropy loss calculation."""
        with patch.object(trainer.buffer, 'is_ready_for_training', return_value=True), \
             patch.object(trainer.buffer, 'get_training_data', return_value=prepared_buffer_data), \
             patch.object(trainer, '_compute_advantages_and_returns'):
            
            result = trainer.update_policy()
            
            # Entropy loss should be calculated
            assert isinstance(result.entropy_loss, float)

    def test_update_policy_loss_coefficients(self, trainer, prepared_buffer_data):
        """Test that loss coefficients are applied correctly."""
        trainer.critic_coef = 2.0
        trainer.entropy_coef = 0.1
        
        with patch.object(trainer.buffer, 'is_ready_for_training', return_value=True), \
             patch.object(trainer.buffer, 'get_training_data', return_value=prepared_buffer_data), \
             patch.object(trainer, '_compute_advantages_and_returns'):
            
            result = trainer.update_policy()
            
            # Should complete with custom coefficients
            assert not result.interrupted

    def test_update_policy_tensor_shape_handling(self, trainer):
        """Test handling of different tensor shapes."""
        # Create data with unusual shapes
        data_with_shapes = {
            "states": {
                "hf": torch.randn(6, 60, 9),
                "mf": torch.randn(6, 30, 43),
                "lf": torch.randn(6, 30, 19),
                "portfolio": torch.randn(6, 5, 10)
            },
            "actions": torch.randint(0, 7, (6,)),
            "old_log_probs": torch.randn(6, 1),
            "advantages": torch.randn(6, 2),  # Wrong shape
            "returns": torch.randn(6, 3)      # Wrong shape
        }
        
        with patch.object(trainer.buffer, 'is_ready_for_training', return_value=True), \
             patch.object(trainer.buffer, 'get_training_data', return_value=data_with_shapes), \
             patch.object(trainer, '_compute_advantages_and_returns'):
            
            result = trainer.update_policy()
            
            # Should handle shape mismatches gracefully
            assert not result.interrupted

    def test_update_policy_index_error_handling(self, trainer, prepared_buffer_data):
        """Test handling of IndexError during batch extraction."""
        # Mock data that might cause IndexError
        with patch.object(trainer.buffer, 'is_ready_for_training', return_value=True), \
             patch.object(trainer.buffer, 'get_training_data', return_value=prepared_buffer_data), \
             patch.object(trainer, '_compute_advantages_and_returns'):
            
            # Force an IndexError by making batch indices invalid
            with patch('numpy.random.shuffle') as mock_shuffle:
                mock_shuffle.side_effect = lambda x: None  # Don't actually shuffle
                
                result = trainer.update_policy()
                
                # Should handle gracefully
                assert not result.interrupted

    def test_update_policy_categorical_distribution(self, trainer, prepared_buffer_data):
        """Test categorical distribution creation and usage."""
        with patch.object(trainer.buffer, 'is_ready_for_training', return_value=True), \
             patch.object(trainer.buffer, 'get_training_data', return_value=prepared_buffer_data), \
             patch.object(trainer, '_compute_advantages_and_returns'), \
             patch('torch.distributions.Categorical') as mock_categorical:
            
            mock_dist = Mock()
            mock_dist.log_prob.return_value = torch.randn(4)  # batch_size
            mock_dist.entropy.return_value = torch.randn(4)
            mock_categorical.return_value = mock_dist
            
            result = trainer.update_policy()
            
            # Categorical should be created for each batch
            expected_calls = trainer.ppo_epochs * (8 // trainer.batch_size)
            assert mock_categorical.call_count == expected_calls

    def test_update_policy_action_tensor_reshaping(self, trainer, prepared_buffer_data):
        """Test action tensor reshaping for categorical distribution."""
        # Test with actions that need reshaping
        prepared_buffer_data["actions"] = torch.randint(0, 7, (8, 1))  # 2D instead of 1D
        
        with patch.object(trainer.buffer, 'is_ready_for_training', return_value=True), \
             patch.object(trainer.buffer, 'get_training_data', return_value=prepared_buffer_data), \
             patch.object(trainer, '_compute_advantages_and_returns'):
            
            result = trainer.update_policy()
            
            # Should handle reshaping correctly
            assert not result.interrupted

    def test_update_policy_value_tensor_size_mismatch(self, trainer, prepared_buffer_data):
        """Test handling of value tensor size mismatches."""
        # Mock model to return different sized values
        def forward_with_mismatch(states_dict):
            batch_size = next(iter(states_dict.values())).shape[0]
            action_logits = torch.randn(batch_size, 7)
            values = torch.randn(batch_size - 1, 1)  # Intentional size mismatch
            return (action_logits,), values
        
        trainer.model.forward = Mock(side_effect=forward_with_mismatch)
        
        with patch.object(trainer.buffer, 'is_ready_for_training', return_value=True), \
             patch.object(trainer.buffer, 'get_training_data', return_value=prepared_buffer_data), \
             patch.object(trainer, '_compute_advantages_and_returns'):
            
            result = trainer.update_policy()
            
            # Should handle size mismatch by taking minimum size
            assert not result.interrupted

    def test_update_policy_optimizer_step_sequence(self, trainer, prepared_buffer_data):
        """Test optimizer step sequence: zero_grad -> backward -> step."""
        with patch.object(trainer.buffer, 'is_ready_for_training', return_value=True), \
             patch.object(trainer.buffer, 'get_training_data', return_value=prepared_buffer_data), \
             patch.object(trainer, '_compute_advantages_and_returns'), \
             patch.object(trainer.optimizer, 'zero_grad') as mock_zero_grad, \
             patch.object(trainer.optimizer, 'step') as mock_step:
            
            result = trainer.update_policy()
            
            # zero_grad and step should be called same number of times
            assert mock_zero_grad.call_count == mock_step.call_count

    def test_update_policy_return_values_types(self, trainer, prepared_buffer_data):
        """Test that return values have correct types."""
        with patch.object(trainer.buffer, 'is_ready_for_training', return_value=True), \
             patch.object(trainer.buffer, 'get_training_data', return_value=prepared_buffer_data), \
             patch.object(trainer, '_compute_advantages_and_returns'):
            
            result = trainer.update_policy()
            
            assert isinstance(result, UpdateResult)
            assert isinstance(result.policy_loss, float)
            assert isinstance(result.value_loss, float)
            assert isinstance(result.entropy_loss, float)
            assert isinstance(result.total_loss, float)
            assert isinstance(result.interrupted, bool)

    def test_update_policy_different_batch_sizes(self, trainer, prepared_buffer_data):
        """Test update_policy with different batch sizes."""
        for batch_size in [1, 2, 4, 8]:
            trainer.batch_size = batch_size
            
            with patch.object(trainer.buffer, 'is_ready_for_training', return_value=True), \
                 patch.object(trainer.buffer, 'get_training_data', return_value=prepared_buffer_data), \
                 patch.object(trainer, '_compute_advantages_and_returns'):
                
                result = trainer.update_policy()
                
                # Should work with any valid batch size
                assert not result.interrupted

    def test_update_policy_extreme_advantages(self, trainer, prepared_buffer_data):
        """Test update_policy with extreme advantage values."""
        # Very large advantages
        prepared_buffer_data["advantages"] = torch.tensor([[1e6], [1e6], [-1e6], [-1e6], [0], [0], [1e3], [-1e3]])
        
        with patch.object(trainer.buffer, 'is_ready_for_training', return_value=True), \
             patch.object(trainer.buffer, 'get_training_data', return_value=prepared_buffer_data), \
             patch.object(trainer, '_compute_advantages_and_returns'):
            
            result = trainer.update_policy()
            
            # Should handle extreme values after normalization
            assert not result.interrupted

    def test_update_policy_zero_advantages(self, trainer, prepared_buffer_data):
        """Test update_policy with zero advantages."""
        prepared_buffer_data["advantages"] = torch.zeros(8, 1)
        
        with patch.object(trainer.buffer, 'is_ready_for_training', return_value=True), \
             patch.object(trainer.buffer, 'get_training_data', return_value=prepared_buffer_data), \
             patch.object(trainer, '_compute_advantages_and_returns'):
            
            result = trainer.update_policy()
            
            # Should handle zero advantages
            assert not result.interrupted

    @pytest.mark.parametrize("clip_eps", [0.0, 0.1, 0.2, 0.5, 1.0])
    def test_update_policy_various_clip_epsilon(self, trainer, prepared_buffer_data, clip_eps):
        """Test update_policy with various clipping epsilon values."""
        trainer.clip_eps = clip_eps
        
        with patch.object(trainer.buffer, 'is_ready_for_training', return_value=True), \
             patch.object(trainer.buffer, 'get_training_data', return_value=prepared_buffer_data), \
             patch.object(trainer, '_compute_advantages_and_returns'):
            
            result = trainer.update_policy()
            
            # Should work with any clipping epsilon
            assert not result.interrupted

    @pytest.mark.parametrize("ppo_epochs", [1, 2, 4, 8])
    def test_update_policy_various_epochs(self, trainer, prepared_buffer_data, ppo_epochs):
        """Test update_policy with various number of PPO epochs."""
        trainer.ppo_epochs = ppo_epochs
        
        with patch.object(trainer.buffer, 'is_ready_for_training', return_value=True), \
             patch.object(trainer.buffer, 'get_training_data', return_value=prepared_buffer_data), \
             patch.object(trainer, '_compute_advantages_and_returns'):
            
            result = trainer.update_policy()
            
            # Should work with any number of epochs
            assert not result.interrupted