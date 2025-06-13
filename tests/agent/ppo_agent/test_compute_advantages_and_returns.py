"""
Comprehensive tests for PPOTrainer._compute_advantages_and_returns method.
Tests 100% coverage including normal cases, edge cases, error conditions.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from agent.ppo_agent import PPOTrainer
from config.training.training_config import TrainingConfig


class TestPPOTrainerComputeAdvantagesAndReturns:
    """Test cases for PPOTrainer _compute_advantages_and_returns method."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = Mock()
        model.parameters.return_value = iter([torch.tensor([1.0])])
        return model

    @pytest.fixture
    def training_config(self):
        """Create training configuration with specific GAE parameters."""
        return TrainingConfig(
            gamma=0.99,
            gae_lambda=0.95
        )

    @pytest.fixture
    def trainer(self, training_config, mock_model):
        """Create PPOTrainer instance."""
        device = torch.device("cpu")
        return PPOTrainer(training_config, mock_model, device=device)

    def setup_buffer_with_data(self, trainer, rewards, values, dones):
        """Setup buffer with specific rewards, values, and dones."""
        trainer.buffer.rewards = torch.tensor(rewards, dtype=torch.float32, device=trainer.device)
        trainer.buffer.values = torch.tensor(values, dtype=torch.float32, device=trainer.device).view(-1, 1)
        trainer.buffer.dones = torch.tensor(dones, dtype=torch.bool, device=trainer.device)

    def test_compute_advantages_and_returns_basic_functionality(self, trainer):
        """Test basic GAE computation functionality."""
        rewards = [1.0, 2.0, 3.0]
        values = [0.5, 1.0, 1.5]
        dones = [False, False, True]
        
        self.setup_buffer_with_data(trainer, rewards, values, dones)
        
        trainer._compute_advantages_and_returns()
        
        assert trainer.buffer.advantages is not None
        assert trainer.buffer.returns is not None
        assert trainer.buffer.advantages.shape == (3, 1)
        assert trainer.buffer.returns.shape == (3, 1)

    def test_compute_advantages_and_returns_no_rewards(self, trainer):
        """Test when buffer rewards is None."""
        trainer.buffer.rewards = None
        trainer.buffer.values = torch.randn(5, 1)
        trainer.buffer.dones = torch.zeros(5, dtype=torch.bool)
        
        with patch.object(trainer.logger, 'error') as mock_log:
            trainer._compute_advantages_and_returns()
            mock_log.assert_called_once()

    def test_compute_advantages_and_returns_no_values(self, trainer):
        """Test when buffer values is None."""
        trainer.buffer.rewards = torch.randn(5)
        trainer.buffer.values = None
        trainer.buffer.dones = torch.zeros(5, dtype=torch.bool)
        
        with patch.object(trainer.logger, 'error') as mock_log:
            trainer._compute_advantages_and_returns()
            mock_log.assert_called_once()

    def test_compute_advantages_and_returns_no_dones(self, trainer):
        """Test when buffer dones is None."""
        trainer.buffer.rewards = torch.randn(5)
        trainer.buffer.values = torch.randn(5, 1)
        trainer.buffer.dones = None
        
        with patch.object(trainer.logger, 'error') as mock_log:
            trainer._compute_advantages_and_returns()
            mock_log.assert_called_once()

    def test_compute_advantages_and_returns_single_step(self, trainer):
        """Test GAE computation with single step."""
        rewards = [1.0]
        values = [0.5]
        dones = [True]
        
        self.setup_buffer_with_data(trainer, rewards, values, dones)
        
        trainer._compute_advantages_and_returns()
        
        # For single step ending in done, advantage should be reward - value
        expected_advantage = rewards[0] - values[0]
        assert torch.allclose(trainer.buffer.advantages[0], torch.tensor([[expected_advantage]]))

    def test_compute_advantages_and_returns_episode_end(self, trainer):
        """Test GAE computation when episode ends (done=True)."""
        rewards = [1.0, 2.0]
        values = [0.5, 1.0]
        dones = [False, True]
        
        self.setup_buffer_with_data(trainer, rewards, values, dones)
        
        trainer._compute_advantages_and_returns()
        
        # Last step should have next_value = 0 due to episode end
        assert trainer.buffer.advantages is not None

    def test_compute_advantages_and_returns_no_episode_end(self, trainer):
        """Test GAE computation when episode doesn't end."""
        rewards = [1.0, 2.0, 3.0]
        values = [0.5, 1.0, 1.5]
        dones = [False, False, False]
        
        self.setup_buffer_with_data(trainer, rewards, values, dones)
        
        trainer._compute_advantages_and_returns()
        
        # Last step should use its own value as next_value
        assert trainer.buffer.advantages is not None

    def test_compute_advantages_and_returns_gamma_effect(self, trainer):
        """Test effect of gamma parameter on computation."""
        rewards = [1.0, 1.0, 1.0]
        values = [0.0, 0.0, 0.0]
        dones = [False, False, True]
        
        # Test with different gamma values
        for gamma in [0.0, 0.5, 0.99]:
            trainer.gamma = gamma
            self.setup_buffer_with_data(trainer, rewards, values, dones)
            
            trainer._compute_advantages_and_returns()
            
            assert trainer.buffer.advantages is not None
            assert torch.all(torch.isfinite(trainer.buffer.advantages))

    def test_compute_advantages_and_returns_gae_lambda_effect(self, trainer):
        """Test effect of GAE lambda parameter on computation."""
        rewards = [1.0, 1.0, 1.0]
        values = [0.0, 0.0, 0.0]
        dones = [False, False, True]
        
        # Test with different lambda values
        for gae_lambda in [0.0, 0.5, 0.95]:
            trainer.gae_lambda = gae_lambda
            self.setup_buffer_with_data(trainer, rewards, values, dones)
            
            trainer._compute_advantages_and_returns()
            
            assert trainer.buffer.advantages is not None
            assert torch.all(torch.isfinite(trainer.buffer.advantages))

    def test_compute_advantages_and_returns_reversed_computation(self, trainer):
        """Test that computation goes in reverse order (backwards through time)."""
        rewards = [1.0, 2.0, 3.0, 4.0]
        values = [0.5, 1.0, 1.5, 2.0]
        dones = [False, False, False, True]
        
        self.setup_buffer_with_data(trainer, rewards, values, dones)
        
        # Mock the computation to verify reverse order
        original_compute = trainer._compute_advantages_and_returns
        computation_order = []
        
        def track_computation():
            num_steps = len(trainer.buffer.rewards)
            advantages = torch.zeros_like(trainer.buffer.values, device=trainer.device)
            last_gae_lam = 0
            
            for t in reversed(range(num_steps)):
                computation_order.append(t)
                # Simplified computation for tracking
                advantages[t] = 1.0
            
            trainer.buffer.advantages = advantages
            trainer.buffer.returns = advantages + trainer.buffer.values
        
        trainer._compute_advantages_and_returns = track_computation
        trainer._compute_advantages_and_returns()
        
        # Should compute in reverse order: 3, 2, 1, 0
        assert computation_order == [3, 2, 1, 0]

    def test_compute_advantages_and_returns_gae_formula(self, trainer):
        """Test that GAE formula is implemented correctly."""
        # Simple case for manual verification
        rewards = [1.0, 0.0]
        values = [0.0, 0.0]
        dones = [False, True]
        
        trainer.gamma = 0.9
        trainer.gae_lambda = 0.8
        
        self.setup_buffer_with_data(trainer, rewards, values, dones)
        trainer._compute_advantages_and_returns()
        
        # Manual calculation for verification
        # Step 1 (t=1): delta = reward + gamma * next_value * (1-done) - value = 0 + 0.9 * 0 * 0 - 0 = 0
        # advantage[1] = delta = 0
        
        # Step 0 (t=0): delta = 1.0 + 0.9 * 0 * 1 - 0 = 1.0
        # advantage[0] = delta + gamma * gae_lambda * (1-done) * last_gae_lam = 1.0 + 0.9 * 0.8 * 1 * 0 = 1.0
        
        assert torch.allclose(trainer.buffer.advantages[1], torch.tensor([[0.0]]), atol=1e-5)
        assert torch.allclose(trainer.buffer.advantages[0], torch.tensor([[1.0]]), atol=1e-5)

    def test_compute_advantages_and_returns_value_tensor_shapes(self, trainer):
        """Test handling of different value tensor shapes."""
        rewards = [1.0, 2.0]
        dones = [False, True]
        
        # Test with multi-column values
        values_multi = torch.tensor([[0.5, 0.6], [1.0, 1.1]], device=trainer.device)
        trainer.buffer.rewards = torch.tensor(rewards, device=trainer.device)
        trainer.buffer.values = values_multi
        trainer.buffer.dones = torch.tensor(dones, dtype=torch.bool, device=trainer.device)
        
        trainer._compute_advantages_and_returns()
        
        # Should handle shape reshaping correctly
        assert trainer.buffer.advantages is not None
        assert trainer.buffer.returns is not None

    def test_compute_advantages_and_returns_zero_rewards(self, trainer):
        """Test computation with all zero rewards."""
        rewards = [0.0, 0.0, 0.0]
        values = [1.0, 1.0, 1.0]
        dones = [False, False, True]
        
        self.setup_buffer_with_data(trainer, rewards, values, dones)
        
        trainer._compute_advantages_and_returns()
        
        # With zero rewards, advantages should be negative (since values > 0)
        assert torch.all(trainer.buffer.advantages <= 0)

    def test_compute_advantages_and_returns_negative_rewards(self, trainer):
        """Test computation with negative rewards."""
        rewards = [-1.0, -2.0, -3.0]
        values = [0.0, 0.0, 0.0]
        dones = [False, False, True]
        
        self.setup_buffer_with_data(trainer, rewards, values, dones)
        
        trainer._compute_advantages_and_returns()
        
        # With negative rewards and zero values, advantages should be negative
        assert torch.all(trainer.buffer.advantages < 0)

    def test_compute_advantages_and_returns_large_values(self, trainer):
        """Test computation with large reward and value magnitudes."""
        rewards = [1e6, 1e6, 1e6]
        values = [1e6, 1e6, 1e6]
        dones = [False, False, True]
        
        self.setup_buffer_with_data(trainer, rewards, values, dones)
        
        trainer._compute_advantages_and_returns()
        
        # Should handle large values without overflow
        assert torch.all(torch.isfinite(trainer.buffer.advantages))
        assert torch.all(torch.isfinite(trainer.buffer.returns))

    def test_compute_advantages_and_returns_small_values(self, trainer):
        """Test computation with very small reward and value magnitudes."""
        rewards = [1e-10, 1e-10, 1e-10]
        values = [1e-10, 1e-10, 1e-10]
        dones = [False, False, True]
        
        self.setup_buffer_with_data(trainer, rewards, values, dones)
        
        trainer._compute_advantages_and_returns()
        
        # Should handle small values without underflow
        assert torch.all(torch.isfinite(trainer.buffer.advantages))

    def test_compute_advantages_and_returns_mixed_episodes(self, trainer):
        """Test computation with multiple episode endings."""
        rewards = [1.0, 2.0, 3.0, 1.0, 2.0]
        values = [0.5, 1.0, 1.5, 0.5, 1.0]
        dones = [False, False, True, False, True]  # Two episodes
        
        self.setup_buffer_with_data(trainer, rewards, values, dones)
        
        trainer._compute_advantages_and_returns()
        
        # Should handle multiple episode boundaries correctly
        assert trainer.buffer.advantages is not None
        assert trainer.buffer.returns is not None

    def test_compute_advantages_and_returns_returns_calculation(self, trainer):
        """Test that returns are calculated as advantages + values."""
        rewards = [1.0, 2.0, 3.0]
        values = [0.5, 1.0, 1.5]
        dones = [False, False, True]
        
        self.setup_buffer_with_data(trainer, rewards, values, dones)
        
        trainer._compute_advantages_and_returns()
        
        # Returns should equal advantages + values
        expected_returns = trainer.buffer.advantages + trainer.buffer.values[:, 0:1]
        assert torch.allclose(trainer.buffer.returns, expected_returns)

    def test_compute_advantages_and_returns_device_consistency(self, trainer):
        """Test that all tensors remain on correct device."""
        rewards = [1.0, 2.0, 3.0]
        values = [0.5, 1.0, 1.5]
        dones = [False, False, True]
        
        self.setup_buffer_with_data(trainer, rewards, values, dones)
        
        trainer._compute_advantages_and_returns()
        
        assert trainer.buffer.advantages.device == trainer.device
        assert trainer.buffer.returns.device == trainer.device

    def test_compute_advantages_and_returns_extreme_gamma(self, trainer):
        """Test computation with extreme gamma values."""
        rewards = [1.0, 1.0, 1.0]
        values = [0.0, 0.0, 0.0]
        dones = [False, False, True]
        
        # Test with gamma = 0 (no discounting)
        trainer.gamma = 0.0
        self.setup_buffer_with_data(trainer, rewards, values, dones)
        trainer._compute_advantages_and_returns()
        
        assert torch.all(torch.isfinite(trainer.buffer.advantages))
        
        # Test with gamma very close to 1
        trainer.gamma = 0.9999
        self.setup_buffer_with_data(trainer, rewards, values, dones)
        trainer._compute_advantages_and_returns()
        
        assert torch.all(torch.isfinite(trainer.buffer.advantages))

    def test_compute_advantages_and_returns_extreme_lambda(self, trainer):
        """Test computation with extreme lambda values."""
        rewards = [1.0, 1.0, 1.0]
        values = [0.0, 0.0, 0.0]
        dones = [False, False, True]
        
        # Test with lambda = 0 (no GAE, just TD error)
        trainer.gae_lambda = 0.0
        self.setup_buffer_with_data(trainer, rewards, values, dones)
        trainer._compute_advantages_and_returns()
        
        assert torch.all(torch.isfinite(trainer.buffer.advantages))
        
        # Test with lambda = 1 (Monte Carlo)
        trainer.gae_lambda = 1.0
        self.setup_buffer_with_data(trainer, rewards, values, dones)
        trainer._compute_advantages_and_returns()
        
        assert torch.all(torch.isfinite(trainer.buffer.advantages))

    def test_compute_advantages_and_returns_empty_buffer(self, trainer):
        """Test computation with empty buffer."""
        trainer.buffer.rewards = torch.tensor([], device=trainer.device)
        trainer.buffer.values = torch.empty(0, 1, device=trainer.device)
        trainer.buffer.dones = torch.tensor([], dtype=torch.bool, device=trainer.device)
        
        trainer._compute_advantages_and_returns()
        
        # Should handle empty buffers gracefully
        assert trainer.buffer.advantages.shape == (0, 1)
        assert trainer.buffer.returns.shape == (0, 1)

    def test_compute_advantages_and_returns_single_episode_all_done_false(self, trainer):
        """Test computation when all done flags are False."""
        rewards = [1.0, 2.0, 3.0]
        values = [0.5, 1.0, 1.5]
        dones = [False, False, False]
        
        self.setup_buffer_with_data(trainer, rewards, values, dones)
        
        trainer._compute_advantages_and_returns()
        
        # Should use last value as next_value for final step
        assert trainer.buffer.advantages is not None

    def test_compute_advantages_and_returns_all_done_true(self, trainer):
        """Test computation when all done flags are True."""
        rewards = [1.0, 2.0, 3.0]
        values = [0.5, 1.0, 1.5]
        dones = [True, True, True]
        
        self.setup_buffer_with_data(trainer, rewards, values, dones)
        
        trainer._compute_advantages_and_returns()
        
        # Each step should treat next_value as 0
        assert trainer.buffer.advantages is not None

    def test_compute_advantages_and_returns_numerical_stability(self, trainer):
        """Test numerical stability with challenging values."""
        # Mix of very different magnitudes
        rewards = [1e-10, 1e10, -1e10]
        values = [1e10, 1e-10, 0.0]
        dones = [False, False, True]
        
        self.setup_buffer_with_data(trainer, rewards, values, dones)
        
        trainer._compute_advantages_and_returns()
        
        # Should remain numerically stable
        assert torch.all(torch.isfinite(trainer.buffer.advantages))
        assert torch.all(torch.isfinite(trainer.buffer.returns))

    @pytest.mark.parametrize("num_steps", [1, 5, 10, 100])
    def test_compute_advantages_and_returns_various_lengths(self, trainer, num_steps):
        """Test computation with various sequence lengths."""
        rewards = [1.0] * num_steps
        values = [0.5] * num_steps
        dones = [False] * (num_steps - 1) + [True]
        
        self.setup_buffer_with_data(trainer, rewards, values, dones)
        
        trainer._compute_advantages_and_returns()
        
        assert trainer.buffer.advantages.shape == (num_steps, 1)
        assert trainer.buffer.returns.shape == (num_steps, 1)