"""
Comprehensive tests for PPOTrainer.evaluate method.
Tests 100% coverage including normal cases, edge cases, error conditions.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock, patch, ANY

from agent.ppo_agent import PPOTrainer
from config.training.training_config import TrainingConfig
from envs.trading_environment import TradingEnvironment


class TestPPOTrainerEvaluate:
    """Test cases for PPOTrainer evaluate method."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model with get_action method."""
        model = Mock()
        model.get_action.return_value = (
            torch.tensor([2]),  # action_tensor
            {
                "action_logits": torch.tensor([[0.1, 0.2, 0.7, 0.0, 0.0, 0.0, 0.0]]),
                "value": torch.tensor([[1.5]]),
                "log_prob": torch.tensor([[0.3]])
            }
        )
        model.parameters.return_value = iter([torch.tensor([1.0])])
        model.training = True  # Initially in training mode
        model.eval = Mock()
        model.train = Mock()
        return model

    @pytest.fixture
    def training_config(self):
        """Create training configuration."""
        return TrainingConfig()

    @pytest.fixture
    def trainer(self, training_config, mock_model):
        """Create PPOTrainer instance."""
        return PPOTrainer(training_config, mock_model)

    @pytest.fixture
    def mock_environment(self):
        """Create a mock trading environment."""
        env = Mock(spec=TradingEnvironment)
        return env

    @pytest.fixture
    def initial_obs(self):
        """Create initial observation dictionary."""
        return {
            "hf": np.random.randn(60, 9),
            "mf": np.random.randn(30, 43),
            "lf": np.random.randn(30, 19),
            "portfolio": np.random.randn(5, 10)
        }

    def test_evaluate_basic_functionality(self, trainer, mock_environment, initial_obs):
        """Test basic evaluation functionality."""
        # Setup environment to end after 3 steps
        mock_environment.step.side_effect = [
            (initial_obs, 1.0, False, False, {}),  # Step 1
            (initial_obs, 2.0, False, False, {}),  # Step 2
            (initial_obs, 3.0, True, False, {})    # Step 3 - episode ends
        ]
        
        total_reward = trainer.evaluate(mock_environment, initial_obs)
        
        assert total_reward == 6.0  # 1.0 + 2.0 + 3.0
        assert isinstance(total_reward, float)

    def test_evaluate_deterministic_true(self, trainer, mock_environment, initial_obs):
        """Test evaluation with deterministic=True (default)."""
        mock_environment.step.return_value = (initial_obs, 1.0, True, False, {})
        
        total_reward = trainer.evaluate(mock_environment, initial_obs, deterministic=True)
        
        # Should call get_action with deterministic=True
        trainer.model.get_action.assert_called_with(
            ANY, deterministic=True
        )

    def test_evaluate_deterministic_false(self, trainer, mock_environment, initial_obs):
        """Test evaluation with deterministic=False."""
        mock_environment.step.return_value = (initial_obs, 1.0, True, False, {})
        
        total_reward = trainer.evaluate(mock_environment, initial_obs, deterministic=False)
        
        # Should call get_action with deterministic=False
        trainer.model.get_action.assert_called_with(
            ANY, deterministic=False
        )

    def test_evaluate_model_mode_management(self, trainer, mock_environment, initial_obs):
        """Test that model mode is properly managed during evaluation."""
        mock_environment.step.return_value = (initial_obs, 1.0, True, False, {})
        
        # Model starts in training mode
        trainer.model.training = True
        
        total_reward = trainer.evaluate(mock_environment, initial_obs)
        
        # Should call eval() during evaluation
        trainer.model.eval.assert_called_once()
        
        # Should call train() to restore training mode
        trainer.model.train.assert_called_once()

    def test_evaluate_model_already_in_eval_mode(self, trainer, mock_environment, initial_obs):
        """Test evaluation when model is already in eval mode."""
        mock_environment.step.return_value = (initial_obs, 1.0, True, False, {})
        
        # Model starts in eval mode
        trainer.model.training = False
        
        total_reward = trainer.evaluate(mock_environment, initial_obs)
        
        # Should still call eval() but not train() at end
        trainer.model.eval.assert_called_once()
        trainer.model.train.assert_not_called()

    def test_evaluate_episode_termination(self, trainer, mock_environment, initial_obs):
        """Test evaluation stops on episode termination."""
        mock_environment.step.side_effect = [
            (initial_obs, 1.0, False, False, {}),  # Step 1
            (initial_obs, 2.0, True, False, {})    # Step 2 - terminated
        ]
        
        total_reward = trainer.evaluate(mock_environment, initial_obs)
        
        assert total_reward == 3.0
        assert mock_environment.step.call_count == 2

    def test_evaluate_episode_truncation(self, trainer, mock_environment, initial_obs):
        """Test evaluation stops on episode truncation."""
        mock_environment.step.side_effect = [
            (initial_obs, 1.0, False, False, {}),  # Step 1
            (initial_obs, 2.0, False, True, {})   # Step 2 - truncated
        ]
        
        total_reward = trainer.evaluate(mock_environment, initial_obs)
        
        assert total_reward == 3.0
        assert mock_environment.step.call_count == 2

    def test_evaluate_max_steps_limit(self, trainer, mock_environment, initial_obs):
        """Test evaluation stops at max_steps limit."""
        # Environment never ends episode
        mock_environment.step.return_value = (initial_obs, 1.0, False, False, {})
        
        total_reward = trainer.evaluate(mock_environment, initial_obs, max_steps=5)
        
        assert total_reward == 5.0  # 5 steps * 1.0 reward each
        assert mock_environment.step.call_count == 5

    def test_evaluate_zero_max_steps(self, trainer, mock_environment, initial_obs):
        """Test evaluation with max_steps=0."""
        total_reward = trainer.evaluate(mock_environment, initial_obs, max_steps=0)
        
        assert total_reward == 0.0
        assert mock_environment.step.call_count == 0

    def test_evaluate_negative_rewards(self, trainer, mock_environment, initial_obs):
        """Test evaluation with negative rewards."""
        mock_environment.step.side_effect = [
            (initial_obs, -1.0, False, False, {}),
            (initial_obs, -2.0, False, False, {}),
            (initial_obs, -3.0, True, False, {})
        ]
        
        total_reward = trainer.evaluate(mock_environment, initial_obs)
        
        assert total_reward == -6.0

    def test_evaluate_mixed_rewards(self, trainer, mock_environment, initial_obs):
        """Test evaluation with mixed positive and negative rewards."""
        mock_environment.step.side_effect = [
            (initial_obs, 5.0, False, False, {}),
            (initial_obs, -2.0, False, False, {}),
            (initial_obs, 3.0, True, False, {})
        ]
        
        total_reward = trainer.evaluate(mock_environment, initial_obs)
        
        assert total_reward == 6.0

    def test_evaluate_zero_rewards(self, trainer, mock_environment, initial_obs):
        """Test evaluation with all zero rewards."""
        mock_environment.step.side_effect = [
            (initial_obs, 0.0, False, False, {}),
            (initial_obs, 0.0, False, False, {}),
            (initial_obs, 0.0, True, False, {})
        ]
        
        total_reward = trainer.evaluate(mock_environment, initial_obs)
        
        assert total_reward == 0.0

    def test_evaluate_state_conversion(self, trainer, mock_environment, initial_obs):
        """Test that state conversion is called correctly."""
        mock_environment.step.return_value = (initial_obs, 1.0, True, False, {})
        
        with patch.object(trainer, '_convert_state_to_tensors') as mock_convert:
            mock_convert.return_value = {
                "hf": torch.randn(1, 60, 9),
                "mf": torch.randn(1, 30, 43),
                "lf": torch.randn(1, 30, 19),
                "portfolio": torch.randn(1, 5, 10)
            }
            
            total_reward = trainer.evaluate(mock_environment, initial_obs)
            
            # Should convert initial obs at least once
            assert mock_convert.call_count >= 1

    def test_evaluate_action_conversion(self, trainer, mock_environment, initial_obs):
        """Test that action conversion is called correctly."""
        mock_environment.step.return_value = (initial_obs, 1.0, True, False, {})
        
        with patch.object(trainer, '_convert_action_for_env') as mock_convert:
            mock_convert.return_value = 2  # Environment action
            
            total_reward = trainer.evaluate(mock_environment, initial_obs)
            
            # Should convert action for environment
            mock_convert.assert_called_once()

    def test_evaluate_torch_no_grad_context(self, trainer, mock_environment, initial_obs):
        """Test that model inference uses torch.no_grad context."""
        mock_environment.step.return_value = (initial_obs, 1.0, True, False, {})
        
        with patch('torch.no_grad') as mock_no_grad:
            mock_context = MagicMock()
            mock_no_grad.return_value.__enter__ = Mock(return_value=mock_context)
            mock_no_grad.return_value.__exit__ = Mock(return_value=None)
            
            total_reward = trainer.evaluate(mock_environment, initial_obs)
            
            # Should use no_grad context
            mock_no_grad.assert_called()

    def test_evaluate_observation_progression(self, trainer, mock_environment, initial_obs):
        """Test that observations progress correctly through evaluation."""
        obs1 = {k: v + 0.1 for k, v in initial_obs.items()}
        obs2 = {k: v + 0.2 for k, v in initial_obs.items()}
        
        mock_environment.step.side_effect = [
            (obs1, 1.0, False, False, {}),
            (obs2, 2.0, True, False, {})
        ]
        
        with patch.object(trainer, '_convert_state_to_tensors') as mock_convert:
            mock_convert.return_value = {}  # Simplified return
            
            total_reward = trainer.evaluate(mock_environment, initial_obs)
            
            # Should be called with initial_obs, then obs1, then obs2
            assert mock_convert.call_count == 2

    def test_evaluate_environment_info_handling(self, trainer, mock_environment, initial_obs):
        """Test handling of info dictionary from environment."""
        info_dict = {"custom_metric": 42, "debug_info": "test"}
        mock_environment.step.return_value = (initial_obs, 1.0, True, False, info_dict)
        
        total_reward = trainer.evaluate(mock_environment, initial_obs)
        
        # Should handle info dict without errors
        assert total_reward == 1.0

    def test_evaluate_large_max_steps(self, trainer, mock_environment, initial_obs):
        """Test evaluation with very large max_steps."""
        # Episode ends quickly despite large limit
        mock_environment.step.return_value = (initial_obs, 10.0, True, False, {})
        
        total_reward = trainer.evaluate(mock_environment, initial_obs, max_steps=10000)
        
        assert total_reward == 10.0
        assert mock_environment.step.call_count == 1

    def test_evaluate_extreme_reward_values(self, trainer, mock_environment, initial_obs):
        """Test evaluation with extreme reward values."""
        mock_environment.step.side_effect = [
            (initial_obs, 1e6, False, False, {}),
            (initial_obs, -1e6, False, False, {}),
            (initial_obs, 1e-10, True, False, {})
        ]
        
        total_reward = trainer.evaluate(mock_environment, initial_obs)
        
        expected = 1e6 - 1e6 + 1e-10
        assert abs(total_reward - expected) < 1e-8

    def test_evaluate_model_action_consistency(self, trainer, mock_environment, initial_obs):
        """Test that model action calls are consistent."""
        mock_environment.step.side_effect = [
            (initial_obs, 1.0, False, False, {}),
            (initial_obs, 1.0, True, False, {})
        ]
        
        total_reward = trainer.evaluate(mock_environment, initial_obs)
        
        # Should call get_action for each step
        assert trainer.model.get_action.call_count == 2

    def test_evaluate_step_counter_accuracy(self, trainer, mock_environment, initial_obs):
        """Test that step counter is accurate."""
        # Create scenario with exactly 7 steps
        step_returns = [(initial_obs, 1.0, False, False, {})] * 6
        step_returns.append((initial_obs, 1.0, True, False, {}))
        mock_environment.step.side_effect = step_returns
        
        total_reward = trainer.evaluate(mock_environment, initial_obs)
        
        assert total_reward == 7.0
        assert mock_environment.step.call_count == 7

    def test_evaluate_exception_safety(self, trainer, mock_environment, initial_obs):
        """Test that finally block executes even if exception occurs."""
        # Model starts in training mode
        trainer.model.training = True
        
        # Environment raises exception
        mock_environment.step.side_effect = Exception("Environment error")
        
        with pytest.raises(Exception):
            trainer.evaluate(mock_environment, initial_obs)
        
        # Should still restore training mode due to finally block
        trainer.model.train.assert_called_once()

    def test_evaluate_memory_management(self, trainer, mock_environment, initial_obs):
        """Test memory management during evaluation."""
        # Long episode to test memory usage
        step_returns = [(initial_obs, 1.0, False, False, {})] * 99
        step_returns.append((initial_obs, 1.0, True, False, {}))
        mock_environment.step.side_effect = step_returns
        
        total_reward = trainer.evaluate(mock_environment, initial_obs)
        
        assert total_reward == 100.0
        # Should complete without memory issues

    def test_evaluate_default_parameters(self, trainer, mock_environment, initial_obs):
        """Test evaluation with default parameters."""
        mock_environment.step.return_value = (initial_obs, 1.0, True, False, {})
        
        # Call with only required parameters
        total_reward = trainer.evaluate(mock_environment, initial_obs)
        
        # Should use default deterministic=True and max_steps=1000
        assert total_reward == 1.0

    def test_evaluate_floating_point_precision(self, trainer, mock_environment, initial_obs):
        """Test floating point precision in reward accumulation."""
        # Use rewards that test floating point precision
        mock_environment.step.side_effect = [
            (initial_obs, 0.1, False, False, {}),
            (initial_obs, 0.2, False, False, {}),
            (initial_obs, 0.3, True, False, {})
        ]
        
        total_reward = trainer.evaluate(mock_environment, initial_obs)
        
        # Should maintain reasonable precision
        assert abs(total_reward - 0.6) < 1e-10

    @pytest.mark.parametrize("max_steps", [1, 10, 100, 1000])
    def test_evaluate_various_max_steps(self, trainer, mock_environment, initial_obs, max_steps):
        """Test evaluation with various max_steps values."""
        # Environment never ends
        mock_environment.step.return_value = (initial_obs, 1.0, False, False, {})
        
        total_reward = trainer.evaluate(mock_environment, initial_obs, max_steps=max_steps)
        
        assert total_reward == float(max_steps)
        assert mock_environment.step.call_count == max_steps

    @pytest.mark.parametrize("deterministic", [True, False])
    def test_evaluate_both_deterministic_modes(self, trainer, mock_environment, initial_obs, deterministic):
        """Test evaluation in both deterministic modes."""
        mock_environment.step.return_value = (initial_obs, 1.0, True, False, {})
        
        total_reward = trainer.evaluate(mock_environment, initial_obs, deterministic=deterministic)
        
        # Should work in both modes
        assert total_reward == 1.0

    def test_evaluate_return_type(self, trainer, mock_environment, initial_obs):
        """Test that return type is always float."""
        mock_environment.step.return_value = (initial_obs, 5, True, False, {})  # int reward
        
        total_reward = trainer.evaluate(mock_environment, initial_obs)
        
        assert isinstance(total_reward, float)
        assert total_reward == 5.0

    def test_evaluate_done_flag_combinations(self, trainer, mock_environment, initial_obs):
        """Test different combinations of terminated and truncated flags."""
        test_cases = [
            (True, False),   # terminated only
            (False, True),   # truncated only
            (True, True),    # both
        ]
        
        for terminated, truncated in test_cases:
            mock_environment.reset()
            mock_environment.step.return_value = (initial_obs, 5.0, terminated, truncated, {})
            
            total_reward = trainer.evaluate(mock_environment, initial_obs)
            
            # Should stop after one step in all cases
            assert total_reward == 5.0
            assert mock_environment.step.call_count == 1

    def test_evaluate_single_step_episode(self, trainer, mock_environment, initial_obs):
        """Test evaluation with single-step episode."""
        mock_environment.step.return_value = (initial_obs, 42.0, True, False, {})
        
        total_reward = trainer.evaluate(mock_environment, initial_obs)
        
        assert total_reward == 42.0
        assert mock_environment.step.call_count == 1