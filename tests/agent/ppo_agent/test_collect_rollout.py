"""
Comprehensive tests for PPOTrainer.collect_rollout method.
Tests 100% coverage including normal cases, edge cases, error conditions.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import logging

from agent.ppo_agent import PPOTrainer
from agent.replay_buffer import ReplayBuffer
from config.training.training_config import TrainingConfig, TrainingManagerConfig
from core.types import RolloutResult
from callbacks.core import CallbackManager


class TestPPOTrainerCollectRollout:
    """Test cases for PPOTrainer collect_rollout method."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model with get_action method."""
        model = Mock()
        model.get_action.return_value = (
            torch.tensor([1]),  # action_tensor
            {
                "action_logits": torch.tensor([[0.1, 0.2, 0.7]]),
                "value": torch.tensor([[0.5]]),
                "log_prob": torch.tensor([[0.3]])
            }
        )
        model.parameters.return_value = iter([torch.tensor([1.0])])
        return model

    @pytest.fixture
    def training_config(self):
        """Create training configuration."""
        return TrainingConfig(
            batch_size=32,
            training_manager=TrainingManagerConfig(rollout_steps=10)
        )

    @pytest.fixture
    def mock_environment(self):
        """Create a mock trading environment."""
        env = Mock()
        env.step.return_value = (
            {"hf": np.random.randn(60, 9), "mf": np.random.randn(30, 43), 
             "lf": np.random.randn(30, 19), "portfolio": np.random.randn(5, 10)},  # next_state
            1.0,  # reward
            False,  # terminated
            False,  # truncated
            {}    # info
        )
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

    @pytest.fixture
    def trainer(self, training_config, mock_model):
        """Create PPOTrainer instance."""
        return PPOTrainer(training_config, mock_model)

    @pytest.fixture
    def callback_manager(self):
        """Create mock callback manager."""
        callback_manager = Mock(spec=CallbackManager)
        return callback_manager

    def test_collect_rollout_basic_functionality(self, trainer, mock_environment, initial_obs):
        """Test basic rollout collection functionality."""
        result = trainer.collect_rollout(mock_environment, num_steps=5, initial_obs=initial_obs)
        
        assert isinstance(result, RolloutResult)
        assert result.steps_collected >= 0
        assert result.episodes_completed >= 0
        assert isinstance(result.buffer_ready, bool)

    def test_collect_rollout_no_initial_obs(self, trainer, mock_environment):
        """Test collect_rollout when no initial observation is provided."""
        result = trainer.collect_rollout(mock_environment, num_steps=5, initial_obs=None)
        
        assert result.steps_collected == 0
        assert result.episodes_completed == 0
        assert result.buffer_ready == False
        assert result.interrupted == True

    def test_collect_rollout_default_num_steps(self, trainer, mock_environment, initial_obs):
        """Test collect_rollout uses default rollout_steps when num_steps is None."""
        result = trainer.collect_rollout(mock_environment, initial_obs=initial_obs)
        
        # Should use trainer.rollout_steps
        assert result.steps_collected <= trainer.rollout_steps

    def test_collect_rollout_environment_step_success(self, trainer, mock_environment, initial_obs):
        """Test successful environment steps during rollout."""
        mock_environment.step.return_value = (
            initial_obs,  # next_state (same as initial for simplicity)
            2.5,          # reward
            False,        # terminated
            False,        # truncated
            {"info": "test"}
        )
        
        result = trainer.collect_rollout(mock_environment, num_steps=3, initial_obs=initial_obs)
        
        assert result.steps_collected == 3
        assert result.episodes_completed == 0  # No episode completed
        assert mock_environment.step.call_count == 3

    def test_collect_rollout_episode_termination(self, trainer, mock_environment, initial_obs):
        """Test rollout stops when episode terminates."""
        # First step normal, second step terminates
        mock_environment.step.side_effect = [
            (initial_obs, 1.0, False, False, {}),  # First step
            (initial_obs, 1.0, True, False, {})    # Second step terminates
        ]
        
        result = trainer.collect_rollout(mock_environment, num_steps=5, initial_obs=initial_obs)
        
        assert result.steps_collected == 2
        assert result.episodes_completed == 1
        assert mock_environment.step.call_count == 2

    def test_collect_rollout_episode_truncation(self, trainer, mock_environment, initial_obs):
        """Test rollout stops when episode is truncated."""
        mock_environment.step.return_value = (initial_obs, 1.0, False, True, {})
        
        result = trainer.collect_rollout(mock_environment, num_steps=5, initial_obs=initial_obs)
        
        assert result.steps_collected == 1
        assert result.episodes_completed == 1

    def test_collect_rollout_model_action_selection(self, trainer, mock_environment, initial_obs):
        """Test that model action selection is called correctly."""
        result = trainer.collect_rollout(mock_environment, num_steps=2, initial_obs=initial_obs)
        
        # Model should be called for action selection
        assert trainer.model.get_action.call_count == 2
        
        # Check that get_action was called with correct parameters
        for call in trainer.model.get_action.call_args_list:
            args, kwargs = call
            state_tensors = args[0]
            assert isinstance(state_tensors, dict)
            assert kwargs.get('deterministic') == False

    def test_collect_rollout_buffer_operations(self, trainer, mock_environment, initial_obs):
        """Test buffer operations during rollout."""
        with patch.object(trainer.buffer, 'clear') as mock_clear, \
             patch.object(trainer.buffer, 'add') as mock_add, \
             patch.object(trainer.buffer, 'prepare_data_for_training') as mock_prepare:
            
            result = trainer.collect_rollout(mock_environment, num_steps=3, initial_obs=initial_obs)
            
            # Buffer should be cleared at start
            mock_clear.assert_called_once()
            
            # Buffer should have experiences added
            assert mock_add.call_count == 3
            
            # Buffer should be prepared for training at end
            mock_prepare.assert_called_once()

    def test_collect_rollout_with_callback_manager(self, training_config, mock_model, mock_environment, initial_obs, callback_manager):
        """Test rollout with callback manager triggering events."""
        trainer = PPOTrainer(training_config, mock_model, callback_manager=callback_manager)
        
        result = trainer.collect_rollout(mock_environment, num_steps=2, initial_obs=initial_obs)
        
        # Should trigger action_selected, step_start, and step_end events
        assert callback_manager.trigger_event.call_count >= 4  # At least 2 action_selected + 2 step_start
        assert callback_manager.trigger_step_end.call_count == 2

    def test_collect_rollout_callback_action_selected(self, training_config, mock_model, mock_environment, initial_obs, callback_manager):
        """Test action_selected callback is triggered correctly."""
        trainer = PPOTrainer(training_config, mock_model, callback_manager=callback_manager)
        
        trainer.collect_rollout(mock_environment, num_steps=1, initial_obs=initial_obs)
        
        # Check action_selected callback
        action_selected_calls = [call for call in callback_manager.trigger_event.call_args_list 
                               if call[0][0] == "action_selected"]
        assert len(action_selected_calls) == 1
        
        event_data = action_selected_calls[0][0][1]
        assert "action" in event_data
        assert "action_info" in event_data
        assert "state" in event_data
        assert "step" in event_data

    def test_collect_rollout_callback_step_start(self, training_config, mock_model, mock_environment, initial_obs, callback_manager):
        """Test step_start callback is triggered correctly."""
        trainer = PPOTrainer(training_config, mock_model, callback_manager=callback_manager)
        
        trainer.collect_rollout(mock_environment, num_steps=1, initial_obs=initial_obs)
        
        # Check step_start callback
        step_start_calls = [call for call in callback_manager.trigger_event.call_args_list 
                          if call[0][0] == "step_start"]
        assert len(step_start_calls) == 1
        
        event_data = step_start_calls[0][0][1]
        assert "action" in event_data
        assert "state" in event_data
        assert "step" in event_data

    def test_collect_rollout_callback_step_end(self, training_config, mock_model, mock_environment, initial_obs, callback_manager):
        """Test step_end callback is triggered correctly."""
        trainer = PPOTrainer(training_config, mock_model, callback_manager=callback_manager)
        
        trainer.collect_rollout(mock_environment, num_steps=1, initial_obs=initial_obs)
        
        # Check step_end callback
        assert callback_manager.trigger_step_end.call_count == 1
        
        call_args = callback_manager.trigger_step_end.call_args[0][0]
        assert "action" in call_args
        assert "state" in call_args
        assert "next_state" in call_args
        assert "reward" in call_args
        assert "done" in call_args
        assert "info" in call_args
        assert "step" in call_args

    def test_collect_rollout_no_callback_manager(self, trainer, mock_environment, initial_obs):
        """Test rollout works correctly without callback manager."""
        trainer.callback_manager = None
        
        result = trainer.collect_rollout(mock_environment, num_steps=2, initial_obs=initial_obs)
        
        # Should complete successfully without callbacks
        assert result.steps_collected == 2

    def test_collect_rollout_environment_step_exception(self, trainer, mock_environment, initial_obs):
        """Test handling of environment step exceptions."""
        mock_environment.step.side_effect = Exception("Environment error")
        
        with patch.object(trainer.logger, 'error') as mock_log_error:
            result = trainer.collect_rollout(mock_environment, num_steps=5, initial_obs=initial_obs)
            
            # Should log error and stop rollout
            mock_log_error.assert_called_once()
            assert "Error during environment step" in mock_log_error.call_args[0][0]
            assert result.steps_collected == 0

    def test_collect_rollout_state_tensor_conversion(self, trainer, mock_environment, initial_obs):
        """Test state to tensor conversion during rollout."""
        with patch.object(trainer, '_convert_state_to_tensors') as mock_convert:
            mock_convert.return_value = {
                "hf": torch.randn(1, 60, 9),
                "mf": torch.randn(1, 30, 43),
                "lf": torch.randn(1, 30, 19),
                "portfolio": torch.randn(1, 5, 10)
            }
            
            result = trainer.collect_rollout(mock_environment, num_steps=2, initial_obs=initial_obs)
            
            # Should convert state to tensors for each step
            assert mock_convert.call_count == 2

    def test_collect_rollout_action_conversion(self, trainer, mock_environment, initial_obs):
        """Test action conversion for environment."""
        with patch.object(trainer, '_convert_action_for_env') as mock_convert:
            mock_convert.return_value = 1  # Environment action
            
            result = trainer.collect_rollout(mock_environment, num_steps=2, initial_obs=initial_obs)
            
            # Should convert action for environment for each step
            assert mock_convert.call_count == 2

    def test_collect_rollout_buffer_ready_condition(self, trainer, mock_environment, initial_obs):
        """Test buffer_ready condition in result."""
        trainer.batch_size = 5
        
        # Mock buffer size
        with patch.object(trainer.buffer, 'get_size', return_value=10):
            result = trainer.collect_rollout(mock_environment, num_steps=3, initial_obs=initial_obs)
            assert result.buffer_ready == True  # 10 >= 5
        
        with patch.object(trainer.buffer, 'get_size', return_value=3):
            result = trainer.collect_rollout(mock_environment, num_steps=3, initial_obs=initial_obs)
            assert result.buffer_ready == False  # 3 < 5

    def test_collect_rollout_zero_steps(self, trainer, mock_environment, initial_obs):
        """Test rollout with zero steps."""
        result = trainer.collect_rollout(mock_environment, num_steps=0, initial_obs=initial_obs)
        
        assert result.steps_collected == 0
        assert result.episodes_completed == 0
        assert mock_environment.step.call_count == 0

    def test_collect_rollout_large_num_steps(self, trainer, mock_environment, initial_obs):
        """Test rollout with large number of steps."""
        # Set episode to end after 5 steps
        mock_environment.step.side_effect = [
            (initial_obs, 1.0, False, False, {}),  # Steps 1-4
            (initial_obs, 1.0, False, False, {}),
            (initial_obs, 1.0, False, False, {}),
            (initial_obs, 1.0, False, False, {}),
            (initial_obs, 1.0, True, False, {})    # Step 5 - episode ends
        ]
        
        result = trainer.collect_rollout(mock_environment, num_steps=100, initial_obs=initial_obs)
        
        # Should stop when episode ends, not when num_steps reached
        assert result.steps_collected == 5
        assert result.episodes_completed == 1

    def test_collect_rollout_reward_values(self, trainer, mock_environment, initial_obs):
        """Test rollout with different reward values."""
        rewards = [1.5, -0.5, 2.0]
        mock_environment.step.side_effect = [
            (initial_obs, rewards[0], False, False, {}),
            (initial_obs, rewards[1], False, False, {}),
            (initial_obs, rewards[2], True, False, {})  # Episode ends
        ]
        
        with patch.object(trainer.buffer, 'add') as mock_add:
            result = trainer.collect_rollout(mock_environment, num_steps=5, initial_obs=initial_obs)
            
            # Check that rewards were passed to buffer
            for i, call in enumerate(mock_add.call_args_list):
                assert call[0][2] == rewards[i]  # reward is 3rd argument

    def test_collect_rollout_done_flag_combinations(self, trainer, mock_environment, initial_obs):
        """Test different combinations of terminated and truncated flags."""
        test_cases = [
            (True, False),   # terminated only
            (False, True),   # truncated only
            (True, True),    # both
            (False, False)   # neither
        ]
        
        for terminated, truncated in test_cases:
            mock_environment.reset()
            mock_environment.step.return_value = (initial_obs, 1.0, terminated, truncated, {})
            
            result = trainer.collect_rollout(mock_environment, num_steps=2, initial_obs=initial_obs)
            
            if terminated or truncated:
                assert result.episodes_completed == 1
                assert result.steps_collected == 1
            else:
                assert result.steps_collected == 2

    def test_collect_rollout_info_dict_handling(self, trainer, mock_environment, initial_obs):
        """Test handling of info dictionary from environment."""
        info_dict = {"custom_metric": 42, "debug_info": "test"}
        mock_environment.step.return_value = (initial_obs, 1.0, False, False, info_dict)
        
        result = trainer.collect_rollout(mock_environment, num_steps=1, initial_obs=initial_obs)
        
        # Should handle info dict without errors
        assert result.steps_collected == 1

    def test_collect_rollout_state_progression(self, trainer, mock_environment, initial_obs):
        """Test that state progresses correctly through rollout."""
        # Create different states for progression
        state1 = {k: v + 0.1 for k, v in initial_obs.items()}
        state2 = {k: v + 0.2 for k, v in initial_obs.items()}
        
        mock_environment.step.side_effect = [
            (state1, 1.0, False, False, {}),
            (state2, 1.0, True, False, {})
        ]
        
        with patch.object(trainer, '_convert_state_to_tensors') as mock_convert:
            result = trainer.collect_rollout(mock_environment, num_steps=5, initial_obs=initial_obs)
            
            # Should be called with initial_obs, then state1, then state2
            assert mock_convert.call_count == 2
            
            # First call should be with initial_obs
            first_call_state = mock_convert.call_args_list[0][0][0]
            assert first_call_state is initial_obs
            
            # Second call should be with state1
            second_call_state = mock_convert.call_args_list[1][0][0]
            assert second_call_state is state1

    def test_collect_rollout_torch_no_grad_context(self, trainer, mock_environment, initial_obs):
        """Test that model inference uses torch.no_grad context."""
        with patch('torch.no_grad') as mock_no_grad:
            mock_context = MagicMock()
            mock_no_grad.return_value.__enter__ = Mock(return_value=mock_context)
            mock_no_grad.return_value.__exit__ = Mock(return_value=None)
            
            result = trainer.collect_rollout(mock_environment, num_steps=2, initial_obs=initial_obs)
            
            # Should use no_grad context for each action
            assert mock_no_grad.call_count == 2

    def test_collect_rollout_logging_error_case(self, trainer, mock_environment):
        """Test error logging when no initial observation provided."""
        with patch.object(trainer.logger, 'error') as mock_log_error:
            result = trainer.collect_rollout(mock_environment, num_steps=5, initial_obs=None)
            
            mock_log_error.assert_called_once_with("No initial observation provided for rollout")

    def test_collect_rollout_interrupted_flag(self, trainer, mock_environment):
        """Test interrupted flag is set correctly."""
        # Test with no initial obs
        result = trainer.collect_rollout(mock_environment, num_steps=5, initial_obs=None)
        assert result.interrupted == True
        
        # Test with normal execution
        initial_obs = {
            "hf": np.random.randn(60, 9),
            "mf": np.random.randn(30, 43),
            "lf": np.random.randn(30, 19),
            "portfolio": np.random.randn(5, 10)
        }
        result = trainer.collect_rollout(mock_environment, num_steps=2, initial_obs=initial_obs)
        assert result.interrupted == False

    def test_collect_rollout_buffer_add_call_signature(self, trainer, mock_environment, initial_obs):
        """Test that buffer.add is called with correct signature."""
        with patch.object(trainer.buffer, 'add') as mock_add:
            result = trainer.collect_rollout(mock_environment, num_steps=1, initial_obs=initial_obs)
            
            # Should be called once
            mock_add.assert_called_once()
            
            # Check call signature: current_state, action_tensor, reward, next_state, done, action_info
            args = mock_add.call_args[0]
            assert len(args) == 6
            assert isinstance(args[0], dict)  # current_state
            assert isinstance(args[1], torch.Tensor)  # action_tensor
            assert isinstance(args[2], (int, float))  # reward
            assert isinstance(args[3], dict)  # next_state
            assert isinstance(args[4], bool)  # done
            assert isinstance(args[5], dict)  # action_info

    @pytest.mark.parametrize("num_steps", [1, 5, 10, 50])
    def test_collect_rollout_various_step_counts(self, trainer, mock_environment, initial_obs, num_steps):
        """Test rollout with various step counts."""
        result = trainer.collect_rollout(mock_environment, num_steps=num_steps, initial_obs=initial_obs)
        
        # Should collect up to num_steps (may be less if episode ends)
        assert result.steps_collected <= num_steps
        assert result.steps_collected >= 0