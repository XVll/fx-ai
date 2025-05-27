import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, List, Optional, Any
import asyncio

from envs.training_manager import (
    TrainingManager,
    TrainingConfig,
    TrainingState,
    EpisodeResult,
    TrainingMetrics,
    CheckpointManager
)


class TestTrainingManager:
    """Test suite for the complete training loop integration."""
    
    @pytest.fixture
    def training_config(self):
        """Configuration for training manager."""
        return TrainingConfig(
            num_episodes=10000,
            checkpoint_frequency=100,
            validation_frequency=500,
            early_stopping_patience=1000,
            early_stopping_metric='sharpe_ratio',
            early_stopping_threshold=1.5,
            parallel_data_loading=True,
            num_data_workers=4,
            episode_batch_size=10,
            log_frequency=10,
            save_best_model=True,
            model_selection_metric='total_reward',
            training_data_start='2024-01-01',
            training_data_end='2024-12-31',
            validation_split=0.2,
            random_seed=42
        )
    
    @pytest.fixture
    def mock_environment(self):
        """Mock trading environment."""
        env = Mock()
        env.reset.return_value = np.array([0.5] * 100)  # Mock observation
        env.step.return_value = (
            np.array([0.5] * 100),  # obs
            1.0,                     # reward
            False,                   # done
            False,                   # truncated
            {'pnl': 100}            # info
        )
        env.action_space = Mock(n=12)
        env.observation_space = Mock(shape=(100,))
        return env
    
    @pytest.fixture
    def mock_agent(self):
        """Mock PPO agent."""
        agent = Mock()
        agent.predict.return_value = (5, None)  # action, state
        agent.learn.return_value = None
        agent.save.return_value = None
        agent.load.return_value = None
        return agent
    
    @pytest.fixture
    def mock_episode_scanner(self):
        """Mock episode scanner."""
        scanner = Mock()
        scanner.scan_date_range.return_value = {
            datetime(2024, 1, 15): {
                'prime_momentum': [Mock(quality_score=0.9) for _ in range(10)],
                'secondary_momentum': [Mock(quality_score=0.7) for _ in range(5)],
                'risk_scenarios': [Mock(quality_score=0.5) for _ in range(5)]
            }
        }
        return scanner
    
    @pytest.fixture
    def mock_curriculum_selector(self):
        """Mock curriculum selector."""
        selector = Mock()
        selector.select_reset_point.return_value = Mock(
            timestamp=datetime(2024, 1, 15, 9, 30),
            quality_score=0.9,
            momentum_phase='front_breakout'
        )
        selector.get_current_stage.return_value = Mock(name='stage_1')
        return selector
    
    @pytest.fixture
    def mock_position_handler(self):
        """Mock position handler."""
        handler = Mock()
        handler.handle_episode_end.return_value = {
            'had_position': True,
            'forced_exit': False,
            'realized_pnl': 100,
            'position_continues': False
        }
        return handler
    
    @pytest.fixture
    def mock_data_manager(self):
        """Mock data manager."""
        manager = Mock()
        manager.get_available_dates.return_value = [
            datetime(2024, 1, i) for i in range(1, 32)
        ]
        return manager
    
    @pytest.fixture
    def training_manager(self, training_config, mock_environment, mock_agent,
                        mock_episode_scanner, mock_curriculum_selector,
                        mock_position_handler, mock_data_manager):
        """Create training manager instance."""
        return TrainingManager(
            config=training_config,
            environment=mock_environment,
            agent=mock_agent,
            episode_scanner=mock_episode_scanner,
            curriculum_selector=mock_curriculum_selector,
            position_handler=mock_position_handler,
            data_manager=mock_data_manager
        )
    
    def test_training_manager_initialization(self, training_manager, training_config):
        """Test training manager initialization."""
        assert training_manager.config == training_config
        assert training_manager.state == TrainingState.INITIALIZED
        assert training_manager.current_episode == 0
        assert training_manager.best_metric_value == -float('inf')
        assert training_manager.episodes_without_improvement == 0
    
    def test_training_loop_basic_flow(self, training_manager, mock_environment, mock_agent):
        """Test basic training loop flow."""
        # Run short training
        training_manager.config.num_episodes = 10
        
        training_manager.run_training()
        
        # Verify state progression
        assert training_manager.state == TrainingState.COMPLETED
        assert training_manager.current_episode == 10
        
        # Verify environment and agent interactions
        assert mock_environment.reset.call_count >= 10
        assert mock_agent.predict.call_count > 0
        assert mock_agent.learn.call_count > 0
    
    def test_episode_execution(self, training_manager, mock_environment, mock_agent):
        """Test single episode execution."""
        reset_point = Mock(
            timestamp=datetime(2024, 1, 15, 9, 30),
            quality_score=0.9
        )
        
        # Set up termination after 100 steps
        step_count = 0
        def step_side_effect(*args):
            nonlocal step_count
            step_count += 1
            done = step_count >= 100
            return np.array([0.5] * 100), 1.0, done, False, {'pnl': 10}
        
        mock_environment.step.side_effect = step_side_effect
        
        result = training_manager._run_episode(reset_point)
        
        assert isinstance(result, EpisodeResult)
        assert result.total_reward == 100  # 100 steps * 1.0 reward
        assert result.episode_length == 100
        assert result.reset_point == reset_point
    
    def test_checkpoint_saving(self, training_manager, mock_agent):
        """Test checkpoint saving functionality."""
        training_manager.config.num_episodes = 200
        training_manager.config.checkpoint_frequency = 50
        
        # Track save calls
        save_calls = []
        mock_agent.save.side_effect = lambda path: save_calls.append(path)
        
        training_manager.run_training()
        
        # Should save at episodes 50, 100, 150, 200
        assert len(save_calls) >= 4
        assert any('checkpoint_50' in call for call in save_calls)
        assert any('checkpoint_100' in call for call in save_calls)
    
    def test_best_model_tracking(self, training_manager, mock_agent, mock_environment):
        """Test best model saving based on performance."""
        training_manager.config.num_episodes = 100
        
        # Simulate improving performance
        rewards = [0.5, 1.0, 1.5, 2.0, 1.8]  # Peak at episode 3
        reward_idx = 0
        
        def step_with_varying_reward(*args):
            nonlocal reward_idx
            reward = rewards[min(reward_idx // 20, len(rewards) - 1)]
            reward_idx += 1
            return np.array([0.5] * 100), reward, reward_idx % 50 == 0, False, {}
        
        mock_environment.step.side_effect = step_with_varying_reward
        
        training_manager.run_training()
        
        # Should have saved best model
        assert training_manager.best_metric_value > 0
        save_calls = [call[0][0] for call in mock_agent.save.call_args_list]
        assert any('best_model' in call for call in save_calls)
    
    def test_early_stopping(self, training_manager, mock_environment):
        """Test early stopping mechanism."""
        training_manager.config.num_episodes = 5000
        training_manager.config.early_stopping_patience = 100
        training_manager.config.early_stopping_metric = 'total_reward'
        
        # Simulate no improvement
        mock_environment.step.return_value = (
            np.array([0.5] * 100), 0.1, False, False, {}
        )
        
        training_manager.run_training()
        
        # Should stop early
        assert training_manager.state == TrainingState.EARLY_STOPPED
        assert training_manager.current_episode < 5000
        assert training_manager.episodes_without_improvement >= 100
    
    def test_validation_episodes(self, training_manager, mock_environment):
        """Test validation episode execution."""
        training_manager.config.num_episodes = 1000
        training_manager.config.validation_frequency = 200
        
        validation_results = []
        
        def track_validation(result):
            if result.is_validation:
                validation_results.append(result)
        
        training_manager._process_episode_result = track_validation
        
        training_manager.run_training()
        
        # Should have run validation episodes
        assert len(validation_results) >= 4  # At 200, 400, 600, 800
    
    def test_parallel_data_loading(self, training_manager, mock_data_manager):
        """Test parallel data loading functionality."""
        training_manager.config.parallel_data_loading = True
        training_manager.config.num_data_workers = 4
        
        # Mock async data loading
        async def mock_load_data(date):
            await asyncio.sleep(0.01)  # Simulate loading time
            return {'date': date, 'data': 'mock_data'}
        
        mock_data_manager.load_data_async = mock_load_data
        
        # Test data prefetching
        dates = [datetime(2024, 1, i) for i in range(1, 11)]
        training_manager._prefetch_data(dates)
        
        # Should have data ready
        assert len(training_manager.data_cache) > 0
    
    def test_episode_batching(self, training_manager):
        """Test episode batching for efficiency."""
        training_manager.config.episode_batch_size = 5
        training_manager.config.num_episodes = 20
        
        batch_starts = []
        
        original_run = training_manager._run_episode
        def track_batches(*args, **kwargs):
            batch_starts.append(training_manager.current_episode)
            return original_run(*args, **kwargs)
        
        training_manager._run_episode = track_batches
        
        training_manager.run_training()
        
        # Should process in batches
        assert len(batch_starts) == 20
        # Check batching pattern
        for i in range(0, 20, 5):
            batch = batch_starts[i:i+5]
            assert len(set(batch)) <= 2  # Should be close together
    
    def test_metrics_collection(self, training_manager):
        """Test comprehensive metrics collection."""
        training_manager.config.num_episodes = 50
        
        training_manager.run_training()
        
        metrics = training_manager.get_training_metrics()
        
        assert isinstance(metrics, TrainingMetrics)
        assert metrics.total_episodes == 50
        assert metrics.total_training_time > 0
        assert 'episode_rewards' in metrics.history
        assert 'win_rates' in metrics.history
        assert 'sharpe_ratios' in metrics.history
        
        # Check aggregated metrics
        assert hasattr(metrics, 'avg_episode_reward')
        assert hasattr(metrics, 'best_episode_reward')
        assert hasattr(metrics, 'final_win_rate')
    
    def test_position_inheritance_handling(self, training_manager, mock_position_handler):
        """Test handling of position inheritance between episodes."""
        # First episode ends with position
        mock_position_handler.handle_episode_end.return_value = {
            'had_position': True,
            'position_continues': True,
            'continuation_info': {
                'symbol': 'MLGO',
                'quantity': 500,
                'entry_price': 10.0
            }
        }
        
        training_manager.config.num_episodes = 2
        training_manager.run_training()
        
        # Check that position info was passed to next episode
        assert training_manager.inherited_position is not None
        assert training_manager.inherited_position['symbol'] == 'MLGO'
    
    def test_error_handling_and_recovery(self, training_manager, mock_environment):
        """Test error handling during training."""
        training_manager.config.num_episodes = 100
        
        # Simulate occasional errors
        error_count = 0
        def step_with_errors(*args):
            nonlocal error_count
            error_count += 1
            if error_count % 30 == 0:
                raise RuntimeError("Simulated environment error")
            return np.array([0.5] * 100), 1.0, False, False, {}
        
        mock_environment.step.side_effect = step_with_errors
        
        # Should handle errors gracefully
        training_manager.run_training()
        
        assert training_manager.state == TrainingState.COMPLETED
        assert training_manager.error_count > 0
        assert training_manager.current_episode < 100  # Some episodes failed
    
    def test_curriculum_progression_tracking(self, training_manager, mock_curriculum_selector):
        """Test tracking of curriculum progression."""
        training_manager.config.num_episodes = 3000
        
        # Simulate stage changes
        stage_names = ['stage_1', 'stage_1', 'stage_2', 'stage_2', 'stage_3']
        call_count = 0
        
        def get_stage():
            nonlocal call_count
            stage_idx = min(call_count // 1000, len(stage_names) - 1)
            call_count += 1
            return Mock(name=stage_names[stage_idx])
        
        mock_curriculum_selector.get_current_stage.side_effect = get_stage
        
        training_manager.run_training()
        
        # Check curriculum progression history
        assert len(training_manager.curriculum_progression) > 0
        assert training_manager.curriculum_progression[0]['stage'] == 'stage_1'
        assert any(p['stage'] == 'stage_2' for p in training_manager.curriculum_progression)
    
    def test_training_state_persistence(self, training_manager):
        """Test saving and loading training state."""
        training_manager.config.num_episodes = 100
        
        # Run partial training
        training_manager.config.num_episodes = 50
        training_manager.run_training()
        
        # Save state
        state_dict = training_manager.get_state_dict()
        
        assert state_dict['current_episode'] == 50
        assert state_dict['best_metric_value'] == training_manager.best_metric_value
        assert 'episode_history' in state_dict
        
        # Create new manager and load state
        new_manager = TrainingManager(
            training_manager.config,
            training_manager.environment,
            training_manager.agent,
            training_manager.episode_scanner,
            training_manager.curriculum_selector,
            training_manager.position_handler,
            training_manager.data_manager
        )
        
        new_manager.load_state_dict(state_dict)
        
        assert new_manager.current_episode == 50
        assert new_manager.best_metric_value == training_manager.best_metric_value
    
    def test_multi_symbol_support(self, training_manager, mock_environment):
        """Test training with multiple symbols."""
        symbols = ['MLGO', 'NVDA', 'TSLA']
        training_manager.config.symbols = symbols
        training_manager.config.num_episodes = 30
        
        # Track symbol usage
        symbol_counts = {s: 0 for s in symbols}
        
        def reset_with_symbol(symbol=None, **kwargs):
            if symbol:
                symbol_counts[symbol] += 1
            return np.array([0.5] * 100)
        
        mock_environment.reset.side_effect = reset_with_symbol
        
        training_manager.run_training()
        
        # Should have used all symbols
        assert all(count > 0 for count in symbol_counts.values())
    
    def test_wandb_integration(self, training_manager):
        """Test Weights & Biases integration."""
        with patch('wandb.init') as mock_init, \
             patch('wandb.log') as mock_log:
            
            training_manager.config.use_wandb = True
            training_manager.config.num_episodes = 10
            
            training_manager.run_training()
            
            # Should have initialized wandb
            mock_init.assert_called_once()
            
            # Should have logged metrics
            assert mock_log.call_count > 0
            logged_data = [call[0][0] for call in mock_log.call_args_list]
            assert any('episode_reward' in data for data in logged_data)
    
    def test_dashboard_updates(self, training_manager):
        """Test live dashboard update mechanism."""
        training_manager.config.num_episodes = 20
        training_manager.config.enable_dashboard = True
        
        # Mock dashboard queue
        dashboard_updates = []
        training_manager.dashboard_queue = Mock()
        training_manager.dashboard_queue.put = lambda x: dashboard_updates.append(x)
        
        training_manager.run_training()
        
        # Should have sent updates
        assert len(dashboard_updates) > 0
        
        # Check update structure
        update = dashboard_updates[0]
        assert 'episode' in update
        assert 'metrics' in update
        assert 'timestamp' in update
    
    def test_adaptive_exploration(self, training_manager, mock_agent):
        """Test adaptive exploration rate adjustment."""
        training_manager.config.num_episodes = 1000
        training_manager.config.adaptive_exploration = True
        
        # Track exploration rate changes
        exploration_rates = []
        
        def track_exploration(rate):
            exploration_rates.append(rate)
        
        mock_agent.set_exploration_rate = track_exploration
        
        training_manager.run_training()
        
        # Should have adjusted exploration
        assert len(exploration_rates) > 0
        assert exploration_rates[0] > exploration_rates[-1]  # Decreasing over time


class TestTrainingMetrics:
    """Test training metrics collection and analysis."""
    
    def test_metrics_calculation(self):
        """Test metrics calculation from episode results."""
        results = [
            EpisodeResult(
                episode_num=i,
                total_reward=100 + i * 10,
                episode_length=1000,
                final_pnl=50 + i * 5,
                num_trades=10,
                win_trades=6,
                reset_point=Mock(quality_score=0.8)
            )
            for i in range(10)
        ]
        
        metrics = TrainingMetrics.from_episode_results(results)
        
        assert metrics.total_episodes == 10
        assert metrics.avg_episode_reward == 145  # Average of 100 to 190
        assert metrics.best_episode_reward == 190
        assert metrics.final_win_rate == 0.6
    
    def test_metrics_export(self):
        """Test metrics export functionality."""
        metrics = TrainingMetrics(
            total_episodes=1000,
            total_training_time=3600,
            avg_episode_reward=150,
            best_episode_reward=500,
            final_win_rate=0.65,
            final_sharpe_ratio=1.8,
            curriculum_stages_completed=4
        )
        
        # Export to dict
        export_dict = metrics.to_dict()
        assert export_dict['total_episodes'] == 1000
        assert export_dict['metrics']['sharpe_ratio'] == 1.8
        
        # Export to DataFrame
        df = metrics.to_dataframe()
        assert len(df) > 0
        assert 'metric' in df.columns
        assert 'value' in df.columns