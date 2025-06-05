"""
Test Training Manager and Continuous Training Integration
"""

import pytest
from unittest.mock import Mock, MagicMock
from training.training_manager import TrainingManager, TrainingMode, TerminationReason, TrainingState
from training.continuous_training import ContinuousTraining


class TestTrainingManager:
    """Test TrainingManager functionality"""
    
    def test_training_manager_initialization(self):
        """Test TrainingManager initialization"""
        config = {
            'termination': {
                'max_episodes': 100,
                'max_updates': 50,
                'max_hours': 1.0,
                'intelligent_termination': True
            },
            'episodes': {
                'episode_length': 256,
                'episodes_per_day': 1
            },
            'continuous': {
                'performance_window': 50,
                'recommendation_frequency': 10
            }
        }
        
        manager = TrainingManager(config, "production")
        
        assert manager.mode == TrainingMode.PRODUCTION
        assert manager.termination_controller.max_episodes == 100
        assert manager.termination_controller.max_updates == 50
        assert manager.episode_controller.episode_length == 256
        assert isinstance(manager.continuous_training, ContinuousTraining)
    
    def test_termination_conditions(self):
        """Test termination decision logic"""
        config = {
            'termination': {
                'max_episodes': 100,
                'max_updates': 50,
                'intelligent_termination': False
            }
        }
        
        manager = TrainingManager(config, "optuna")
        state = TrainingState()
        
        # Test no termination initially
        reason = manager.termination_controller.should_terminate(state)
        assert reason is None
        
        # Test episode limit
        state.episodes = 100
        reason = manager.termination_controller.should_terminate(state)
        assert reason == TerminationReason.MAX_EPISODES_REACHED
        
        # Reset and test update limit
        state.episodes = 50
        state.updates = 50
        reason = manager.termination_controller.should_terminate(state)
        assert reason == TerminationReason.MAX_UPDATES_REACHED
    
    def test_intelligent_termination(self):
        """Test intelligent termination in production mode"""
        config = {
            'termination': {
                'intelligent_termination': True,
                'plateau_patience': 5,
                'degradation_threshold': 0.05
            }
        }
        
        manager = TrainingManager(config, "production")
        termination_controller = manager.termination_controller
        
        # Test plateau detection
        for i in range(10):
            termination_controller.update_performance(10.0, i)
        
        # Should detect plateau after patience is exceeded
        state = TrainingState()
        reason = termination_controller.should_terminate(state)
        assert reason == TerminationReason.PERFORMANCE_PLATEAU
    
    def test_optuna_vs_production_mode(self):
        """Test differences between optuna and production modes"""
        base_config = {
            'termination': {
                'max_episodes': 100,
                'intelligent_termination': True
            }
        }
        
        # Optuna mode should disable intelligent termination
        optuna_manager = TrainingManager(base_config, "optuna")
        assert optuna_manager.mode == TrainingMode.OPTUNA
        
        # Production mode should enable intelligent termination
        prod_manager = TrainingManager(base_config, "production")
        assert prod_manager.mode == TrainingMode.PRODUCTION
        assert prod_manager.termination_controller.enable_intelligent_termination


class TestContinuousTraining:
    """Test ContinuousTraining functionality"""
    
    def test_continuous_training_initialization(self):
        """Test ContinuousTraining initialization"""
        config = {
            'performance_window': 50,
            'recommendation_frequency': 10,
            'initial_quality_range': [0.7, 1.0]
        }
        
        ct = ContinuousTraining(config, "production", enabled=True)
        
        assert ct.mode == "production"
        assert ct.enabled is True
        assert ct.recommendation_frequency == 10
        assert ct.difficulty_manager.current_quality_range == [0.7, 1.0]
    
    def test_performance_analysis(self):
        """Test performance trend analysis"""
        config = {'performance_window': 20}
        ct = ContinuousTraining(config, "production")
        
        analyzer = ct.performance_analyzer
        
        # Add performance data showing improvement
        for i in range(25):
            analyzer.add_performance(i * 0.1)  # Steadily improving
        
        analysis = analyzer.analyze_performance()
        assert analysis.trend == "excelling"
        assert analysis.confidence > 0.5
    
    def test_data_difficulty_adaptation(self):
        """Test adaptive data difficulty"""
        config = {'initial_quality_range': [0.7, 1.0]}
        ct = ContinuousTraining(config, "production")
        
        difficulty_manager = ct.difficulty_manager
        
        # Mock performance analysis showing excellence
        from training.continuous_training import PerformanceAnalysis
        analysis = PerformanceAnalysis(
            trend="excelling",
            confidence=0.8,
            recent_performance=1.0,
            baseline_performance=0.5,
            improvement_rate=0.2,
            stability_score=0.9
        )
        
        # Should adapt difficulty (lower quality threshold)
        change = difficulty_manager.adapt_difficulty(analysis)
        assert change is not None
        assert change['quality_range'][0] < 0.7  # Lower threshold = harder data
    
    def test_recommendation_generation(self):
        """Test recommendation generation"""
        config = {
            'recommendation_frequency': 1,  # Every update
            'initial_quality_range': [0.7, 1.0]
        }
        ct = ContinuousTraining(config, "production")
        
        # Mock training state
        from training.training_manager import TrainingState
        state = TrainingState()
        state.updates = 10
        
        performance_metrics = {'mean_reward': 1.5}
        
        # Should generate recommendations
        recommendations = ct.get_recommendations(state, performance_metrics)
        
        # In production mode with good performance, should get recommendations
        assert recommendations is not None


class TestIntegration:
    """Test integration between TrainingManager and ContinuousTraining"""
    
    def test_training_lifecycle(self):
        """Test complete training lifecycle"""
        config = {
            'termination': {
                'max_episodes': 5,  # Small number for testing
                'intelligent_termination': False
            },
            'episodes': {
                'episode_length': 256,
                'episodes_per_day': 1
            },
            'continuous': {
                'recommendation_frequency': 1,
                'initial_quality_range': [0.7, 1.0]
            }
        }
        
        manager = TrainingManager(config, "optuna")
        
        # Mock trainer
        trainer = Mock()
        trainer.run_training_step = Mock(return_value=True)
        trainer.global_episode_counter = 0
        trainer.global_update_counter = 0
        trainer.global_step_counter = 0
        trainer.mean_episode_reward = 1.0
        
        # Should initialize properly
        manager.continuous_training.initialize(trainer)
        
        # Test one step of training lifecycle
        manager._update_training_state(trainer)
        assert manager.state.episodes == 0
        assert manager.state.updates == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])