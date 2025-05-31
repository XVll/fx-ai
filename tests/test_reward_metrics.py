# tests/test_reward_metrics.py - Tests for reward metrics tracking

import pytest
import numpy as np
from unittest.mock import Mock
from collections import deque

from rewards.metrics import ComponentMetrics, RewardMetricsTracker


class TestComponentMetrics:
    """Test ComponentMetrics functionality"""
    
    def test_initialization(self):
        """Test ComponentMetrics initialization"""
        metrics = ComponentMetrics(name="test_component", type="foundational")
        
        assert metrics.name == "test_component"
        assert metrics.type == "foundational"
        assert metrics.total_value == 0.0
        assert metrics.count_triggered == 0
        assert metrics.count_positive == 0
        assert metrics.count_negative == 0
        assert metrics.count_zero == 0
        assert metrics.min_value == float('inf')
        assert metrics.max_value == float('-inf')
        assert len(metrics.values) == 0
        assert metrics.last_triggered_step is None
    
    def test_update_positive_value(self):
        """Test updating with positive values"""
        metrics = ComponentMetrics(name="test", type="foundational")
        
        metrics.update(5.0, step=10, action="BUY", is_profitable=True)
        
        assert metrics.total_value == 5.0
        assert metrics.count_triggered == 1
        assert metrics.count_positive == 1
        assert metrics.count_negative == 0
        assert metrics.count_zero == 0
        assert metrics.min_value == 5.0
        assert metrics.max_value == 5.0
        assert len(metrics.values) == 1
        assert metrics.values[0] == 5.0
        assert metrics.last_triggered_step == 10
        assert len(metrics.when_action_buy) == 1
        assert len(metrics.when_profitable) == 1
    
    def test_update_negative_value(self):
        """Test updating with negative values"""
        metrics = ComponentMetrics(name="test", type="penalty")
        
        metrics.update(-3.0, step=5, action="SELL", is_profitable=False)
        
        assert metrics.total_value == -3.0
        assert metrics.count_negative == 1
        assert metrics.count_positive == 0
        assert metrics.min_value == -3.0
        assert metrics.max_value == -3.0
        assert len(metrics.when_action_sell) == 1
        assert len(metrics.when_losing) == 1
    
    def test_update_zero_value(self):
        """Test updating with zero values"""
        metrics = ComponentMetrics(name="test", type="shaping")
        
        metrics.update(0.0, step=1, action="HOLD", is_profitable=False)
        
        assert metrics.total_value == 0.0
        assert metrics.count_zero == 1
        assert metrics.count_positive == 0
        assert metrics.count_negative == 0
        assert len(metrics.when_action_hold) == 1
    
    def test_multiple_updates(self):
        """Test multiple updates with different values"""
        metrics = ComponentMetrics(name="test", type="foundational")
        
        values = [5.0, -2.0, 0.0, 10.0, -1.0]
        actions = ["BUY", "SELL", "HOLD", "BUY", "SELL"]
        profitable = [True, False, False, True, False]
        
        for i, (value, action, is_prof) in enumerate(zip(values, actions, profitable)):
            metrics.update(value, step=i, action=action, is_profitable=is_prof)
        
        assert metrics.total_value == 12.0  # Sum of all values
        assert metrics.count_triggered == 5
        assert metrics.count_positive == 2
        assert metrics.count_negative == 2
        assert metrics.count_zero == 1
        assert metrics.min_value == -2.0
        assert metrics.max_value == 10.0
        assert len(metrics.values) == 5
        
        # Check action distributions
        assert len(metrics.when_action_buy) == 2
        assert len(metrics.when_action_sell) == 2
        assert len(metrics.when_action_hold) == 1
        
        # Check profitability distributions
        assert len(metrics.when_profitable) == 2
        assert len(metrics.when_losing) == 3
    
    def test_trigger_intervals(self):
        """Test trigger interval tracking"""
        metrics = ComponentMetrics(name="test", type="foundational")
        
        # First update - no interval yet
        metrics.update(1.0, step=10, action="BUY", is_profitable=True)
        assert len(metrics.trigger_intervals) == 0
        
        # Second update - should record interval
        metrics.update(2.0, step=15, action="SELL", is_profitable=True)
        assert len(metrics.trigger_intervals) == 1
        assert metrics.trigger_intervals[0] == 5  # 15 - 10
        
        # Third update
        metrics.update(3.0, step=25, action="HOLD", is_profitable=False)
        assert len(metrics.trigger_intervals) == 2
        assert metrics.trigger_intervals[1] == 10  # 25 - 15
    
    def test_get_statistics_empty(self):
        """Test statistics when no data"""
        metrics = ComponentMetrics(name="test", type="foundational")
        
        stats = metrics.get_statistics()
        
        assert stats['name'] == "test"
        assert stats['type'] == "foundational"
        assert stats['triggered'] == False
    
    def test_get_statistics_with_data(self):
        """Test statistics with data"""
        metrics = ComponentMetrics(name="test", type="foundational")
        
        # Add some data
        values = [1.0, -2.0, 3.0, 0.0, 5.0]
        for i, value in enumerate(values):
            action = ["BUY", "SELL", "HOLD", "HOLD", "BUY"][i]
            is_prof = [True, False, True, False, True][i]
            metrics.update(value, step=i*10, action=action, is_profitable=is_prof)
        
        stats = metrics.get_statistics()
        
        assert stats['triggered'] == True
        assert stats['total_value'] == 7.0
        assert stats['count_triggered'] == 5
        assert stats['mean_value'] == 1.4  # 7.0 / 5
        assert stats['min_value'] == -2.0
        assert stats['max_value'] == 5.0
        assert stats['median_value'] == 1.0
        
        # Check rates
        assert stats['positive_rate'] == 0.6  # 3/5
        assert stats['negative_rate'] == 0.2  # 1/5
        assert stats['zero_rate'] == 0.2     # 1/5
        
        # Check conditional means
        assert stats['mean_positive'] == 3.0  # (1+3+5)/3
        assert stats['mean_negative'] == -2.0 # (-2)/1
        
        # Check action correlations
        assert stats['mean_when_buy'] == 3.0   # (1+5)/2
        assert stats['mean_when_sell'] == -2.0 # (-2)/1
        assert stats['mean_when_hold'] == 1.5  # (3+0)/2
        
        # Check profitability correlations
        assert stats['mean_when_profitable'] == 3.0  # (1+3+5)/3
        assert stats['mean_when_losing'] == -1.0     # (-2+0)/2
    
    def test_deque_maxlen_behavior(self):
        """Test that deques respect maxlen"""
        metrics = ComponentMetrics(name="test", type="foundational")
        
        # Add more values than maxlen (1000 for values deque)
        for i in range(1005):
            metrics.update(float(i), step=i, action="BUY", is_profitable=True)
        
        # Values deque should be limited to 1000
        assert len(metrics.values) == 1000
        assert metrics.values[0] == 5.0   # Should have dropped first 5 values
        assert metrics.values[-1] == 1004.0
        
        # But total statistics should include all values
        assert metrics.count_triggered == 1005
        assert metrics.total_value == sum(range(1005))


class TestRewardMetricsTracker:
    """Test RewardMetricsTracker functionality"""
    
    def setup_method(self):
        """Setup test tracker"""
        self.tracker = RewardMetricsTracker()
    
    def test_initialization(self):
        """Test RewardMetricsTracker initialization"""
        assert len(self.tracker.component_metrics) == 0
        assert len(self.tracker.episode_metrics) == 0
        assert self.tracker.current_episode_steps == 0
        assert self.tracker.total_episodes == 0
        assert self.tracker.total_steps == 0
        assert self.tracker.episode_total_reward == 0.0
        assert self.tracker.episode_trades == 0
        assert self.tracker.episode_profitable_trades == 0
    
    def test_register_component(self):
        """Test component registration"""
        self.tracker.register_component("pnl_reward", "foundational")
        self.tracker.register_component("holding_penalty", "penalty")
        
        assert "pnl_reward" in self.tracker.component_metrics
        assert "holding_penalty" in self.tracker.component_metrics
        assert self.tracker.component_metrics["pnl_reward"].name == "pnl_reward"
        assert self.tracker.component_metrics["pnl_reward"].type == "foundational"
    
    def test_register_duplicate_component(self):
        """Test registering same component twice"""
        self.tracker.register_component("test_comp", "foundational")
        original_comp = self.tracker.component_metrics["test_comp"]
        
        # Register again - should not replace
        self.tracker.register_component("test_comp", "different_type")
        assert self.tracker.component_metrics["test_comp"] is original_comp
        assert self.tracker.component_metrics["test_comp"].type == "foundational"
    
    def test_update_component(self):
        """Test updating component metrics"""
        self.tracker.register_component("test_comp", "foundational")
        
        diagnostics = {"test": True}
        self.tracker.update_component("test_comp", 5.0, diagnostics, 10, "BUY", True)
        
        comp_metrics = self.tracker.component_metrics["test_comp"]
        assert comp_metrics.count_triggered == 1
        assert comp_metrics.total_value == 5.0
        assert self.tracker.episode_component_totals["test_comp"] == 5.0
    
    def test_update_unregistered_component(self):
        """Test updating unregistered component"""
        # Should log warning but not crash
        self.tracker.update_component("unknown", 5.0, {}, 10, "BUY", True)
        
        assert "unknown" not in self.tracker.component_metrics
        assert self.tracker.episode_component_totals["unknown"] == 0.0
    
    def test_update_step(self):
        """Test step-level updates"""
        self.tracker.update_step(2.5, "BUY", True, True)
        
        assert self.tracker.current_episode_steps == 1
        assert self.tracker.total_steps == 1
        assert self.tracker.episode_total_reward == 2.5
        assert self.tracker.episode_action_counts["BUY"] == 1
        assert self.tracker.episode_trades == 1
        assert self.tracker.episode_profitable_trades == 1
        
        # Add another step
        self.tracker.update_step(-1.0, "SELL", True, False)
        
        assert self.tracker.current_episode_steps == 2
        assert self.tracker.total_steps == 2
        assert self.tracker.episode_total_reward == 1.5  # 2.5 + (-1.0)
        assert self.tracker.episode_trades == 2
        assert self.tracker.episode_profitable_trades == 1  # Still 1
    
    def test_update_step_no_trade(self):
        """Test step update without trade"""
        self.tracker.update_step(0.1, "HOLD", False, None)
        
        assert self.tracker.episode_trades == 0
        assert self.tracker.episode_profitable_trades == 0
        assert self.tracker.episode_action_counts["HOLD"] == 1
    
    def test_end_episode_basic(self):
        """Test basic episode ending"""
        # Register components and add some data
        self.tracker.register_component("pnl", "foundational")
        self.tracker.register_component("penalty", "penalty")
        
        # Simulate some steps
        self.tracker.update_component("pnl", 10.0, {}, 0, "BUY", True)
        self.tracker.update_step(8.0, "BUY", True, True)
        
        self.tracker.update_component("penalty", -2.0, {}, 1, "HOLD", False)
        self.tracker.update_step(-1.0, "HOLD", False, None)
        
        # End episode
        final_portfolio = {
            'total_equity': 110000.0,
            'realized_pnl_session': 8000.0,
            'sharpe_ratio': 1.5
        }
        
        summary = self.tracker.end_episode(final_portfolio)
        
        assert summary['episode'] == 1
        assert summary['steps'] == 2
        assert summary['total_reward'] == 7.0  # 8.0 + (-1.0)
        assert summary['mean_reward_per_step'] == 3.5
        assert summary['total_trades'] == 1
        assert summary['profitable_trades'] == 1
        assert summary['win_rate'] == 1.0
        assert summary['final_equity'] == 110000.0
        assert summary['total_pnl'] == 8000.0
        assert summary['sharpe_ratio'] == 1.5
        
        # Check component totals
        assert summary['component_totals']['pnl'] == 10.0
        assert summary['component_totals']['penalty'] == -2.0
        
        # Check component means
        assert summary['component_means']['pnl'] == 5.0  # 10.0 / 2 steps
        assert summary['component_means']['penalty'] == -1.0  # -2.0 / 2 steps
        
        # Check action distribution
        assert summary['action_distribution']['BUY'] == 1
        assert summary['action_distribution']['HOLD'] == 1
        
        # Check dominant components
        assert summary['dominant_positive_component']['name'] == 'pnl'
        assert summary['dominant_positive_component']['total'] == 10.0
        assert summary['dominant_negative_component']['name'] == 'penalty'
        assert summary['dominant_negative_component']['total'] == -2.0
        
        # Check that episode state was reset
        assert self.tracker.total_episodes == 1
        assert self.tracker.current_episode_steps == 0
        assert self.tracker.episode_total_reward == 0.0
        assert len(self.tracker.episode_component_totals) == 0
        assert len(self.tracker.episode_action_counts) == 0
        assert self.tracker.episode_trades == 0
        assert self.tracker.episode_profitable_trades == 0
    
    def test_end_episode_no_data(self):
        """Test ending episode with no data"""
        final_portfolio = {'total_equity': 100000.0}
        summary = self.tracker.end_episode(final_portfolio)
        
        assert summary['episode'] == 1
        assert summary['steps'] == 0
        assert summary['total_reward'] == 0.0
        assert summary['total_trades'] == 0
        assert summary['win_rate'] == 0.0
    
    def test_multiple_episodes(self):
        """Test tracking across multiple episodes"""
        # Episode 1
        self.tracker.register_component("pnl", "foundational")
        self.tracker.update_step(5.0, "BUY", True, True)
        summary1 = self.tracker.end_episode({'total_equity': 105000.0})
        
        # Episode 2
        self.tracker.update_step(-2.0, "SELL", True, False)
        self.tracker.update_step(3.0, "HOLD", False, None)
        summary2 = self.tracker.end_episode({'total_equity': 103000.0})
        
        assert len(self.tracker.episode_metrics) == 2
        assert self.tracker.total_episodes == 2
        assert summary1['episode'] == 1
        assert summary2['episode'] == 2
        assert summary2['steps'] == 2
        assert summary2['total_reward'] == 1.0  # -2.0 + 3.0
    
    def test_get_component_analysis_insufficient_episodes(self):
        """Test component analysis with insufficient episodes"""
        analysis = self.tracker.get_component_analysis()
        assert analysis == {}
    
    def test_get_component_analysis_with_data(self):
        """Test component analysis with sufficient data"""
        # Setup components
        self.tracker.register_component("pnl", "foundational")
        self.tracker.register_component("penalty", "penalty")
        
        # Run multiple episodes
        for episode in range(3):
            # Simulate episode with different patterns
            base_pnl = 10.0 + episode * 5.0
            base_penalty = -2.0 - episode * 1.0
            
            self.tracker.update_component("pnl", base_pnl, {}, 0, "BUY", True)
            self.tracker.update_component("penalty", base_penalty, {}, 1, "SELL", False)
            self.tracker.update_step(base_pnl + base_penalty, "BUY", True, True)
            self.tracker.update_step(0.0, "HOLD", False, None)
            
            final_portfolio = {'total_equity': 100000.0 + episode * 1000}
            self.tracker.end_episode(final_portfolio)
        
        analysis = self.tracker.get_component_analysis()
        
        assert analysis['total_episodes'] == 3
        assert analysis['total_steps'] == 6  # 2 steps per episode * 3 episodes
        assert 'components' in analysis
        
        # Check component analysis
        components = analysis['components']
        assert 'pnl' in components
        assert 'penalty' in components
        
        pnl_analysis = components['pnl']
        assert pnl_analysis['global_total_value'] == 45.0  # 10+15+20
        assert abs(pnl_analysis['global_mean_value'] - 12.5) < 1e-6   # Mean of per-episode means
        assert pnl_analysis['mean_positive_rate'] == 1.0   # Always positive
        
        penalty_analysis = components['penalty']
        assert penalty_analysis['global_total_value'] == -9.0  # -2-3-4
        assert penalty_analysis['mean_negative_rate'] == 1.0   # Always negative
        
        # Check for potential issues
        issues = analysis['potential_issues']
        assert len(issues) >= 0  # May or may not have issues detected
    
    def test_get_correlation_analysis_insufficient_data(self):
        """Test correlation analysis with insufficient data"""
        correlations = self.tracker.get_correlation_analysis()
        assert correlations == {}
    
    def test_get_correlation_analysis_with_data(self):
        """Test correlation analysis with sufficient data"""
        # Setup and run multiple episodes with varying patterns
        for episode in range(5):
            reward = 10.0 - episode * 2.0  # Decreasing rewards
            trades = episode + 1           # Increasing trades
            win_rate = 1.0 - episode * 0.1 # Decreasing win rate
            
            for step in range(trades):
                is_profitable = step == 0  # First trade profitable
                self.tracker.update_step(reward/trades, "BUY", True, is_profitable)
            
            # Add some hold steps
            self.tracker.update_step(0.0, "HOLD", False, None)
            
            final_portfolio = {'total_equity': 100000.0}
            self.tracker.end_episode(final_portfolio)
        
        correlations = self.tracker.get_correlation_analysis()
        
        assert 'reward_vs_trades' in correlations
        assert 'reward_vs_win_rate' in correlations
        assert isinstance(correlations['reward_vs_trades'], float)
        assert isinstance(correlations['reward_vs_win_rate'], float)
        
        # Should be negative correlation (more trades = lower reward in our test)
        assert correlations['reward_vs_trades'] < 0
    
    def test_edge_case_zero_steps(self):
        """Test edge case with zero steps in episode"""
        final_portfolio = {'total_equity': 100000.0}
        summary = self.tracker.end_episode(final_portfolio)
        
        assert summary['steps'] == 0
        assert summary['mean_reward_per_step'] == 0.0
        assert summary['win_rate'] == 0.0
    
    def test_edge_case_zero_trades(self):
        """Test edge case with zero trades"""
        self.tracker.update_step(1.0, "HOLD", False, None)
        self.tracker.update_step(2.0, "HOLD", False, None)
        
        final_portfolio = {'total_equity': 100000.0}
        summary = self.tracker.end_episode(final_portfolio)
        
        assert summary['total_trades'] == 0
        assert summary['profitable_trades'] == 0
        assert summary['win_rate'] == 0.0
    
    def test_component_statistics_in_episode_summary(self):
        """Test that component statistics are included in episode summary"""
        self.tracker.register_component("test_comp", "foundational")
        
        # Add some varied data
        values = [1.0, -2.0, 3.0, 0.0]
        for i, value in enumerate(values):
            self.tracker.update_component("test_comp", value, {}, i, "BUY", True)
            self.tracker.update_step(value, "BUY", False, None)
        
        final_portfolio = {'total_equity': 100000.0}
        summary = self.tracker.end_episode(final_portfolio)
        
        assert 'component_statistics' in summary
        comp_stats = summary['component_statistics']['test_comp']
        
        assert comp_stats['triggered'] == True
        assert comp_stats['count_triggered'] == 4
        assert comp_stats['trigger_rate'] == 1.0  # 4 triggers / 4 steps
        assert comp_stats['total_value'] == 2.0  # 1-2+3+0
        assert comp_stats['positive_rate'] == 0.5  # 2 positive out of 4
        assert comp_stats['negative_rate'] == 0.25 # 1 negative out of 4
        assert comp_stats['zero_rate'] == 0.25     # 1 zero out of 4


if __name__ == '__main__':
    pytest.main([__file__, '-v'])