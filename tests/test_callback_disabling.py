"""Test callback disabling mechanism to ensure no calculations when disabled."""

import unittest
from unittest.mock import Mock, patch
from agent.callbacks import BaseCallback, CallbackManager, create_callback_manager
from agent.optuna_callback import OptunaCallback
from agent.wandb_callback import WandBCallback
from agent.dashboard_callback import DashboardCallback


class TestCallback(BaseCallback):
    """Test callback to verify disabling mechanism."""
    
    def __init__(self, enabled: bool = True):
        super().__init__(enabled)
        self.call_count = 0
        self.events = []
    
    def on_episode_start(self, episode_num: int, reset_info: dict):
        self.call_count += 1
        self.events.append(('episode_start', episode_num))
    
    def on_episode_end(self, episode_num: int, episode_data: dict):
        self.call_count += 1
        self.events.append(('episode_end', episode_num))
    
    def on_custom_event(self, event_name: str, event_data: dict):
        self.call_count += 1
        self.events.append(('custom', event_name))


class TestCallbackDisabling(unittest.TestCase):
    """Test suite for callback disabling mechanism."""
    
    def test_disabled_callback_no_operations(self):
        """Test that disabled callbacks perform no operations."""
        # Create disabled callback
        callback = TestCallback(enabled=False)
        
        # Try to trigger events
        callback.on_episode_start(1, {})
        callback.on_episode_end(1, {'reward': 100})
        callback.on_custom_event('test', {})
        
        # Verify no operations were performed
        self.assertEqual(callback.call_count, 0)
        self.assertEqual(len(callback.events), 0)
    
    def test_enabled_callback_operations(self):
        """Test that enabled callbacks perform operations."""
        # Create enabled callback
        callback = TestCallback(enabled=True)
        
        # Trigger events
        callback.on_episode_start(1, {})
        callback.on_episode_end(1, {'reward': 100})
        callback.on_custom_event('test', {})
        
        # Verify operations were performed
        self.assertEqual(callback.call_count, 3)
        self.assertEqual(len(callback.events), 3)
    
    def test_callback_manager_respects_disabled(self):
        """Test that CallbackManager doesn't call disabled callbacks."""
        # Create callbacks
        enabled_callback = TestCallback(enabled=True)
        disabled_callback = TestCallback(enabled=False)
        
        # Create manager with both callbacks
        manager = CallbackManager([enabled_callback, disabled_callback])
        
        # Trigger events
        manager.trigger('on_episode_start', 1, {})
        manager.trigger('on_episode_end', 1, {'reward': 100})
        
        # Only enabled callback should have been called
        self.assertEqual(enabled_callback.call_count, 2)
        self.assertEqual(disabled_callback.call_count, 0)
    
    def test_optuna_callback_disabled(self):
        """Test OptunaCallback respects disabled state."""
        # Create disabled Optuna callback
        callback = OptunaCallback(trial=None, enabled=False)
        
        # Should not raise any errors when disabled
        callback.on_episode_end(1, {'episode_reward': 100})
        callback.on_training_end({})
    
    @patch('wandb.init')
    @patch('wandb.log')
    def test_wandb_callback_disabled(self, mock_log, mock_init):
        """Test WandBCallback respects disabled state."""
        # Create disabled WandB callback
        callback = WandBCallback(config={'project': 'test'}, enabled=False)
        
        # Should not call wandb functions when disabled
        callback.on_training_start({})
        callback.on_episode_end(1, {'episode_reward': 100})
        
        # Verify wandb was not called
        mock_init.assert_not_called()
        mock_log.assert_not_called()
    
    def test_dashboard_callback_disabled(self):
        """Test DashboardCallback respects disabled state."""
        # Create mock dashboard state
        mock_state = Mock()
        
        # Create disabled dashboard callback
        callback = DashboardCallback(
            config={},
            dashboard_state=mock_state,
            enabled=False
        )
        
        # Should not update dashboard state when disabled
        callback.on_episode_end(1, {'episode_reward': 100})
        callback.on_training_end({})
        
        # Verify dashboard state was not modified
        self.assertFalse(mock_state.called)
    
    def test_create_callback_manager_with_disabled_systems(self):
        """Test callback manager creation with all systems disabled."""
        config = {
            'wandb': {'enabled': False},
            'dashboard': {'enabled': False},
            'optuna_trial': None
        }
        
        manager = create_callback_manager(config)
        
        # Should have no callbacks when all systems disabled
        self.assertEqual(len(manager.callbacks), 0)
    
    def test_create_callback_manager_with_enabled_systems(self):
        """Test callback manager creation with systems enabled."""
        config = {
            'wandb': {'enabled': True, 'project': 'test'},
            'dashboard': {'enabled': True},
            'optuna_trial': Mock()  # Mock trial object
        }
        
        with patch('wandb.init'), patch('dashboard.shared_state.dashboard_state', Mock()):
            manager = create_callback_manager(config)
        
        # Should have callbacks for enabled systems
        self.assertTrue(any(isinstance(cb, WandBCallback) for cb in manager.callbacks))
        self.assertTrue(any(isinstance(cb, DashboardCallback) for cb in manager.callbacks))
        self.assertTrue(any(isinstance(cb, OptunaCallback) for cb in manager.callbacks))
    
    def test_callback_manager_disable_all(self):
        """Test disabling all callbacks at once."""
        # Create manager with enabled callbacks
        callback1 = TestCallback(enabled=True)
        callback2 = TestCallback(enabled=True)
        manager = CallbackManager([callback1, callback2])
        
        # Disable all
        manager.disable_all()
        
        # Trigger events
        manager.trigger('on_episode_start', 1, {})
        
        # No callbacks should have been called
        self.assertEqual(callback1.call_count, 0)
        self.assertEqual(callback2.call_count, 0)
    
    def test_callback_manager_enable_all(self):
        """Test enabling all callbacks at once."""
        # Create manager with disabled callbacks
        callback1 = TestCallback(enabled=False)
        callback2 = TestCallback(enabled=False)
        manager = CallbackManager([callback1, callback2])
        
        # Enable all
        manager.enable_all()
        
        # Trigger events
        manager.trigger('on_episode_start', 1, {})
        
        # All callbacks should have been called
        self.assertEqual(callback1.call_count, 1)
        self.assertEqual(callback2.call_count, 1)


if __name__ == '__main__':
    unittest.main()