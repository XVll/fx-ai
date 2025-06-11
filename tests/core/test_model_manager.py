"""
Tests for the simplified ModelManager.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import shutil
from datetime import datetime

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.model_manager import ModelManager, ModelManagerError, ModelNotFoundError
from config.model.model_storage_config import ModelStorageConfig


class SimpleModel(nn.Module):
    """Simple test model."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        
    def forward(self, x):
        return self.linear(x)


class TestModelManager:
    """Test suite for simplified ModelManager."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
        
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ModelStorageConfig(
            checkpoint_dir="checkpoints",
            best_models_dir="best_models",
            temp_dir="temp",
            max_best_models=3,
            save_metadata=True,
        )
        
    @pytest.fixture
    def model_manager(self, config, temp_dir):
        """Create ModelManager instance."""
        return ModelManager(config=config, base_dir=temp_dir)
        
    @pytest.fixture
    def model(self):
        """Create test model."""
        return SimpleModel()
        
    @pytest.fixture
    def optimizer(self, model):
        """Create test optimizer."""
        return torch.optim.Adam(model.parameters(), lr=0.001)
        
    def test_directory_creation(self, model_manager, temp_dir, config):
        """Test that directories are created properly."""
        assert (temp_dir / config.checkpoint_dir).exists()
        assert (temp_dir / config.best_models_dir).exists()
        assert (temp_dir / config.temp_dir).exists()
        
    def test_get_model_version(self, model_manager):
        """Test version numbering."""
        # First version should be 1
        assert model_manager.get_model_version() == 1
        
    def test_save_and_find_best_model(self, model_manager, model, optimizer, temp_dir):
        """Test saving and finding best models."""
        # Save a checkpoint first
        checkpoint_path = model_manager.save_checkpoint(
            model, optimizer, 1000, 50, 100, 2, {"test": "data"}
        )
        
        # Save as best model
        saved_path = model_manager.save_best_model(
            checkpoint_path, 
            {"mean_reward": 0.95, "std_reward": 0.1},
            0.95
        )
        assert Path(saved_path).exists()
        
        # Find best model
        best_model = model_manager.find_best_model()
        assert best_model is not None
        assert best_model['metadata']['reward'] == 0.95
        assert best_model['version'] == 1
        
    def test_load_model(self, model_manager, model, optimizer, temp_dir):
        """Test loading models."""
        # Save a checkpoint first
        checkpoint_path = model_manager.save_checkpoint(
            model, optimizer, 1000, 50, 100, 2, {"test": "data"}
        )
        
        # Save as best model
        model_manager.save_best_model(
            checkpoint_path, 
            {"mean_reward": 0.95},
            0.95
        )
        
        # Load model
        new_model = SimpleModel()
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        
        loaded_model, training_state = model_manager.load_model(new_model, new_optimizer)
        assert training_state['global_step'] == 1000
        assert training_state['global_episode'] == 50
        
        # Verify weights were loaded
        original_weights = model.state_dict()
        loaded_weights = new_model.state_dict()
        for key in original_weights:
            assert torch.allclose(original_weights[key], loaded_weights[key])
            
    def test_multiple_best_models_cleanup(self, model_manager, model, optimizer, temp_dir):
        """Test that only max_best_models are kept."""
        # Save 5 models (max is 3)
        for i in range(5):
            checkpoint_path = model_manager.save_checkpoint(
                model, optimizer, i*100, i*10, i*5, 1, {}
            )
            model_manager.save_best_model(
                checkpoint_path,
                {"mean_reward": float(i)},
                float(i)
            )
            
        # Check only 3 models remain
        best_models_dir = temp_dir / "best_models"
        model_files = list(best_models_dir.glob("*.pt"))
        assert len(model_files) == 3
        
    def test_checkpoint_save_and_load(self, model_manager, model, optimizer):
        """Test checkpoint functionality."""
        # Save checkpoint
        checkpoint_path = model_manager.save_checkpoint(
            model, optimizer, 500, 25, 50, 1, {"test": "checkpoint"}
        )
        assert Path(checkpoint_path).exists()
        
        # Load checkpoint
        new_model = SimpleModel()
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        
        loaded_model, training_state = model_manager.load_checkpoint(new_model, new_optimizer)
        assert training_state['global_step'] == 500
        assert training_state['metadata']['test'] == "checkpoint"
        
    def test_model_not_found_error(self, model_manager, model):
        """Test proper error when model not found."""
        with pytest.raises(ModelNotFoundError):
            model_manager.load_model(model, model_path="nonexistent.pt")
            
    def test_backward_compatibility(self, model_manager, model, optimizer, temp_dir):
        """Test get_best_model_info backward compatibility."""
        # Save a model
        checkpoint_path = model_manager.save_checkpoint(
            model, optimizer, 1000, 50, 100, 2, {}
        )
        model_manager.save_best_model(
            checkpoint_path,
            {"mean_reward": 0.75},
            0.75
        )
        
        # Use backward compatible method
        info = model_manager.get_best_model_info()
        assert info is not None
        assert info['reward'] == 0.75
        assert info['version'] == 1
        assert 'path' in info
        assert 'metadata' in info
        
    def test_default_config(self, temp_dir):
        """Test that ModelManager works with default config."""
        # Should work without explicit config
        manager = ModelManager(base_dir=temp_dir)
        assert manager.config is not None
        assert manager.get_model_version() == 1