"""
Tests for the improved ModelManager with typed state.
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

from core.model_manager import ModelManager, ModelManagerError, ModelNotFoundError, ModelValidationError
from core.model_state import ModelState, TrainingState, ModelMetadata
from config.model.model_storage_config import ModelStorageConfig


class SimpleModel(nn.Module):
    """Simple test model."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        
    def forward(self, x):
        return self.linear(x)


class TestModelManager:
    """Test suite for ModelManager."""
    
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
            atomic_saves=True,
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
        
    def test_save_and_load_best_model(self, model_manager, model, optimizer):
        """Test saving and loading best models."""
        # Create model state
        model_state = ModelState(
            training_state=TrainingState(
                global_step=1000,
                global_episode=50,
                global_update=100,
                global_cycle=2,
            ),
            metadata=ModelMetadata(
                version=1,
                timestamp=datetime.now(),
                reward=0.95,
                symbol="TEST",
                day_quality=0.8,
                metrics={"mean_reward": 0.95, "std_reward": 0.1},
            )
        )
        
        # Save model
        saved_path = model_manager.save_best_model(model, optimizer, model_state)
        assert saved_path.exists()
        
        # Find best model
        best_model = model_manager.find_best_model()
        assert best_model is not None
        assert best_model.metadata.reward == 0.95
        assert best_model.metadata.version == 1
        
        # Load model
        new_model = SimpleModel()
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        
        loaded_state = model_manager.load_model(new_model, new_optimizer)
        assert loaded_state.training_state.global_step == 1000
        assert loaded_state.metadata.reward == 0.95
        
        # Verify weights were loaded
        original_weights = model.state_dict()
        loaded_weights = new_model.state_dict()
        for key in original_weights:
            assert torch.allclose(original_weights[key], loaded_weights[key])
            
    def test_multiple_best_models(self, model_manager, model, optimizer):
        """Test that only max_best_models are kept."""
        # Save 5 models (max is 3)
        for i in range(5):
            model_state = ModelState(
                metadata=ModelMetadata(
                    version=i+1,
                    timestamp=datetime.now(),
                    reward=float(i),
                )
            )
            model_manager.save_best_model(model, optimizer, model_state)
            
        # Check only 3 models remain
        models = model_manager.get_best_models()
        assert len(models) == 3
        
        # Check they are the highest reward models
        rewards = [m.metadata.reward for m in models]
        assert rewards == [4.0, 3.0, 2.0]
        
    def test_checkpoint_save_and_load(self, model_manager, model, optimizer):
        """Test checkpoint functionality."""
        # Create model state
        model_state = ModelState(
            training_state=TrainingState(global_step=500),
            metadata=ModelMetadata(
                version=0,
                timestamp=datetime.now(),
                reward=0.5,
            )
        )
        
        # Save checkpoint
        checkpoint_path = model_manager.save_checkpoint(model, optimizer, model_state)
        assert checkpoint_path.exists()
        
        # Load checkpoint
        new_model = SimpleModel()
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        
        loaded_state = model_manager.load_checkpoint(new_model, new_optimizer)
        assert loaded_state is not None
        assert loaded_state.training_state.global_step == 500
        
    def test_model_not_found_error(self, model_manager, model):
        """Test proper error when model not found."""
        with pytest.raises(ModelNotFoundError):
            model_manager.load_model(model, model_path="nonexistent.pt")
            
    def test_backward_compatibility(self, model_manager, model, optimizer):
        """Test get_best_model_info backward compatibility."""
        # Save a model
        model_state = ModelState(
            metadata=ModelMetadata(
                version=1,
                timestamp=datetime.now(),
                reward=0.75,
            )
        )
        model_manager.save_best_model(model, optimizer, model_state)
        
        # Use backward compatible method
        info = model_manager.get_best_model_info()
        assert info is not None
        assert info['reward'] == 0.75
        assert info['version'] == 1
        assert 'path' in info
        assert 'metadata' in info
        
    def test_metadata_persistence(self, model_manager, model, optimizer):
        """Test that metadata is properly saved and loaded."""
        # Create detailed metadata
        model_state = ModelState(
            metadata=ModelMetadata(
                version=1,
                timestamp=datetime.now(),
                reward=0.85,
                symbol="AAPL",
                day_quality=0.9,
                episode_count=100,
                update_count=50,
                model_class="MultiBranchTransformer",
                metrics={"sharpe": 1.5, "win_rate": 0.65},
                notes="Test model with good performance",
                tags=["test", "high-quality"],
            )
        )
        
        # Save and reload
        model_manager.save_best_model(model, optimizer, model_state)
        
        # Load through find_best_model
        best_model = model_manager.find_best_model()
        assert best_model.metadata.symbol == "AAPL"
        assert best_model.metadata.day_quality == 0.9
        assert best_model.metadata.metrics["sharpe"] == 1.5
        assert "test" in best_model.metadata.tags
        
    def test_atomic_write_safety(self, model_manager, model, optimizer, temp_dir):
        """Test atomic write prevents corruption."""
        model_state = ModelState(
            metadata=ModelMetadata(
                version=1,
                timestamp=datetime.now(),
                reward=0.7,
            )
        )
        
        # Monkey patch to simulate failure during save
        original_save = torch.save
        def failing_save(*args, **kwargs):
            raise Exception("Simulated save failure")
            
        torch.save = failing_save
        
        # Attempt save - should fail but not corrupt existing files
        with pytest.raises(ModelManagerError):
            model_manager.save_best_model(model, optimizer, model_state)
            
        # Restore original save
        torch.save = original_save
        
        # Verify no partial files were left
        temp_files = list((temp_dir / "temp").glob("*.tmp"))
        assert len(temp_files) == 0