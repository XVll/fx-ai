"""Feature registry for dynamic feature discovery"""
from typing import Dict, Type, List, Optional
from .feature_base import BaseFeature


class FeatureRegistry:
    """Registry for all available features"""
    
    def __init__(self):
        self._features: Dict[str, Type[BaseFeature]] = {}
        self._categories: Dict[str, List[str]] = {}
    
    def register(self, name: str, category: Optional[str] = None):
        """Decorator to register a feature class"""
        def decorator(cls: Type[BaseFeature]):
            if name in self._features:
                raise ValueError(f"Feature '{name}' already registered")
            
            self._features[name] = cls
            
            if category:
                if category not in self._categories:
                    self._categories[category] = []
                self._categories[category].append(name)
            
            return cls
        return decorator
    
    def has_feature(self, name: str) -> bool:
        """Check if a feature is registered"""
        return name in self._features
    
    def get_feature_class(self, name: str) -> Type[BaseFeature]:
        """Get a feature class by name"""
        if name not in self._features:
            raise ValueError(f"Feature '{name}' not found in registry")
        return self._features[name]
    
    def list_features(self) -> List[str]:
        """List all registered feature names"""
        return list(self._features.keys())
    
    def get_features_by_category(self, category: str) -> List[str]:
        """Get all features in a category"""
        return self._categories.get(category, [])


# Global registry instance
feature_registry = FeatureRegistry()