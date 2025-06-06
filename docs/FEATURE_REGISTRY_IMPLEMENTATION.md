# Feature Registry Implementation - Single Source of Truth

## Overview

We have successfully implemented a single source of truth for all feature definitions in the FxAIv2 system through the `FeatureRegistry` class. This ensures consistency across all components that use features.

## Key Components

### 1. FeatureRegistry (`feature/feature_registry.py`)

The central registry that defines:
- **Feature names** - Exact names for all features
- **Feature descriptions** - What each feature represents
- **Feature grouping** - How features are organized (e.g., price_action, volume_dynamics)
- **Active/Inactive status** - Which features are currently used by the model
- **Feature dimensions** - Expected counts for each category

Current active dimensions:
- HF: 9 features (all active)
- MF: 43 features (out of 52 total)
- LF: 19 features (out of 22 total)
- Portfolio: 10 features (all active)

### 2. SimpleFeatureManager Updates

Updated to use FeatureRegistry for:
- Creating features in the exact order defined by the registry
- Only creating active features (skipping inactive ones)
- Validating feature counts match expectations
- Using real feature names instead of generic ones

### 3. Config Schema Updates

`ModelConfig` now:
- Has a validator that checks dimensions against FeatureRegistry
- Provides `from_registry()` class method to create config with registry dimensions
- Logs warnings if hardcoded dimensions don't match registry

### 4. Captum Integration

The Captum attribution system now:
- Uses FeatureRegistry for all feature names
- Automatically generates feature groups from registry
- Shows meaningful names like "price_velocity" instead of "hf_feat_0"
- Groups features correctly for analysis

## Benefits

1. **Consistency** - All components use the same feature definitions
2. **Maintainability** - Add/remove features in one place
3. **Clarity** - Meaningful feature names throughout the system
4. **Flexibility** - Easy to mark features as active/inactive
5. **Validation** - Automatic checking for mismatches

## Testing

Run the consistency test to verify everything is working:

```bash
poetry run python scripts/test_feature_consistency.py
```

This test checks:
- FeatureRegistry dimensions
- SimpleFeatureManager feature creation
- ModelConfig dimensions
- Feature name ordering
- Captum integration

## Adding New Features

To add a new feature:

1. Add it to the appropriate list in FeatureRegistry (HF_FEATURES, MF_FEATURES, etc.)
2. Mark it as active=True or active=False
3. Implement the feature class in the appropriate module
4. Add the class mapping in SimpleFeatureManager

The system will automatically:
- Include it in feature extraction (if active)
- Add it to Captum attribution analysis
- Validate dimensions across components

## Migration Notes

- The system maintains backward compatibility with existing configs
- Warnings are logged for dimension mismatches but training continues
- All existing models will work without changes