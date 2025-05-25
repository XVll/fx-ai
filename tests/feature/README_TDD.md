# Feature Tests - TDD Approach

## Overview
This directory contains tests for all features in the FxAIv2 feature system. We follow Test-Driven Development (TDD) principles where tests are written before implementation.

## Test Coverage Status

### ✅ Implemented Features with Tests
All 28 implemented features have comprehensive tests:

#### High-Frequency (HF) - 6/6 tested
- `test_hf_features.py`:
  - PriceVelocityFeature
  - PriceAccelerationFeature
  - TapeImbalanceFeature
  - TapeAggressionRatioFeature
  - SpreadCompressionFeature
  - QuoteVelocityFeature

#### Medium-Frequency (MF) - 12/12 tested
- `test_mf_features.py`:
  - All velocity features (1m/5m price/volume)
  - All EMA distance features (9/20 period, 1m/5m)
  - Current candle position features (1m/5m)
  - Body size relative features (1m/5m)

#### Low-Frequency (LF) - 7/7 tested
- `test_lf_features.py`:
  - Daily/previous day range positions
  - Price change from previous close
  - Support/resistance distances
  - Dollar level proximity features

#### Static - 3/3 tested
- `test_static_features.py`:
  - Time of day sine/cosine encoding
  - Market session type

### 🚧 Missing Features with Tests (TDD)
Tests written but features not yet implemented:

#### Missing HF Features - 3 features
- `test_missing_hf_features.py`:
  - VolumeVelocityFeature - 1s volume change rate
  - VolumeAccelerationFeature - 1s volume acceleration
  - QuoteImbalanceFeature - Bid/ask size imbalance

#### Missing MF Features - 14 features
- `test_missing_mf_features.py`:
  - Price acceleration (1m/5m)
  - Volume acceleration (1m/5m)
  - Previous candle positions (1m/5m)
  - Wick analysis (upper/lower, 1m/5m)
  - Swing high/low distances (1m/5m)

#### Portfolio Features - 5 features
- `test_portfolio_features.py`:
  - Position size as % of portfolio
  - Average entry price distance
  - Unrealized P&L
  - Time in position
  - Maximum adverse excursion (MAE)

## Test Design Principles

### 1. Normalization Contract
Every feature MUST:
- Never return NaN values
- Always return values within specified ranges:
  - [-1, 1] for symmetric features (velocities, imbalances)
  - [0, 1] for positive features (positions, distances)
- Handle edge cases gracefully with sensible defaults

### 2. Edge Case Coverage
Each feature test includes:
- Normal operation tests
- Missing data scenarios
- Zero/empty data handling
- Extreme value handling (clipping)
- Invalid data type handling

### 3. Integration Tests
- `test_feature_integration.py` - Tests feature manager and pipeline
- Multiple features from same data source
- Cross-timeframe consistency
- Performance and caching

## Running Tests

```bash
# Run all feature tests
poetry run pytest tests/feature/ -v

# Run only implemented feature tests
poetry run pytest tests/feature/test_*_features.py -v

# Run missing feature tests (will skip)
poetry run pytest tests/feature/test_missing_*.py -v

# Run specific test class
poetry run pytest tests/feature/test_hf_features.py::TestPriceFeatures -v
```

## Adding New Features

1. **Write tests first** in appropriate test file
2. Use `pytest.skip("Feature not implemented yet")` 
3. Define expected behavior and edge cases
4. Implement feature to make tests pass
5. Remove skip decorators once implemented

## Key Test Patterns

### Basic Feature Test
```python
def test_feature_name(self):
    feature = FeatureClass()
    
    market_data = {...}  # Setup test data
    result = feature.calculate(market_data)
    
    assert not np.isnan(result)  # No NaN
    assert -1.0 <= result <= 1.0  # Normalized
    assert result == expected_value  # Correct calculation
```

### Edge Case Test
```python
def test_feature_edge_cases(self):
    feature = FeatureClass()
    
    # Missing data
    result = feature.calculate({})
    assert not np.isnan(result)
    
    # Zero values
    result = feature.calculate({"value": 0})
    assert result == feature.get_default_value()
```

### Requirements Test
```python
def test_feature_requirements(self):
    feature = FeatureClass()
    requirements = feature.get_requirements()
    
    assert requirements["data_type"] == "expected_type"
    assert requirements["lookback"] >= min_lookback
    assert "required_field" in requirements["fields"]
```