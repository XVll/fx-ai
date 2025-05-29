# Complete Feature Engineering & Model Architecture Guide
## FxAIv2 - Professional Momentum Trading System

*Last Updated: 2025-05-29 after extensive architecture refactoring*

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Recent Major Refactoring](#recent-major-refactoring)
3. [Current Feature System](#current-feature-system)
4. [Model Architecture](#model-architecture)
5. [Feature Implementation Guide](#feature-implementation-guide)
6. [Professional Tools Integration](#professional-tools-integration)
7. [Performance & Configuration](#performance--configuration)
8. [Testing & Validation](#testing--validation)

---

## Architecture Overview

FxAIv2 is a **reinforcement learning-based momentum trading system** specializing in **squeeze/breakout patterns** in low-float stocks (primarily MLGO). The system uses a **Multi-Branch Transformer** with **professional feature engineering**.

### Core Philosophy
- **Momentum/Squeeze Trading**: 3s-1m decision timeframe, ~40 trades/day
- **Professional Standards**: pandas/ta library instead of manual calculations
- **Sequence Efficiency**: Features utilize full temporal windows, not just point-in-time
- **Clean Architecture**: 4-branch system (HF, MF, LF, Portfolio) with honest categorization

---

## Recent Major Refactoring

### ✅ **Completed Improvements (May 2025)**

#### 1. **Sequence Utilization Fix**
**Problem**: Features built expensive 60-timestep sequences but only used 1-3 timesteps

**Solution**: 
- ❌ Removed: `price_velocity`, `price_acceleration`, `volume_velocity`, `volume_acceleration` (HF)
- ❌ Removed: `1m_price_velocity`, `5m_price_velocity`, acceleration variants (MF) 
- ✅ Added: Professional aggregated features using full sequence windows

#### 2. **Professional Tools Integration**
**Problem**: Manual calculations prone to errors and division warnings

**Solution**:
- ✅ **pandas**: Vectorized time series operations
- ✅ **ta library**: Industry-standard indicators (EMA, RSI, MACD, Bollinger Bands, ATR)
- ✅ **Robust error handling**: No more division by zero warnings

#### 3. **Static Branch Elimination**
**Problem**: "Static" features that actually changed continuously

**Solution**:
- ❌ Removed: Entire static branch (was confusing and dishonest)
- ✅ Moved: Session/time context features to LF branch where they belong
- ✅ Clean: 4-category architecture instead of confusing 5

#### 4. **Feature Category Cleanup**
**Before**: Redundant swing features, manual calculations
**After**: Clean professional implementation with proper categorization

---

## Current Feature System

### **Final Feature Architecture**

| Branch | Count | Description | Professional Tools |
|--------|-------|-------------|-------------------|
| **HF** | 7 | 1s aggregated microstructure | Custom aggregated |
| **MF** | 43 | 1m technical analysis | pandas + ta library |
| **LF** | 19 | Session/daily context | Custom + regulatory |
| **Portfolio** | 5 | Position/P&L tracking | Custom |

### **High-Frequency (HF) Features - 7 Total**
*1-second aggregated features over 60-second windows*

| Feature | Implementation | Range | Description |
|---------|----------------|-------|-------------|
| `spread_compression` | Custom | [-1, 1] | Bid-ask spread changes |
| `tape_imbalance` | Custom | [-1, 1] | Buy vs sell volume ratio |
| `tape_aggression_ratio` | Custom | [-1, 1] | Market vs limit order ratio |
| `quote_imbalance` | Custom | [-1, 1] | Bid vs ask size imbalance |
| `hf_momentum_summary` | **Aggregated** | [-1, 1] | **Comprehensive momentum using entire 60s window** |
| `hf_volume_dynamics` | **Aggregated** | [-1, 1] | **Volume patterns across HF window** |
| `hf_microstructure_quality` | **Aggregated** | [-1, 1] | **Market quality from spread/quote data** |

### **Medium-Frequency (MF) Features - 43 Total**  
*1-5 minute technical analysis using professional tools*

#### **Professional Technical Analysis (4 features)**
| Feature | Implementation | Range | Description |
|---------|----------------|-------|-------------|
| `professional_ema_system` | **ta.trend.EMAIndicator** | [-1, 1] | **EMA alignment & trend strength** |
| `professional_momentum_quality` | **ta.momentum.RSI + ta.trend.MACD** | [-1, 1] | **RSI + MACD momentum quality** |
| `professional_volatility_regime` | **ta.volatility.BollingerBands + ATR** | [0, 1] | **Volatility regime detection** |
| `professional_vwap_analysis` | **pandas vectorized** | [-1, 1] | **VWAP relationship analysis** |

#### **Candle Analysis (10 features)**
| Feature | Range | Description |
|---------|-------|-------------|
| `1m_position_in_current_candle` | [0, 1] | Position within current 1m candle |
| `5m_position_in_current_candle` | [0, 1] | Position within current 5m candle |
| `1m_position_in_previous_candle` | [0, 1] | Position in previous 1m candle |
| `5m_position_in_previous_candle` | [0, 1] | Position in previous 5m candle |
| `1m_body_size_relative` | [0, 1] | Body size relative to range |
| `5m_body_size_relative` | [0, 1] | 5m body size relative |
| `1m_upper_wick_relative` | [0, 1] | Upper wick relative to range |
| `5m_upper_wick_relative` | [0, 1] | 5m upper wick relative |
| `1m_lower_wick_relative` | [0, 1] | Lower wick relative to range |
| `5m_lower_wick_relative` | [0, 1] | 5m lower wick relative |

#### **Pattern Detection (16 features)**
| Feature | Range | Description |
|---------|-------|-------------|
| `swing_high_distance` | [-1, 1] | Distance to major swing high |
| `swing_low_distance` | [-1, 1] | Distance to major swing low |
| `swing_high_price_pct` | [-1, 1] | % from current to swing high |
| `swing_low_price_pct` | [-1, 1] | % from current to swing low |
| `bars_since_swing_high` | [0, 1] | Time since last swing high |
| `bars_since_swing_low` | [0, 1] | Time since last swing low |
| `higher_highs_count` | [0, 1] | Number of higher highs |
| `higher_lows_count` | [0, 1] | Number of higher lows |
| `lower_highs_count` | [0, 1] | Number of lower highs |
| `lower_lows_count` | [0, 1] | Number of lower lows |
| `range_compression` | [0, 1] | Squeeze indicator |
| `consolidation_score` | [0, 1] | Price consolidation tightness |
| `triangle_apex_distance` | [0, 1] | Distance to triangle apex |
| `momentum_alignment` | [-1, 1] | Multi-timeframe alignment |
| `breakout_potential` | [0, 1] | Breakout probability |
| `squeeze_intensity` | [0, 1] | Combined squeeze indicators |

#### **Volume Analysis (6 features)**
| Feature | Range | Description |
|---------|-------|-------------|
| `relative_volume` | [0, 3] | Volume vs average |
| `volume_surge` | [0, 5] | Recent volume surge |
| `cumulative_volume_delta` | [-1, 1] | Buy vs sell pressure |
| `volume_momentum` | [-1, 1] | Volume trend strength |

#### **Sequence-Aware Features (4 features)**
| Feature | Range | Description |
|---------|-------|-------------|
| `trend_acceleration` | [-1, 1] | Acceleration using full sequence |
| `volume_pattern_evolution` | [-1, 1] | How volume patterns evolve |
| `momentum_quality` | [-1, 1] | Pattern-based momentum quality |
| `pattern_maturation` | [0, 1] | How close patterns are to completion |

#### **Adaptive Features (2 features)**
| Feature | Range | Description |
|---------|-------|-------------|
| `volatility_adjusted_momentum` | [-1, 1] | Momentum adjusted for market conditions |
| `regime_relative_volume` | [0, 3] | Volume relative to current regime |

#### **Aggregated Sequence Features (3 features)**
| Feature | Range | Description |
|---------|-------|-------------|
| `mf_trend_consistency` | [-1, 1] | Trend consistency across MF window |
| `mf_volume_price_divergence` | [-1, 1] | Volume-price divergence analysis |
| `mf_momentum_persistence` | [-1, 1] | How well momentum persists |

### **Low-Frequency (LF) Features - 19 Total**
*Daily/session context and regulatory features*

#### **Support/Resistance (7 features)**
| Feature | Range | Description |
|---------|-------|-------------|
| `support_distance` | [0, 1] | Distance to nearest support |
| `resistance_distance` | [0, 1] | Distance to nearest resistance |
| `whole_dollar_proximity` | [0, 1] | Distance to $1.00 levels |
| `half_dollar_proximity` | [0, 1] | Distance to $0.50 levels |
| `daily_range_position` | [0, 1] | Position in today's range |
| `position_in_prev_day_range` | [0, 1] | Position vs yesterday's range |
| `price_change_from_prev_close` | [-1, 1] | % change from previous close |

#### **Regulatory/Market Structure (6 features)**
| Feature | Range | Description |
|---------|-------|-------------|
| `distance_to_luld_up` | [0, 1] | Distance to upper LULD band |
| `distance_to_luld_down` | [0, 1] | Distance to lower LULD band |
| `luld_band_width` | [0, 1] | Width of LULD bands |
| `is_halted` | {0, 1} | Trading halt status |
| `time_since_halt` | [0, 1] | Time since last halt |
| `adaptive_support_resistance` | [-1, 1] | Adaptive S/R calculation |

#### **Session/Time Context (6 features - moved from "static")**
| Feature | Range | Description |
|---------|-------|-------------|
| `market_session_type` | [0, 1] | Pre/regular/post market |
| `time_of_day_sin` | [-1, 1] | Sine encoding of time |
| `time_of_day_cos` | [-1, 1] | Cosine encoding of time |
| `session_progress` | [0, 1] | Progress through session |
| `market_stress_level` | [0, 1] | Dynamic market stress |
| `session_volume_profile` | [0, 1] | Volume profile evolution |

### **Portfolio Features - 5 Total**
*Position and P&L tracking*

| Feature | Range | Description |
|---------|-------|-------------|
| `portfolio_position_size` | [-1, 1] | Current position (normalized) |
| `portfolio_average_price` | [-1, 1] | Entry price vs current |
| `portfolio_unrealized_pnl` | [-1, 1] | Current P&L |
| `portfolio_time_in_position` | [0, 1] | Time holding position |
| `portfolio_max_adverse_excursion` | [-1, 0] | Maximum drawdown |

---

## Model Architecture

### **4-Branch Transformer with Professional Features**

```
Input Features (74 total)
    ↓
┌─ HF Branch (7×60) ──┐
├─ MF Branch (43×30) ─┤
├─ LF Branch (19×30) ─┤    → Multi-Head Attention Fusion → Action Output
├─ Portfolio (5×5) ───┤       (5 branches total)
└─ Cross-Attention ───┘
```

#### **Key Architectural Features**

1. **Temporal Pooling**: Exponentially weighted averaging preserves all temporal information
2. **Cross-Timeframe Attention**: HF features attend to MF/LF patterns for context
3. **Professional Integration**: pandas/ta calculations ensure robustness
4. **Clean 4-Category System**: HF, MF, LF, Portfolio (no confusing "static")

#### **Fusion Strategy**
- **5-way fusion**: HF + MF + LF + Portfolio + Cross-Attention
- **Attention-based**: Model learns which branches matter when
- **Pattern extraction**: Convolutional layers identify key patterns

#### **Model Dimensions**
```python
# Feature dimensions (sequence_length × feature_count)
hf_feat_dim: int = 7     # 60-second window, 1s intervals
mf_feat_dim: int = 43    # 30-minute window, 1m intervals  
lf_feat_dim: int = 19    # 30-day context, daily intervals
portfolio_feat_dim: int = 5  # 5-step position history

# Sequence lengths
hf_seq_len: int = 60     # 60 seconds
mf_seq_len: int = 30     # 30 minutes
lf_seq_len: int = 30     # 30 timepoints (varies by feature)
portfolio_seq_len: int = 5  # 5 timesteps

# Architecture
d_model: int = 64        # Feature projection dimension
d_fused: int = 256       # Final fused representation
n_heads: int = 8         # Attention heads
n_layers: int = 4        # Transformer layers
```

---

## Feature Implementation Guide

### **Professional Implementation Standards**

#### **1. Use Professional Libraries**
```python
# ✅ Good: Use pandas + ta library
import pandas as pd
import ta

df = pd.DataFrame(bars_1m)
df['ema9'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

# ❌ Bad: Manual calculations
ema9 = calculate_ema_manually(closes, 9)  # Error-prone, slower
```

#### **2. Sequence-Aware Implementation**
```python
# ✅ Good: Use entire sequence window
def calculate_momentum_quality(bars_window):
    # Uses all 30 bars to assess momentum persistence
    closes = np.array([bar['close'] for bar in bars_window])
    returns = np.diff(closes) / closes[:-1]
    
    # Analyze consistency across full window
    trend_consistency = analyze_direction_changes(returns)
    momentum_persistence = measure_acceleration(returns)
    
    return combine_metrics(trend_consistency, momentum_persistence)

# ❌ Bad: Point-in-time only  
def calculate_velocity(current_bar, previous_bar):
    # Only uses 2 points, wastes 58 other timesteps
    return (current_bar['close'] - previous_bar['close']) / previous_bar['close']
```

#### **3. Robust Error Handling**
```python
def calculate_raw(self, market_data: Dict[str, Any]) -> float:
    try:
        bars = market_data.get('1m_bars_window', [])
        if len(bars) < 10:
            return self.get_default_value()
        
        df = pd.DataFrame(bars)
        if df.empty or 'close' not in df.columns:
            return self.get_default_value()
        
        # Professional calculation using ta library
        result = ta.momentum.RSIIndicator(df['close']).rsi().iloc[-1]
        
        # Robust validation
        if pd.isna(result) or np.isinf(result):
            return self.get_default_value()
            
        return float(np.clip(result / 100.0, 0.0, 1.0))  # Normalize to [0,1]
        
    except Exception:
        return self.get_default_value()
```

### **Feature Registration Pattern**
```python
from feature.feature_base import BaseFeature
from feature.feature_registry import feature_registry
import pandas as pd
import ta

@feature_registry.register("feature_name", category="mf")
class YourProfessionalFeature(BaseFeature):
    """Professional feature using pandas + ta library."""
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        # Implementation using professional tools
        pass
    
    def get_default_value(self) -> float:
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        return {'min': -1.0, 'max': 1.0, 'range_type': 'symmetric'}
    
    def get_requirements(self) -> Dict[str, Any]:
        return {'data_type': 'mf_data', 'lookback': 30, 'fields': ['close', 'volume']}
```

---

## Professional Tools Integration

### **pandas for Time Series**
```python
# Vectorized operations for efficiency
df = pd.DataFrame(market_data)
df['returns'] = df['close'].pct_change()
df['rolling_vol'] = df['returns'].rolling(20).std()
df['volume_sma'] = df['volume'].rolling(10).mean()

# Professional time series analysis
price_momentum = df['close'].rolling(5).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0])
volume_trend = df['volume'].rolling(10).apply(lambda x: np.corrcoef(np.arange(len(x)), x)[0,1])
```

### **ta Library for Technical Indicators**
```python
import ta

# Industry-standard calculations
bb = ta.volatility.BollingerBands(df['close'], window=20)
df['bb_upper'] = bb.bollinger_hband()
df['bb_lower'] = bb.bollinger_lband()
df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']

macd = ta.trend.MACD(df['close'])
df['macd'] = macd.macd()
df['macd_signal'] = macd.macd_signal()
df['macd_histogram'] = macd.macd_diff()

df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
```

### **Available Professional Tools**
| Library | Version | Usage |
|---------|---------|-------|
| **pandas** | 2.2.3 | Time series operations, vectorized calculations |
| **ta** | Latest | RSI, MACD, EMA, Bollinger Bands, ATR, volume indicators |
| **numpy** | 2.2.6 | Array operations, mathematical functions |

---

## Performance & Configuration

### **Current Performance Metrics**
- **Total Features**: 74 (reduced from 80+ through professional optimization)
- **Model Parameters**: ~319k (efficient transformer design)
- **Feature Computation**: Vectorized pandas operations
- **Memory Usage**: Optimized sequence windows

### **Configuration Files**
```yaml
# config/schemas.py - Feature dimensions
hf_feat_dim: 7   # Professional aggregated features
mf_feat_dim: 43  # pandas/ta library features  
lf_feat_dim: 19  # Session context + regulatory
portfolio_feat_dim: 5

# No more static_feat_dim - eliminated!
```

### **Feature Loading**
```python
# feature/load_features.py - Automatic discovery
def load_all_features():
    # Core features
    from feature.hf import price_features, tape_features, quote_features
    from feature.mf import candle_features, ema_features, swing_features
    from feature.lf import range_features, level_features
    from feature.portfolio import portfolio_features
    
    # Professional features
    from feature.professional import ta_features
    from feature.aggregated import aggregated_features
    
    # Market structure
    from feature.market_structure import halt_features, luld_features
    from feature.volume_analysis import vwap_features, relative_volume_features
```

---

## Testing & Validation

### **Comprehensive Testing Strategy**

#### **1. Feature Unit Tests**
```python
def test_professional_features():
    from feature.professional.ta_features import ProfessionalEMASystemFeature
    
    feature = ProfessionalEMASystemFeature()
    
    # Test with synthetic data
    market_data = create_test_market_data()
    result = feature.calculate(market_data)
    
    # Validate bounds
    assert -1.0 <= result <= 1.0
    
    # Test edge cases
    assert feature.calculate({}) == feature.get_default_value()
```

#### **2. Model Integration Tests**
```python
def test_model_with_professional_features():
    config = Config()
    model = MultiBranchTransformer(config.model)
    
    # Test forward pass
    state_dict = create_test_state_dict()
    with torch.no_grad():
        result = model(state_dict)
    
    assert len(result) == 2  # (actions, values)
```

#### **3. Performance Validation**
```bash
# Run comprehensive tests
poetry run poe test

# Test specific feature categories
poetry run pytest tests/test_professional_features.py -v
poetry run pytest tests/test_aggregated_features.py -v

# Performance profiling
poetry run python scripts/profile_features.py
```

### **Validation Checklist**

- [x] **No NaN/Infinity**: All features return valid floats
- [x] **Normalized Ranges**: Features respect declared bounds
- [x] **Professional Tools**: Using pandas/ta instead of manual calculations
- [x] **Sequence Efficiency**: Features utilize full temporal windows
- [x] **Clean Architecture**: 4-branch system with honest categorization
- [x] **Error Handling**: Robust fallbacks for edge cases
- [x] **Performance**: Vectorized operations for efficiency

---

## Migration & Maintenance

### **Completed Migrations**

#### **✅ May 2025 - Professional Refactoring**
1. **Sequence Utilization**: Replaced point-in-time with sequence-aware features
2. **Professional Tools**: Integrated pandas/ta library for robust calculations  
3. **Static Elimination**: Removed confusing static branch, moved to LF
4. **Feature Optimization**: Reduced from 80+ to 74 high-quality features

### **Future Enhancements**

#### **Phase 1: Additional Professional Features**
- More ta library indicators (Williams %R, Stochastic, CCI)
- Advanced volume analysis (accumulation/distribution, OBV)
- Market breadth indicators

#### **Phase 2: Performance Optimization**
- Feature importance tracking via attention weights
- Dynamic feature selection based on market regimes
- Caching optimizations for repeated calculations

#### **Phase 3: Advanced Analytics**
- Inter-symbol correlation features
- Options flow integration (if data available)
- Alternative data sources (news sentiment, social media)

---

## Conclusion

The FxAIv2 feature system has undergone extensive professional refactoring to create a **robust, efficient, and maintainable** architecture:

### **Key Achievements**
✅ **Professional Standards**: pandas/ta library integration  
✅ **Sequence Efficiency**: Full temporal window utilization  
✅ **Clean Architecture**: Honest 4-branch categorization  
✅ **Robust Implementation**: No division warnings or edge case failures  
✅ **Momentum Focus**: Optimized for squeeze/breakout pattern detection  

### **System Strengths**
- **Proven Tools**: Industry-standard technical analysis library
- **Efficient Design**: 74 high-quality features vs 80+ redundant features
- **Professional Implementation**: pandas vectorized operations
- **Clear Organization**: Features properly categorized by update frequency
- **Robust Error Handling**: Graceful fallbacks for all edge cases

The system is now **production-ready** for professional momentum trading with a clean, maintainable codebase that follows industry best practices.

---

*This document represents the complete state of the FxAIv2 feature and model architecture as of May 2025 after extensive refactoring. All implementation details, architectural decisions, and professional standards are documented for future development and maintenance.*