# FxAI Dashboard

A comprehensive real-time trading dashboard with dark mode for monitoring training and trading activity.

## Architecture

The dashboard uses a **hybrid architecture** that combines:

1. **Metrics System** - Aggregated metrics from training (W&B compatible)
2. **Event Stream** - Detailed trading events (trades, positions, market data)

### Components

- **`event_stream.py`** - High-frequency event streaming for detailed trading data
- **`shared_state.py`** - Thread-safe shared state that both systems write to
- **`dashboard_server.py`** - Dash web server that reads from shared state
- **`dashboard_transmitter.py`** - Metrics system integration

## Features

### Real-Time Display
- **Header**: Model name, session time, symbol
- **Market Info**: NY time, trading hours, price, bid/ask/spread, volume
- **Position**: Side, quantity, avg entry, P&L ($ and %)
- **Portfolio**: Total equity, cash, session/realized/unrealized P&L, drawdown, Sharpe, win rate
- **Recent Trades**: Table with time/side/qty/price/P&L
- **Actions Analysis**: Distribution and recent actions
- **Episode Info**: Current step, cumulative reward, progress
- **Training Progress**: Mode, stage, episodes, steps
- **PPO Metrics**: All key training metrics
- **Reward Components**: Breakdown with visual analysis
- **Environment Info**: Data quality, momentum, volatility
- **Performance Footer**: Steps/sec, updates/sec, episodes/hr
- **Candlestick Chart**: Full-width price chart with trade markers

### Dark Theme
GitHub-inspired dark theme with:
- Primary background: `#0d1117`
- Secondary panels: `#161b22`
- Green for profits: `#56d364`
- Red for losses: `#f85149`

## Usage

### Starting the Dashboard

```python
from dashboard import start_dashboard

# Start dashboard server
start_dashboard(port=8050, open_browser=True)
```

### During Training

The dashboard automatically receives data from:
1. **MetricsManager** → DashboardTransmitter → Shared State
2. **Trading Environment** → Event Stream → Shared State

No additional code needed - just ensure metrics are enabled in your config.

### Manual Testing

```python
from dashboard import dashboard_state, event_stream

# Send metrics (like W&B)
dashboard_state.update_metrics({
    'policy_loss': 0.05,
    'win_rate': 0.55,
    'sharpe_ratio': 1.5
})

# Send market data
event_stream.emit_market_update(
    symbol="MLGO",
    price=15.50,
    bid=15.49,
    ask=15.51,
    volume=100000
)

# Send trade execution
event_stream.emit_trade(
    side="BUY",
    quantity=100,
    price=15.50,
    fill_price=15.51,
    pnl=25.50,
    commission=0.50
)
```

## Data Flow

```
Training Loop
    ├── MetricsManager
    │   └── DashboardTransmitter → dashboard_state.update_metrics()
    │
    └── TradingEnvironment
        ├── MarketSimulator → event_stream.emit_market_update()
        ├── ExecutionSimulator → event_stream.emit_trade()
        └── PortfolioSimulator → event_stream.emit_position_update()
                    ↓
            Shared State (dashboard_state)
                    ↓
            Dashboard Server (Dash callbacks)
                    ↓
            Web UI (http://localhost:8050)
```

## Benefits

1. **No Duplicate Calculations** - Metrics calculated once, used by both W&B and dashboard
2. **High-Frequency Updates** - Event stream handles tick-by-tick data
3. **Thread-Safe** - Proper locking for concurrent updates
4. **Modular** - Easy to add new event types or metrics
5. **Real-Time** - Sub-second update latency

## Adding New Data

### New Metric
```python
# In your metrics collector
dashboard_state.update_metrics({
    'new_metric': value
})
```

### New Event Type
```python
# In event_stream.py
class EventType(Enum):
    NEW_EVENT = "new_event"

# Emit event
event_stream.emit(EventType.NEW_EVENT, {
    'data': 'value'
})
```

### Handle in Dashboard
```python
# In shared_state.py _handle_event()
elif event.event_type == EventType.NEW_EVENT:
    # Update state
```

## Troubleshooting

1. **Dashboard not updating**: Check that both DashboardTransmitter and event streams are active
2. **Port already in use**: Change port in `start_dashboard(port=8051)`
3. **Missing data**: Verify metrics are being sent and events are being emitted

## Future Enhancements

- WebSocket support for lower latency
- Historical data replay
- Multiple symbol support
- Custom indicator overlays
- Trade analytics panel
- Model decision explanations