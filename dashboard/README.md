# FX-AI Live Trading Dashboard

A real-time monitoring dashboard for the FX-AI trading system using Plotly Dash.

## Features

- **Real-time Metrics**: Current price, position, equity, P&L, and episode tracking
- **Interactive Charts**:
  - Price chart with buy/sell markers
  - Volume and position tracking
  - Action distribution pie chart
  - Reward tracking (step and cumulative)
  - Portfolio equity evolution
  - Feature heatmap for model insights
- **Recent Trades Table**: Shows last 10 trades with P&L
- **Automatic Updates**: Refreshes every second (configurable)

## Usage

### Enable Dashboard in Training

Add these flags when running training:

```bash
# Enable dashboard with default settings
poetry run python main.py --config-path=config --config-name=config \
    training=continuous ++training.load_best_model=true \
    ++enable_dashboard=true \
    data.symbol=MLGO

# Custom port and disable auto-browser
poetry run python main.py --config-path=config --config-name=config \
    training=continuous ++training.load_best_model=true \
    ++enable_dashboard=true \
    ++dashboard_port=8051 \
    ++dashboard_open_browser=false \
    data.symbol=MLGO
```

### Programmatic Usage

```python
from dashboard.live_dashboard import LiveTradingDashboard

# Create dashboard
dashboard = LiveTradingDashboard(
    port=8050,              # Web server port
    update_interval=1000,   # Update interval in milliseconds
    max_points=1000        # Maximum data points to display
)

# Start dashboard
dashboard.start(open_browser=True)

# Update dashboard with data
dashboard.update_step({
    'step': 100,
    'price': 10.5432,
    'volume': 12500,
    'position': 1000,
    'reward': 0.05,
    'equity': 25500,
    'action': 'BUY'
})

dashboard.update_trade({
    'step': 100,
    'action': 'BUY',
    'price': 10.5432,
    'quantity': 1000,
    'pnl': 0
})

dashboard.update_episode({
    'episode': 1,
    'total_reward': 5.23,
    'total_pnl': 500,
    'steps': 1000,
    'win_rate': 65.5,
    'reset': True
})

# Stop dashboard
dashboard.stop()
```

## Dashboard Components

### Main Metrics Row
- **Current Price**: Real-time price display
- **Position**: Current position size (green=long, red=short, yellow=flat)
- **Equity**: Total portfolio value
- **Total P&L**: Cumulative profit/loss
- **Episode/Step**: Current episode and step counter

### Price Chart
- Main price line with buy/sell trade markers
- Trade markers colored by profit (green) or loss (red)
- Position indicator showing long/short exposure

### Volume & Actions
- Volume bars for market activity
- Action distribution pie chart (Buy/Sell/Hold percentages)

### Performance Charts
- **Reward Chart**: Step rewards (bars) and cumulative reward (line)
- **Equity Chart**: Portfolio value over time with initial capital baseline

### Feature Heatmap
- Shows top 20 features and their values over recent steps
- Helps understand what the model is "looking at"
- Red/blue color scale centered at zero

### Recent Trades Table
- Last 10 trades with step, action, price, quantity, and P&L
- Color-coded P&L (green=profit, red=loss)

## Integration with Metrics System

The dashboard is integrated with the FX-AI metrics system:

1. **MetricsManager** has dashboard control methods:
   - `enable_dashboard()`: Start the dashboard
   - `disable_dashboard()`: Stop the dashboard
   - `update_dashboard_*()`: Send data updates

2. **TradingEnvironment** automatically sends:
   - Step-by-step price, volume, position data
   - Trade executions with P&L
   - Episode summaries

3. **DashboardMetricsCollector** handles:
   - Data formatting for visualization
   - Thread-safe updates via queue
   - Feature selection for heatmap

## Technical Details

- Built with Plotly Dash for web framework
- Uses thread-safe deques for data storage
- Runs in separate thread to avoid blocking training
- Dark theme optimized for trading
- Responsive layout that works on different screen sizes

## Troubleshooting

- **Port already in use**: Change the port with `++dashboard_port=8051`
- **Browser doesn't open**: Set `++dashboard_open_browser=false` and manually navigate to `http://localhost:8050`
- **No data showing**: Ensure training has started and episodes are running
- **Performance issues**: Reduce `update_interval` or `max_points` in dashboard config