# Equity Curves, Drawdowns, Cumulative Returns, and Performance Metrics
"""
- **Visualizations:** We will build visualization tools to interpret the agent’s performance:
    - **Equity Curve Plotter:** After an episode or during evaluation, plot the portfolio value over time against the price chart. Mark points where the agent entered or exited positions. This helps in qualitatively assessing if the agent is trading in sensible places (e.g. buying dips and selling rallies) or if it’s doing something odd.
    - **Trade Distribution:** Plot histograms of trade returns, duration, etc. to see the strategy profile (e.g. lots of small wins and occasional big loss? We’d aim to minimize big losses).
    - **Feature/Signal Tracking:** We can also plot certain features or the agent’s internal signals. For instance, if using an LSTM, we might not directly interpret neurons, but if we have an auxiliary prediction (like value function), we can plot the value function over time to see if it correlates with actual future outcomes.
    - These visualization modules could be implemented with Matplotlib/Plotly. We might also integrate the environment with **Gymnasium’s rendering** interface: e.g. `env.render()` could show a live-updating chart of the last N seconds of price with the agent’s position marked, which could be used in a Jupyter notebook to watch the agent trade in real time.
"""