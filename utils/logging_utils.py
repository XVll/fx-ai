"""
Might set up Python loggers or wrappers for experiment tracking (e.g. via TensorBoard or Weights & Biases).

For a complex trading agent, robust logging and monitoring are essential both for debugging and for later analysis:

- **Environment Logging:** The environment will include detailed logging of each step’s important events. We can implement this in `logging_utils.py` or within the environment:
    - Each trade executed (time, price, size, action taken).
    - P&L updates after each step.
    - When significant events happen (e.g. hitting a stop-loss, a LULD halt starts/ends, large slippage occurred on an order fill), the environment can log a message.

        These can be written to a log file or stored in memory for analysis after an episode. Having a chronological log of actions and outcomes greatly aids debugging (we can trace why the agent had a big loss, for example).


        Error and Debug Logs: We will use Python’s logging to capture errors or unusual conditions. For example, if the data feed runs out of data unexpectedly or if the agent tries an invalid action (shouldn’t happen if spaces are correct, but just in case), we log it. The environment can have an option like verbose=True to print debug info during development.

        - **Performance Tracking:** In a trading context, **post-trade analysis** is very important. We plan to create a module (or just use the logs) for computing after each episode:
    - Total return, average return per trade.
    - Max drawdown, Sharpe ratio of the episode.
    - Hit rate (% of profitable trades), profit factor (sum of profits / sum of losses).
    - These can be output to console or saved, and during hyperparam tuning, the evaluation of these metrics guides the optimization objective.
"""
