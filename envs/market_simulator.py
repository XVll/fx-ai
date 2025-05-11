# This will simulate market dynamics, slippage, latency, luld halts etc.
"""
- If the agent issues a market order, the fill price will suffer slippage based on current spread and recent volatility/liquidity.
For example, we can simulate slippage by moving the execution price a few ticks against the order or sampling from a distribution proportional to volume imbalance.

- If using limit orders, we simulate a **latency** and **queue**: the order may not fill immediately, and if the price moves away (or a halt occurs), the order could miss. Latency can be a fixed delay (e.g. 1 second) or random within a range to mimic network delays.
Bid/Ask Spread: If the observation includes mid-price but agent trades on market orders, the simulator will incur half-spread cost on each round-trip. This encourages the agent to consider the cost of crossing the spread, similar to a real settinggithub.com.
"""