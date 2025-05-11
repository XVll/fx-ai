"""
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

from strategies.sma_crossover import calculate_sma
from utils.data_loader import load_price_data
from envs.trading_metrics import calculate_metrics


class BasicTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, data_file: str, short_window=5, long_window=20):
        super().__init__()
        self.data = load_price_data(data_file)
        self.prices = self.data['close'].values
        self.sma_short = calculate_sma(self.data['close'], short_window).bfill().values
        self.sma_long = calculate_sma(self.data['close'], long_window).bfill().values
        self.n_steps = len(self.prices)

        # Action: 0 Hold, 1 Buy, 2 Sell
        self.action_space = spaces.Discrete(3)
        # Observations: price, short SMA, long SMA
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1000.0, 1000.0, 1000.0], dtype=np.float32),
            shape=(3,), dtype=np.float32
        )
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0
        self.entry_price = 0.0
        self.profit = 0.0
        self.unrealized_pnl = 0.0
        self.trades = []
        obs = np.array([
            self.prices[0],
            self.sma_short[0],
            self.sma_long[0]
        ], dtype=np.float32)
        return obs, {}

    def step(self, action):
        terminated = False
        truncated = False
        reward = 0.0

        price = self.prices[self.current_step]
        next_price = self.prices[min(self.current_step + 1, self.n_steps - 1)]

        # Trading logic
        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = price
            self.trades.append((self.current_step, price, 'buy'))
        elif action == 2 and self.position == 1:
            pnl = next_price - self.entry_price
            reward = pnl
            self.profit += pnl
            self.position = 0
            self.unrealized_pnl = 0.0
            self.trades.append((self.current_step, next_price, 'sell'))

        # Unrealized PnL
        if self.position == 1:
            self.unrealized_pnl = next_price - self.entry_price
        else:
            self.unrealized_pnl = 0.0

        self.current_step += 1
        if self.current_step >= self.n_steps - 1:
            terminated = True
            reward += self.unrealized_pnl

        obs = np.array([
            next_price,
            self.sma_short[min(self.current_step, self.n_steps - 1)],
            self.sma_long[min(self.current_step, self.n_steps - 1)]
        ], dtype=np.float32)


        return obs, reward, terminated, truncated, {
            "info": "This is Auxiliary diagnostic information",
            "debug": "You can add metrics that describe agent's performance state, variables that are hidden from observations."
        }

    def render(self):
        plt.figure(figsize=(14, 6))
        plt.plot(self.data.index, self.data['close'], label='Price')
        plt.plot(self.data.index, self.sma_short, label='SMA Short')
        plt.plot(self.data.index, self.sma_long, label='SMA Long')
        for step, price, action in self.trades:
            color = 'g' if action == 'buy' else 'r'
            marker = '^' if action == 'buy' else 'v'
            plt.scatter(self.data.index[step], price, color=color, marker=marker)
        plt.legend()
        plt.title(f'Total Profit: {self.profit:.2f}')
        plt.show()
        metrics = calculate_metrics(self.trades)
        print(metrics)
