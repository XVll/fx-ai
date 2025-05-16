# envs/wrappers.py

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Dict, Box
from typing import Dict as DictType, Any


class NormalizeDictObservation(gym.ObservationWrapper):
    def __init__(self, env, epsilon=1e-8):
        super().__init__(env)
        self.epsilon = epsilon
        self.update_running_mean = True

        # Create running mean/std tracker for each component
        self.running_stats = {}

        # Create new observation space with float32 dtype for all components
        spaces = {}

        for key, space in self.observation_space.spaces.items():
            if isinstance(space, Box):
                # Replace with unbounded float32 space of same shape
                spaces[key] = Box(
                    low=-np.inf, high=np.inf,
                    shape=space.shape,
                    dtype=np.float32
                )

                # Initialize running stats for this component
                self.running_stats[key] = {
                    'mean': np.zeros(space.shape, dtype=np.float32),
                    'var': np.ones(space.shape, dtype=np.float32),
                    'count': 0
                }
            else:
                # Keep non-Box spaces unchanged
                spaces[key] = space

        # Update observation space
        self.observation_space = Dict(spaces)

    def observation(self, obs: DictType[str, Any]) -> DictType[str, Any]:
        """
        Normalize each component of the observation.

        Args:
            obs: Dict observation from the environment

        Returns:
            Dict with normalized components
        """
        normalized_obs = {}

        for key, value in obs.items():
            if key in self.running_stats:
                # Convert to float32 numpy array
                value = np.asarray(value, dtype=np.float32)

                if self.update_running_mean:
                    # Update running stats (Welford's algorithm)
                    self.running_stats[key]['count'] += 1
                    count = self.running_stats[key]['count']

                    # Update mean
                    delta = value - self.running_stats[key]['mean']
                    self.running_stats[key]['mean'] += delta / count

                    # Update variance
                    delta2 = value - self.running_stats[key]['mean']
                    self.running_stats[key]['var'] *= (count - 1) / count
                    self.running_stats[key]['var'] += delta * delta2 / count

                # Normalize using current stats
                mean = self.running_stats[key]['mean']
                var = self.running_stats[key]['var']
                std = np.sqrt(var) + self.epsilon

                normalized_obs[key] = (value - mean) / std
            else:
                # Pass through unchanged
                normalized_obs[key] = value

        return normalized_obs