"""Normalization classes for features"""

import numpy as np


class MinMaxNormalizer:
    """Min-max normalization to [0, 1] range"""

    def __init__(self, min_val: float, max_val: float):
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, value: float) -> float:
        """Normalize value to [0, 1] range"""
        # Handle edge cases
        if np.isnan(value):
            return 0.5  # Default to middle

        if np.isinf(value):
            return 1.0 if value > 0 else 0.0

        # Clip to range
        if value <= self.min_val:
            return 0.0
        if value >= self.max_val:
            return 1.0

        # Normalize
        if self.max_val == self.min_val:
            return 0.5

        return (value - self.min_val) / (self.max_val - self.min_val)


class StandardNormalizer:
    """Standard (z-score) normalization with clipping"""

    def __init__(self, mean: float, std: float, clip_range: float = 3.0):
        self.mean = mean
        self.std = std
        self.clip_range = clip_range

    def normalize(self, value: float) -> float:
        """Normalize value using z-score with clipping"""
        # Handle edge cases
        if np.isnan(value):
            return 0.0  # Default to mean

        if np.isinf(value):
            return self.clip_range if value > 0 else -self.clip_range

        # Avoid division by zero
        if self.std == 0:
            return 0.0

        # Calculate z-score
        z_score = (value - self.mean) / self.std

        # Clip to range
        return np.clip(z_score, -self.clip_range, self.clip_range)
