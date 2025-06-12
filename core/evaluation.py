"""
Evaluation result data structures and exports.

This module provides backward compatibility by re-exporting
evaluation types from the evaluation package.
"""

# Re-export from the evaluation package for backward compatibility
from core.evaluation import EvaluationResult, EvaluationEpisodeResult

__all__ = [
    "EvaluationEpisodeResult",
    "EvaluationResult"
]