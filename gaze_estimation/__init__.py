"""
Gaze Estimation module.

Provides gaze estimation using various regression models,
all implemented from scratch for CSE381.

Models:
- Linear Regression (baseline)
- Ridge Regression (L2 regularization)
- Polynomial Regression (degree 2-3)
- Neural Network (MLP)
"""

from .model_comparison import GazeModelComparison, compare_gaze_models

__all__ = [
    'GazeModelComparison',
    'compare_gaze_models',
]
