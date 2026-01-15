"""
Dataset module for eye tracking ML project.

Provides data collection and management utilities for:
- Blink detection training data
- Eye movement classification data
- Train/test splitting and cross-validation
"""

from .data_collector import LabeledSample, BlinkDataCollector, MovementDataCollector
from .dataset_manager import DatasetManager

__all__ = [
    'LabeledSample',
    'BlinkDataCollector',
    'MovementDataCollector',
    'DatasetManager',
]
