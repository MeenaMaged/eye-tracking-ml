"""
Classification module for blink detection and movement classification.

Provides comparison utilities for evaluating multiple classifiers
on eye tracking classification tasks.

Modules:
- classifier_comparison: General classifier comparison (binary and multi-class)
- movement_classifier: Specialized 3-class eye movement classification
"""

from .classifier_comparison import (
    ClassifierComparison,
    compare_classifiers,
    ClassifierResult,
    confusion_matrix,
    precision_recall_f1
)

from .movement_classifier import (
    MovementClassifierComparison,
    compare_movement_classifiers,
    MovementClassifierResult,
    PerClassMetrics,
    generate_movement_data,
    FIXATION,
    SACCADE,
    BLINK,
    CLASS_NAMES,
    MOVEMENT_FEATURE_NAMES
)

__all__ = [
    # General classifier comparison
    'ClassifierComparison',
    'compare_classifiers',
    'ClassifierResult',
    'confusion_matrix',
    'precision_recall_f1',

    # Movement classification (Phase 5)
    'MovementClassifierComparison',
    'compare_movement_classifiers',
    'MovementClassifierResult',
    'PerClassMetrics',
    'generate_movement_data',

    # Movement class constants
    'FIXATION',
    'SACCADE',
    'BLINK',
    'CLASS_NAMES',
    'MOVEMENT_FEATURE_NAMES',
]
