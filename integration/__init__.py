"""
Integration module for Eye Tracking ML Project - Phase 8.

Provides live demonstration and batch evaluation capabilities:
- LiveDemo: Real-time eye tracking with ML predictions
- BatchEvaluation: Complete evaluation pipeline for all ML tasks

This is the final phase that integrates all previous phases
into a cohesive system.
"""

from .live_demo import (
    LiveDemo,
    run_live_demo,
)

from .batch_evaluation import (
    BatchEvaluation,
    ResultsDocumenter,
    run_full_evaluation,
)

__all__ = [
    # Live demo
    'LiveDemo',
    'run_live_demo',

    # Batch evaluation
    'BatchEvaluation',
    'ResultsDocumenter',
    'run_full_evaluation',
]
