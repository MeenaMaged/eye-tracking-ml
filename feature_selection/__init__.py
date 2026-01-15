"""
Feature Selection module for Eye Tracking ML Project.

Provides feature selection methods and comparison utilities:
- PCA: Principal Component Analysis (dimensionality reduction)
- GA: Genetic Algorithm (feature subset selection)
- LDA: Linear Discriminant Analysis (supervised reduction)
- Comparison: Evaluate all methods on classification tasks

All methods are implemented from scratch.
"""

from .comparison import (
    FeatureSelectionComparison,
    FeatureSelectionResult,
    compare_feature_selection,
    generate_test_data
)

# Import from ml_from_scratch for convenience
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_from_scratch.pca import PCA, IncrementalPCA, select_n_components
from ml_from_scratch.genetic_algorithm import (
    GeneticAlgorithmFeatureSelector,
    SimpleGA
)
from ml_from_scratch.fishers_discriminant import (
    FishersLinearDiscriminant,
    LinearDiscriminantAnalysis
)

__all__ = [
    # Comparison utilities
    'FeatureSelectionComparison',
    'FeatureSelectionResult',
    'compare_feature_selection',
    'generate_test_data',

    # PCA
    'PCA',
    'IncrementalPCA',
    'select_n_components',

    # Genetic Algorithm
    'GeneticAlgorithmFeatureSelector',
    'SimpleGA',

    # Fisher's / LDA
    'FishersLinearDiscriminant',
    'LinearDiscriminantAnalysis',
]
