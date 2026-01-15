"""
ML From Scratch - Machine Learning algorithms implemented from scratch.

This module contains implementations of various ML algorithms for the
CSE381 Eye Tracking project, all built without using sklearn or similar libraries.

Regressors:
- LinearRegression: OLS and Ridge regression using normal equation
- PolynomialFeatures: Feature expansion for polynomial regression
- PolynomialRegression: Combined polynomial + linear regression

Classifiers:
- DecisionTreeClassifier: CART with entropy/gini
- RandomForestClassifier: Bagging + feature sampling
- XGBoostClassifier: Gradient boosting
- KNeighborsClassifier: K-nearest neighbors
- SVC: Support Vector Machine (linear/RBF kernel)
- GaussianNB: Gaussian Naive Bayes

Neural Networks:
- NeuralNetwork: Multi-layer perceptron for regression and classification

Feature Selection (Phase 6):
- PCA: Principal Component Analysis
- GeneticAlgorithmFeatureSelector: Feature subset selection via GA
- FishersLinearDiscriminant: Binary classification projection
- LinearDiscriminantAnalysis: Multi-class LDA
"""

# Regressors
from .linear_regression import LinearRegression, RidgeRegression
from .polynomial_features import PolynomialFeatures, PolynomialRegression

# Classifiers
from .decision_tree import DecisionTreeClassifier
from .random_forest import RandomForestClassifier
from .xgboost import XGBoostClassifier
from .knn import KNeighborsClassifier
from .svm import SVC
from .naive_bayes import GaussianNB, MultinomialNB

# Neural Networks
from .neural_network import NeuralNetwork, create_gaze_estimator, create_classifier

# Feature Selection (Phase 6)
from .pca import PCA, IncrementalPCA, select_n_components
from .genetic_algorithm import GeneticAlgorithmFeatureSelector, SimpleGA
from .fishers_discriminant import FishersLinearDiscriminant, LinearDiscriminantAnalysis

__all__ = [
    # Regressors
    'LinearRegression',
    'RidgeRegression',
    'PolynomialFeatures',
    'PolynomialRegression',

    # Classifiers
    'DecisionTreeClassifier',
    'RandomForestClassifier',
    'XGBoostClassifier',
    'KNeighborsClassifier',
    'SVC',
    'GaussianNB',
    'MultinomialNB',

    # Neural Networks
    'NeuralNetwork',
    'create_gaze_estimator',
    'create_classifier',

    # Feature Selection (Phase 6)
    'PCA',
    'IncrementalPCA',
    'select_n_components',
    'GeneticAlgorithmFeatureSelector',
    'SimpleGA',
    'FishersLinearDiscriminant',
    'LinearDiscriminantAnalysis',
]
