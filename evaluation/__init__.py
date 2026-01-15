"""
Evaluation module for Eye Tracking ML Project.

Provides comprehensive evaluation metrics and visualization utilities:
- Classification metrics: confusion matrix, accuracy, precision, recall, F1
- ROC-AUC: Binary and multi-class ROC curves and AUC scores
- Regression metrics: MSE, MAE, R2, mean pixel error
- Cross-validation: K-fold cross-validation with stratification
- Visualization: Confusion matrix heatmaps, ROC curves

All metrics are implemented from scratch using only NumPy.
"""

from .metrics import (
    # Classification metrics
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_f1,
    classification_report,

    # ROC-AUC
    roc_curve,
    roc_auc_score,
    multi_class_roc_auc,

    # Regression metrics
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    mean_pixel_error,
    regression_report,

    # Cross-validation
    cross_validate,
    cross_val_score,
    stratified_k_fold,

    # Visualization
    plot_confusion_matrix,
    plot_roc_curve,
    plot_multi_class_roc,
    print_confusion_matrix,
)

__all__ = [
    # Classification metrics
    'confusion_matrix',
    'accuracy_score',
    'precision_score',
    'recall_score',
    'f1_score',
    'precision_recall_f1',
    'classification_report',

    # ROC-AUC
    'roc_curve',
    'roc_auc_score',
    'multi_class_roc_auc',

    # Regression metrics
    'mean_squared_error',
    'mean_absolute_error',
    'root_mean_squared_error',
    'r2_score',
    'mean_pixel_error',
    'regression_report',

    # Cross-validation
    'cross_validate',
    'cross_val_score',
    'stratified_k_fold',

    # Visualization
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_multi_class_roc',
    'print_confusion_matrix',
]
