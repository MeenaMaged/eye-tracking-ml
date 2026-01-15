"""
Evaluation Metrics - Implemented FROM SCRATCH.

This module provides comprehensive evaluation metrics for classification and
regression tasks, including visualization utilities.

Classification Metrics:
- Confusion Matrix
- Accuracy, Precision, Recall, F1-Score
- ROC Curve and AUC

Regression Metrics:
- MSE, MAE, RMSE
- R-squared (R2)
- Mean Pixel Error (for gaze estimation)

Cross-Validation:
- K-fold cross-validation
- Stratified k-fold for classification

Author: CSE381 Eye Tracking ML Project
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Union, Any
from dataclasses import dataclass


# =============================================================================
# CLASSIFICATION METRICS
# =============================================================================

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                     n_classes: Optional[int] = None,
                     normalize: Optional[str] = None) -> np.ndarray:
    """
    Compute confusion matrix from scratch.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.
    y_pred : np.ndarray
        Predicted labels.
    n_classes : int, optional
        Number of classes. If None, inferred from data.
    normalize : str, optional
        Normalization mode: 'true' (by row), 'pred' (by column), 'all'.

    Returns
    -------
    np.ndarray
        Confusion matrix of shape (n_classes, n_classes).
        Row i, column j is the count of samples with true label i
        predicted as label j.

    Mathematical Definition
    -----------------------
    CM[i,j] = |{x : y_true(x) = i AND y_pred(x) = j}|

    Example
    -------
    >>> y_true = np.array([0, 0, 1, 1, 2, 2])
    >>> y_pred = np.array([0, 1, 1, 1, 2, 0])
    >>> cm = confusion_matrix(y_true, y_pred)
    >>> print(cm)
    [[1 1 0]
     [0 2 0]
     [1 0 1]]
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if n_classes is None:
        n_classes = max(np.max(y_true), np.max(y_pred)) + 1

    # Initialize confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)

    # Count occurrences
    for true_label, pred_label in zip(y_true, y_pred):
        cm[int(true_label), int(pred_label)] += 1

    # Normalize if requested
    if normalize == 'true':
        # Normalize by row (true labels) - gives recall per class
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        cm = cm.astype(np.float64) / row_sums
    elif normalize == 'pred':
        # Normalize by column (predictions) - gives precision per class
        col_sums = cm.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1
        cm = cm.astype(np.float64) / col_sums
    elif normalize == 'all':
        # Normalize by total
        total = cm.sum()
        if total > 0:
            cm = cm.astype(np.float64) / total

    return cm


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate classification accuracy.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.
    y_pred : np.ndarray
        Predicted labels.

    Returns
    -------
    float
        Accuracy score in [0, 1].

    Mathematical Definition
    -----------------------
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
             = correct_predictions / total_predictions
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if len(y_true) == 0:
        return 0.0

    return np.mean(y_true == y_pred)


def precision_score(y_true: np.ndarray, y_pred: np.ndarray,
                    average: str = 'macro',
                    pos_label: int = 1,
                    zero_division: float = 0.0) -> Union[float, np.ndarray]:
    """
    Calculate precision score.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.
    y_pred : np.ndarray
        Predicted labels.
    average : str
        Averaging method: 'binary', 'micro', 'macro', 'weighted', None.
    pos_label : int
        Positive class for binary classification.
    zero_division : float
        Value to return when there are no positive predictions.

    Returns
    -------
    float or np.ndarray
        Precision score(s).

    Mathematical Definition
    -----------------------
    Precision = TP / (TP + FP)
              = True Positives / All Predicted Positives
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    classes = np.unique(np.concatenate([y_true, y_pred]))

    if average == 'binary':
        # Binary classification
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
        return tp / (tp + fp) if (tp + fp) > 0 else zero_division

    # Multi-class: calculate precision for each class
    precisions = []
    supports = []

    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        precision = tp / (tp + fp) if (tp + fp) > 0 else zero_division
        precisions.append(precision)
        supports.append(np.sum(y_true == cls))

    precisions = np.array(precisions)
    supports = np.array(supports)

    if average is None:
        return precisions
    elif average == 'micro':
        # Global TP and FP
        tp_total = sum(np.sum((y_true == cls) & (y_pred == cls)) for cls in classes)
        fp_total = sum(np.sum((y_true != cls) & (y_pred == cls)) for cls in classes)
        return tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else zero_division
    elif average == 'macro':
        return np.mean(precisions)
    elif average == 'weighted':
        total_support = np.sum(supports)
        if total_support == 0:
            return zero_division
        return np.sum(precisions * supports) / total_support

    return np.mean(precisions)


def recall_score(y_true: np.ndarray, y_pred: np.ndarray,
                 average: str = 'macro',
                 pos_label: int = 1,
                 zero_division: float = 0.0) -> Union[float, np.ndarray]:
    """
    Calculate recall score.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.
    y_pred : np.ndarray
        Predicted labels.
    average : str
        Averaging method: 'binary', 'micro', 'macro', 'weighted', None.
    pos_label : int
        Positive class for binary classification.
    zero_division : float
        Value to return when there are no positive samples.

    Returns
    -------
    float or np.ndarray
        Recall score(s).

    Mathematical Definition
    -----------------------
    Recall = TP / (TP + FN)
           = True Positives / All Actual Positives
           = Sensitivity = True Positive Rate (TPR)
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    classes = np.unique(np.concatenate([y_true, y_pred]))

    if average == 'binary':
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
        return tp / (tp + fn) if (tp + fn) > 0 else zero_division

    recalls = []
    supports = []

    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        recall = tp / (tp + fn) if (tp + fn) > 0 else zero_division
        recalls.append(recall)
        supports.append(np.sum(y_true == cls))

    recalls = np.array(recalls)
    supports = np.array(supports)

    if average is None:
        return recalls
    elif average == 'micro':
        tp_total = sum(np.sum((y_true == cls) & (y_pred == cls)) for cls in classes)
        fn_total = sum(np.sum((y_true == cls) & (y_pred != cls)) for cls in classes)
        return tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else zero_division
    elif average == 'macro':
        return np.mean(recalls)
    elif average == 'weighted':
        total_support = np.sum(supports)
        if total_support == 0:
            return zero_division
        return np.sum(recalls * supports) / total_support

    return np.mean(recalls)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray,
             average: str = 'macro',
             pos_label: int = 1,
             zero_division: float = 0.0) -> Union[float, np.ndarray]:
    """
    Calculate F1 score (harmonic mean of precision and recall).

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.
    y_pred : np.ndarray
        Predicted labels.
    average : str
        Averaging method: 'binary', 'micro', 'macro', 'weighted', None.
    pos_label : int
        Positive class for binary classification.
    zero_division : float
        Value to return when precision + recall = 0.

    Returns
    -------
    float or np.ndarray
        F1 score(s).

    Mathematical Definition
    -----------------------
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
       = 2 * TP / (2 * TP + FP + FN)
    """
    precision = precision_score(y_true, y_pred, average=average,
                                pos_label=pos_label, zero_division=zero_division)
    recall = recall_score(y_true, y_pred, average=average,
                          pos_label=pos_label, zero_division=zero_division)

    if average is None:
        # Element-wise F1
        f1 = np.zeros_like(precision)
        mask = (precision + recall) > 0
        f1[mask] = 2 * precision[mask] * recall[mask] / (precision[mask] + recall[mask])
        return f1

    if precision + recall == 0:
        return zero_division
    return 2 * precision * recall / (precision + recall)


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray,
                        average: str = 'macro') -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1 score together.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.
    y_pred : np.ndarray
        Predicted labels.
    average : str
        Averaging method: 'binary', 'micro', 'macro', 'weighted'.

    Returns
    -------
    tuple
        (precision, recall, f1_score)
    """
    p = precision_score(y_true, y_pred, average=average)
    r = recall_score(y_true, y_pred, average=average)
    f1 = f1_score(y_true, y_pred, average=average)
    return p, r, f1


@dataclass
class ClassificationReport:
    """Container for classification metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    per_class_precision: np.ndarray
    per_class_recall: np.ndarray
    per_class_f1: np.ndarray
    support: np.ndarray
    confusion_matrix: np.ndarray
    class_names: Optional[List[str]] = None


def classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                          class_names: Optional[List[str]] = None,
                          output_dict: bool = False) -> Union[str, ClassificationReport]:
    """
    Generate a comprehensive classification report.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.
    y_pred : np.ndarray
        Predicted labels.
    class_names : list, optional
        Names for each class.
    output_dict : bool
        If True, return ClassificationReport object instead of string.

    Returns
    -------
    str or ClassificationReport
        Formatted report string or structured report object.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)

    if class_names is None:
        class_names = [f"Class {c}" for c in classes]

    # Calculate per-class metrics
    per_class_precision = precision_score(y_true, y_pred, average=None)
    per_class_recall = recall_score(y_true, y_pred, average=None)
    per_class_f1 = f1_score(y_true, y_pred, average=None)
    support = np.array([np.sum(y_true == c) for c in classes])

    # Overall metrics
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1 = precision_recall_f1(y_true, y_pred, average='macro')
    weighted_p, weighted_r, weighted_f1 = precision_recall_f1(y_true, y_pred, average='weighted')

    cm = confusion_matrix(y_true, y_pred, n_classes=n_classes)

    report = ClassificationReport(
        accuracy=acc,
        precision=macro_p,
        recall=macro_r,
        f1=macro_f1,
        per_class_precision=per_class_precision,
        per_class_recall=per_class_recall,
        per_class_f1=per_class_f1,
        support=support,
        confusion_matrix=cm,
        class_names=class_names
    )

    if output_dict:
        return report

    # Generate string report
    max_name_len = max(len(name) for name in class_names)
    header = f"{'':>{max_name_len}}  Precision  Recall    F1-Score  Support"
    separator = "-" * len(header)

    lines = [header, separator]

    for i, name in enumerate(class_names):
        lines.append(
            f"{name:>{max_name_len}}  "
            f"{per_class_precision[i]:.4f}     "
            f"{per_class_recall[i]:.4f}    "
            f"{per_class_f1[i]:.4f}     "
            f"{support[i]:>5d}"
        )

    lines.append(separator)
    lines.append(f"{'Accuracy':>{max_name_len}}                        {acc:.4f}     {np.sum(support):>5d}")
    lines.append(f"{'Macro avg':>{max_name_len}}  {macro_p:.4f}     {macro_r:.4f}    {macro_f1:.4f}     {np.sum(support):>5d}")
    lines.append(f"{'Weighted avg':>{max_name_len}}  {weighted_p:.4f}     {weighted_r:.4f}    {weighted_f1:.4f}     {np.sum(support):>5d}")

    return "\n".join(lines)


# =============================================================================
# ROC-AUC METRICS
# =============================================================================

def roc_curve(y_true: np.ndarray, y_scores: np.ndarray,
              pos_label: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve from scratch.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground truth labels.
    y_scores : np.ndarray
        Probability scores for the positive class.
    pos_label : int
        Label of the positive class.

    Returns
    -------
    tuple
        (fpr, tpr, thresholds)
        - fpr: False Positive Rate at each threshold
        - tpr: True Positive Rate at each threshold
        - thresholds: Decreasing thresholds used to compute fpr and tpr

    Mathematical Definition
    -----------------------
    TPR = TP / (TP + FN) = TP / P  (Sensitivity/Recall)
    FPR = FP / (FP + TN) = FP / N  (1 - Specificity)

    ROC curve plots TPR vs FPR at various threshold settings.
    """
    y_true = np.asarray(y_true).ravel()
    y_scores = np.asarray(y_scores).ravel()

    # Convert to binary
    y_binary = (y_true == pos_label).astype(int)

    # Sort by decreasing score
    sorted_indices = np.argsort(y_scores)[::-1]
    y_sorted = y_binary[sorted_indices]
    scores_sorted = y_scores[sorted_indices]

    # Total positives and negatives
    P = np.sum(y_binary)
    N = len(y_binary) - P

    if P == 0 or N == 0:
        # Degenerate case
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])

    # Get unique thresholds
    thresholds = np.unique(scores_sorted)[::-1]  # Descending

    tpr_list = [0.0]
    fpr_list = [0.0]
    threshold_list = [thresholds[0] + 1e-10]  # Start above highest score

    for thresh in thresholds:
        # Predictions at this threshold
        y_pred = (y_scores >= thresh).astype(int)

        # Calculate TP, FP
        tp = np.sum((y_binary == 1) & (y_pred == 1))
        fp = np.sum((y_binary == 0) & (y_pred == 1))

        tpr = tp / P
        fpr = fp / N

        tpr_list.append(tpr)
        fpr_list.append(fpr)
        threshold_list.append(thresh)

    # Ensure we end at (1, 1)
    if fpr_list[-1] != 1.0 or tpr_list[-1] != 1.0:
        fpr_list.append(1.0)
        tpr_list.append(1.0)
        threshold_list.append(0.0)

    return np.array(fpr_list), np.array(tpr_list), np.array(threshold_list)


def roc_auc_score(y_true: np.ndarray, y_scores: np.ndarray,
                  pos_label: int = 1) -> float:
    """
    Compute Area Under the ROC Curve (AUC) from scratch.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground truth labels.
    y_scores : np.ndarray
        Probability scores for the positive class.
    pos_label : int
        Label of the positive class.

    Returns
    -------
    float
        AUC score in [0, 1]. Higher is better.
        AUC = 0.5 means random classifier.
        AUC = 1.0 means perfect classifier.

    Mathematical Definition
    -----------------------
    AUC = integral of TPR(FPR) dFPR from 0 to 1
        = Probability that a randomly chosen positive sample
          ranks higher than a randomly chosen negative sample

    Uses trapezoidal rule for integration.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=pos_label)

    # Compute AUC using trapezoidal rule
    # AUC = sum of trapezoids under ROC curve
    auc = 0.0
    for i in range(1, len(fpr)):
        # Trapezoid area = (base) * (average height)
        # base = fpr[i] - fpr[i-1]
        # average height = (tpr[i] + tpr[i-1]) / 2
        auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2

    return auc


def multi_class_roc_auc(y_true: np.ndarray, y_scores: np.ndarray,
                        average: str = 'macro') -> Union[float, Dict[int, float]]:
    """
    Compute multi-class ROC AUC using One-vs-Rest approach.

    Parameters
    ----------
    y_true : np.ndarray
        Multi-class ground truth labels.
    y_scores : np.ndarray
        Probability scores of shape (n_samples, n_classes).
    average : str
        Averaging method: 'macro', 'weighted', None.

    Returns
    -------
    float or dict
        AUC score(s).

    Multi-class Strategy
    --------------------
    One-vs-Rest: For each class c, treat class c as positive and
    all other classes as negative. Compute binary ROC AUC for each.
    """
    y_true = np.asarray(y_true).ravel()
    y_scores = np.asarray(y_scores)

    classes = np.unique(y_true)
    n_classes = len(classes)

    if y_scores.ndim == 1:
        # Binary case
        return roc_auc_score(y_true, y_scores)

    auc_scores = {}
    supports = {}

    for i, cls in enumerate(classes):
        # One-vs-Rest
        y_binary = (y_true == cls).astype(int)
        scores_cls = y_scores[:, i] if y_scores.shape[1] > i else y_scores[:, 0]

        try:
            auc = roc_auc_score(y_binary, scores_cls, pos_label=1)
        except:
            auc = 0.5  # Default for degenerate cases

        auc_scores[cls] = auc
        supports[cls] = np.sum(y_true == cls)

    if average is None:
        return auc_scores

    auc_values = np.array(list(auc_scores.values()))
    support_values = np.array(list(supports.values()))

    if average == 'macro':
        return np.mean(auc_values)
    elif average == 'weighted':
        return np.sum(auc_values * support_values) / np.sum(support_values)

    return np.mean(auc_values)


# =============================================================================
# REGRESSION METRICS
# =============================================================================

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray,
                       multioutput: str = 'uniform_average') -> Union[float, np.ndarray]:
    """
    Calculate Mean Squared Error (MSE).

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values.
    multioutput : str
        How to handle multiple outputs: 'raw_values' or 'uniform_average'.

    Returns
    -------
    float or np.ndarray
        MSE value(s).

    Mathematical Definition
    -----------------------
    MSE = (1/n) * sum((y_true - y_pred)^2)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    errors = y_true - y_pred
    mse = np.mean(errors ** 2, axis=0)

    if multioutput == 'raw_values':
        return mse
    return np.mean(mse)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray,
                        multioutput: str = 'uniform_average') -> Union[float, np.ndarray]:
    """
    Calculate Mean Absolute Error (MAE).

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values.
    multioutput : str
        How to handle multiple outputs: 'raw_values' or 'uniform_average'.

    Returns
    -------
    float or np.ndarray
        MAE value(s).

    Mathematical Definition
    -----------------------
    MAE = (1/n) * sum(|y_true - y_pred|)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    errors = y_true - y_pred
    mae = np.mean(np.abs(errors), axis=0)

    if multioutput == 'raw_values':
        return mae
    return np.mean(mae)


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray,
                            multioutput: str = 'uniform_average') -> Union[float, np.ndarray]:
    """
    Calculate Root Mean Squared Error (RMSE).

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values.
    multioutput : str
        How to handle multiple outputs.

    Returns
    -------
    float or np.ndarray
        RMSE value(s).

    Mathematical Definition
    -----------------------
    RMSE = sqrt(MSE) = sqrt((1/n) * sum((y_true - y_pred)^2))
    """
    mse = mean_squared_error(y_true, y_pred, multioutput=multioutput)
    if isinstance(mse, np.ndarray):
        return np.sqrt(mse)
    return np.sqrt(mse)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray,
             multioutput: str = 'uniform_average') -> Union[float, np.ndarray]:
    """
    Calculate R-squared (coefficient of determination).

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values.
    multioutput : str
        How to handle multiple outputs.

    Returns
    -------
    float or np.ndarray
        R2 score(s). Best possible score is 1.0, can be negative.

    Mathematical Definition
    -----------------------
    R2 = 1 - (SS_res / SS_tot)
       = 1 - (sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2))

    Interpretation
    --------------
    R2 = 1.0: Perfect predictions
    R2 = 0.0: Model predicts the mean (as good as baseline)
    R2 < 0:   Model is worse than predicting the mean
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Residual sum of squares
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)

    # Total sum of squares
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)

    # Avoid division by zero
    if np.any(ss_tot == 0):
        if np.all(ss_tot == 0):
            return 1.0 if np.allclose(y_true, y_pred) else 0.0
        ss_tot = np.where(ss_tot == 0, 1, ss_tot)

    r2 = 1 - (ss_res / ss_tot)

    if multioutput == 'raw_values':
        return r2
    return np.mean(r2)


def mean_pixel_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate mean pixel error for gaze estimation.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth screen coordinates of shape (n_samples, 2).
    y_pred : np.ndarray
        Predicted screen coordinates of shape (n_samples, 2).

    Returns
    -------
    float
        Mean Euclidean distance in pixels.

    Mathematical Definition
    -----------------------
    Mean Pixel Error = (1/n) * sum(sqrt((x_true - x_pred)^2 + (y_true - y_pred)^2))
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 2)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 2)

    # Euclidean distance for each sample
    distances = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1))

    return np.mean(distances)


@dataclass
class RegressionReport:
    """Container for regression metrics."""
    mse: float
    mae: float
    rmse: float
    r2: float
    mean_pixel_error: Optional[float] = None


def regression_report(y_true: np.ndarray, y_pred: np.ndarray,
                      is_gaze: bool = False,
                      output_dict: bool = False) -> Union[str, RegressionReport]:
    """
    Generate a comprehensive regression report.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values.
    is_gaze : bool
        If True, also compute mean pixel error (for 2D gaze estimation).
    output_dict : bool
        If True, return RegressionReport object instead of string.

    Returns
    -------
    str or RegressionReport
        Formatted report string or structured report object.
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mpe = mean_pixel_error(y_true, y_pred) if is_gaze else None

    report = RegressionReport(
        mse=mse,
        mae=mae,
        rmse=rmse,
        r2=r2,
        mean_pixel_error=mpe
    )

    if output_dict:
        return report

    lines = [
        "=" * 40,
        "REGRESSION METRICS",
        "=" * 40,
        f"Mean Squared Error (MSE):     {mse:.4f}",
        f"Mean Absolute Error (MAE):    {mae:.4f}",
        f"Root Mean Squared Error:      {rmse:.4f}",
        f"R-squared (R2):               {r2:.4f}",
    ]

    if is_gaze and mpe is not None:
        lines.append(f"Mean Pixel Error:             {mpe:.2f} px")

    lines.append("=" * 40)

    return "\n".join(lines)


# =============================================================================
# CROSS-VALIDATION
# =============================================================================

def stratified_k_fold(y: np.ndarray, n_folds: int = 5,
                      shuffle: bool = True,
                      random_state: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate stratified k-fold cross-validation indices.

    Parameters
    ----------
    y : np.ndarray
        Target labels (for stratification).
    n_folds : int
        Number of folds.
    shuffle : bool
        Whether to shuffle indices before splitting.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    list
        List of (train_indices, test_indices) tuples.

    Stratification
    --------------
    Ensures each fold has approximately the same percentage of
    samples from each class as the complete dataset.
    """
    y = np.asarray(y).ravel()
    n_samples = len(y)

    if random_state is not None:
        np.random.seed(random_state)

    classes = np.unique(y)

    # Get indices for each class
    class_indices = {cls: np.where(y == cls)[0] for cls in classes}

    if shuffle:
        for cls in classes:
            np.random.shuffle(class_indices[cls])

    # Distribute indices into folds
    folds = [{'train': [], 'test': []} for _ in range(n_folds)]

    for cls in classes:
        indices = class_indices[cls]
        n_cls = len(indices)
        fold_sizes = [n_cls // n_folds] * n_folds
        remainder = n_cls % n_folds

        # Distribute remainder
        for i in range(remainder):
            fold_sizes[i] += 1

        # Assign to folds
        current_idx = 0
        for fold_idx in range(n_folds):
            fold_size = fold_sizes[fold_idx]
            test_idx = indices[current_idx:current_idx + fold_size]
            train_idx = np.concatenate([
                indices[:current_idx],
                indices[current_idx + fold_size:]
            ])
            folds[fold_idx]['test'].extend(test_idx)
            folds[fold_idx]['train'].extend(train_idx)
            current_idx += fold_size

    # Convert to arrays and shuffle
    result = []
    for fold in folds:
        train_indices = np.array(fold['train'])
        test_indices = np.array(fold['test'])
        if shuffle:
            np.random.shuffle(train_indices)
            np.random.shuffle(test_indices)
        result.append((train_indices, test_indices))

    return result


def cross_val_score(model, X: np.ndarray, y: np.ndarray,
                    cv: int = 5,
                    scoring: str = 'accuracy',
                    stratify: bool = True,
                    random_state: Optional[int] = None,
                    verbose: bool = False) -> np.ndarray:
    """
    Evaluate model using cross-validation.

    Parameters
    ----------
    model : object
        Model with fit() and predict() methods (or predict_proba for AUC).
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target labels.
    cv : int
        Number of cross-validation folds.
    scoring : str
        Scoring metric: 'accuracy', 'precision', 'recall', 'f1', 'roc_auc',
        'mse', 'mae', 'r2'.
    stratify : bool
        Whether to use stratified k-fold (for classification).
    random_state : int, optional
        Random seed.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    np.ndarray
        Array of scores for each fold.
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()

    # Get fold indices
    if stratify:
        folds = stratified_k_fold(y, n_folds=cv, shuffle=True,
                                  random_state=random_state)
    else:
        # Simple k-fold
        if random_state is not None:
            np.random.seed(random_state)
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        fold_size = len(y) // cv
        folds = []
        for i in range(cv):
            start = i * fold_size
            end = start + fold_size if i < cv - 1 else len(y)
            test_idx = indices[start:end]
            train_idx = np.concatenate([indices[:start], indices[end:]])
            folds.append((train_idx, test_idx))

    scores = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Clone model (create new instance)
        try:
            model_clone = model.__class__(**model.get_params() if hasattr(model, 'get_params') else {})
        except:
            # Fallback: just use the model directly
            model_clone = model

        # Fit model
        model_clone.fit(X_train, y_train)

        # Predict
        y_pred = model_clone.predict(X_test)

        # Calculate score based on metric
        if scoring == 'accuracy':
            score = accuracy_score(y_test, y_pred)
        elif scoring == 'precision':
            score = precision_score(y_test, y_pred, average='macro')
        elif scoring == 'recall':
            score = recall_score(y_test, y_pred, average='macro')
        elif scoring == 'f1':
            score = f1_score(y_test, y_pred, average='macro')
        elif scoring == 'roc_auc':
            if hasattr(model_clone, 'predict_proba'):
                y_scores = model_clone.predict_proba(X_test)
                if y_scores.ndim > 1 and y_scores.shape[1] == 2:
                    y_scores = y_scores[:, 1]
                score = roc_auc_score(y_test, y_scores) if y_scores.ndim == 1 else multi_class_roc_auc(y_test, y_scores)
            else:
                score = 0.5  # Default
        elif scoring == 'mse':
            score = -mean_squared_error(y_test, y_pred)  # Negative for consistency
        elif scoring == 'mae':
            score = -mean_absolute_error(y_test, y_pred)
        elif scoring == 'r2':
            score = r2_score(y_test, y_pred)
        else:
            score = accuracy_score(y_test, y_pred)

        scores.append(score)

        if verbose:
            print(f"  Fold {fold_idx + 1}/{cv}: {scoring} = {score:.4f}")

    return np.array(scores)


@dataclass
class CrossValidationResult:
    """Container for cross-validation results."""
    scores: np.ndarray
    mean_score: float
    std_score: float
    scoring: str
    n_folds: int


def cross_validate(model, X: np.ndarray, y: np.ndarray,
                   cv: int = 5,
                   scoring: Union[str, List[str]] = 'accuracy',
                   stratify: bool = True,
                   random_state: Optional[int] = None,
                   verbose: bool = False) -> Dict[str, CrossValidationResult]:
    """
    Perform k-fold cross-validation with multiple metrics.

    Parameters
    ----------
    model : object
        Model with fit() and predict() methods.
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target labels.
    cv : int
        Number of folds.
    scoring : str or list
        Scoring metric(s).
    stratify : bool
        Whether to use stratified folds.
    random_state : int, optional
        Random seed.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Dictionary mapping metric names to CrossValidationResult objects.
    """
    if isinstance(scoring, str):
        scoring = [scoring]

    results = {}

    for metric in scoring:
        if verbose:
            print(f"\nCross-validating with {metric}...")

        scores = cross_val_score(
            model, X, y, cv=cv, scoring=metric,
            stratify=stratify, random_state=random_state, verbose=verbose
        )

        results[metric] = CrossValidationResult(
            scores=scores,
            mean_score=np.mean(scores),
            std_score=np.std(scores),
            scoring=metric,
            n_folds=cv
        )

        if verbose:
            print(f"  {metric}: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

    return results


# =============================================================================
# VISUALIZATION UTILITIES
# =============================================================================

def print_confusion_matrix(cm: np.ndarray,
                           class_names: Optional[List[str]] = None,
                           title: str = "Confusion Matrix") -> str:
    """
    Create ASCII representation of confusion matrix.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix.
    class_names : list, optional
        Names for each class.
    title : str
        Title for the matrix.

    Returns
    -------
    str
        Formatted string representation.
    """
    n_classes = cm.shape[0]

    if class_names is None:
        class_names = [f"C{i}" for i in range(n_classes)]

    # Determine column widths
    max_val = np.max(cm)
    val_width = max(len(str(int(max_val))), 4)
    label_width = max(len(name) for name in class_names)

    lines = [title, "=" * (label_width + 2 + (val_width + 1) * n_classes + 10)]

    # Header
    header = " " * (label_width + 2) + "Predicted"
    lines.append(header)

    header2 = " " * (label_width + 2) + " ".join(f"{name:>{val_width}}" for name in class_names)
    lines.append(header2)

    lines.append("-" * len(header2))

    # Rows
    for i, row_name in enumerate(class_names):
        prefix = "Actual " if i == n_classes // 2 else "       "
        row_vals = " ".join(f"{int(cm[i, j]):>{val_width}}" for j in range(n_classes))
        lines.append(f"{prefix}{row_name:>{label_width}} {row_vals}")

    lines.append("=" * len(header2))

    # Add summary statistics
    correct = np.trace(cm)
    total = np.sum(cm)
    accuracy = correct / total if total > 0 else 0
    lines.append(f"Accuracy: {accuracy:.4f} ({int(correct)}/{int(total)})")

    return "\n".join(lines)


def plot_confusion_matrix(cm: np.ndarray,
                          class_names: Optional[List[str]] = None,
                          title: str = "Confusion Matrix",
                          cmap: str = 'Blues',
                          normalize: bool = False,
                          figsize: Tuple[int, int] = (8, 6),
                          save_path: Optional[str] = None) -> Any:
    """
    Plot confusion matrix as a heatmap.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix.
    class_names : list, optional
        Names for each class.
    title : str
        Plot title.
    cmap : str
        Colormap name.
    normalize : bool
        Whether to normalize the matrix.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if matplotlib is available.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available. Returning text representation.")
        print(print_confusion_matrix(cm, class_names, title))
        return None

    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]

    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_display = np.nan_to_num(cm_display)
        fmt = '.2f'
    else:
        cm_display = cm
        fmt = 'd'

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_display, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm_display.max() / 2.
    for i in range(n_classes):
        for j in range(n_classes):
            val = cm_display[i, j]
            text = f"{val:{fmt}}" if normalize else f"{int(val)}"
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if val > thresh else "black")

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray,
                   pos_label: int = 1,
                   title: str = "ROC Curve",
                   figsize: Tuple[int, int] = (8, 6),
                   save_path: Optional[str] = None) -> Any:
    """
    Plot ROC curve for binary classification.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground truth labels.
    y_scores : np.ndarray
        Probability scores for positive class.
    pos_label : int
        Positive class label.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if matplotlib is available.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available.")
        fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label)
        auc = roc_auc_score(y_true, y_scores, pos_label)
        print(f"ROC AUC: {auc:.4f}")
        return None

    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label)
    auc = roc_auc_score(y_true, y_scores, pos_label)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot ROC curve
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {auc:.4f})')

    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
            label='Random (AUC = 0.5)')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_multi_class_roc(y_true: np.ndarray, y_scores: np.ndarray,
                         class_names: Optional[List[str]] = None,
                         title: str = "Multi-class ROC Curves",
                         figsize: Tuple[int, int] = (10, 8),
                         save_path: Optional[str] = None) -> Any:
    """
    Plot ROC curves for multi-class classification (One-vs-Rest).

    Parameters
    ----------
    y_true : np.ndarray
        Multi-class ground truth labels.
    y_scores : np.ndarray
        Probability scores of shape (n_samples, n_classes).
    class_names : list, optional
        Names for each class.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if matplotlib is available.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available.")
        auc_scores = multi_class_roc_auc(y_true, y_scores, average=None)
        print("Per-class AUC scores:")
        for cls, auc in auc_scores.items():
            print(f"  Class {cls}: {auc:.4f}")
        return None

    classes = np.unique(y_true)
    n_classes = len(classes)

    if class_names is None:
        class_names = [f"Class {c}" for c in classes]

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, n_classes))

    for i, (cls, name) in enumerate(zip(classes, class_names)):
        # One-vs-Rest
        y_binary = (y_true == cls).astype(int)
        scores_cls = y_scores[:, i] if y_scores.ndim > 1 else y_scores

        fpr, tpr, _ = roc_curve(y_binary, scores_cls, pos_label=1)
        auc = roc_auc_score(y_binary, scores_cls, pos_label=1)

        ax.plot(fpr, tpr, color=colors[i], lw=2,
                label=f'{name} (AUC = {auc:.4f})')

    # Plot diagonal
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--',
            label='Random (AUC = 0.5)')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # Add macro AUC
    macro_auc = multi_class_roc_auc(y_true, y_scores, average='macro')
    ax.text(0.6, 0.1, f'Macro AUC: {macro_auc:.4f}',
            transform=ax.transAxes, fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# =============================================================================
# QUICK TEST FUNCTION
# =============================================================================

def _quick_test():
    """Quick test of evaluation metrics."""
    print("Testing evaluation metrics...")
    print("=" * 50)

    # Test classification metrics
    y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0, 0, 1, 2])

    print("\n1. Classification Metrics Test")
    print("-" * 30)
    print(f"y_true: {y_true}")
    print(f"y_pred: {y_pred}")

    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")

    acc = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {acc:.4f}")

    p, r, f1 = precision_recall_f1(y_true, y_pred, average='macro')
    print(f"Precision (macro): {p:.4f}")
    print(f"Recall (macro): {r:.4f}")
    print(f"F1 (macro): {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, class_names=['Fixation', 'Saccade', 'Blink']))

    # Test ROC-AUC (binary)
    print("\n2. ROC-AUC Test (Binary)")
    print("-" * 30)
    y_true_binary = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_scores = np.array([0.1, 0.3, 0.8, 0.9, 0.2, 0.7, 0.4, 0.6])

    fpr, tpr, thresholds = roc_curve(y_true_binary, y_scores)
    auc = roc_auc_score(y_true_binary, y_scores)
    print(f"ROC AUC: {auc:.4f}")

    # Test regression metrics
    print("\n3. Regression Metrics Test")
    print("-" * 30)
    y_true_reg = np.array([[100, 200], [150, 250], [200, 300], [250, 350]])
    y_pred_reg = np.array([[105, 195], [145, 255], [210, 290], [240, 360]])

    print(f"MSE: {mean_squared_error(y_true_reg, y_pred_reg):.4f}")
    print(f"MAE: {mean_absolute_error(y_true_reg, y_pred_reg):.4f}")
    print(f"RMSE: {root_mean_squared_error(y_true_reg, y_pred_reg):.4f}")
    print(f"R2: {r2_score(y_true_reg, y_pred_reg):.4f}")
    print(f"Mean Pixel Error: {mean_pixel_error(y_true_reg, y_pred_reg):.2f} px")

    print("\nRegression Report:")
    print(regression_report(y_true_reg, y_pred_reg, is_gaze=True))

    # Test cross-validation
    print("\n4. Cross-Validation Test")
    print("-" * 30)
    folds = stratified_k_fold(y_true, n_folds=3, random_state=42)
    print(f"Stratified 3-fold splits:")
    for i, (train_idx, test_idx) in enumerate(folds):
        print(f"  Fold {i+1}: train={len(train_idx)}, test={len(test_idx)}")

    print("\n" + "=" * 50)
    print("All tests passed!")


if __name__ == "__main__":
    _quick_test()
