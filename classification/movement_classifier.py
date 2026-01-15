"""
Eye Movement Classification - 3-Class Problem.

Classifies eye movements into:
- Class 0: Fixation (stable gaze, low velocity)
- Class 1: Saccade (rapid movement, high velocity)
- Class 2: Blink (eye closed, low EAR)

Uses the same classifiers from blink detection but with multi-class support
and per-class evaluation metrics.

All models are implemented from scratch.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_from_scratch.decision_tree import DecisionTreeClassifier
from ml_from_scratch.random_forest import RandomForestClassifier
from ml_from_scratch.xgboost import XGBoostClassifier
from ml_from_scratch.knn import KNeighborsClassifier
from ml_from_scratch.svm import SVC
from ml_from_scratch.naive_bayes import GaussianNB
from ml_from_scratch.neural_network import NeuralNetwork


# Movement class labels
FIXATION = 0
SACCADE = 1
BLINK = 2

CLASS_NAMES = {
    FIXATION: 'Fixation',
    SACCADE: 'Saccade',
    BLINK: 'Blink'
}

# Feature names for the 14-dimensional movement vector
MOVEMENT_FEATURE_NAMES = [
    'current_velocity',
    'mean_velocity',
    'max_velocity',
    'current_acceleration',
    'mean_abs_acceleration',
    'dispersion',
    'rms_deviation',
    'direction_consistency',
    'mean_ear',
    'min_ear',
    'std_ear',
    'velocity_std',
    'x_range',
    'y_range'
]


@dataclass
class PerClassMetrics:
    """Metrics for a single class."""
    class_id: int
    class_name: str
    precision: float
    recall: float
    f1_score: float
    support: int  # Number of true samples


@dataclass
class MovementClassifierResult:
    """Results from training and evaluating a movement classifier."""
    name: str
    train_accuracy: float
    test_accuracy: float
    overall_precision: float
    overall_recall: float
    overall_f1: float
    per_class_metrics: List[PerClassMetrics]
    confusion_matrix: np.ndarray
    train_time: float
    model: object
    feature_importances: Optional[np.ndarray] = None


def confusion_matrix_multiclass(y_true: np.ndarray, y_pred: np.ndarray,
                                 n_classes: int = 3) -> np.ndarray:
    """
    Compute confusion matrix for multi-class classification.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        n_classes: Number of classes (default 3 for movement)

    Returns:
        Confusion matrix of shape (n_classes, n_classes)
        cm[i, j] = count of samples with true label i and predicted label j
    """
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1

    return cm


def per_class_precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray,
                                   n_classes: int = 3) -> List[PerClassMetrics]:
    """
    Calculate per-class precision, recall, and F1 score.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        n_classes: Number of classes

    Returns:
        List of PerClassMetrics for each class
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = []
    for cls in range(n_classes):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        support = np.sum(y_true == cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics.append(PerClassMetrics(
            class_id=cls,
            class_name=CLASS_NAMES.get(cls, f'Class {cls}'),
            precision=precision,
            recall=recall,
            f1_score=f1,
            support=int(support)
        ))

    return metrics


def macro_average_metrics(per_class: List[PerClassMetrics]) -> Tuple[float, float, float]:
    """Calculate macro-averaged precision, recall, F1 from per-class metrics."""
    if not per_class:
        return 0.0, 0.0, 0.0

    precision = np.mean([m.precision for m in per_class])
    recall = np.mean([m.recall for m in per_class])
    f1 = np.mean([m.f1_score for m in per_class])

    return precision, recall, f1


def weighted_average_metrics(per_class: List[PerClassMetrics]) -> Tuple[float, float, float]:
    """Calculate weighted-averaged precision, recall, F1 from per-class metrics."""
    if not per_class:
        return 0.0, 0.0, 0.0

    total_support = sum(m.support for m in per_class)
    if total_support == 0:
        return 0.0, 0.0, 0.0

    precision = sum(m.precision * m.support for m in per_class) / total_support
    recall = sum(m.recall * m.support for m in per_class) / total_support
    f1 = sum(m.f1_score * m.support for m in per_class) / total_support

    return precision, recall, f1


class MovementClassifierComparison:
    """
    Compare multiple classifiers on eye movement classification (3-class).

    Classes:
        0: Fixation - Stable gaze
        1: Saccade - Rapid eye movement
        2: Blink - Eye closure

    Usage:
        comparison = MovementClassifierComparison()
        comparison.load_data(X_train, y_train, X_test, y_test)
        results = comparison.run_comparison()
        comparison.print_results()
        comparison.print_per_class_results()
        comparison.print_confusion_matrices()
    """

    def __init__(self, normalize: bool = True):
        """
        Initialize comparison.

        Args:
            normalize: If True, normalize features using training statistics
        """
        self.normalize = normalize
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.results: List[MovementClassifierResult] = []
        self._norm_params: Dict = {}
        self.n_classes: int = 3
        self.feature_names: List[str] = MOVEMENT_FEATURE_NAMES

    def load_data(self, X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray,
                  feature_names: List[str] = None):
        """
        Load training and test data.

        Args:
            X_train: Training features (n_train, n_features)
            y_train: Training labels (n_train,) with values in {0, 1, 2}
            X_test: Test features (n_test, n_features)
            y_test: Test labels (n_test,)
            feature_names: Optional list of feature names
        """
        X_train = np.array(X_train, dtype=np.float64)
        y_train = np.array(y_train, dtype=int)
        X_test = np.array(X_test, dtype=np.float64)
        y_test = np.array(y_test, dtype=int)

        if self.normalize:
            mean = X_train.mean(axis=0)
            std = X_train.std(axis=0)
            std[std == 0] = 1.0

            self.X_train = (X_train - mean) / std
            self.X_test = (X_test - mean) / std
            self._norm_params = {'mean': mean, 'std': std}
        else:
            self.X_train = X_train
            self.X_test = X_test

        self.y_train = y_train
        self.y_test = y_test
        self.n_classes = len(np.unique(np.concatenate([y_train, y_test])))

        if feature_names:
            self.feature_names = feature_names

        # Print data statistics
        print(f"\nData loaded for Eye Movement Classification:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Classes: {self.n_classes}")

        # Class distribution
        print("\n  Class distribution (Training):")
        for cls in range(self.n_classes):
            count = np.sum(y_train == cls)
            pct = 100 * count / len(y_train)
            print(f"    {CLASS_NAMES.get(cls, f'Class {cls}')}: {count} ({pct:.1f}%)")

    def _evaluate(self, model, name: str, train_time: float,
                  feature_importances: np.ndarray = None) -> MovementClassifierResult:
        """Evaluate a trained model with per-class metrics."""
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)

        train_acc = np.mean(y_train_pred == self.y_train)
        test_acc = np.mean(y_test_pred == self.y_test)

        # Per-class metrics
        per_class = per_class_precision_recall_f1(self.y_test, y_test_pred, self.n_classes)

        # Overall metrics (macro average)
        overall_precision, overall_recall, overall_f1 = macro_average_metrics(per_class)

        # Confusion matrix
        cm = confusion_matrix_multiclass(self.y_test, y_test_pred, self.n_classes)

        return MovementClassifierResult(
            name=name,
            train_accuracy=train_acc,
            test_accuracy=test_acc,
            overall_precision=overall_precision,
            overall_recall=overall_recall,
            overall_f1=overall_f1,
            per_class_metrics=per_class,
            confusion_matrix=cm,
            train_time=train_time,
            model=model,
            feature_importances=feature_importances
        )

    def run_comparison(self, include_svm: bool = True,
                       include_nn: bool = True,
                       nn_epochs: int = 100,
                       verbose: bool = True) -> List[MovementClassifierResult]:
        """
        Run comparison of all classifiers on movement data.

        Args:
            include_svm: Include SVM (can be slow on large datasets)
            include_nn: Include Neural Network
            nn_epochs: Training epochs for neural network
            verbose: Print progress messages

        Returns:
            List of MovementClassifierResult
        """
        if self.X_train is None:
            raise ValueError("No data loaded. Call load_data() first.")

        self.results = []
        n_features = self.X_train.shape[1]

        # 1. Decision Tree
        if verbose:
            print("\nTraining Decision Tree...")
        start = time.time()
        model = DecisionTreeClassifier(max_depth=10, criterion='entropy', min_samples_split=5)
        model.fit(self.X_train, self.y_train)
        feature_imp = model.feature_importances_ if hasattr(model, 'feature_importances_') else None
        result = self._evaluate(model, "Decision Tree", time.time() - start, feature_imp)
        self.results.append(result)
        if verbose:
            print(f"  -> Test Accuracy: {result.test_accuracy:.4f}, F1: {result.overall_f1:.4f}")

        # 2. Random Forest
        if verbose:
            print("Training Random Forest...")
        start = time.time()
        model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5)
        model.fit(self.X_train, self.y_train)
        feature_imp = model.feature_importances_ if hasattr(model, 'feature_importances_') else None
        result = self._evaluate(model, "Random Forest", time.time() - start, feature_imp)
        self.results.append(result)
        if verbose:
            print(f"  -> Test Accuracy: {result.test_accuracy:.4f}, F1: {result.overall_f1:.4f}")

        # 3. XGBoost
        if verbose:
            print("Training XGBoost...")
        start = time.time()
        model = XGBoostClassifier(n_estimators=100, max_depth=4, learning_rate=0.1)
        model.fit(self.X_train, self.y_train)
        feature_imp = model.feature_importances_ if hasattr(model, 'feature_importances_') else None
        result = self._evaluate(model, "XGBoost", time.time() - start, feature_imp)
        self.results.append(result)
        if verbose:
            print(f"  -> Test Accuracy: {result.test_accuracy:.4f}, F1: {result.overall_f1:.4f}")

        # 4. KNN
        if verbose:
            print("Training KNN...")
        start = time.time()
        model = KNeighborsClassifier(n_neighbors=5, weights='distance')
        model.fit(self.X_train, self.y_train)
        result = self._evaluate(model, "KNN (k=5)", time.time() - start)
        self.results.append(result)
        if verbose:
            print(f"  -> Test Accuracy: {result.test_accuracy:.4f}, F1: {result.overall_f1:.4f}")

        # 5. SVM
        if include_svm:
            if verbose:
                print("Training SVM...")
            start = time.time()
            model = SVC(kernel='rbf', C=1.0, gamma='scale', max_iter=300)
            model.fit(self.X_train, self.y_train)
            result = self._evaluate(model, "SVM (RBF)", time.time() - start)
            self.results.append(result)
            if verbose:
                print(f"  -> Test Accuracy: {result.test_accuracy:.4f}, F1: {result.overall_f1:.4f}")

        # 6. Naive Bayes
        if verbose:
            print("Training Naive Bayes...")
        start = time.time()
        model = GaussianNB()
        model.fit(self.X_train, self.y_train)
        result = self._evaluate(model, "Naive Bayes", time.time() - start)
        self.results.append(result)
        if verbose:
            print(f"  -> Test Accuracy: {result.test_accuracy:.4f}, F1: {result.overall_f1:.4f}")

        # 7. Neural Network
        if include_nn:
            if verbose:
                print("Training Neural Network...")
            start = time.time()

            # Multi-class output with softmax
            layer_sizes = [n_features, 64, 32, self.n_classes]

            nn_model = NeuralNetwork(
                layer_sizes=layer_sizes,
                activation='relu',
                output_activation='softmax',
                learning_rate=0.01,
                l2_lambda=0.001
            )

            # One-hot encode labels
            y_train_onehot = np.zeros((len(self.y_train), self.n_classes))
            for i, label in enumerate(self.y_train):
                y_train_onehot[i, int(label)] = 1

            nn_model.fit(self.X_train, y_train_onehot, epochs=nn_epochs, batch_size=32, verbose=0)

            # Wrapper for consistent predict interface
            class NNWrapper:
                def __init__(self, nn):
                    self.nn = nn

                def predict(self, X):
                    return self.nn.predict_classes(X)

            wrapper = NNWrapper(nn_model)
            result = self._evaluate(wrapper, "ANN (64-32)", time.time() - start)
            result.model = nn_model  # Store actual model
            self.results.append(result)
            if verbose:
                print(f"  -> Test Accuracy: {result.test_accuracy:.4f}, F1: {result.overall_f1:.4f}")

        if verbose:
            print("\nComparison complete!")

        return self.results

    def print_results(self):
        """Print overall comparison results as a formatted table."""
        if not self.results:
            print("No results. Run run_comparison() first.")
            return

        print("\n" + "=" * 100)
        print("EYE MOVEMENT CLASSIFICATION COMPARISON")
        print("Classes: Fixation (0) | Saccade (1) | Blink (2)")
        print("=" * 100)
        print(f"{'Model':<16} {'Train Acc':<10} {'Test Acc':<10} "
              f"{'Precision':<10} {'Recall':<10} {'F1 (macro)':<10} {'Time (s)':<10}")
        print("-" * 100)

        for r in self.results:
            print(f"{r.name:<16} {r.train_accuracy:<10.4f} {r.test_accuracy:<10.4f} "
                  f"{r.overall_precision:<10.4f} {r.overall_recall:<10.4f} "
                  f"{r.overall_f1:<10.4f} {r.train_time:<10.3f}")

        print("-" * 100)

        best = max(self.results, key=lambda x: x.overall_f1)
        print(f"\nBest Model: {best.name} (Overall F1: {best.overall_f1:.4f})")

    def print_per_class_results(self):
        """Print per-class F1 scores for all models."""
        if not self.results:
            return

        print("\n" + "=" * 80)
        print("PER-CLASS F1 SCORES")
        print("=" * 80)

        # Header
        header = f"{'Model':<16}"
        for cls in range(self.n_classes):
            header += f" {CLASS_NAMES.get(cls, f'Class {cls}')+' F1':<12}"
        header += f" {'Overall':<10}"
        print(header)
        print("-" * 80)

        for r in self.results:
            row = f"{r.name:<16}"
            for m in r.per_class_metrics:
                row += f" {m.f1_score:<12.4f}"
            row += f" {r.overall_f1:<10.4f}"
            print(row)

        print("-" * 80)

        # Best per class
        print("\nBest model per class:")
        for cls in range(self.n_classes):
            best_model = max(self.results, key=lambda r: r.per_class_metrics[cls].f1_score)
            best_f1 = best_model.per_class_metrics[cls].f1_score
            print(f"  {CLASS_NAMES.get(cls, f'Class {cls}')}: {best_model.name} (F1: {best_f1:.4f})")

    def print_confusion_matrices(self):
        """Print confusion matrices for all models."""
        if not self.results:
            return

        for r in self.results:
            print(f"\n{'='*50}")
            print(f"{r.name} - Confusion Matrix")
            print(f"{'='*50}")

            # Header row
            print(f"{'Actual \\ Pred':<12}", end="")
            for cls in range(self.n_classes):
                print(f" {CLASS_NAMES.get(cls, f'Cls{cls}')[:8]:>8}", end="")
            print(f" {'Total':>8}")
            print("-" * 50)

            # Matrix rows
            for i in range(self.n_classes):
                row_name = CLASS_NAMES.get(i, f'Class {i}')[:12]
                print(f"{row_name:<12}", end="")
                for j in range(self.n_classes):
                    print(f" {r.confusion_matrix[i, j]:>8}", end="")
                row_total = np.sum(r.confusion_matrix[i, :])
                print(f" {row_total:>8}")

            # Column totals
            print("-" * 50)
            print(f"{'Predicted':<12}", end="")
            for j in range(self.n_classes):
                col_total = np.sum(r.confusion_matrix[:, j])
                print(f" {col_total:>8}", end="")
            total = np.sum(r.confusion_matrix)
            print(f" {total:>8}")

            # Per-class accuracy
            print(f"\nPer-class accuracy:")
            for i, m in enumerate(r.per_class_metrics):
                correct = r.confusion_matrix[i, i]
                total = np.sum(r.confusion_matrix[i, :])
                acc = correct / total if total > 0 else 0
                print(f"  {m.class_name}: {acc:.2%} ({correct}/{total})")

    def print_detailed_analysis(self):
        """Print detailed analysis including misclassification patterns."""
        if not self.results:
            return

        best = max(self.results, key=lambda x: x.overall_f1)

        print("\n" + "=" * 70)
        print("DETAILED ANALYSIS - Best Model:", best.name)
        print("=" * 70)

        cm = best.confusion_matrix
        total = np.sum(cm)

        # Misclassification analysis
        print("\nMisclassification Patterns:")
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                if i != j and cm[i, j] > 0:
                    pct = 100 * cm[i, j] / np.sum(cm[i, :]) if np.sum(cm[i, :]) > 0 else 0
                    print(f"  {CLASS_NAMES[i]} -> {CLASS_NAMES[j]}: "
                          f"{cm[i, j]} samples ({pct:.1f}% of {CLASS_NAMES[i]})")

        # Class-specific insights
        print("\nClass-specific Insights:")
        for i, m in enumerate(best.per_class_metrics):
            print(f"\n  {m.class_name}:")
            print(f"    Precision: {m.precision:.4f} (of predicted {m.class_name}, {m.precision*100:.1f}% were correct)")
            print(f"    Recall: {m.recall:.4f} (of actual {m.class_name}, {m.recall*100:.1f}% were detected)")
            print(f"    F1 Score: {m.f1_score:.4f}")
            print(f"    Support: {m.support} samples in test set")

    def get_best_model(self) -> MovementClassifierResult:
        """Get the best performing model (highest overall F1)."""
        if not self.results:
            raise ValueError("No results. Run run_comparison() first.")
        return max(self.results, key=lambda x: x.overall_f1)

    def get_best_per_class(self) -> Dict[int, MovementClassifierResult]:
        """Get the best model for each class."""
        if not self.results:
            raise ValueError("No results. Run run_comparison() first.")

        best_per_class = {}
        for cls in range(self.n_classes):
            best_per_class[cls] = max(
                self.results,
                key=lambda r: r.per_class_metrics[cls].f1_score
            )
        return best_per_class

    def to_dict(self) -> Dict:
        """Export results as dictionary for saving."""
        return {
            'models': [
                {
                    'name': r.name,
                    'train_accuracy': r.train_accuracy,
                    'test_accuracy': r.test_accuracy,
                    'overall_precision': r.overall_precision,
                    'overall_recall': r.overall_recall,
                    'overall_f1': r.overall_f1,
                    'train_time': r.train_time,
                    'per_class': [
                        {
                            'class': m.class_name,
                            'precision': m.precision,
                            'recall': m.recall,
                            'f1': m.f1_score,
                            'support': m.support
                        }
                        for m in r.per_class_metrics
                    ],
                    'confusion_matrix': r.confusion_matrix.tolist()
                }
                for r in self.results
            ]
        }


def compare_movement_classifiers(X: np.ndarray, y: np.ndarray,
                                  test_size: float = 0.2,
                                  random_state: int = 42,
                                  stratify: bool = True,
                                  verbose: bool = True) -> MovementClassifierComparison:
    """
    Convenience function to run full movement classifier comparison.

    Args:
        X: Feature matrix (n_samples, 14) - movement features
        y: Labels (n_samples,) with values in {0: Fixation, 1: Saccade, 2: Blink}
        test_size: Fraction for testing
        random_state: Random seed
        stratify: If True, maintain class proportions in split
        verbose: Print results

    Returns:
        MovementClassifierComparison with results
    """
    np.random.seed(random_state)
    X = np.array(X)
    y = np.array(y)
    n_samples = len(X)

    if stratify:
        # Stratified split
        train_idx, test_idx = [], []
        for cls in np.unique(y):
            cls_idx = np.where(y == cls)[0]
            np.random.shuffle(cls_idx)
            n_test = int(len(cls_idx) * test_size)
            test_idx.extend(cls_idx[:n_test])
            train_idx.extend(cls_idx[n_test:])
        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)
        np.random.shuffle(train_idx)
        np.random.shuffle(test_idx)
    else:
        indices = np.random.permutation(n_samples)
        n_test = int(n_samples * test_size)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    comparison = MovementClassifierComparison(normalize=True)
    comparison.load_data(X_train, y_train, X_test, y_test)
    comparison.run_comparison(include_svm=True, include_nn=True, nn_epochs=100, verbose=verbose)

    if verbose:
        comparison.print_results()
        comparison.print_per_class_results()

    return comparison


# =============================================================================
# Synthetic Data Generation for Testing
# =============================================================================

def generate_movement_data(n_samples: int = 900, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic eye movement classification data.

    The 14-dimensional feature vector represents:
    - Velocity features (0-2): current, mean, max velocity
    - Acceleration features (3-4): current, mean absolute
    - Spatial features (5-7): dispersion, RMS deviation, direction consistency
    - EAR features (8-10): mean, min, std
    - Additional (11-13): velocity std, x_range, y_range

    Class characteristics:
    - Fixation: Low velocity, low dispersion, normal EAR
    - Saccade: High velocity, high dispersion, normal EAR
    - Blink: Any velocity, low EAR

    Args:
        n_samples: Total number of samples (divided equally among classes)
        random_state: Random seed

    Returns:
        X: Feature matrix (n_samples, 14)
        y: Labels (n_samples,)
    """
    np.random.seed(random_state)

    n_per_class = n_samples // 3

    # Class 0: Fixation - low velocity, low dispersion, normal EAR
    X0 = np.random.randn(n_per_class, 14) * 0.3
    X0[:, 0:3] -= 0.8    # Low velocity features
    X0[:, 5:8] -= 0.5    # Low dispersion features
    X0[:, 8:11] += 0.3   # Normal EAR (eyes open)
    y0 = np.zeros(n_per_class, dtype=int)

    # Class 1: Saccade - high velocity, high dispersion, normal EAR
    X1 = np.random.randn(n_per_class, 14) * 0.3
    X1[:, 0:3] += 1.0    # High velocity features
    X1[:, 3:5] += 0.5    # Higher acceleration
    X1[:, 5:8] += 0.8    # High dispersion
    X1[:, 8:11] += 0.3   # Normal EAR (eyes open)
    y1 = np.ones(n_per_class, dtype=int)

    # Class 2: Blink - variable velocity, low EAR
    X2 = np.random.randn(n_per_class, 14) * 0.3
    X2[:, 8:11] -= 1.0   # Low EAR (eyes closed/closing)
    y2 = np.full(n_per_class, 2, dtype=int)

    X = np.vstack([X0, X1, X2])
    y = np.concatenate([y0, y1, y2])

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    return X, y


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Run demo of movement classification comparison."""
    print("=" * 70)
    print("EYE MOVEMENT CLASSIFICATION DEMO")
    print("3-Class Problem: Fixation | Saccade | Blink")
    print("=" * 70)

    # Generate synthetic data
    print("\nGenerating synthetic movement data...")
    X, y = generate_movement_data(n_samples=900, random_state=42)

    # Run comparison
    comparison = compare_movement_classifiers(X, y, test_size=0.2, random_state=42)

    # Print detailed results
    comparison.print_confusion_matrices()
    comparison.print_detailed_analysis()

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    demo()
