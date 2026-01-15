"""
Classifier Comparison for Blink Detection and Movement Classification.

Compares multiple classification models:
- Decision Tree
- Random Forest
- XGBoost
- KNN
- SVM
- Naive Bayes
- Neural Network (ANN)

All models are implemented from scratch.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
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


@dataclass
class ClassifierResult:
    """Results from training and evaluating a classifier."""
    name: str
    train_accuracy: float
    test_accuracy: float
    precision: float
    recall: float
    f1_score: float
    train_time: float
    confusion_matrix: np.ndarray
    model: object


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                     n_classes: int = None) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        n_classes: Number of classes (auto-detected if None)

    Returns:
        Confusion matrix of shape (n_classes, n_classes)
        cm[i, j] = count of samples with true label i and predicted label j
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if n_classes is None:
        n_classes = max(len(np.unique(y_true)), len(np.unique(y_pred)))

    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    return cm


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray,
                        average: str = 'macro') -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1 score.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: 'macro' (unweighted mean), 'weighted', or 'binary'

    Returns:
        Tuple of (precision, recall, f1)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)

    if n_classes == 2 and average == 'binary':
        # Binary classification - report for positive class
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return precision, recall, f1

    # Per-class metrics
    precisions, recalls, f1s, supports = [], [], [], []

    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(np.sum(y_true == cls))

    if average == 'macro':
        return np.mean(precisions), np.mean(recalls), np.mean(f1s)
    elif average == 'weighted':
        total = sum(supports)
        if total == 0:
            return 0, 0, 0
        return (np.average(precisions, weights=supports),
                np.average(recalls, weights=supports),
                np.average(f1s, weights=supports))
    else:
        return np.mean(precisions), np.mean(recalls), np.mean(f1s)


class ClassifierComparison:
    """
    Compare multiple classifiers on a dataset.

    Usage:
        comparison = ClassifierComparison()
        comparison.load_data(X_train, y_train, X_test, y_test)
        results = comparison.run_comparison()
        comparison.print_results()
    """

    def __init__(self, normalize: bool = True, task: str = 'blink'):
        """
        Initialize comparison.

        Args:
            normalize: If True, normalize features before training
            task: 'blink' (binary) or 'movement' (3-class)
        """
        self.normalize = normalize
        self.task = task
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.results: List[ClassifierResult] = []
        self._norm_params: Dict = {}
        self.n_classes: int = 2

    def load_data(self, X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray):
        """Load training and test data."""
        X_train = np.array(X_train, dtype=np.float64)
        y_train = np.array(y_train)
        X_test = np.array(X_test, dtype=np.float64)
        y_test = np.array(y_test)

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
        self.n_classes = len(np.unique(y_train))

        print(f"Data loaded: {len(X_train)} train, {len(X_test)} test, {self.n_classes} classes")

    def _evaluate(self, model, name: str, train_time: float) -> ClassifierResult:
        """Evaluate a trained model."""
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)

        train_acc = np.mean(y_train_pred == self.y_train)
        test_acc = np.mean(y_test_pred == self.y_test)

        avg = 'binary' if self.n_classes == 2 else 'macro'
        precision, recall, f1 = precision_recall_f1(self.y_test, y_test_pred, average=avg)

        cm = confusion_matrix(self.y_test, y_test_pred, self.n_classes)

        return ClassifierResult(
            name=name,
            train_accuracy=train_acc,
            test_accuracy=test_acc,
            precision=precision,
            recall=recall,
            f1_score=f1,
            train_time=train_time,
            confusion_matrix=cm,
            model=model
        )

    def run_comparison(self, include_svm: bool = True,
                       include_nn: bool = True,
                       nn_epochs: int = 50,
                       verbose: bool = True) -> List[ClassifierResult]:
        """
        Run comparison of all classifiers.

        Args:
            include_svm: Include SVM (can be slow)
            include_nn: Include Neural Network
            nn_epochs: Number of epochs for NN
            verbose: Print progress

        Returns:
            List of ClassifierResult
        """
        if self.X_train is None:
            raise ValueError("No data loaded. Call load_data() first.")

        self.results = []

        # 1. Decision Tree
        if verbose:
            print("Training Decision Tree...")
        start = time.time()
        model = DecisionTreeClassifier(max_depth=8, criterion='entropy')
        model.fit(self.X_train, self.y_train)
        result = self._evaluate(model, "Decision Tree", time.time() - start)
        self.results.append(result)

        # 2. Random Forest
        if verbose:
            print("Training Random Forest...")
        start = time.time()
        model = RandomForestClassifier(n_estimators=50, max_depth=8)
        model.fit(self.X_train, self.y_train)
        result = self._evaluate(model, "Random Forest", time.time() - start)
        self.results.append(result)

        # 3. XGBoost
        if verbose:
            print("Training XGBoost...")
        start = time.time()
        model = XGBoostClassifier(n_estimators=50, max_depth=3, learning_rate=0.1)
        model.fit(self.X_train, self.y_train)
        result = self._evaluate(model, "XGBoost", time.time() - start)
        self.results.append(result)

        # 4. KNN
        if verbose:
            print("Training KNN...")
        start = time.time()
        model = KNeighborsClassifier(n_neighbors=5, weights='distance')
        model.fit(self.X_train, self.y_train)
        result = self._evaluate(model, "KNN (k=5)", time.time() - start)
        self.results.append(result)

        # 5. SVM
        if include_svm:
            if verbose:
                print("Training SVM...")
            start = time.time()
            model = SVC(kernel='rbf', C=1.0, max_iter=200)
            model.fit(self.X_train, self.y_train)
            result = self._evaluate(model, "SVM (RBF)", time.time() - start)
            self.results.append(result)

        # 6. Naive Bayes
        if verbose:
            print("Training Naive Bayes...")
        start = time.time()
        model = GaussianNB()
        model.fit(self.X_train, self.y_train)
        result = self._evaluate(model, "Naive Bayes", time.time() - start)
        self.results.append(result)

        # 7. Neural Network
        if include_nn:
            if verbose:
                print("Training Neural Network...")
            start = time.time()

            n_features = self.X_train.shape[1]
            if self.n_classes == 2:
                layer_sizes = [n_features, 64, 32, 1]
                output_activation = 'sigmoid'
            else:
                layer_sizes = [n_features, 64, 32, self.n_classes]
                output_activation = 'softmax'

            model = NeuralNetwork(
                layer_sizes=layer_sizes,
                activation='relu',
                output_activation=output_activation,
                learning_rate=0.01,
                l2_lambda=0.001
            )

            # Prepare labels
            if self.n_classes == 2:
                y_train_nn = self.y_train.reshape(-1, 1)
            else:
                y_train_nn = np.zeros((len(self.y_train), self.n_classes))
                for i, label in enumerate(self.y_train):
                    y_train_nn[i, int(label)] = 1

            model.fit(self.X_train, y_train_nn, epochs=nn_epochs, batch_size=32, verbose=0)

            # Custom prediction for NN
            class NNWrapper:
                def __init__(self, nn, n_classes):
                    self.nn = nn
                    self.n_classes = n_classes

                def predict(self, X):
                    return self.nn.predict_classes(X)

            wrapper = NNWrapper(model, self.n_classes)
            result = self._evaluate(wrapper, "ANN (64-32)", time.time() - start)
            result.model = model  # Store actual model
            self.results.append(result)

        if verbose:
            print("Comparison complete!")

        return self.results

    def print_results(self):
        """Print comparison results as a formatted table."""
        if not self.results:
            print("No results. Run run_comparison() first.")
            return

        print("\n" + "=" * 95)
        print(f"CLASSIFIER COMPARISON ({self.task.upper()} DETECTION)")
        print("=" * 95)
        print(f"{'Model':<16} {'Train Acc':<10} {'Test Acc':<10} "
              f"{'Precision':<10} {'Recall':<10} {'F1':<8} {'Time (s)':<10}")
        print("-" * 95)

        for r in self.results:
            print(f"{r.name:<16} {r.train_accuracy:<10.4f} {r.test_accuracy:<10.4f} "
                  f"{r.precision:<10.4f} {r.recall:<10.4f} {r.f1_score:<8.4f} {r.train_time:<10.3f}")

        print("-" * 95)

        best = max(self.results, key=lambda x: x.f1_score)
        print(f"\nBest Model: {best.name} (F1: {best.f1_score:.4f})")

    def print_confusion_matrices(self):
        """Print confusion matrices for all models."""
        if not self.results:
            return

        class_names = ['No Blink', 'Blink'] if self.n_classes == 2 else \
                      ['Fixation', 'Saccade', 'Blink']

        for r in self.results:
            print(f"\n{r.name} Confusion Matrix:")
            print("Predicted ->", end="")
            for name in class_names:
                print(f" {name[:8]:>8}", end="")
            print()

            for i, row in enumerate(r.confusion_matrix):
                print(f"{class_names[i]:<10}", end="")
                for val in row:
                    print(f" {val:>8}", end="")
                print()

    def get_best_model(self) -> ClassifierResult:
        """Get the best performing model (highest F1)."""
        if not self.results:
            raise ValueError("No results. Run run_comparison() first.")
        return max(self.results, key=lambda x: x.f1_score)

    def to_dict(self) -> List[Dict]:
        """Export results as list of dictionaries."""
        return [
            {
                'name': r.name,
                'train_accuracy': r.train_accuracy,
                'test_accuracy': r.test_accuracy,
                'precision': r.precision,
                'recall': r.recall,
                'f1_score': r.f1_score,
                'train_time': r.train_time
            }
            for r in self.results
        ]


def compare_classifiers(X: np.ndarray, y: np.ndarray,
                        test_size: float = 0.2,
                        task: str = 'blink',
                        random_state: int = 42) -> ClassifierComparison:
    """
    Convenience function to run full classifier comparison.

    Args:
        X: Feature matrix
        y: Labels
        test_size: Fraction for testing
        task: 'blink' or 'movement'
        random_state: Random seed

    Returns:
        ClassifierComparison with results
    """
    np.random.seed(random_state)
    n_samples = len(X)
    indices = np.random.permutation(n_samples)

    n_test = int(n_samples * test_size)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    comparison = ClassifierComparison(normalize=True, task=task)
    comparison.load_data(X_train, y_train, X_test, y_test)
    comparison.run_comparison(include_svm=True, include_nn=True, nn_epochs=50)
    comparison.print_results()

    return comparison


# =============================================================================
# Demo
# =============================================================================

def generate_blink_data(n_samples: int = 600) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic blink detection data."""
    np.random.seed(42)

    # 19-dimensional feature vector (9 temporal + 10 base)
    # Class 0: No blink (high EAR values)
    X0 = np.random.randn(n_samples // 2, 19) * 0.3
    X0[:, 0:4] += 0.3  # EAR features
    y0 = np.zeros(n_samples // 2)

    # Class 1: Blink (low EAR values)
    X1 = np.random.randn(n_samples // 2, 19) * 0.3
    X1[:, 0:4] -= 0.4  # EAR features
    y1 = np.ones(n_samples // 2)

    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])

    return X, y


def generate_movement_data(n_samples: int = 600) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic movement classification data."""
    np.random.seed(42)

    n_per_class = n_samples // 3

    # Class 0: Fixation (low velocity, low dispersion)
    X0 = np.random.randn(n_per_class, 14) * 0.3
    X0[:, 0:3] -= 0.5  # Low velocity
    y0 = np.zeros(n_per_class)

    # Class 1: Saccade (high velocity)
    X1 = np.random.randn(n_per_class, 14) * 0.3
    X1[:, 0:3] += 0.8  # High velocity
    y1 = np.ones(n_per_class)

    # Class 2: Blink (low EAR)
    X2 = np.random.randn(n_per_class, 14) * 0.3
    X2[:, 8:11] -= 0.6  # Low EAR
    y2 = np.full(n_per_class, 2)

    X = np.vstack([X0, X1, X2])
    y = np.concatenate([y0, y1, y2])

    return X, y


def demo():
    """Run demo comparisons."""
    print("=" * 60)
    print("CLASSIFIER COMPARISON DEMO")
    print("=" * 60)

    # Blink Detection
    print("\n1. BLINK DETECTION (Binary)")
    print("-" * 40)
    X, y = generate_blink_data(600)
    comparison = compare_classifiers(X, y, task='blink')
    comparison.print_confusion_matrices()

    # Movement Classification
    print("\n\n2. MOVEMENT CLASSIFICATION (3-class)")
    print("-" * 40)
    X, y = generate_movement_data(600)
    comparison = compare_classifiers(X, y, task='movement')
    comparison.print_confusion_matrices()


if __name__ == "__main__":
    demo()
