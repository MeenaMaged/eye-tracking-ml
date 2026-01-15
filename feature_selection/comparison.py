"""
Feature Selection Comparison for Eye Tracking ML Project.

Compares different feature selection methods:
- No selection (baseline)
- PCA (variance threshold)
- Genetic Algorithm (wrapper method)
- Fisher's LDA (supervised)

Evaluates each method by downstream classifier accuracy.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_from_scratch.pca import PCA
from ml_from_scratch.genetic_algorithm import GeneticAlgorithmFeatureSelector
from ml_from_scratch.fishers_discriminant import LinearDiscriminantAnalysis


@dataclass
class FeatureSelectionResult:
    """Results from a feature selection method."""
    method: str
    n_features_original: int
    n_features_selected: int
    train_accuracy: float
    test_accuracy: float
    transform_time: float
    classifier_name: str
    feature_indices: Optional[np.ndarray] = None
    explained_variance: Optional[float] = None


class FeatureSelectionComparison:
    """
    Compare feature selection methods for classification tasks.

    Methods compared:
    1. Baseline (all features)
    2. PCA with different variance thresholds (95%, 90%, 80%)
    3. Genetic Algorithm
    4. LDA (Linear Discriminant Analysis)

    Usage:
        comparison = FeatureSelectionComparison()
        comparison.load_data(X_train, y_train, X_test, y_test)
        comparison.run_comparison(classifier_class)
        comparison.print_results()
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize comparison.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.results: List[FeatureSelectionResult] = []
        self.n_classes: int = 2

    def load_data(self, X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray):
        """Load training and test data."""
        self.X_train = np.array(X_train, dtype=np.float64)
        self.X_test = np.array(X_test, dtype=np.float64)
        self.y_train = np.array(y_train)
        self.y_test = np.array(y_test)
        self.n_classes = len(np.unique(y_train))

        print(f"Data loaded: {len(X_train)} train, {len(X_test)} test")
        print(f"Features: {X_train.shape[1]}, Classes: {self.n_classes}")

    def _normalize(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize data using training statistics."""
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std == 0] = 1.0

        X_train_norm = (X_train - mean) / std
        X_test_norm = (X_test - mean) / std

        return X_train_norm, X_test_norm

    def _evaluate_classifier(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_test: np.ndarray, y_test: np.ndarray,
                             classifier_class) -> Tuple[float, float]:
        """Train and evaluate classifier."""
        clf = classifier_class()
        clf.fit(X_train, y_train)

        train_acc = np.mean(clf.predict(X_train) == y_train)
        test_acc = np.mean(clf.predict(X_test) == y_test)

        return train_acc, test_acc

    def run_comparison(self, classifier_class,
                       include_ga: bool = True,
                       ga_generations: int = 30,
                       verbose: bool = True) -> List[FeatureSelectionResult]:
        """
        Run comparison of all feature selection methods.

        Args:
            classifier_class: Classifier class to use for evaluation
            include_ga: Include Genetic Algorithm (slow)
            ga_generations: Number of GA generations
            verbose: Print progress

        Returns:
            List of FeatureSelectionResult
        """
        if self.X_train is None:
            raise ValueError("No data loaded. Call load_data() first.")

        self.results = []
        n_features = self.X_train.shape[1]
        classifier_name = classifier_class.__name__

        # Normalize data
        X_train_norm, X_test_norm = self._normalize(self.X_train, self.X_test)

        # 1. Baseline - All features
        if verbose:
            print("\n1. Evaluating Baseline (all features)...")
        start = time.time()
        train_acc, test_acc = self._evaluate_classifier(
            X_train_norm, self.y_train, X_test_norm, self.y_test, classifier_class
        )
        self.results.append(FeatureSelectionResult(
            method="Baseline (All)",
            n_features_original=n_features,
            n_features_selected=n_features,
            train_accuracy=train_acc,
            test_accuracy=test_acc,
            transform_time=time.time() - start,
            classifier_name=classifier_name
        ))
        if verbose:
            print(f"   Accuracy: {test_acc:.4f} ({n_features} features)")

        # 2. PCA with different variance thresholds
        for var_threshold in [0.95, 0.90, 0.80]:
            if verbose:
                print(f"\n2. Evaluating PCA ({int(var_threshold*100)}% variance)...")
            start = time.time()

            pca = PCA(n_components=var_threshold)
            X_train_pca = pca.fit_transform(X_train_norm)
            X_test_pca = pca.transform(X_test_norm)

            train_acc, test_acc = self._evaluate_classifier(
                X_train_pca, self.y_train, X_test_pca, self.y_test, classifier_class
            )

            self.results.append(FeatureSelectionResult(
                method=f"PCA ({int(var_threshold*100)}%)",
                n_features_original=n_features,
                n_features_selected=pca.n_components_,
                train_accuracy=train_acc,
                test_accuracy=test_acc,
                transform_time=time.time() - start,
                classifier_name=classifier_name,
                explained_variance=np.sum(pca.explained_variance_ratio_)
            ))
            if verbose:
                print(f"   Accuracy: {test_acc:.4f} ({pca.n_components_} components, "
                      f"{np.sum(pca.explained_variance_ratio_)*100:.1f}% var)")

        # 3. LDA (Linear Discriminant Analysis)
        if verbose:
            print("\n3. Evaluating LDA...")
        start = time.time()

        max_components = min(self.n_classes - 1, n_features)
        lda = LinearDiscriminantAnalysis(n_components=max_components)
        X_train_lda = lda.fit_transform(X_train_norm, self.y_train)
        X_test_lda = lda.transform(X_test_norm)

        train_acc, test_acc = self._evaluate_classifier(
            X_train_lda, self.y_train, X_test_lda, self.y_test, classifier_class
        )

        self.results.append(FeatureSelectionResult(
            method="LDA",
            n_features_original=n_features,
            n_features_selected=lda.n_components_,
            train_accuracy=train_acc,
            test_accuracy=test_acc,
            transform_time=time.time() - start,
            classifier_name=classifier_name,
            explained_variance=np.sum(lda.explained_variance_ratio_) if lda.explained_variance_ratio_ is not None else None
        ))
        if verbose:
            print(f"   Accuracy: {test_acc:.4f} ({lda.n_components_} components)")

        # 4. Genetic Algorithm (wrapper method)
        if include_ga:
            if verbose:
                print(f"\n4. Evaluating Genetic Algorithm ({ga_generations} generations)...")
            start = time.time()

            ga = GeneticAlgorithmFeatureSelector(
                n_features=n_features,
                population_size=30,
                n_generations=ga_generations,
                random_state=self.random_state
            )
            ga.fit(X_train_norm, self.y_train, classifier_class, verbose=False)

            X_train_ga = ga.transform(X_train_norm)
            X_test_ga = ga.transform(X_test_norm)

            train_acc, test_acc = self._evaluate_classifier(
                X_train_ga, self.y_train, X_test_ga, self.y_test, classifier_class
            )

            self.results.append(FeatureSelectionResult(
                method="Genetic Algorithm",
                n_features_original=n_features,
                n_features_selected=ga.n_features_selected_,
                train_accuracy=train_acc,
                test_accuracy=test_acc,
                transform_time=time.time() - start,
                classifier_name=classifier_name,
                feature_indices=ga.get_selected_features()
            ))
            if verbose:
                print(f"   Accuracy: {test_acc:.4f} ({ga.n_features_selected_} features)")
                print(f"   Selected: {ga.get_selected_features()}")

        if verbose:
            print("\nComparison complete!")

        return self.results

    def print_results(self):
        """Print comparison results as formatted table."""
        if not self.results:
            print("No results. Run run_comparison() first.")
            return

        print("\n" + "=" * 85)
        print("FEATURE SELECTION COMPARISON")
        print("=" * 85)
        print(f"{'Method':<20} {'Features':<10} {'Train Acc':<10} {'Test Acc':<10} "
              f"{'Time (s)':<10} {'Notes':<20}")
        print("-" * 85)

        for r in self.results:
            notes = ""
            if r.explained_variance is not None:
                notes = f"Var: {r.explained_variance*100:.1f}%"
            elif r.feature_indices is not None:
                notes = f"Indices: {len(r.feature_indices)}"

            print(f"{r.method:<20} {r.n_features_selected:<10} {r.train_accuracy:<10.4f} "
                  f"{r.test_accuracy:<10.4f} {r.transform_time:<10.3f} {notes:<20}")

        print("-" * 85)

        # Best method
        best = max(self.results, key=lambda x: x.test_accuracy)
        print(f"\nBest Method: {best.method} (Test Accuracy: {best.test_accuracy:.4f})")

        # Best accuracy with fewest features
        sorted_by_features = sorted(self.results, key=lambda x: x.n_features_selected)
        for r in sorted_by_features:
            if r.test_accuracy >= best.test_accuracy - 0.01:  # Within 1% of best
                print(f"Best Trade-off: {r.method} ({r.n_features_selected} features, "
                      f"{r.test_accuracy:.4f} accuracy)")
                break

    def get_best_method(self) -> FeatureSelectionResult:
        """Get method with highest test accuracy."""
        if not self.results:
            raise ValueError("No results. Run run_comparison() first.")
        return max(self.results, key=lambda x: x.test_accuracy)

    def get_most_efficient(self, accuracy_threshold: float = 0.01) -> FeatureSelectionResult:
        """
        Get method with fewest features within threshold of best accuracy.

        Args:
            accuracy_threshold: Maximum drop from best accuracy

        Returns:
            Most efficient method
        """
        if not self.results:
            raise ValueError("No results. Run run_comparison() first.")

        best_acc = max(r.test_accuracy for r in self.results)
        threshold = best_acc - accuracy_threshold

        efficient = [r for r in self.results if r.test_accuracy >= threshold]
        return min(efficient, key=lambda x: x.n_features_selected)

    def to_dict(self) -> List[Dict]:
        """Export results as list of dictionaries."""
        return [
            {
                'method': r.method,
                'n_features_original': r.n_features_original,
                'n_features_selected': r.n_features_selected,
                'train_accuracy': r.train_accuracy,
                'test_accuracy': r.test_accuracy,
                'transform_time': r.transform_time,
                'classifier': r.classifier_name
            }
            for r in self.results
        ]


def compare_feature_selection(X: np.ndarray, y: np.ndarray,
                               classifier_class,
                               test_size: float = 0.2,
                               include_ga: bool = True,
                               random_state: int = 42) -> FeatureSelectionComparison:
    """
    Convenience function to run feature selection comparison.

    Args:
        X: Feature matrix
        y: Labels
        classifier_class: Classifier to use for evaluation
        test_size: Fraction for test set
        include_ga: Include Genetic Algorithm
        random_state: Random seed

    Returns:
        FeatureSelectionComparison with results
    """
    np.random.seed(random_state)

    # Stratified split
    n_samples = len(X)
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

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    comparison = FeatureSelectionComparison(random_state=random_state)
    comparison.load_data(X_train, y_train, X_test, y_test)
    comparison.run_comparison(classifier_class, include_ga=include_ga)
    comparison.print_results()

    return comparison


# =============================================================================
# Demo
# =============================================================================

def generate_test_data(n_samples: int = 600, n_features: int = 19,
                       n_informative: int = 8, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic classification data with some irrelevant features.

    Args:
        n_samples: Total samples
        n_features: Total features
        n_informative: Number of informative features
        random_state: Random seed

    Returns:
        X, y
    """
    np.random.seed(random_state)

    n_per_class = n_samples // 2

    # Informative features
    X_info_0 = np.random.randn(n_per_class, n_informative)
    X_info_0[:, :4] -= 1.5  # Class separation

    X_info_1 = np.random.randn(n_per_class, n_informative)
    X_info_1[:, :4] += 1.5

    # Noise features
    n_noise = n_features - n_informative
    X_noise_0 = np.random.randn(n_per_class, n_noise) * 2  # High variance noise
    X_noise_1 = np.random.randn(n_per_class, n_noise) * 2

    # Combine
    X0 = np.hstack([X_info_0, X_noise_0])
    X1 = np.hstack([X_info_1, X_noise_1])

    X = np.vstack([X0, X1])
    y = np.array([0] * n_per_class + [1] * n_per_class)

    # Shuffle
    idx = np.random.permutation(n_samples)
    X, y = X[idx], y[idx]

    return X, y


def demo():
    """Run feature selection comparison demo."""
    print("=" * 70)
    print("FEATURE SELECTION COMPARISON DEMO")
    print("=" * 70)

    # Generate data
    print("\nGenerating test data (19 features, 8 informative)...")
    X, y = generate_test_data(n_samples=600, n_features=19, n_informative=8)

    # Run comparison
    from ml_from_scratch.random_forest import RandomForestClassifier

    print("\nUsing Random Forest as classifier...")
    comparison = compare_feature_selection(
        X, y,
        classifier_class=RandomForestClassifier,
        include_ga=True
    )

    print("\n" + "=" * 70)
    print("Demo complete!")


if __name__ == "__main__":
    demo()
