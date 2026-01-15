"""
Gaze Estimation Model Comparison.

Compares multiple regression models for predicting screen coordinates
from eye tracking features. All models are implemented from scratch.

Models compared:
- Linear Regression (OLS)
- Ridge Regression (L2)
- Polynomial Regression (degree 2)
- Polynomial Regression (degree 3)
- Neural Network (MLP)
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_from_scratch.linear_regression import LinearRegression, RidgeRegression
from ml_from_scratch.polynomial_features import PolynomialRegression
from ml_from_scratch.neural_network import NeuralNetwork, create_gaze_estimator


@dataclass
class ModelResult:
    """Results from training and evaluating a model."""
    name: str
    train_error: float  # Pixel error on training set
    test_error: float   # Pixel error on test set
    train_r2: float     # R² on training set
    test_r2: float      # R² on test set
    train_time: float   # Training time in seconds
    model: object       # Trained model


class GazeModelComparison:
    """
    Compare multiple regression models for gaze estimation.

    Usage:
        comparison = GazeModelComparison()
        comparison.load_data(X_train, y_train, X_test, y_test)
        results = comparison.run_comparison()
        comparison.print_results()
    """

    def __init__(self, normalize: bool = True):
        """
        Initialize comparison.

        Args:
            normalize: If True, normalize features before training
        """
        self.normalize = normalize
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.results: List[ModelResult] = []
        self._norm_params: Dict = {}

    def load_data(self, X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray):
        """
        Load training and test data.

        Args:
            X_train: Training features (n_samples, 10)
            y_train: Training targets (n_samples, 2) - screen (x, y)
            X_test: Test features
            y_test: Test targets
        """
        X_train = np.array(X_train, dtype=np.float64)
        y_train = np.array(y_train, dtype=np.float64)
        X_test = np.array(X_test, dtype=np.float64)
        y_test = np.array(y_test, dtype=np.float64)

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

        print(f"Data loaded: {len(X_train)} train, {len(X_test)} test samples")

    def _train_and_evaluate(self, model, name: str,
                            fit_fn=None, **fit_kwargs) -> ModelResult:
        """Train a model and evaluate it."""
        start_time = time.time()

        if fit_fn:
            fit_fn(self.X_train, self.y_train, **fit_kwargs)
        else:
            model.fit(self.X_train, self.y_train)

        train_time = time.time() - start_time

        train_error = model.pixel_error(self.X_train, self.y_train)
        test_error = model.pixel_error(self.X_test, self.y_test)
        train_r2 = model.score(self.X_train, self.y_train)
        test_r2 = model.score(self.X_test, self.y_test)

        return ModelResult(
            name=name,
            train_error=train_error,
            test_error=test_error,
            train_r2=train_r2,
            test_r2=test_r2,
            train_time=train_time,
            model=model
        )

    def run_comparison(self, include_nn: bool = True,
                       nn_epochs: int = 100,
                       verbose: bool = True) -> List[ModelResult]:
        """
        Run comparison of all models.

        Args:
            include_nn: If True, include neural network (slower)
            nn_epochs: Number of epochs for neural network training
            verbose: If True, print progress

        Returns:
            List of ModelResult for each model
        """
        if self.X_train is None:
            raise ValueError("No data loaded. Call load_data() first.")

        self.results = []

        # 1. Linear Regression (OLS)
        if verbose:
            print("Training Linear Regression...")
        model = LinearRegression()
        result = self._train_and_evaluate(model, "Linear Regression")
        self.results.append(result)

        # 2. Ridge Regression
        if verbose:
            print("Training Ridge Regression...")
        model = RidgeRegression(alpha=1.0)
        result = self._train_and_evaluate(model, "Ridge (a=1.0)")
        self.results.append(result)

        # 3. Polynomial Regression (degree 2)
        if verbose:
            print("Training Polynomial (deg=2)...")
        model = PolynomialRegression(degree=2, alpha=0.1)
        result = self._train_and_evaluate(model, "Polynomial (deg=2)")
        self.results.append(result)

        # 4. Polynomial Regression (degree 3)
        if verbose:
            print("Training Polynomial (deg=3)...")
        model = PolynomialRegression(degree=3, alpha=1.0)
        result = self._train_and_evaluate(model, "Polynomial (deg=3)")
        self.results.append(result)

        # 5. Neural Network
        if include_nn:
            if verbose:
                print("Training Neural Network...")
            model = create_gaze_estimator(
                n_features=self.X_train.shape[1],
                hidden_layers=[64, 32],
                learning_rate=0.01,
                l2_lambda=0.001
            )
            result = self._train_and_evaluate(
                model, "ANN (64-32)",
                fit_fn=model.fit,
                epochs=nn_epochs,
                batch_size=32,
                verbose=0
            )
            self.results.append(result)

        if verbose:
            print("Comparison complete!")

        return self.results

    def print_results(self):
        """Print comparison results as a formatted table."""
        if not self.results:
            print("No results. Run run_comparison() first.")
            return

        print("\n" + "=" * 85)
        print("GAZE ESTIMATION MODEL COMPARISON")
        print("=" * 85)
        print(f"{'Model':<22} {'Train Err (px)':<15} {'Test Err (px)':<15} "
              f"{'R²':<8} {'Time (s)':<10}")
        print("-" * 85)

        for r in self.results:
            print(f"{r.name:<22} {r.train_error:<15.2f} {r.test_error:<15.2f} "
                  f"{r.test_r2:<8.4f} {r.train_time:<10.3f}")

        print("-" * 85)

        # Find best model
        best = min(self.results, key=lambda x: x.test_error)
        print(f"\nBest Model: {best.name} (Test Error: {best.test_error:.2f} px)")

    def get_best_model(self) -> ModelResult:
        """Get the best performing model (lowest test error)."""
        if not self.results:
            raise ValueError("No results. Run run_comparison() first.")
        return min(self.results, key=lambda x: x.test_error)

    def to_dict(self) -> List[Dict]:
        """Export results as list of dictionaries."""
        return [
            {
                'name': r.name,
                'train_error': r.train_error,
                'test_error': r.test_error,
                'train_r2': r.train_r2,
                'test_r2': r.test_r2,
                'train_time': r.train_time
            }
            for r in self.results
        ]


def compare_gaze_models(X: np.ndarray, y: np.ndarray,
                        test_size: float = 0.2,
                        random_state: int = 42,
                        include_nn: bool = True,
                        nn_epochs: int = 100) -> GazeModelComparison:
    """
    Convenience function to run full model comparison.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target coordinates (n_samples, 2)
        test_size: Fraction of data for testing
        random_state: Random seed
        include_nn: Include neural network
        nn_epochs: NN training epochs

    Returns:
        GazeModelComparison with results
    """
    # Split data
    np.random.seed(random_state)
    n_samples = len(X)
    indices = np.random.permutation(n_samples)

    n_test = int(n_samples * test_size)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Run comparison
    comparison = GazeModelComparison(normalize=True)
    comparison.load_data(X_train, y_train, X_test, y_test)
    comparison.run_comparison(include_nn=include_nn, nn_epochs=nn_epochs)
    comparison.print_results()

    return comparison


# =============================================================================
# Demo and testing
# =============================================================================

def generate_synthetic_gaze_data(n_samples: int = 500,
                                  n_features: int = 10,
                                  noise_level: float = 20.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic gaze estimation data for testing.

    The relationship is non-linear to test polynomial and NN models.
    """
    np.random.seed(42)

    X = np.random.randn(n_samples, n_features)

    # Non-linear relationship (simulating real gaze patterns)
    # Screen X depends on iris X ratios and head yaw
    y_x = (500 +
           300 * X[:, 0] +  # left iris X
           250 * X[:, 2] +  # right iris X
           100 * np.sin(X[:, 7] * 0.5) +  # head yaw (non-linear)
           50 * X[:, 0] * X[:, 2])  # interaction

    # Screen Y depends on iris Y ratios and head pitch
    y_y = (400 +
           200 * X[:, 1] +  # left iris Y
           180 * X[:, 3] +  # right iris Y
           80 * np.sin(X[:, 6] * 0.5) +  # head pitch (non-linear)
           40 * X[:, 1] * X[:, 3])  # interaction

    y = np.column_stack([y_x, y_y])

    # Add noise
    y += np.random.randn(n_samples, 2) * noise_level

    return X, y


def demo():
    """Run demo comparison with synthetic data."""
    print("=" * 60)
    print("GAZE ESTIMATION MODEL COMPARISON DEMO")
    print("=" * 60)
    print("\nGenerating synthetic gaze data...")

    X, y = generate_synthetic_gaze_data(n_samples=500, noise_level=25.0)
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Target range: X=[{y[:,0].min():.0f}, {y[:,0].max():.0f}], "
          f"Y=[{y[:,1].min():.0f}, {y[:,1].max():.0f}]")

    comparison = compare_gaze_models(
        X, y,
        test_size=0.2,
        include_nn=True,
        nn_epochs=100
    )

    # Export results
    results_dict = comparison.to_dict()
    print("\nExported Results:")
    for r in results_dict:
        print(f"  {r['name']}: {r['test_error']:.2f} px")


if __name__ == "__main__":
    demo()
