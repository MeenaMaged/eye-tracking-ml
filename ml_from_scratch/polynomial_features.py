"""
Polynomial Features for non-linear regression.

Transforms input features into polynomial combinations, enabling
linear models to learn non-linear relationships.

For gaze estimation, this captures non-linear relationships between
eye features and screen coordinates.
"""

import numpy as np
from typing import List, Tuple, Optional
from itertools import combinations_with_replacement


class PolynomialFeatures:
    """
    Generate polynomial and interaction features.

    For degree=2 and input [a, b]:
        Output: [1, a, b, a², ab, b²]

    For degree=3 and input [a, b]:
        Output: [1, a, b, a², ab, b², a³, a²b, ab², b³]

    This allows linear regression to fit polynomial curves.
    """

    def __init__(self, degree: int = 2, include_bias: bool = True,
                 interaction_only: bool = False):
        """
        Initialize PolynomialFeatures.

        Args:
            degree: Maximum polynomial degree (1, 2, 3, ...)
            include_bias: If True, include a column of ones (bias term)
            interaction_only: If True, only produce interaction features
                             (no powers like x², x³)
        """
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only
        self.n_input_features: int = 0
        self.n_output_features: int = 0
        self._powers: List[Tuple[int, ...]] = []
        self._feature_names: List[str] = []

    def _compute_powers(self, n_features: int) -> List[Tuple[int, ...]]:
        """
        Compute all polynomial power combinations.

        Returns list of tuples where each tuple represents the power
        of each input feature. E.g., (2, 1) means x1² * x2¹
        """
        powers = []

        for d in range(0 if self.include_bias else 1, self.degree + 1):
            if self.interaction_only and d > 1:
                # Only include combinations where each feature appears at most once
                for combo in combinations_with_replacement(range(n_features), d):
                    power = [0] * n_features
                    for idx in combo:
                        power[idx] += 1
                    # Check if any power > 1 (interaction_only excludes these)
                    if max(power) <= 1:
                        powers.append(tuple(power))
            else:
                # Include all combinations
                for combo in combinations_with_replacement(range(n_features), d):
                    power = [0] * n_features
                    for idx in combo:
                        power[idx] += 1
                    powers.append(tuple(power))

        return powers

    def fit(self, X: np.ndarray) -> 'PolynomialFeatures':
        """
        Fit transformer to learn the number of features.

        Args:
            X: Input feature matrix of shape (n_samples, n_features)

        Returns:
            self
        """
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n_input_features = X.shape[1]
        self._powers = self._compute_powers(self.n_input_features)
        self.n_output_features = len(self._powers)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features to polynomial features.

        Args:
            X: Input feature matrix of shape (n_samples, n_features)

        Returns:
            Transformed matrix of shape (n_samples, n_output_features)
        """
        X = np.array(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if len(self._powers) == 0:
            raise ValueError("Transformer not fitted. Call fit() first.")

        n_samples = X.shape[0]
        X_poly = np.empty((n_samples, self.n_output_features), dtype=np.float64)

        for i, power in enumerate(self._powers):
            # Compute x1^p1 * x2^p2 * ... * xn^pn
            X_poly[:, i] = np.prod(X ** power, axis=1)

        return X_poly

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def get_feature_names(self, input_names: Optional[List[str]] = None) -> List[str]:
        """
        Get feature names for the output features.

        Args:
            input_names: Names for input features (default: x0, x1, ...)

        Returns:
            List of feature names like ['1', 'x0', 'x1', 'x0^2', 'x0*x1', ...]
        """
        if len(self._powers) == 0:
            return []

        if input_names is None:
            input_names = [f'x{i}' for i in range(self.n_input_features)]

        names = []
        for power in self._powers:
            if sum(power) == 0:
                names.append('1')
            else:
                terms = []
                for name, p in zip(input_names, power):
                    if p == 1:
                        terms.append(name)
                    elif p > 1:
                        terms.append(f'{name}^{p}')
                names.append('*'.join(terms) if terms else '1')

        return names


class PolynomialRegression:
    """
    Polynomial Regression combining PolynomialFeatures with LinearRegression.

    Convenience class that wraps feature expansion and linear regression
    into a single interface for gaze estimation.
    """

    def __init__(self, degree: int = 2, alpha: float = 0.0,
                 include_bias: bool = True):
        """
        Initialize Polynomial Regression.

        Args:
            degree: Polynomial degree
            alpha: Ridge regularization strength (0 = OLS)
            include_bias: Include bias in polynomial features
        """
        from .linear_regression import LinearRegression, RidgeRegression

        self.degree = degree
        self.alpha = alpha
        self.include_bias = include_bias

        self.poly = PolynomialFeatures(degree=degree, include_bias=include_bias)

        if alpha > 0:
            self.regressor = RidgeRegression(alpha=alpha, fit_intercept=False)
        else:
            self.regressor = LinearRegression(fit_intercept=False)

        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'PolynomialRegression':
        """
        Fit polynomial regression.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target values

        Returns:
            self
        """
        X_poly = self.poly.fit_transform(X)
        self.regressor.fit(X_poly, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using fitted model."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        X_poly = self.poly.transform(X)
        return self.regressor.predict(X_poly)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R² score."""
        X_poly = self.poly.transform(X)
        return self.regressor.score(X_poly, y)

    def mean_squared_error(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate MSE."""
        X_poly = self.poly.transform(X)
        return self.regressor.mean_squared_error(X_poly, y)

    def pixel_error(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate mean pixel error for gaze estimation."""
        X_poly = self.poly.transform(X)
        return self.regressor.pixel_error(X_poly, y)

    @property
    def n_features_out(self) -> int:
        """Number of polynomial features."""
        return self.poly.n_output_features


# =============================================================================
# Test functions
# =============================================================================

def test_polynomial_features():
    """Test PolynomialFeatures transformation."""
    print("Testing PolynomialFeatures...")

    # Simple test with 2 features
    X = np.array([[1, 2], [3, 4], [5, 6]])

    # Degree 2
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    print(f"Input shape: {X.shape}")
    print(f"Output shape: {X_poly.shape}")
    print(f"Feature names: {poly.get_feature_names(['a', 'b'])}")

    # Verify first sample: [1, 2] -> [1, 1, 2, 1, 2, 4]
    print(f"First sample: {X[0]} -> {X_poly[0]}")

    # Degree 3
    poly3 = PolynomialFeatures(degree=3)
    X_poly3 = poly3.fit_transform(X)
    print(f"\nDegree 3 output shape: {X_poly3.shape}")
    print(f"Degree 3 features: {poly3.get_feature_names(['a', 'b'])}")


def test_polynomial_regression():
    """Test PolynomialRegression on non-linear data."""
    print("\nTesting PolynomialRegression...")

    np.random.seed(42)

    # Generate non-linear data: y = x² + noise
    X = np.linspace(-3, 3, 100).reshape(-1, 1)
    y = X.ravel() ** 2 + np.random.randn(100) * 0.5

    # Split
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]

    # Linear (should perform poorly)
    from .linear_regression import LinearRegression
    linear = LinearRegression()
    linear.fit(X_train, y_train)
    print(f"Linear R² (test): {linear.score(X_test, y_test):.4f}")

    # Polynomial degree 2 (should fit well)
    poly2 = PolynomialRegression(degree=2)
    poly2.fit(X_train, y_train)
    print(f"Poly(2) R² (test): {poly2.score(X_test, y_test):.4f}")

    # Polynomial degree 5 with regularization
    poly5 = PolynomialRegression(degree=5, alpha=0.1)
    poly5.fit(X_train, y_train)
    print(f"Poly(5) R² (test): {poly5.score(X_test, y_test):.4f}")


def test_gaze_estimation():
    """Test polynomial regression for gaze estimation (2D output)."""
    print("\nTesting Polynomial Regression for Gaze Estimation...")

    np.random.seed(42)

    # Simulate gaze data
    n_samples = 200
    n_features = 10

    X = np.random.randn(n_samples, n_features)

    # Non-linear relationship
    y_x = 500 + 200 * X[:, 0] + 100 * X[:, 1]**2 + 50 * X[:, 0] * X[:, 1]
    y_y = 400 + 150 * X[:, 2] + 80 * X[:, 3]**2 - 30 * X[:, 2] * X[:, 3]
    y = np.column_stack([y_x, y_y]) + np.random.randn(n_samples, 2) * 20

    # Split
    X_train, X_test = X[:160], X[160:]
    y_train, y_test = y[:160], y[160:]

    # Compare models
    from .linear_regression import LinearRegression

    linear = LinearRegression()
    linear.fit(X_train, y_train)

    poly2 = PolynomialRegression(degree=2, alpha=0.1)
    poly2.fit(X_train, y_train)

    print(f"Linear - R²: {linear.score(X_test, y_test):.4f}, "
          f"Pixel Error: {linear.pixel_error(X_test, y_test):.2f}")
    print(f"Poly(2) - R²: {poly2.score(X_test, y_test):.4f}, "
          f"Pixel Error: {poly2.pixel_error(X_test, y_test):.2f}")
    print(f"Poly features: {poly2.n_features_out}")


if __name__ == "__main__":
    test_polynomial_features()
    test_polynomial_regression()
    test_gaze_estimation()
