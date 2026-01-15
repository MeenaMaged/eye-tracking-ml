"""
Linear Regression from scratch using Normal Equation.

Implements:
- Ordinary Least Squares (OLS) regression
- Ridge Regression (L2 regularization)

Both support multi-output regression for gaze estimation (predicting x, y coordinates).
"""

import numpy as np
from typing import Optional, Tuple


class LinearRegression:
    """
    Ordinary Least Squares Linear Regression using the Normal Equation.

    The normal equation provides a closed-form solution:
        θ = (X^T X)^(-1) X^T y

    For gaze estimation:
        Input: 10-dimensional feature vector
        Output: 2-dimensional (screen_x, screen_y)
    """

    def __init__(self, fit_intercept: bool = True):
        """
        Initialize Linear Regression.

        Args:
            fit_intercept: If True, add bias term (column of ones)
        """
        self.fit_intercept = fit_intercept
        self.weights: Optional[np.ndarray] = None
        self.n_features: int = 0
        self.n_outputs: int = 0

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """Add column of ones for intercept term."""
        if self.fit_intercept:
            ones = np.ones((X.shape[0], 1))
            return np.hstack([ones, X])
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Fit linear regression using normal equation.

        Normal Equation: θ = (X^T X)^(-1) X^T y

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target values of shape (n_samples,) or (n_samples, n_outputs)

        Returns:
            self
        """
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.n_features = X.shape[1]
        self.n_outputs = y.shape[1]

        # Add intercept term
        X_b = self._add_intercept(X)

        # Normal equation: θ = (X^T X)^(-1) X^T y
        # Use pseudo-inverse for numerical stability
        XTX = X_b.T @ X_b
        XTy = X_b.T @ y

        try:
            # Try direct inverse first
            self.weights = np.linalg.solve(XTX, XTy)
        except np.linalg.LinAlgError:
            # Fall back to pseudo-inverse if singular
            self.weights = np.linalg.pinv(XTX) @ XTy

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Predictions of shape (n_samples, n_outputs)
        """
        if self.weights is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = np.array(X, dtype=np.float64)
        X_b = self._add_intercept(X)

        predictions = X_b @ self.weights

        # Return 1D if single output
        if self.n_outputs == 1:
            return predictions.ravel()
        return predictions

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R² (coefficient of determination).

        R² = 1 - (SS_res / SS_tot)

        Args:
            X: Feature matrix
            y: True target values

        Returns:
            R² score (1.0 is perfect, 0.0 is baseline)
        """
        y = np.array(y, dtype=np.float64)
        y_pred = self.predict(X)

        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)

        if ss_tot == 0:
            return 0.0

        return 1 - (ss_res / ss_tot)

    def mean_squared_error(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate Mean Squared Error."""
        y = np.array(y, dtype=np.float64)
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)

    def mean_absolute_error(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        y = np.array(y, dtype=np.float64)
        y_pred = self.predict(X)
        return np.mean(np.abs(y - y_pred))

    def pixel_error(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate mean Euclidean distance error in pixels.

        For gaze estimation where y = [x, y] coordinates.
        """
        y = np.array(y, dtype=np.float64)
        y_pred = self.predict(X)

        if y.ndim == 1:
            return self.mean_absolute_error(X, y)

        # Euclidean distance for each sample
        distances = np.sqrt(np.sum((y - y_pred) ** 2, axis=1))
        return np.mean(distances)

    @property
    def coef_(self) -> np.ndarray:
        """Get coefficients (excluding intercept)."""
        if self.weights is None:
            return np.array([])
        if self.fit_intercept:
            return self.weights[1:]
        return self.weights

    @property
    def intercept_(self) -> np.ndarray:
        """Get intercept term."""
        if self.weights is None or not self.fit_intercept:
            return np.zeros(self.n_outputs)
        return self.weights[0]


class RidgeRegression(LinearRegression):
    """
    Ridge Regression (Linear Regression with L2 regularization).

    Adds regularization term to prevent overfitting:
        θ = (X^T X + λI)^(-1) X^T y

    The regularization term λ (alpha) penalizes large weights,
    leading to more stable and generalizable models.
    """

    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True):
        """
        Initialize Ridge Regression.

        Args:
            alpha: Regularization strength (λ). Higher = more regularization.
                   - 0: equivalent to OLS
                   - 0.1-1.0: light regularization
                   - 1.0-10.0: moderate regularization
                   - 10.0+: strong regularization
            fit_intercept: If True, add bias term
        """
        super().__init__(fit_intercept)
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RidgeRegression':
        """
        Fit Ridge regression using regularized normal equation.

        Ridge Normal Equation: θ = (X^T X + λI)^(-1) X^T y

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target values of shape (n_samples,) or (n_samples, n_outputs)

        Returns:
            self
        """
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.n_features = X.shape[1]
        self.n_outputs = y.shape[1]

        # Add intercept term
        X_b = self._add_intercept(X)
        n_features_with_bias = X_b.shape[1]

        # Ridge regularization: (X^T X + λI)
        XTX = X_b.T @ X_b
        regularization = self.alpha * np.eye(n_features_with_bias)

        # Don't regularize the intercept term
        if self.fit_intercept:
            regularization[0, 0] = 0

        XTX_reg = XTX + regularization
        XTy = X_b.T @ y

        # Solve the regularized system
        self.weights = np.linalg.solve(XTX_reg, XTy)

        return self


# =============================================================================
# Utility functions
# =============================================================================

def train_test_metrics(model, X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Calculate comprehensive metrics for a regression model.

    Returns:
        Dictionary with train/test MSE, MAE, R², and pixel error
    """
    return {
        'train_mse': model.mean_squared_error(X_train, y_train),
        'test_mse': model.mean_squared_error(X_test, y_test),
        'train_mae': model.mean_absolute_error(X_train, y_train),
        'test_mae': model.mean_absolute_error(X_test, y_test),
        'train_r2': model.score(X_train, y_train),
        'test_r2': model.score(X_test, y_test),
        'train_pixel_error': model.pixel_error(X_train, y_train),
        'test_pixel_error': model.pixel_error(X_test, y_test),
    }


# =============================================================================
# Test functions
# =============================================================================

def test_linear_regression():
    """Test LinearRegression with synthetic data."""
    print("Testing LinearRegression...")

    np.random.seed(42)

    # Generate synthetic gaze data
    n_samples = 200
    n_features = 10

    # True weights
    true_weights = np.random.randn(n_features, 2) * 100

    X = np.random.randn(n_samples, n_features)
    y = X @ true_weights + np.random.randn(n_samples, 2) * 10  # Add noise

    # Split
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train OLS
    model = LinearRegression()
    model.fit(X_train, y_train)

    print(f"OLS R² (train): {model.score(X_train, y_train):.4f}")
    print(f"OLS R² (test): {model.score(X_test, y_test):.4f}")
    print(f"OLS Pixel Error (test): {model.pixel_error(X_test, y_test):.2f}")

    # Train Ridge
    ridge = RidgeRegression(alpha=1.0)
    ridge.fit(X_train, y_train)

    print(f"\nRidge R² (train): {ridge.score(X_train, y_train):.4f}")
    print(f"Ridge R² (test): {ridge.score(X_test, y_test):.4f}")
    print(f"Ridge Pixel Error (test): {ridge.pixel_error(X_test, y_test):.2f}")


def test_single_output():
    """Test with single output regression."""
    print("\nTesting single output regression...")

    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = 3 * X[:, 0] + 2 * X[:, 1] - X[:, 2] + np.random.randn(100) * 0.1

    model = LinearRegression()
    model.fit(X, y)

    print(f"R²: {model.score(X, y):.4f}")
    print(f"Coefficients: {model.coef_.ravel()[:3]}...")


if __name__ == "__main__":
    test_linear_regression()
    test_single_output()
