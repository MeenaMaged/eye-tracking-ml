"""
Naive Bayes Classifier from scratch.

Implements:
- Gaussian Naive Bayes (continuous features)
- Class priors and conditional probabilities
- Laplace smoothing option

Naive Bayes assumes features are conditionally independent given the class:
P(y|x1,...,xn) ∝ P(y) * ∏ P(xi|y)
"""

import numpy as np
from typing import Optional


class GaussianNB:
    """
    Gaussian Naive Bayes Classifier.

    Assumes features follow a Gaussian (normal) distribution within each class:
    P(xi|y) = (1 / sqrt(2π * σ²)) * exp(-(xi - μ)² / (2σ²))

    Prediction:
    y_pred = argmax_y [ log P(y) + Σ log P(xi|y) ]
    """

    def __init__(self, var_smoothing: float = 1e-9):
        """
        Initialize Gaussian Naive Bayes.

        Args:
            var_smoothing: Added to variance for numerical stability
        """
        self.var_smoothing = var_smoothing

        self.classes_: Optional[np.ndarray] = None
        self.n_classes: int = 0
        self.class_prior_: Optional[np.ndarray] = None
        self.theta_: Optional[np.ndarray] = None  # Mean of each feature per class
        self.var_: Optional[np.ndarray] = None    # Variance of each feature per class

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianNB':
        """
        Fit Gaussian Naive Bayes model.

        Computes:
        - Class priors P(y) from class frequencies
        - Mean μ and variance σ² of each feature per class

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)

        Returns:
            self
        """
        X = np.array(X, dtype=np.float64)
        y = np.array(y)

        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        n_features = X.shape[1]

        # Compute class priors
        self.class_prior_ = np.zeros(self.n_classes)
        for i, cls in enumerate(self.classes_):
            self.class_prior_[i] = np.sum(y == cls) / len(y)

        # Compute mean and variance per class
        self.theta_ = np.zeros((self.n_classes, n_features))
        self.var_ = np.zeros((self.n_classes, n_features))

        for i, cls in enumerate(self.classes_):
            X_cls = X[y == cls]
            self.theta_[i] = np.mean(X_cls, axis=0)
            self.var_[i] = np.var(X_cls, axis=0) + self.var_smoothing

        return self

    def _log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Compute log-likelihood of each class for each sample.

        log P(xi|y) = -0.5 * [log(2π) + log(σ²) + (xi - μ)²/σ²]

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Log-likelihood matrix (n_samples, n_classes)
        """
        n_samples = len(X)
        log_likelihood = np.zeros((n_samples, self.n_classes))

        for i in range(self.n_classes):
            # Gaussian log-likelihood for each feature
            mean = self.theta_[i]
            var = self.var_[i]

            # log P(X|y=i) = sum over features of log P(x_j|y=i)
            log_prob = -0.5 * (np.log(2 * np.pi * var) + (X - mean)**2 / var)
            log_likelihood[:, i] = np.sum(log_prob, axis=1)

        return log_likelihood

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict log probabilities.

        log P(y|X) ∝ log P(y) + log P(X|y)

        Args:
            X: Feature matrix

        Returns:
            Log probability matrix (n_samples, n_classes)
        """
        X = np.array(X, dtype=np.float64)

        # log P(y) + log P(X|y)
        log_prior = np.log(self.class_prior_)
        log_likelihood = self._log_likelihood(X)

        log_posterior = log_prior + log_likelihood

        # Normalize for proper probabilities
        log_sum = np.max(log_posterior, axis=1, keepdims=True) + \
                  np.log(np.sum(np.exp(log_posterior - np.max(log_posterior, axis=1, keepdims=True)),
                               axis=1, keepdims=True))
        log_proba = log_posterior - log_sum

        return log_proba

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix

        Returns:
            Probability matrix (n_samples, n_classes)
        """
        return np.exp(self.predict_log_proba(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Feature matrix

        Returns:
            Predicted class labels
        """
        X = np.array(X, dtype=np.float64)

        log_prior = np.log(self.class_prior_)
        log_likelihood = self._log_likelihood(X)
        log_posterior = log_prior + log_likelihood

        indices = np.argmax(log_posterior, axis=1)
        return self.classes_[indices]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            'class_prior': self.class_prior_,
            'theta': self.theta_,
            'var': self.var_,
            'classes': self.classes_
        }


class MultinomialNB:
    """
    Multinomial Naive Bayes (for count/frequency features).

    Useful for text classification or discrete count data.
    """

    def __init__(self, alpha: float = 1.0):
        """
        Initialize Multinomial Naive Bayes.

        Args:
            alpha: Laplace smoothing parameter
        """
        self.alpha = alpha

        self.classes_: Optional[np.ndarray] = None
        self.class_prior_: Optional[np.ndarray] = None
        self.feature_log_prob_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultinomialNB':
        """
        Fit Multinomial Naive Bayes model.

        Args:
            X: Training features (counts, non-negative)
            y: Training labels

        Returns:
            self
        """
        X = np.array(X, dtype=np.float64)
        y = np.array(y)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # Class priors
        self.class_prior_ = np.zeros(n_classes)
        for i, cls in enumerate(self.classes_):
            self.class_prior_[i] = np.sum(y == cls) / len(y)

        # Feature log probabilities with Laplace smoothing
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, cls in enumerate(self.classes_):
            X_cls = X[y == cls]
            count = np.sum(X_cls, axis=0) + self.alpha
            total = np.sum(count)
            self.feature_log_prob_[i] = np.log(count / total)

        return self

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict log probabilities."""
        X = np.array(X, dtype=np.float64)
        log_prior = np.log(self.class_prior_)

        # log P(X|y) = sum(xi * log P(feature_i|y))
        log_likelihood = X @ self.feature_log_prob_.T

        log_posterior = log_prior + log_likelihood
        log_sum = np.max(log_posterior, axis=1, keepdims=True) + \
                  np.log(np.sum(np.exp(log_posterior - np.max(log_posterior, axis=1, keepdims=True)),
                               axis=1, keepdims=True))

        return log_posterior - log_sum

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return np.exp(self.predict_log_proba(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        log_proba = self.predict_log_proba(X)
        indices = np.argmax(log_proba, axis=1)
        return self.classes_[indices]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy."""
        return np.mean(self.predict(X) == y)


# =============================================================================
# Test functions
# =============================================================================

def test_gaussian_nb():
    """Test GaussianNB."""
    print("Testing GaussianNB...")

    np.random.seed(42)

    # Generate Gaussian data
    n_samples = 400

    # Class 0: No blink (different mean/variance)
    X0 = np.random.randn(n_samples // 2, 10) * 0.5
    X0[:, 4:6] += 0.3
    y0 = np.zeros(n_samples // 2)

    # Class 1: Blink
    X1 = np.random.randn(n_samples // 2, 10) * 0.5
    X1[:, 4:6] -= 0.5
    y1 = np.ones(n_samples // 2)

    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])

    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    X_train, X_test = X[:320], X[320:]
    y_train, y_test = y[:320], y[320:]

    # Train
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    print(f"Train Accuracy: {gnb.score(X_train, y_train):.4f}")
    print(f"Test Accuracy: {gnb.score(X_test, y_test):.4f}")

    # Check probabilities sum to 1
    proba = gnb.predict_proba(X_test[:5])
    print(f"Probability sums: {np.sum(proba, axis=1)}")


def test_multiclass():
    """Test with 3 classes."""
    print("\nTesting multi-class GaussianNB...")

    np.random.seed(42)

    # 3 well-separated classes
    X0 = np.random.randn(100, 6) + [0, 0, 0, 0, 0, 0]
    X1 = np.random.randn(100, 6) + [3, 0, 0, 0, 0, 0]
    X2 = np.random.randn(100, 6) + [0, 3, 0, 0, 0, 0]

    X = np.vstack([X0, X1, X2])
    y = np.array([0]*100 + [1]*100 + [2]*100)

    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    X_train, X_test = X[:240], X[240:]
    y_train, y_test = y[:240], y[240:]

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    print(f"Train Accuracy: {gnb.score(X_train, y_train):.4f}")
    print(f"Test Accuracy: {gnb.score(X_test, y_test):.4f}")
    print(f"Class priors: {gnb.class_prior_}")


def test_multinomial_nb():
    """Test MultinomialNB on count data."""
    print("\nTesting MultinomialNB...")

    np.random.seed(42)

    # Simulate count data
    X0 = np.random.poisson(5, (100, 10))
    X1 = np.random.poisson(10, (100, 10))

    X = np.vstack([X0, X1]).astype(float)
    y = np.array([0]*100 + [1]*100)

    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    X_train, X_test = X[:160], X[160:]
    y_train, y_test = y[:160], y[160:]

    mnb = MultinomialNB(alpha=1.0)
    mnb.fit(X_train, y_train)

    print(f"Train Accuracy: {mnb.score(X_train, y_train):.4f}")
    print(f"Test Accuracy: {mnb.score(X_test, y_test):.4f}")


if __name__ == "__main__":
    test_gaussian_nb()
    test_multiclass()
    test_multinomial_nb()
