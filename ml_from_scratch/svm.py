"""
Support Vector Machine (SVM) Classifier from scratch.

Implements:
- Linear and RBF (Gaussian) kernels
- Sequential Minimal Optimization (SMO) algorithm
- Soft-margin SVM with C parameter
- Multi-class via One-vs-Rest

SVM finds the optimal hyperplane that maximizes margin between classes.
"""

import numpy as np
from typing import Optional, Literal, Tuple


class SVC:
    """
    Support Vector Classifier.

    Uses SMO (Sequential Minimal Optimization) algorithm for training.
    Supports linear and RBF kernels.

    Primal problem:
        min 1/2 ||w||^2 + C * sum(max(0, 1 - y_i * (w.x_i + b)))

    Dual problem (with kernel):
        max sum(alpha) - 1/2 * sum(alpha_i * alpha_j * y_i * y_j * K(x_i, x_j))
        s.t. 0 <= alpha_i <= C, sum(alpha_i * y_i) = 0
    """

    def __init__(self,
                 C: float = 1.0,
                 kernel: Literal['linear', 'rbf'] = 'rbf',
                 gamma: float = 'scale',
                 tol: float = 1e-3,
                 max_iter: int = 1000,
                 random_state: int = 42):
        """
        Initialize SVC.

        Args:
            C: Regularization parameter (soft margin)
            kernel: 'linear' or 'rbf' (Gaussian)
            gamma: RBF kernel coefficient ('scale' = 1/(n_features * X.var()))
            tol: Tolerance for stopping criterion
            max_iter: Maximum number of iterations
            random_state: Random seed
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state

        self.alpha: Optional[np.ndarray] = None
        self.b: float = 0.0
        self.support_vectors_: Optional[np.ndarray] = None
        self.support_labels_: Optional[np.ndarray] = None
        self.support_alpha_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None
        self._gamma: float = 1.0

        # For multi-class
        self.classifiers: list = []

    def _compute_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute kernel matrix K(X1, X2)."""
        if self.kernel == 'linear':
            return X1 @ X2.T
        elif self.kernel == 'rbf':
            # K(x, y) = exp(-gamma * ||x - y||^2)
            # ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x.y
            X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
            X2_sq = np.sum(X2 ** 2, axis=1)
            distances_sq = X1_sq + X2_sq - 2 * X1 @ X2.T
            distances_sq = np.maximum(distances_sq, 0)
            return np.exp(-self._gamma * distances_sq)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def _smo_step(self, i: int, j: int, K: np.ndarray, y: np.ndarray,
                  E: np.ndarray) -> bool:
        """
        Perform one step of SMO algorithm.

        Updates alpha[i] and alpha[j] while maintaining constraints.
        """
        if i == j:
            return False

        alpha_i, alpha_j = self.alpha[i], self.alpha[j]
        y_i, y_j = y[i], y[j]
        E_i, E_j = E[i], E[j]

        # Compute bounds for alpha[j]
        if y_i != y_j:
            L = max(0, alpha_j - alpha_i)
            H = min(self.C, self.C + alpha_j - alpha_i)
        else:
            L = max(0, alpha_i + alpha_j - self.C)
            H = min(self.C, alpha_i + alpha_j)

        if L >= H:
            return False

        # Compute eta (second derivative of objective)
        eta = 2 * K[i, j] - K[i, i] - K[j, j]
        if eta >= 0:
            return False

        # Update alpha[j]
        alpha_j_new = alpha_j - y_j * (E_i - E_j) / eta
        alpha_j_new = np.clip(alpha_j_new, L, H)

        if abs(alpha_j_new - alpha_j) < 1e-8:
            return False

        # Update alpha[i]
        alpha_i_new = alpha_i + y_i * y_j * (alpha_j - alpha_j_new)

        # Update bias
        b1 = self.b - E_i - y_i * (alpha_i_new - alpha_i) * K[i, i] \
             - y_j * (alpha_j_new - alpha_j) * K[i, j]
        b2 = self.b - E_j - y_i * (alpha_i_new - alpha_i) * K[i, j] \
             - y_j * (alpha_j_new - alpha_j) * K[j, j]

        if 0 < alpha_i_new < self.C:
            self.b = b1
        elif 0 < alpha_j_new < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2

        self.alpha[i] = alpha_i_new
        self.alpha[j] = alpha_j_new

        return True

    def _fit_binary(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit binary SVM using SMO algorithm."""
        n_samples = len(X)

        # Set gamma for RBF kernel
        if self.gamma == 'scale':
            self._gamma = 1.0 / (X.shape[1] * X.var()) if X.var() > 0 else 1.0
        else:
            self._gamma = self.gamma

        # Compute kernel matrix
        K = self._compute_kernel(X, X)

        # Initialize alphas
        self.alpha = np.zeros(n_samples)
        self.b = 0.0

        # SMO main loop
        np.random.seed(self.random_state)
        passes = 0
        max_passes = self.max_iter

        while passes < max_passes:
            num_changed = 0

            for i in range(n_samples):
                # Compute error E_i
                E_i = np.sum(self.alpha * y * K[:, i]) + self.b - y[i]

                # Check KKT conditions
                if (y[i] * E_i < -self.tol and self.alpha[i] < self.C) or \
                   (y[i] * E_i > self.tol and self.alpha[i] > 0):

                    # Select j randomly
                    j = i
                    while j == i:
                        j = np.random.randint(n_samples)

                    # Compute all errors for selection
                    E = np.array([np.sum(self.alpha * y * K[:, k]) + self.b - y[k]
                                  for k in range(n_samples)])

                    if self._smo_step(i, j, K, y, E):
                        num_changed += 1

            if num_changed == 0:
                passes += 1
            else:
                passes = 0

        # Store support vectors
        sv_mask = self.alpha > 1e-7
        self.support_vectors_ = X[sv_mask]
        self.support_labels_ = y[sv_mask]
        self.support_alpha_ = self.alpha[sv_mask]

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVC':
        """
        Fit SVM model.

        Uses One-vs-Rest for multi-class classification.

        Args:
            X: Training features
            y: Training labels

        Returns:
            self
        """
        X = np.array(X, dtype=np.float64)
        y = np.array(y)

        self.classes_ = np.unique(y)

        if len(self.classes_) == 2:
            # Binary classification
            y_binary = np.where(y == self.classes_[1], 1, -1)
            self._fit_binary(X, y_binary)
            self._X_train = X
            self._y_train = y_binary
        else:
            # Multi-class: One-vs-Rest
            self.classifiers = []
            for cls in self.classes_:
                y_binary = np.where(y == cls, 1, -1)

                clf = SVC(
                    C=self.C,
                    kernel=self.kernel,
                    gamma=self.gamma,
                    tol=self.tol,
                    max_iter=self.max_iter,
                    random_state=self.random_state
                )
                clf._fit_binary(X, y_binary)
                clf._X_train = X
                clf._y_train = y_binary
                self.classifiers.append(clf)

        return self

    def _decision_function_binary(self, X: np.ndarray) -> np.ndarray:
        """Compute decision function for binary classification."""
        K = self._compute_kernel(X, self.support_vectors_)
        return np.sum(self.support_alpha_ * self.support_labels_ * K, axis=1) + self.b

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute decision function."""
        X = np.array(X, dtype=np.float64)

        if len(self.classes_) == 2:
            return self._decision_function_binary(X)
        else:
            # One-vs-Rest: return scores for each class
            scores = np.zeros((len(X), len(self.classes_)))
            for i, clf in enumerate(self.classifiers):
                scores[:, i] = clf._decision_function_binary(X)
            return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        decision = self.decision_function(X)

        if len(self.classes_) == 2:
            indices = (decision > 0).astype(int)
        else:
            indices = np.argmax(decision, axis=1)

        return self.classes_[indices]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


# =============================================================================
# Test functions
# =============================================================================

def test_svm_binary():
    """Test SVM on binary classification."""
    print("Testing SVM (Binary)...")

    np.random.seed(42)

    # Generate linearly separable data
    n_samples = 200
    X0 = np.random.randn(n_samples // 2, 4) + [-2, -2, 0, 0]
    X1 = np.random.randn(n_samples // 2, 4) + [2, 2, 0, 0]

    X = np.vstack([X0, X1])
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    X_train, X_test = X[:160], X[160:]
    y_train, y_test = y[:160], y[160:]

    # Test linear kernel
    svm_linear = SVC(kernel='linear', C=1.0, max_iter=100)
    svm_linear.fit(X_train, y_train)
    print(f"Linear: Train Acc={svm_linear.score(X_train, y_train):.4f}, "
          f"Test Acc={svm_linear.score(X_test, y_test):.4f}")

    # Test RBF kernel
    svm_rbf = SVC(kernel='rbf', C=1.0, max_iter=100)
    svm_rbf.fit(X_train, y_train)
    print(f"RBF: Train Acc={svm_rbf.score(X_train, y_train):.4f}, "
          f"Test Acc={svm_rbf.score(X_test, y_test):.4f}")
    print(f"Support vectors: {len(svm_rbf.support_vectors_)}")


def test_svm_nonlinear():
    """Test SVM on non-linearly separable data."""
    print("\nTesting SVM on non-linear data...")

    np.random.seed(42)

    # XOR-like pattern
    n = 100
    X0 = np.random.randn(n, 2) + [1, 1]
    X1 = np.random.randn(n, 2) + [-1, -1]
    X2 = np.random.randn(n, 2) + [1, -1]
    X3 = np.random.randn(n, 2) + [-1, 1]

    X = np.vstack([X0, X1, X2, X3])
    y = np.array([0]*n + [0]*n + [1]*n + [1]*n)

    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    X_train, X_test = X[:320], X[320:]
    y_train, y_test = y[:320], y[320:]

    svm = SVC(kernel='rbf', C=10.0, gamma=0.5, max_iter=200)
    svm.fit(X_train, y_train)

    print(f"RBF: Train Acc={svm.score(X_train, y_train):.4f}, "
          f"Test Acc={svm.score(X_test, y_test):.4f}")


def test_svm_multiclass():
    """Test SVM on multi-class data."""
    print("\nTesting SVM (Multi-class)...")

    np.random.seed(42)

    X0 = np.random.randn(80, 4) + [0, 0, 0, 0]
    X1 = np.random.randn(80, 4) + [3, 0, 0, 0]
    X2 = np.random.randn(80, 4) + [0, 3, 0, 0]

    X = np.vstack([X0, X1, X2])
    y = np.array([0]*80 + [1]*80 + [2]*80)

    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    X_train, X_test = X[:200], X[200:]
    y_train, y_test = y[:200], y[200:]

    svm = SVC(kernel='rbf', C=1.0, max_iter=100)
    svm.fit(X_train, y_train)

    print(f"Train Accuracy: {svm.score(X_train, y_train):.4f}")
    print(f"Test Accuracy: {svm.score(X_test, y_test):.4f}")


if __name__ == "__main__":
    test_svm_binary()
    test_svm_nonlinear()
    test_svm_multiclass()
