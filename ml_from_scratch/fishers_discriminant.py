"""
Fisher's Linear Discriminant Analysis (LDA) from scratch.

Implements:
- Binary Fisher's Linear Discriminant
- Multi-class LDA (generalized eigenvalue problem)
- Dimensionality reduction for classification
- Class-conditional Gaussian classification

Fisher's LDA finds projections that maximize class separation.
"""

import numpy as np
from typing import Optional, Tuple, List


class FishersLinearDiscriminant:
    """
    Fisher's Linear Discriminant for binary classification.

    Finds the projection w that maximizes:
        J(w) = (w^T * S_B * w) / (w^T * S_W * w)

    Where:
        S_B = between-class scatter matrix
        S_W = within-class scatter matrix

    Solution: w = S_W^(-1) * (mu_1 - mu_0)

    Attributes:
        w_: Projection vector
        threshold_: Classification threshold
        classes_: Class labels
    """

    def __init__(self):
        """Initialize Fisher's Discriminant."""
        self.w_: Optional[np.ndarray] = None
        self.threshold_: float = 0.0
        self.classes_: Optional[np.ndarray] = None
        self.mean_0_: Optional[np.ndarray] = None
        self.mean_1_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'FishersLinearDiscriminant':
        """
        Fit Fisher's discriminant.

        Args:
            X: Features (n_samples, n_features)
            y: Binary labels (n_samples,)

        Returns:
            self
        """
        X = np.array(X, dtype=np.float64)
        y = np.array(y)

        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("Fisher's discriminant requires exactly 2 classes")

        # Split by class
        X0 = X[y == self.classes_[0]]
        X1 = X[y == self.classes_[1]]

        # Class means
        self.mean_0_ = np.mean(X0, axis=0)
        self.mean_1_ = np.mean(X1, axis=0)

        # Within-class scatter matrix
        S_W = np.cov(X0.T) * (len(X0) - 1) + np.cov(X1.T) * (len(X1) - 1)

        # Handle 1D case
        if S_W.ndim == 0:
            S_W = np.array([[S_W]])

        # Regularize for numerical stability
        S_W += np.eye(S_W.shape[0]) * 1e-6

        # Fisher's solution: w = S_W^(-1) * (mu_1 - mu_0)
        mean_diff = self.mean_1_ - self.mean_0_
        self.w_ = np.linalg.solve(S_W, mean_diff)

        # Normalize
        self.w_ = self.w_ / np.linalg.norm(self.w_)

        # Compute threshold (midpoint of projected means)
        proj_mean_0 = np.dot(self.mean_0_, self.w_)
        proj_mean_1 = np.dot(self.mean_1_, self.w_)
        self.threshold_ = (proj_mean_0 + proj_mean_1) / 2

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project data onto discriminant direction.

        Args:
            X: Features

        Returns:
            Projected values (n_samples,)
        """
        if self.w_ is None:
            raise ValueError("Not fitted. Call fit() first.")

        X = np.array(X, dtype=np.float64)
        return X @ self.w_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Features

        Returns:
            Predicted labels
        """
        projections = self.transform(X)

        # Class 1 if projection > threshold (assuming mean_1 > mean_0 after projection)
        proj_mean_0 = np.dot(self.mean_0_, self.w_)
        proj_mean_1 = np.dot(self.mean_1_, self.w_)

        if proj_mean_1 > proj_mean_0:
            predictions = (projections > self.threshold_).astype(int)
        else:
            predictions = (projections < self.threshold_).astype(int)

        return self.classes_[predictions]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy."""
        return np.mean(self.predict(X) == y)

    def get_discriminant_ratio(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute Fisher's criterion (between/within variance ratio).

        Args:
            X: Features
            y: Labels

        Returns:
            J(w) = S_B / S_W for the learned projection
        """
        projections = self.transform(X)

        proj_0 = projections[y == self.classes_[0]]
        proj_1 = projections[y == self.classes_[1]]

        # Between-class variance
        overall_mean = np.mean(projections)
        s_b = len(proj_0) * (np.mean(proj_0) - overall_mean) ** 2 + \
              len(proj_1) * (np.mean(proj_1) - overall_mean) ** 2

        # Within-class variance
        s_w = np.sum((proj_0 - np.mean(proj_0)) ** 2) + \
              np.sum((proj_1 - np.mean(proj_1)) ** 2)

        return s_b / (s_w + 1e-10)


class LinearDiscriminantAnalysis:
    """
    Multi-class Linear Discriminant Analysis.

    Finds projections that maximize between-class variance while
    minimizing within-class variance.

    For C classes and D features, LDA finds at most min(C-1, D) components.

    Solves: S_W^(-1) * S_B * w = lambda * w

    Attributes:
        components_: LDA components (n_components, n_features)
        explained_variance_ratio_: Variance explained by each component
        means_: Class means
        priors_: Class prior probabilities
        classes_: Unique class labels
    """

    def __init__(self, n_components: Optional[int] = None):
        """
        Initialize LDA.

        Args:
            n_components: Number of components (default: n_classes - 1)
        """
        self.n_components = n_components
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.means_: Optional[np.ndarray] = None
        self.overall_mean_: Optional[np.ndarray] = None
        self.priors_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None
        self.covariance_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearDiscriminantAnalysis':
        """
        Fit LDA model.

        Args:
            X: Features (n_samples, n_features)
            y: Class labels (n_samples,)

        Returns:
            self
        """
        X = np.array(X, dtype=np.float64)
        y = np.array(y)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape

        # Maximum components
        max_components = min(n_classes - 1, n_features)
        n_components = self.n_components or max_components
        n_components = min(n_components, max_components)

        # Overall mean
        self.overall_mean_ = np.mean(X, axis=0)

        # Class statistics
        self.means_ = np.zeros((n_classes, n_features))
        self.priors_ = np.zeros(n_classes)

        # Within-class scatter matrix
        S_W = np.zeros((n_features, n_features))

        for i, cls in enumerate(self.classes_):
            X_cls = X[y == cls]
            self.means_[i] = np.mean(X_cls, axis=0)
            self.priors_[i] = len(X_cls) / n_samples

            # Add to within-class scatter
            X_centered = X_cls - self.means_[i]
            S_W += X_centered.T @ X_centered

        # Between-class scatter matrix
        S_B = np.zeros((n_features, n_features))
        for i, cls in enumerate(self.classes_):
            n_cls = np.sum(y == cls)
            mean_diff = (self.means_[i] - self.overall_mean_).reshape(-1, 1)
            S_B += n_cls * (mean_diff @ mean_diff.T)

        # Regularize S_W for numerical stability
        S_W += np.eye(n_features) * 1e-6

        # Solve generalized eigenvalue problem: S_B * w = lambda * S_W * w
        # Equivalent to: S_W^(-1) * S_B * w = lambda * w
        S_W_inv = np.linalg.inv(S_W)
        A = S_W_inv @ S_B

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eig(A)

        # Sort by eigenvalue (descending)
        # Take real parts (eigenvalues should be real for symmetric matrices)
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Keep positive eigenvalues
        positive_mask = eigenvalues > 1e-10
        n_positive = np.sum(positive_mask)
        n_components = min(n_components, n_positive)

        # Store components
        self.components_ = eigenvectors[:, :n_components].T
        self.n_components_ = n_components

        # Explained variance ratio
        total_var = np.sum(np.abs(eigenvalues[positive_mask]))
        self.explained_variance_ratio_ = np.abs(eigenvalues[:n_components]) / total_var

        # Store pooled covariance for prediction
        self.covariance_ = S_W / n_samples

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project data onto LDA components.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Transformed data (n_samples, n_components)
        """
        if self.components_ is None:
            raise ValueError("LDA not fitted. Call fit() first.")

        X = np.array(X, dtype=np.float64)
        return (X - self.overall_mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit LDA and transform data."""
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels using Gaussian discriminant analysis.

        Uses the learned class means and shared covariance.

        Args:
            X: Features

        Returns:
            Predicted labels
        """
        if self.means_ is None:
            raise ValueError("LDA not fitted. Call fit() first.")

        X = np.array(X, dtype=np.float64)
        n_samples = len(X)

        # Compute discriminant function for each class
        # g_k(x) = log(prior_k) - 0.5 * (x - mu_k)^T * Sigma^(-1) * (x - mu_k)
        # For shared covariance, this simplifies to linear discriminant

        scores = np.zeros((n_samples, len(self.classes_)))

        # Inverse covariance (regularized)
        cov_reg = self.covariance_ + np.eye(self.covariance_.shape[0]) * 1e-6
        cov_inv = np.linalg.inv(cov_reg)

        for k, cls in enumerate(self.classes_):
            # Linear discriminant
            w_k = cov_inv @ self.means_[k]
            b_k = -0.5 * self.means_[k] @ cov_inv @ self.means_[k] + np.log(self.priors_[k])
            scores[:, k] = X @ w_k + b_k

        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using softmax.

        Args:
            X: Features

        Returns:
            Probability matrix (n_samples, n_classes)
        """
        if self.means_ is None:
            raise ValueError("LDA not fitted. Call fit() first.")

        X = np.array(X, dtype=np.float64)
        n_samples = len(X)

        scores = np.zeros((n_samples, len(self.classes_)))
        cov_reg = self.covariance_ + np.eye(self.covariance_.shape[0]) * 1e-6
        cov_inv = np.linalg.inv(cov_reg)

        for k, cls in enumerate(self.classes_):
            w_k = cov_inv @ self.means_[k]
            b_k = -0.5 * self.means_[k] @ cov_inv @ self.means_[k] + np.log(self.priors_[k])
            scores[:, k] = X @ w_k + b_k

        # Softmax
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        proba = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return proba

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy."""
        return np.mean(self.predict(X) == y)

    def get_feature_importance(self) -> np.ndarray:
        """
        Compute feature importance from LDA components.

        Returns:
            Feature importance scores
        """
        if self.components_ is None:
            raise ValueError("LDA not fitted. Call fit() first.")

        # Sum of absolute loadings weighted by explained variance
        importance = np.sum(
            np.abs(self.components_) * self.explained_variance_ratio_.reshape(-1, 1),
            axis=0
        )
        return importance / np.sum(importance)


# =============================================================================
# Test functions
# =============================================================================

def test_fisher_binary():
    """Test Fisher's discriminant on binary classification."""
    print("Testing Fisher's Linear Discriminant (Binary)...")

    np.random.seed(42)

    # Generate two-class data
    n_samples = 200
    n_features = 10

    # Class 0
    X0 = np.random.randn(n_samples // 2, n_features)
    X0[:, 0:2] -= 2

    # Class 1
    X1 = np.random.randn(n_samples // 2, n_features)
    X1[:, 0:2] += 2

    X = np.vstack([X0, X1])
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

    # Shuffle
    idx = np.random.permutation(n_samples)
    X, y = X[idx], y[idx]

    # Split
    X_train, X_test = X[:160], X[160:]
    y_train, y_test = y[:160], y[160:]

    # Fit Fisher's
    fisher = FishersLinearDiscriminant()
    fisher.fit(X_train, y_train)

    print(f"Train accuracy: {fisher.score(X_train, y_train):.4f}")
    print(f"Test accuracy: {fisher.score(X_test, y_test):.4f}")
    print(f"Discriminant ratio: {fisher.get_discriminant_ratio(X_train, y_train):.4f}")


def test_lda_multiclass():
    """Test LDA on multi-class classification."""
    print("\nTesting Linear Discriminant Analysis (Multi-class)...")

    np.random.seed(42)

    # Generate three-class data
    n_samples = 300
    n_features = 10

    X0 = np.random.randn(100, n_features)
    X1 = np.random.randn(100, n_features) + [3, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    X2 = np.random.randn(100, n_features) + [0, 3, 0, 0, 0, 0, 0, 0, 0, 0]

    X = np.vstack([X0, X1, X2])
    y = np.array([0] * 100 + [1] * 100 + [2] * 100)

    idx = np.random.permutation(n_samples)
    X, y = X[idx], y[idx]

    X_train, X_test = X[:240], X[240:]
    y_train, y_test = y[:240], y[240:]

    # Fit LDA
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(X_train, y_train)

    print(f"Components: {lda.n_components_}")
    print(f"Explained variance ratio: {lda.explained_variance_ratio_}")
    print(f"Train accuracy: {lda.score(X_train, y_train):.4f}")
    print(f"Test accuracy: {lda.score(X_test, y_test):.4f}")

    # Transform for visualization
    X_lda = lda.transform(X_train)
    print(f"Transformed shape: {X_lda.shape}")


def test_lda_vs_pca():
    """Compare LDA vs PCA for classification."""
    print("\nComparing LDA vs PCA for Classification...")

    np.random.seed(42)

    # Data where classes differ mainly in one direction
    n_samples = 300
    n_features = 20

    # Class 0: centered at origin with high variance in features 5-20
    X0 = np.random.randn(100, n_features)
    X0[:, 5:] *= 3  # High variance in irrelevant features

    # Class 1: shifted in feature 0
    X1 = np.random.randn(100, n_features)
    X1[:, 0] += 3
    X1[:, 5:] *= 3

    # Class 2: shifted in feature 1
    X2 = np.random.randn(100, n_features)
    X2[:, 1] += 3
    X2[:, 5:] *= 3

    X = np.vstack([X0, X1, X2])
    y = np.array([0] * 100 + [1] * 100 + [2] * 100)

    idx = np.random.permutation(n_samples)
    X, y = X[idx], y[idx]

    X_train, X_test = X[:240], X[240:]
    y_train, y_test = y[:240], y[240:]

    from ml_from_scratch.pca import PCA
    from ml_from_scratch.knn import KNeighborsClassifier as KNN

    # PCA: finds directions of maximum variance (irrelevant features!)
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    knn_pca = KNN(n_neighbors=5)
    knn_pca.fit(X_train_pca, y_train)
    acc_pca = knn_pca.score(X_test_pca, y_test)

    # LDA: finds directions of maximum class separation
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)

    knn_lda = KNN(n_neighbors=5)
    knn_lda.fit(X_train_lda, y_train)
    acc_lda = knn_lda.score(X_test_lda, y_test)

    print(f"KNN with PCA (2 components): {acc_pca:.4f}")
    print(f"KNN with LDA (2 components): {acc_lda:.4f}")
    print(f"LDA directly (classifier): {lda.score(X_test, y_test):.4f}")


if __name__ == "__main__":
    test_fisher_binary()
    test_lda_multiclass()
    test_lda_vs_pca()
