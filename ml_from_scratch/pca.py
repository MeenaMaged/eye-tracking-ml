"""
Principal Component Analysis (PCA) from scratch.

Implements:
- Eigendecomposition of covariance matrix
- Variance explained ratio
- Dimensionality reduction
- Feature reconstruction

PCA finds orthogonal directions of maximum variance in the data.
"""

import numpy as np
from typing import Optional, Tuple, Union


class PCA:
    """
    Principal Component Analysis for dimensionality reduction.

    PCA finds the principal components (eigenvectors) of the data
    covariance matrix and projects data onto these components.

    Algorithm:
    1. Center the data (subtract mean)
    2. Compute covariance matrix: C = (1/n) * X^T * X
    3. Eigendecomposition: C = V * D * V^T
    4. Sort eigenvectors by eigenvalues (descending)
    5. Project data: X_reduced = X * V[:, :k]

    Attributes:
        n_components: Number of components to keep
        components_: Principal components (eigenvectors)
        explained_variance_: Variance explained by each component
        explained_variance_ratio_: Fraction of variance explained
        mean_: Mean of training data (for centering)
    """

    def __init__(self, n_components: Optional[Union[int, float]] = None):
        """
        Initialize PCA.

        Args:
            n_components: Number of components to keep.
                - If int: exact number of components
                - If float (0-1): select components to explain that fraction of variance
                - If None: keep all components
        """
        self.n_components = n_components
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.n_features_: int = 0
        self.n_samples_: int = 0

    def fit(self, X: np.ndarray) -> 'PCA':
        """
        Fit PCA model to data.

        Args:
            X: Data matrix (n_samples, n_features)

        Returns:
            self
        """
        X = np.array(X, dtype=np.float64)
        self.n_samples_, self.n_features_ = X.shape

        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Compute covariance matrix
        # Using X^T X / (n-1) for unbiased estimate
        cov_matrix = np.cov(X_centered, rowvar=False)

        # Handle 1D case
        if cov_matrix.ndim == 0:
            cov_matrix = np.array([[cov_matrix]])

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Handle negative eigenvalues (numerical errors)
        eigenvalues = np.maximum(eigenvalues, 0)

        # Compute variance explained
        total_var = np.sum(eigenvalues)
        self.explained_variance_ = eigenvalues
        self.explained_variance_ratio_ = eigenvalues / total_var if total_var > 0 else eigenvalues

        # Determine number of components
        if self.n_components is None:
            n_components = self.n_features_
        elif isinstance(self.n_components, float):
            # Select components to explain this fraction of variance
            cumsum = np.cumsum(self.explained_variance_ratio_)
            n_components = np.searchsorted(cumsum, self.n_components) + 1
            n_components = min(n_components, self.n_features_)
        else:
            n_components = min(self.n_components, self.n_features_)

        self.n_components_ = n_components
        self.components_ = eigenvectors[:, :n_components].T  # (n_components, n_features)

        # Update explained variance to only selected components
        self.explained_variance_ = self.explained_variance_[:n_components]
        self.explained_variance_ratio_ = self.explained_variance_ratio_[:n_components]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project data onto principal components.

        Args:
            X: Data matrix (n_samples, n_features)

        Returns:
            Transformed data (n_samples, n_components)
        """
        if self.components_ is None:
            raise ValueError("PCA not fitted. Call fit() first.")

        X = np.array(X, dtype=np.float64)
        X_centered = X - self.mean_

        return X_centered @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit PCA and transform data in one step.

        Args:
            X: Data matrix

        Returns:
            Transformed data
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Reconstruct data from principal components.

        Args:
            X_transformed: Transformed data (n_samples, n_components)

        Returns:
            Reconstructed data (n_samples, n_features)
        """
        if self.components_ is None:
            raise ValueError("PCA not fitted. Call fit() first.")

        return X_transformed @ self.components_ + self.mean_

    def get_covariance(self) -> np.ndarray:
        """
        Compute covariance matrix from components.

        Returns:
            Covariance matrix (n_features, n_features)
        """
        if self.components_ is None:
            raise ValueError("PCA not fitted. Call fit() first.")

        return self.components_.T @ np.diag(self.explained_variance_) @ self.components_

    def score(self, X: np.ndarray) -> float:
        """
        Compute reconstruction error (negative MSE).

        Higher is better (sklearn convention).

        Args:
            X: Data matrix

        Returns:
            Negative mean squared reconstruction error
        """
        X_transformed = self.transform(X)
        X_reconstructed = self.inverse_transform(X_transformed)
        mse = np.mean((X - X_reconstructed) ** 2)
        return -mse

    def get_feature_importance(self) -> np.ndarray:
        """
        Compute feature importance based on component loadings.

        Returns:
            Feature importance scores (n_features,)
        """
        if self.components_ is None:
            raise ValueError("PCA not fitted. Call fit() first.")

        # Sum of squared loadings weighted by explained variance
        importance = np.sum(
            self.components_ ** 2 * self.explained_variance_ratio_.reshape(-1, 1),
            axis=0
        )
        return importance / np.sum(importance)


class IncrementalPCA:
    """
    Incremental PCA for large datasets that don't fit in memory.

    Uses SVD updates to compute principal components in batches.
    """

    def __init__(self, n_components: int, batch_size: int = 100):
        """
        Initialize Incremental PCA.

        Args:
            n_components: Number of components to keep
            batch_size: Number of samples per batch
        """
        self.n_components = n_components
        self.batch_size = batch_size
        self.components_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.var_: Optional[np.ndarray] = None
        self.n_samples_seen_: int = 0
        self.explained_variance_: Optional[np.ndarray] = None
        self.singular_values_: Optional[np.ndarray] = None

    def partial_fit(self, X: np.ndarray) -> 'IncrementalPCA':
        """
        Fit model to a batch of data.

        Args:
            X: Batch of data (batch_size, n_features)

        Returns:
            self
        """
        X = np.array(X, dtype=np.float64)
        n_samples, n_features = X.shape

        # Initialize on first batch
        if self.mean_ is None:
            self.mean_ = np.zeros(n_features)
            self.var_ = np.zeros(n_features)
            self.components_ = np.zeros((self.n_components, n_features))
            self.singular_values_ = np.zeros(self.n_components)

        # Update mean and variance (Welford's algorithm)
        col_mean = np.mean(X, axis=0)
        col_var = np.var(X, axis=0)

        # Combined statistics
        total_samples = self.n_samples_seen_ + n_samples
        new_mean = (self.n_samples_seen_ * self.mean_ + n_samples * col_mean) / total_samples

        # Update variance
        self.var_ = (self.n_samples_seen_ * (self.var_ + (self.mean_ - new_mean) ** 2) +
                     n_samples * (col_var + (col_mean - new_mean) ** 2)) / total_samples

        self.mean_ = new_mean
        self.n_samples_seen_ = total_samples

        # Center batch data
        X_centered = X - self.mean_

        # SVD of batch
        if self.n_samples_seen_ == n_samples:
            # First batch - regular SVD
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            n_comp = min(self.n_components, len(S))
            self.components_ = Vt[:n_comp]
            self.singular_values_ = S[:n_comp]
        else:
            # Incremental update using previous components
            # Append new data to existing components
            X_combined = np.vstack([
                self.singular_values_.reshape(-1, 1) * self.components_,
                X_centered
            ])
            U, S, Vt = np.linalg.svd(X_combined, full_matrices=False)
            n_comp = min(self.n_components, len(S))
            self.components_ = Vt[:n_comp]
            self.singular_values_ = S[:n_comp]

        # Compute explained variance
        self.explained_variance_ = self.singular_values_ ** 2 / (self.n_samples_seen_ - 1)

        return self

    def fit(self, X: np.ndarray) -> 'IncrementalPCA':
        """
        Fit model to data in batches.

        Args:
            X: Full data matrix

        Returns:
            self
        """
        X = np.array(X, dtype=np.float64)
        n_samples = len(X)

        for start in range(0, n_samples, self.batch_size):
            end = min(start + self.batch_size, n_samples)
            self.partial_fit(X[start:end])

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data."""
        X = np.array(X, dtype=np.float64)
        return (X - self.mean_) @ self.components_.T


def select_n_components(X: np.ndarray, variance_threshold: float = 0.95) -> Tuple[int, np.ndarray]:
    """
    Select number of PCA components to explain given variance.

    Args:
        X: Data matrix
        variance_threshold: Fraction of variance to explain (0-1)

    Returns:
        Tuple of (n_components, cumulative_variance_ratio)
    """
    pca = PCA()
    pca.fit(X)

    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.searchsorted(cumsum, variance_threshold) + 1

    return min(n_components, X.shape[1]), cumsum


# =============================================================================
# Test functions
# =============================================================================

def test_pca():
    """Test PCA implementation."""
    print("Testing PCA...")

    np.random.seed(42)

    # Create correlated data
    n_samples = 500
    n_features = 10

    # True components
    cov = np.eye(n_features)
    cov[0, 1] = cov[1, 0] = 0.8
    cov[2, 3] = cov[3, 2] = 0.7

    X = np.random.multivariate_normal(np.zeros(n_features), cov, n_samples)

    # Fit PCA
    pca = PCA(n_components=5)
    X_transformed = pca.fit_transform(X)

    print(f"Original shape: {X.shape}")
    print(f"Transformed shape: {X_transformed.shape}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {np.sum(pca.explained_variance_ratio_):.4f}")

    # Test reconstruction
    X_reconstructed = pca.inverse_transform(X_transformed)
    mse = np.mean((X - X_reconstructed) ** 2)
    print(f"Reconstruction MSE: {mse:.6f}")

    # Test with variance threshold
    pca95 = PCA(n_components=0.95)
    pca95.fit(X)
    print(f"\nPCA (95% variance): {pca95.n_components_} components")
    print(f"Variance explained: {np.sum(pca95.explained_variance_ratio_):.4f}")


def test_pca_classification():
    """Test PCA as preprocessing for classification."""
    print("\nTesting PCA for classification...")

    np.random.seed(42)

    # Generate classification data
    n_samples = 300
    n_features = 20

    # Class 0
    X0 = np.random.randn(n_samples // 2, n_features)
    X0[:, :5] += 2

    # Class 1
    X1 = np.random.randn(n_samples // 2, n_features)
    X1[:, :5] -= 2

    X = np.vstack([X0, X1])
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

    # Shuffle
    idx = np.random.permutation(n_samples)
    X, y = X[idx], y[idx]

    # Split
    X_train, X_test = X[:240], X[240:]
    y_train, y_test = y[:240], y[240:]

    # Without PCA
    from ml_from_scratch.knn import KNeighborsClassifier as KNN

    knn = KNN(n_neighbors=5)
    knn.fit(X_train, y_train)
    acc_full = knn.score(X_test, y_test)
    print(f"KNN with all {n_features} features: {acc_full:.4f}")

    # With PCA
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    knn_pca = KNN(n_neighbors=5)
    knn_pca.fit(X_train_pca, y_train)
    acc_pca = knn_pca.score(X_test_pca, y_test)
    print(f"KNN with PCA ({pca.n_components_} components, 95% var): {acc_pca:.4f}")


if __name__ == "__main__":
    test_pca()
    test_pca_classification()
