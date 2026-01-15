"""
K-Nearest Neighbors (KNN) Classifier from scratch.

Implements:
- Euclidean distance computation
- K-nearest neighbor voting
- Distance-weighted voting option
- Support for multi-class classification

KNN is a non-parametric, instance-based learning algorithm.
"""

import numpy as np
from typing import Optional, Literal


class KNeighborsClassifier:
    """
    K-Nearest Neighbors Classifier.

    Classifies samples based on the majority class among
    the k nearest training samples.

    Distance metric: Euclidean distance
    d(x, y) = sqrt(sum((x_i - y_i)^2))
    """

    def __init__(self,
                 n_neighbors: int = 5,
                 weights: Literal['uniform', 'distance'] = 'uniform',
                 p: int = 2):
        """
        Initialize KNN Classifier.

        Args:
            n_neighbors: Number of neighbors to use (k)
            weights: 'uniform' (equal vote) or 'distance' (weighted by 1/d)
            p: Power for Minkowski distance (2 = Euclidean, 1 = Manhattan)
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p

        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_classes: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNeighborsClassifier':
        """
        Store training data (lazy learning).

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)

        Returns:
            self
        """
        self.X_train = np.array(X, dtype=np.float64)
        self.y_train = np.array(y)
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        return self

    def _minkowski_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Minkowski distance between two points."""
        return np.sum(np.abs(x1 - x2) ** self.p) ** (1 / self.p)

    def _euclidean_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate Euclidean distances between X and all training samples.

        Uses efficient vectorized computation:
        ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b

        Args:
            X: Test samples (n_test, n_features)

        Returns:
            Distance matrix (n_test, n_train)
        """
        # ||x||^2 for test samples
        X_sq = np.sum(X ** 2, axis=1, keepdims=True)

        # ||x_train||^2 for training samples
        train_sq = np.sum(self.X_train ** 2, axis=1)

        # -2 * X . X_train^T
        cross = -2 * X @ self.X_train.T

        # ||x - x_train||^2 = ||x||^2 + ||x_train||^2 - 2*x.x_train
        distances_sq = X_sq + train_sq + cross

        # Handle numerical errors
        distances_sq = np.maximum(distances_sq, 0)

        return np.sqrt(distances_sq)

    def _get_k_nearest(self, distances: np.ndarray) -> tuple:
        """
        Get indices and distances of k nearest neighbors.

        Args:
            distances: Distance array (n_train,)

        Returns:
            Tuple of (neighbor_indices, neighbor_distances)
        """
        k = min(self.n_neighbors, len(distances))
        indices = np.argpartition(distances, k)[:k]
        indices = indices[np.argsort(distances[indices])]
        return indices, distances[indices]

    def _vote(self, neighbor_labels: np.ndarray,
              neighbor_distances: np.ndarray) -> np.ndarray:
        """
        Perform voting among neighbors.

        Args:
            neighbor_labels: Labels of k neighbors
            neighbor_distances: Distances to k neighbors

        Returns:
            Class probabilities (n_classes,)
        """
        votes = np.zeros(self.n_classes)

        if self.weights == 'uniform':
            # Equal vote for each neighbor
            for label in neighbor_labels:
                idx = np.where(self.classes_ == label)[0][0]
                votes[idx] += 1
        else:
            # Distance-weighted voting
            for label, dist in zip(neighbor_labels, neighbor_distances):
                idx = np.where(self.classes_ == label)[0][0]
                weight = 1 / (dist + 1e-10)  # Avoid division by zero
                votes[idx] += weight

        # Normalize to probabilities
        total = np.sum(votes)
        if total > 0:
            votes /= total

        return votes

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Test samples (n_test, n_features)

        Returns:
            Probability matrix (n_test, n_classes)
        """
        X = np.array(X, dtype=np.float64)

        if self.p == 2:
            distances = self._euclidean_distances(X)
        else:
            # Fall back to loop for non-Euclidean
            distances = np.array([[self._minkowski_distance(x, x_train)
                                   for x_train in self.X_train] for x in X])

        probas = []
        for i in range(len(X)):
            indices, dists = self._get_k_nearest(distances[i])
            neighbor_labels = self.y_train[indices]
            proba = self._vote(neighbor_labels, dists)
            probas.append(proba)

        return np.array(probas)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Test samples

        Returns:
            Predicted class labels
        """
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def kneighbors(self, X: np.ndarray, n_neighbors: Optional[int] = None,
                   return_distance: bool = True):
        """
        Find k neighbors for each sample.

        Args:
            X: Query samples
            n_neighbors: Number of neighbors (default: self.n_neighbors)
            return_distance: Whether to return distances

        Returns:
            If return_distance: (distances, indices)
            Else: indices
        """
        X = np.array(X, dtype=np.float64)
        k = n_neighbors or self.n_neighbors

        distances = self._euclidean_distances(X)

        all_indices = []
        all_distances = []

        for i in range(len(X)):
            indices = np.argpartition(distances[i], k)[:k]
            indices = indices[np.argsort(distances[i][indices])]
            all_indices.append(indices)
            all_distances.append(distances[i][indices])

        if return_distance:
            return np.array(all_distances), np.array(all_indices)
        return np.array(all_indices)


# =============================================================================
# Test functions
# =============================================================================

def test_knn():
    """Test KNeighborsClassifier."""
    print("Testing KNeighborsClassifier...")

    np.random.seed(42)

    # Generate synthetic blink data
    n_samples = 400

    X0 = np.random.randn(n_samples // 2, 10)
    X0[:, 4:6] += 0.3
    y0 = np.zeros(n_samples // 2)

    X1 = np.random.randn(n_samples // 2, 10)
    X1[:, 4:6] -= 0.5
    y1 = np.ones(n_samples // 2)

    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])

    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    X_train, X_test = X[:320], X[320:]
    y_train, y_test = y[:320], y[320:]

    # Test different k values
    for k in [1, 3, 5, 7]:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        print(f"k={k}: Train Acc={knn.score(X_train, y_train):.4f}, "
              f"Test Acc={knn.score(X_test, y_test):.4f}")

    # Test distance weighting
    print("\nWith distance weighting:")
    knn_dist = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn_dist.fit(X_train, y_train)
    print(f"k=5 (weighted): Test Acc={knn_dist.score(X_test, y_test):.4f}")


def test_multiclass():
    """Test with 3 classes."""
    print("\nTesting multi-class KNN...")

    np.random.seed(42)

    X0 = np.random.randn(100, 6)
    X1 = np.random.randn(100, 6) + [3, 3, 0, 0, 0, 0]
    X2 = np.random.randn(100, 6) + [0, 0, 3, 3, 0, 0]

    X = np.vstack([X0, X1, X2])
    y = np.array([0]*100 + [1]*100 + [2]*100)

    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    X_train, X_test = X[:240], X[240:]
    y_train, y_test = y[:240], y[240:]

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    print(f"Train Accuracy: {knn.score(X_train, y_train):.4f}")
    print(f"Test Accuracy: {knn.score(X_test, y_test):.4f}")


if __name__ == "__main__":
    test_knn()
    test_multiclass()
