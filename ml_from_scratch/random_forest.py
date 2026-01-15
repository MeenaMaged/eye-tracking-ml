"""
Random Forest Classifier from scratch.

Implements:
- Bagging (Bootstrap AGGregatING)
- Random feature selection at each split
- Ensemble voting for predictions
- Out-of-bag (OOB) error estimation

Random Forest = Bagging + Random Feature Selection + Decision Trees
"""

import numpy as np
from typing import List, Optional
from .decision_tree import DecisionTreeClassifier


class RandomForestClassifier:
    """
    Random Forest Classifier.

    Ensemble method combining multiple decision trees trained on:
    1. Bootstrap samples (bagging)
    2. Random subsets of features at each split

    This reduces overfitting and improves generalization.
    """

    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 10,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: str = 'sqrt',
                 criterion: str = 'entropy',
                 bootstrap: bool = True,
                 oob_score: bool = False,
                 random_state: int = 42,
                 n_jobs: int = 1):
        """
        Initialize Random Forest.

        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of each tree
            min_samples_split: Minimum samples to split a node
            min_samples_leaf: Minimum samples in a leaf
            max_features: Features to consider at each split
                         'sqrt' = sqrt(n_features)
                         'log2' = log2(n_features)
                         int = exact number
                         float = fraction of n_features
            criterion: 'entropy' or 'gini'
            bootstrap: Whether to use bootstrap sampling
            oob_score: Whether to compute out-of-bag score
            random_state: Random seed
            n_jobs: Not used (for sklearn compatibility)
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.criterion = criterion
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state

        self.trees: List[DecisionTreeClassifier] = []
        self.feature_indices: List[np.ndarray] = []
        self.classes_: Optional[np.ndarray] = None
        self.n_classes: int = 0
        self.n_features: int = 0
        self.oob_score_: float = 0.0
        self.feature_importances_: Optional[np.ndarray] = None

    def _get_max_features(self, n_features: int) -> int:
        """Calculate number of features to consider at each split."""
        if isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        elif self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            return max(1, int(np.log2(n_features)))
        else:
            return n_features

    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray,
                          rng: np.random.Generator) -> tuple:
        """Create a bootstrap sample."""
        n_samples = len(X)
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices], indices

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestClassifier':
        """
        Build a forest of trees from training data.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Class labels (n_samples,)

        Returns:
            self
        """
        X = np.array(X, dtype=np.float64)
        y = np.array(y)

        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        self.n_features = X.shape[1]

        n_samples = len(X)
        max_features = self._get_max_features(self.n_features)

        rng = np.random.default_rng(self.random_state)

        self.trees = []
        self.feature_indices = []
        self.feature_importances_ = np.zeros(self.n_features)

        # For OOB score
        oob_predictions = np.zeros((n_samples, self.n_classes))
        oob_counts = np.zeros(n_samples)

        for i in range(self.n_estimators):
            # Bootstrap sample
            if self.bootstrap:
                X_sample, y_sample, sample_indices = self._bootstrap_sample(X, y, rng)
                oob_mask = np.ones(n_samples, dtype=bool)
                oob_mask[np.unique(sample_indices)] = False
            else:
                X_sample, y_sample = X, y
                oob_mask = np.zeros(n_samples, dtype=bool)

            # Random feature subset
            feature_idx = rng.choice(self.n_features, size=max_features, replace=False)
            feature_idx = np.sort(feature_idx)
            self.feature_indices.append(feature_idx)

            # Train tree on subset of features
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                criterion=self.criterion,
                random_state=self.random_state + i
            )
            tree.fit(X_sample[:, feature_idx], y_sample)
            self.trees.append(tree)

            # Accumulate feature importances
            for j, idx in enumerate(feature_idx):
                if tree.feature_importances_ is not None:
                    self.feature_importances_[idx] += tree.feature_importances_[j]

            # OOB predictions
            if self.oob_score and np.any(oob_mask):
                oob_proba = tree.predict_proba(X[oob_mask][:, feature_idx])
                oob_predictions[oob_mask] += oob_proba
                oob_counts[oob_mask] += 1

        # Normalize feature importances
        self.feature_importances_ /= self.n_estimators

        # Calculate OOB score
        if self.oob_score:
            valid_mask = oob_counts > 0
            if np.any(valid_mask):
                oob_predictions[valid_mask] /= oob_counts[valid_mask, np.newaxis]
                oob_pred_classes = self.classes_[np.argmax(oob_predictions[valid_mask], axis=1)]
                self.oob_score_ = np.mean(oob_pred_classes == y[valid_mask])

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities by averaging tree predictions.

        Args:
            X: Feature matrix

        Returns:
            Probability matrix (n_samples, n_classes)
        """
        X = np.array(X, dtype=np.float64)

        # Average predictions from all trees
        proba = np.zeros((len(X), self.n_classes))

        for tree, feature_idx in zip(self.trees, self.feature_indices):
            proba += tree.predict_proba(X[:, feature_idx])

        proba /= self.n_estimators
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels using majority voting.

        Args:
            X: Feature matrix

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


# =============================================================================
# Test functions
# =============================================================================

def test_random_forest():
    """Test RandomForestClassifier."""
    print("Testing RandomForestClassifier...")

    np.random.seed(42)

    # Generate synthetic data
    n_samples = 400

    # Class 0: No blink
    X0 = np.random.randn(n_samples // 2, 10)
    X0[:, 4:6] += 0.3  # Higher EAR
    y0 = np.zeros(n_samples // 2)

    # Class 1: Blink
    X1 = np.random.randn(n_samples // 2, 10)
    X1[:, 4:6] -= 0.5  # Lower EAR
    y1 = np.ones(n_samples // 2)

    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])

    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    X_train, X_test = X[:320], X[320:]
    y_train, y_test = y[:320], y[320:]

    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=6,
        max_features='sqrt',
        oob_score=True,
        random_state=42
    )
    rf.fit(X_train, y_train)

    print(f"Train Accuracy: {rf.score(X_train, y_train):.4f}")
    print(f"Test Accuracy: {rf.score(X_test, y_test):.4f}")
    print(f"OOB Score: {rf.oob_score_:.4f}")
    print(f"Feature importances (top 5): {np.argsort(rf.feature_importances_)[-5:]}")

    # Compare with single tree
    from .decision_tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(max_depth=6)
    tree.fit(X_train, y_train)
    print(f"\nSingle Tree Test Accuracy: {tree.score(X_test, y_test):.4f}")


def test_multiclass():
    """Test with 3-class movement classification."""
    print("\nTesting multi-class Random Forest...")

    np.random.seed(42)

    # 3 classes: Fixation, Saccade, Blink
    X0 = np.random.randn(150, 14)
    X1 = np.random.randn(150, 14) + [2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    X2 = np.random.randn(150, 14) + [0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0]

    X = np.vstack([X0, X1, X2])
    y = np.array([0]*150 + [1]*150 + [2]*150)

    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    X_train, X_test = X[:360], X[360:]
    y_train, y_test = y[:360], y[360:]

    rf = RandomForestClassifier(n_estimators=50, max_depth=8)
    rf.fit(X_train, y_train)

    print(f"Train Accuracy: {rf.score(X_train, y_train):.4f}")
    print(f"Test Accuracy: {rf.score(X_test, y_test):.4f}")


if __name__ == "__main__":
    test_random_forest()
    test_multiclass()
