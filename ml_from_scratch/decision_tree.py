"""
Decision Tree Classifier from scratch.

Implements:
- CART algorithm with binary splits
- Entropy and Information Gain for split selection
- Gini impurity (alternative criterion)
- Pre-pruning with max_depth and min_samples
- Support for multi-class classification

For blink detection: Binary classification (0=no blink, 1=blink)
For movement: Multi-class (0=Fixation, 1=Saccade, 2=Blink)
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class Node:
    """Decision tree node."""
    feature_idx: Optional[int] = None  # Split feature index
    threshold: Optional[float] = None   # Split threshold
    left: Optional['Node'] = None       # Left child (feature <= threshold)
    right: Optional['Node'] = None      # Right child (feature > threshold)
    value: Optional[np.ndarray] = None  # Class distribution (leaf node)
    is_leaf: bool = False


class DecisionTreeClassifier:
    """
    Decision Tree Classifier using CART algorithm.

    Uses Information Gain (entropy-based) or Gini impurity
    to select optimal splits.

    Entropy: H(S) = -sum(p_i * log2(p_i))
    Information Gain: IG(S, A) = H(S) - sum(|S_v|/|S| * H(S_v))
    Gini: G(S) = 1 - sum(p_i^2)
    """

    def __init__(self,
                 max_depth: int = 10,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 criterion: str = 'entropy',
                 random_state: int = 42):
        """
        Initialize Decision Tree.

        Args:
            max_depth: Maximum depth of tree (pre-pruning)
            min_samples_split: Minimum samples to split a node
            min_samples_leaf: Minimum samples in a leaf
            criterion: 'entropy' (Information Gain) or 'gini'
            random_state: Random seed for reproducibility
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.random_state = random_state

        self.root: Optional[Node] = None
        self.n_classes: int = 0
        self.n_features: int = 0
        self.classes_: Optional[np.ndarray] = None
        self.feature_importances_: Optional[np.ndarray] = None

    def _entropy(self, y: np.ndarray) -> float:
        """
        Calculate entropy of a set.

        H(S) = -sum(p_i * log2(p_i))
        """
        if len(y) == 0:
            return 0.0

        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)

        # Avoid log(0)
        probabilities = probabilities[probabilities > 0]
        return -np.sum(probabilities * np.log2(probabilities))

    def _gini(self, y: np.ndarray) -> float:
        """
        Calculate Gini impurity.

        G(S) = 1 - sum(p_i^2)
        """
        if len(y) == 0:
            return 0.0

        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def _impurity(self, y: np.ndarray) -> float:
        """Calculate impurity based on criterion."""
        if self.criterion == 'entropy':
            return self._entropy(y)
        return self._gini(y)

    def _information_gain(self, y: np.ndarray, y_left: np.ndarray,
                          y_right: np.ndarray) -> float:
        """
        Calculate information gain from a split.

        IG = H(parent) - (|left|/|parent| * H(left) + |right|/|parent| * H(right))
        """
        n = len(y)
        if n == 0:
            return 0.0

        n_left, n_right = len(y_left), len(y_right)

        parent_impurity = self._impurity(y)
        child_impurity = (n_left / n) * self._impurity(y_left) + \
                         (n_right / n) * self._impurity(y_right)

        return parent_impurity - child_impurity

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
        """
        Find the best feature and threshold to split on.

        Returns:
            Tuple of (best_feature_idx, best_threshold, best_gain)
        """
        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        n_samples, n_features = X.shape

        for feature_idx in range(n_features):
            # Get unique values for this feature
            values = X[:, feature_idx]
            unique_values = np.unique(values)

            # Try thresholds between unique values
            if len(unique_values) > 10:
                # Use percentiles for many unique values
                thresholds = np.percentile(unique_values, [10, 25, 50, 75, 90])
            else:
                thresholds = (unique_values[:-1] + unique_values[1:]) / 2

            for threshold in thresholds:
                # Split data
                left_mask = values <= threshold
                right_mask = ~left_mask

                # Check minimum samples
                if np.sum(left_mask) < self.min_samples_leaf or \
                   np.sum(right_mask) < self.min_samples_leaf:
                    continue

                # Calculate information gain
                gain = self._information_gain(y, y[left_mask], y[right_mask])

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """Recursively build the decision tree."""
        n_samples = len(y)

        # Create leaf node if stopping criteria met
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            len(np.unique(y)) == 1):
            return self._create_leaf(y)

        # Find best split
        feature_idx, threshold, gain = self._find_best_split(X, y)

        # Create leaf if no good split found
        if feature_idx is None or gain <= 0:
            return self._create_leaf(y)

        # Split data
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        # Recursively build children
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        # Update feature importance
        n_left, n_right = np.sum(left_mask), np.sum(right_mask)
        self.feature_importances_[feature_idx] += gain * n_samples

        return Node(
            feature_idx=feature_idx,
            threshold=threshold,
            left=left_child,
            right=right_child,
            is_leaf=False
        )

    def _create_leaf(self, y: np.ndarray) -> Node:
        """Create a leaf node with class distribution."""
        # Count occurrences of each class
        value = np.zeros(self.n_classes)
        for i, cls in enumerate(self.classes_):
            value[i] = np.sum(y == cls)
        value = value / len(y) if len(y) > 0 else value

        return Node(value=value, is_leaf=True)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeClassifier':
        """
        Build decision tree from training data.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Class labels (n_samples,)

        Returns:
            self
        """
        np.random.seed(self.random_state)

        X = np.array(X, dtype=np.float64)
        y = np.array(y)

        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        self.n_features = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features)

        # Build tree
        self.root = self._build_tree(X, y)

        # Normalize feature importances
        total = np.sum(self.feature_importances_)
        if total > 0:
            self.feature_importances_ /= total

        return self

    def _predict_sample(self, x: np.ndarray, node: Node) -> np.ndarray:
        """Predict class probabilities for a single sample."""
        if node.is_leaf:
            return node.value

        if x[node.feature_idx] <= node.threshold:
            return self._predict_sample(x, node.left)
        return self._predict_sample(x, node.right)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix

        Returns:
            Probability matrix (n_samples, n_classes)
        """
        X = np.array(X, dtype=np.float64)
        return np.array([self._predict_sample(x, self.root) for x in X])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

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

    def get_depth(self) -> int:
        """Get the actual depth of the tree."""
        def _depth(node: Node) -> int:
            if node is None or node.is_leaf:
                return 0
            return 1 + max(_depth(node.left), _depth(node.right))
        return _depth(self.root)

    def get_n_leaves(self) -> int:
        """Get the number of leaf nodes."""
        def _count_leaves(node: Node) -> int:
            if node is None:
                return 0
            if node.is_leaf:
                return 1
            return _count_leaves(node.left) + _count_leaves(node.right)
        return _count_leaves(self.root)


# =============================================================================
# Test functions
# =============================================================================

def test_decision_tree():
    """Test DecisionTreeClassifier."""
    print("Testing DecisionTreeClassifier...")

    np.random.seed(42)

    # Generate synthetic blink detection data
    n_samples = 300

    # Class 0: No blink (high EAR)
    X0 = np.random.randn(n_samples // 2, 5) + [0, 0, 0.3, 0.3, 0]
    y0 = np.zeros(n_samples // 2)

    # Class 1: Blink (low EAR)
    X1 = np.random.randn(n_samples // 2, 5) + [0, 0, -0.5, -0.5, 0]
    y1 = np.ones(n_samples // 2)

    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    # Split
    X_train, X_test = X[:240], X[240:]
    y_train, y_test = y[:240], y[240:]

    # Train with entropy
    tree_entropy = DecisionTreeClassifier(max_depth=5, criterion='entropy')
    tree_entropy.fit(X_train, y_train)

    print(f"Entropy - Train Acc: {tree_entropy.score(X_train, y_train):.4f}")
    print(f"Entropy - Test Acc: {tree_entropy.score(X_test, y_test):.4f}")
    print(f"Tree depth: {tree_entropy.get_depth()}, Leaves: {tree_entropy.get_n_leaves()}")

    # Train with gini
    tree_gini = DecisionTreeClassifier(max_depth=5, criterion='gini')
    tree_gini.fit(X_train, y_train)

    print(f"\nGini - Train Acc: {tree_gini.score(X_train, y_train):.4f}")
    print(f"Gini - Test Acc: {tree_gini.score(X_test, y_test):.4f}")

    print(f"\nFeature importances: {tree_entropy.feature_importances_}")


def test_multiclass():
    """Test with 3 classes (movement classification)."""
    print("\nTesting multi-class classification...")

    np.random.seed(42)

    # 3 classes
    X0 = np.random.randn(100, 4) + [0, 0, 0, 0]
    X1 = np.random.randn(100, 4) + [3, 3, 0, 0]
    X2 = np.random.randn(100, 4) + [0, 0, 3, 3]

    X = np.vstack([X0, X1, X2])
    y = np.array([0]*100 + [1]*100 + [2]*100)

    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    X_train, X_test = X[:240], X[240:]
    y_train, y_test = y[:240], y[240:]

    tree = DecisionTreeClassifier(max_depth=6)
    tree.fit(X_train, y_train)

    print(f"Train Acc: {tree.score(X_train, y_train):.4f}")
    print(f"Test Acc: {tree.score(X_test, y_test):.4f}")
    print(f"Classes: {tree.classes_}")


if __name__ == "__main__":
    test_decision_tree()
    test_multiclass()
