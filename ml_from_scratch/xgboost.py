"""
XGBoost (Gradient Boosting) Classifier from scratch.

Implements:
- Gradient Boosting with decision tree base learners
- Second-order Taylor expansion (Newton boosting)
- L2 regularization on leaf weights
- Shrinkage (learning rate)

XGBoost sequentially builds trees, each correcting errors of previous ensemble.
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class XGBNode:
    """XGBoost tree node."""
    feature_idx: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional['XGBNode'] = None
    right: Optional['XGBNode'] = None
    weight: float = 0.0  # Leaf weight (prediction value)
    is_leaf: bool = False


class XGBTree:
    """Single XGBoost tree (weak learner)."""

    def __init__(self, max_depth: int = 3, min_samples_leaf: int = 1,
                 reg_lambda: float = 1.0, gamma: float = 0.0):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.reg_lambda = reg_lambda  # L2 regularization
        self.gamma = gamma  # Minimum loss reduction for split
        self.root: Optional[XGBNode] = None

    def _calc_leaf_weight(self, gradient: np.ndarray, hessian: np.ndarray) -> float:
        """Calculate optimal leaf weight: w = -G / (H + lambda)"""
        G = np.sum(gradient)
        H = np.sum(hessian)
        return -G / (H + self.reg_lambda)

    def _calc_gain(self, gradient: np.ndarray, hessian: np.ndarray,
                   g_left: np.ndarray, h_left: np.ndarray,
                   g_right: np.ndarray, h_right: np.ndarray) -> float:
        """
        Calculate split gain using second-order approximation.

        Gain = 0.5 * [G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - G^2/(H+lambda)] - gamma
        """
        G, H = np.sum(gradient), np.sum(hessian)
        G_L, H_L = np.sum(g_left), np.sum(h_left)
        G_R, H_R = np.sum(g_right), np.sum(h_right)

        gain = 0.5 * (
            G_L**2 / (H_L + self.reg_lambda) +
            G_R**2 / (H_R + self.reg_lambda) -
            G**2 / (H + self.reg_lambda)
        ) - self.gamma

        return gain

    def _find_best_split(self, X: np.ndarray, gradient: np.ndarray,
                         hessian: np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
        """Find best split based on gain."""
        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        n_samples, n_features = X.shape

        for feature_idx in range(n_features):
            values = X[:, feature_idx]
            unique_values = np.unique(values)

            if len(unique_values) > 20:
                thresholds = np.percentile(unique_values, np.linspace(5, 95, 10))
            else:
                thresholds = (unique_values[:-1] + unique_values[1:]) / 2

            for threshold in thresholds:
                left_mask = values <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) < self.min_samples_leaf or \
                   np.sum(right_mask) < self.min_samples_leaf:
                    continue

                gain = self._calc_gain(
                    gradient, hessian,
                    gradient[left_mask], hessian[left_mask],
                    gradient[right_mask], hessian[right_mask]
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X: np.ndarray, gradient: np.ndarray,
                    hessian: np.ndarray, depth: int = 0) -> XGBNode:
        """Build tree recursively."""
        # Stopping criteria
        if depth >= self.max_depth or len(X) < self.min_samples_leaf * 2:
            weight = self._calc_leaf_weight(gradient, hessian)
            return XGBNode(weight=weight, is_leaf=True)

        # Find best split
        feature_idx, threshold, gain = self._find_best_split(X, gradient, hessian)

        if feature_idx is None or gain <= 0:
            weight = self._calc_leaf_weight(gradient, hessian)
            return XGBNode(weight=weight, is_leaf=True)

        # Split data
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        # Build children
        left = self._build_tree(X[left_mask], gradient[left_mask],
                               hessian[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], gradient[right_mask],
                                hessian[right_mask], depth + 1)

        return XGBNode(
            feature_idx=feature_idx,
            threshold=threshold,
            left=left,
            right=right,
            is_leaf=False
        )

    def fit(self, X: np.ndarray, gradient: np.ndarray, hessian: np.ndarray):
        """Build tree from gradients and hessians."""
        self.root = self._build_tree(X, gradient, hessian)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict leaf weights for samples."""
        return np.array([self._predict_sample(x, self.root) for x in X])

    def _predict_sample(self, x: np.ndarray, node: XGBNode) -> float:
        if node.is_leaf:
            return node.weight
        if x[node.feature_idx] <= node.threshold:
            return self._predict_sample(x, node.left)
        return self._predict_sample(x, node.right)


class XGBoostClassifier:
    """
    XGBoost Classifier (Gradient Boosting).

    Uses logistic loss for binary classification and
    softmax for multi-class classification.
    """

    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 3,
                 learning_rate: float = 0.1,
                 min_samples_leaf: int = 1,
                 reg_lambda: float = 1.0,
                 gamma: float = 0.0,
                 random_state: int = 42):
        """
        Initialize XGBoost.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Max depth of each tree
            learning_rate: Shrinkage factor (eta)
            min_samples_leaf: Minimum samples in leaf
            reg_lambda: L2 regularization on weights
            gamma: Minimum gain for split
            random_state: Random seed
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.random_state = random_state

        self.trees: List[List[XGBTree]] = []  # trees[round][class]
        self.classes_: Optional[np.ndarray] = None
        self.n_classes: int = 0
        self.base_score: float = 0.5

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid function with clipping for stability."""
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax function."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def _binary_gradient_hessian(self, y: np.ndarray, pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Gradient and Hessian for binary logistic loss."""
        prob = self._sigmoid(pred)
        gradient = prob - y
        hessian = prob * (1 - prob)
        hessian = np.maximum(hessian, 1e-6)  # Stability
        return gradient, hessian

    def _multiclass_gradient_hessian(self, y_onehot: np.ndarray,
                                      pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Gradient and Hessian for softmax cross-entropy loss."""
        prob = self._softmax(pred)
        gradient = prob - y_onehot
        hessian = prob * (1 - prob)
        hessian = np.maximum(hessian, 1e-6)
        return gradient, hessian

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'XGBoostClassifier':
        """
        Fit XGBoost model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Class labels

        Returns:
            self
        """
        np.random.seed(self.random_state)

        X = np.array(X, dtype=np.float64)
        y = np.array(y)

        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        n_samples = len(X)

        # Initialize predictions
        if self.n_classes == 2:
            # Binary classification
            pred = np.full(n_samples, 0.0)
            y_binary = (y == self.classes_[1]).astype(float)

            self.trees = []
            for _ in range(self.n_estimators):
                gradient, hessian = self._binary_gradient_hessian(y_binary, pred)

                tree = XGBTree(
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    reg_lambda=self.reg_lambda,
                    gamma=self.gamma
                )
                tree.fit(X, gradient, hessian)
                self.trees.append([tree])

                # Update predictions
                pred += self.learning_rate * tree.predict(X)
        else:
            # Multi-class classification
            pred = np.zeros((n_samples, self.n_classes))
            y_onehot = np.zeros((n_samples, self.n_classes))
            for i, cls in enumerate(self.classes_):
                y_onehot[y == cls, i] = 1

            self.trees = []
            for _ in range(self.n_estimators):
                gradient, hessian = self._multiclass_gradient_hessian(y_onehot, pred)

                round_trees = []
                for k in range(self.n_classes):
                    tree = XGBTree(
                        max_depth=self.max_depth,
                        min_samples_leaf=self.min_samples_leaf,
                        reg_lambda=self.reg_lambda,
                        gamma=self.gamma
                    )
                    tree.fit(X, gradient[:, k], hessian[:, k])
                    round_trees.append(tree)

                    pred[:, k] += self.learning_rate * tree.predict(X)

                self.trees.append(round_trees)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        X = np.array(X, dtype=np.float64)
        n_samples = len(X)

        if self.n_classes == 2:
            pred = np.zeros(n_samples)
            for round_trees in self.trees:
                pred += self.learning_rate * round_trees[0].predict(X)
            prob_1 = self._sigmoid(pred)
            return np.column_stack([1 - prob_1, prob_1])
        else:
            pred = np.zeros((n_samples, self.n_classes))
            for round_trees in self.trees:
                for k, tree in enumerate(round_trees):
                    pred[:, k] += self.learning_rate * tree.predict(X)
            return self._softmax(pred)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
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

def test_xgboost_binary():
    """Test XGBoost on binary classification."""
    print("Testing XGBoost (Binary)...")

    np.random.seed(42)

    # Generate data
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

    # Train XGBoost
    xgb = XGBoostClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    xgb.fit(X_train, y_train)

    print(f"Train Accuracy: {xgb.score(X_train, y_train):.4f}")
    print(f"Test Accuracy: {xgb.score(X_test, y_test):.4f}")


def test_xgboost_multiclass():
    """Test XGBoost on multi-class classification."""
    print("\nTesting XGBoost (Multi-class)...")

    np.random.seed(42)

    # 3 classes
    X0 = np.random.randn(100, 8)
    X1 = np.random.randn(100, 8) + [2, 2, 0, 0, 0, 0, 0, 0]
    X2 = np.random.randn(100, 8) + [0, 0, 2, 2, 0, 0, 0, 0]

    X = np.vstack([X0, X1, X2])
    y = np.array([0]*100 + [1]*100 + [2]*100)

    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    X_train, X_test = X[:240], X[240:]
    y_train, y_test = y[:240], y[240:]

    xgb = XGBoostClassifier(n_estimators=50, max_depth=3, learning_rate=0.1)
    xgb.fit(X_train, y_train)

    print(f"Train Accuracy: {xgb.score(X_train, y_train):.4f}")
    print(f"Test Accuracy: {xgb.score(X_test, y_test):.4f}")


if __name__ == "__main__":
    test_xgboost_binary()
    test_xgboost_multiclass()
