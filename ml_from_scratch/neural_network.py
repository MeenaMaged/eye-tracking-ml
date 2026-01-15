"""
Neural Network (Multi-Layer Perceptron) from scratch.

Implements:
- Feedforward neural network with configurable architecture
- Backpropagation with gradient descent
- Multiple activation functions (ReLU, Sigmoid, Tanh)
- Support for both regression and classification

For gaze estimation: Input(10) -> Hidden(64) -> Hidden(32) -> Output(2)
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
from enum import Enum


class Activation(Enum):
    """Supported activation functions."""
    RELU = 'relu'
    SIGMOID = 'sigmoid'
    TANH = 'tanh'
    LINEAR = 'linear'
    SOFTMAX = 'softmax'


class NeuralNetwork:
    """
    Multi-Layer Perceptron (MLP) Neural Network.

    Supports:
    - Arbitrary number of hidden layers
    - ReLU, Sigmoid, Tanh activations
    - Regression (linear output) and Classification (softmax/sigmoid)
    - Mini-batch gradient descent
    - L2 regularization

    Architecture for gaze estimation:
        Input(10) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Output(2, Linear)
    """

    def __init__(self,
                 layer_sizes: List[int],
                 activation: str = 'relu',
                 output_activation: str = 'linear',
                 learning_rate: float = 0.001,
                 l2_lambda: float = 0.0,
                 random_state: int = 42):
        """
        Initialize Neural Network.

        Args:
            layer_sizes: List of layer sizes including input and output.
                        E.g., [10, 64, 32, 2] for gaze estimation
            activation: Hidden layer activation ('relu', 'sigmoid', 'tanh')
            output_activation: Output layer activation ('linear', 'sigmoid', 'softmax')
            learning_rate: Learning rate for gradient descent
            l2_lambda: L2 regularization strength
            random_state: Random seed for weight initialization
        """
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.output_activation = output_activation
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.random_state = random_state

        self.n_layers = len(layer_sizes)
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        self._is_fitted = False
        self.training_history: List[dict] = []

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using He initialization for ReLU, Xavier for others."""
        np.random.seed(self.random_state)

        self.weights = []
        self.biases = []

        for i in range(self.n_layers - 1):
            n_in = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]

            # He initialization for ReLU, Xavier for others
            if self.activation == 'relu':
                std = np.sqrt(2.0 / n_in)  # He init
            else:
                std = np.sqrt(2.0 / (n_in + n_out))  # Xavier init

            W = np.random.randn(n_in, n_out) * std
            b = np.zeros((1, n_out))

            self.weights.append(W)
            self.biases.append(b)

    # =========================================================================
    # Activation functions and derivatives
    # =========================================================================

    def _activation_fn(self, Z: np.ndarray, activation: str) -> np.ndarray:
        """Apply activation function."""
        if activation == 'relu':
            return np.maximum(0, Z)
        elif activation == 'sigmoid':
            # Clip to prevent overflow
            Z = np.clip(Z, -500, 500)
            return 1 / (1 + np.exp(-Z))
        elif activation == 'tanh':
            return np.tanh(Z)
        elif activation == 'softmax':
            # Stable softmax
            exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
            return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        else:  # linear
            return Z

    def _activation_derivative(self, A: np.ndarray, activation: str) -> np.ndarray:
        """Compute derivative of activation function."""
        if activation == 'relu':
            return (A > 0).astype(float)
        elif activation == 'sigmoid':
            return A * (1 - A)
        elif activation == 'tanh':
            return 1 - A ** 2
        else:  # linear
            return np.ones_like(A)

    # =========================================================================
    # Forward propagation
    # =========================================================================

    def _forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Forward propagation through the network.

        Returns:
            Tuple of (activations, pre_activations) for each layer
        """
        activations = [X]  # A[0] = input
        pre_activations = [X]  # Z[0] = input (not used)

        A = X
        for i in range(self.n_layers - 1):
            Z = A @ self.weights[i] + self.biases[i]
            pre_activations.append(Z)

            # Use output activation for last layer
            if i == self.n_layers - 2:
                A = self._activation_fn(Z, self.output_activation)
            else:
                A = self._activation_fn(Z, self.activation)

            activations.append(A)

        return activations, pre_activations

    # =========================================================================
    # Backward propagation
    # =========================================================================

    def _backward(self, y: np.ndarray, activations: List[np.ndarray],
                  pre_activations: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Backward propagation to compute gradients.

        Args:
            y: True labels
            activations: List of layer activations from forward pass
            pre_activations: List of pre-activation values

        Returns:
            Tuple of (weight_gradients, bias_gradients)
        """
        m = y.shape[0]  # Batch size
        n_layers = self.n_layers

        # Initialize gradients
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]

        # Output layer error
        A_out = activations[-1]

        if self.output_activation in ['softmax', 'sigmoid'] and y.ndim == 2:
            # Cross-entropy derivative (simplified for softmax/sigmoid + CE)
            dZ = A_out - y
        else:
            # MSE derivative for regression
            dZ = (A_out - y) * self._activation_derivative(A_out, self.output_activation)

        # Backpropagate through layers
        for i in range(n_layers - 2, -1, -1):
            A_prev = activations[i]

            # Gradients
            dW[i] = (A_prev.T @ dZ) / m
            db[i] = np.sum(dZ, axis=0, keepdims=True) / m

            # Add L2 regularization to weight gradients
            if self.l2_lambda > 0:
                dW[i] += (self.l2_lambda / m) * self.weights[i]

            # Propagate error to previous layer (if not input layer)
            if i > 0:
                dA = dZ @ self.weights[i].T
                dZ = dA * self._activation_derivative(activations[i], self.activation)

        return dW, db

    # =========================================================================
    # Training
    # =========================================================================

    def fit(self, X: np.ndarray, y: np.ndarray,
            epochs: int = 100,
            batch_size: int = 32,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            verbose: int = 1) -> 'NeuralNetwork':
        """
        Train the neural network.

        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training targets
            epochs: Number of training epochs
            batch_size: Mini-batch size
            validation_data: Optional (X_val, y_val) for validation
            verbose: 0=silent, 1=progress bar, 2=one line per epoch

        Returns:
            self
        """
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples = X.shape[0]
        self.training_history = []

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0
            n_batches = 0

            # Mini-batch training
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Forward pass
                activations, pre_activations = self._forward(X_batch)

                # Compute loss
                batch_loss = self._compute_loss(y_batch, activations[-1])
                epoch_loss += batch_loss

                # Backward pass
                dW, db = self._backward(y_batch, activations, pre_activations)

                # Update weights
                for i in range(len(self.weights)):
                    self.weights[i] -= self.learning_rate * dW[i]
                    self.biases[i] -= self.learning_rate * db[i]

                n_batches += 1

            # Record history
            avg_loss = epoch_loss / n_batches
            history = {'epoch': epoch + 1, 'loss': avg_loss}

            if validation_data is not None:
                val_loss = self._compute_loss(validation_data[1],
                                              self.predict(validation_data[0]))
                history['val_loss'] = val_loss

            self.training_history.append(history)

            # Verbose output
            if verbose >= 2:
                msg = f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f}"
                if 'val_loss' in history:
                    msg += f" - val_loss: {history['val_loss']:.4f}"
                print(msg)
            elif verbose == 1 and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f}")

        self._is_fitted = True
        return self

    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute loss (MSE for regression, cross-entropy for classification)."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        if self.output_activation == 'softmax':
            # Cross-entropy loss
            eps = 1e-15
            y_pred = np.clip(y_pred, eps, 1 - eps)
            return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        elif self.output_activation == 'sigmoid' and y_true.shape[1] == 1:
            # Binary cross-entropy
            eps = 1e-15
            y_pred = np.clip(y_pred, eps, 1 - eps)
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            # MSE for regression
            return np.mean((y_true - y_pred) ** 2)

    # =========================================================================
    # Prediction
    # =========================================================================

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input features

        Returns:
            Predictions
        """
        X = np.array(X, dtype=np.float64)
        activations, _ = self._forward(X)
        return activations[-1]

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels (for classification).

        Args:
            X: Input features

        Returns:
            Class labels
        """
        probs = self.predict(X)

        if self.output_activation == 'softmax':
            return np.argmax(probs, axis=1)
        else:  # sigmoid for binary
            return (probs > 0.5).astype(int).ravel()

    # =========================================================================
    # Evaluation metrics
    # =========================================================================

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R² for regression or accuracy for classification.
        """
        y = np.array(y)
        y_pred = self.predict(X)

        if self.output_activation in ['softmax', 'sigmoid']:
            # Accuracy for classification
            if self.output_activation == 'softmax':
                y_pred_classes = np.argmax(y_pred, axis=1)
                y_true_classes = np.argmax(y, axis=1) if y.ndim == 2 else y
            else:
                y_pred_classes = (y_pred > 0.5).astype(int).ravel()
                y_true_classes = y.ravel()
            return np.mean(y_pred_classes == y_true_classes)
        else:
            # R² for regression
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)

            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    def mean_squared_error(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate MSE."""
        y = np.array(y)
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)

    def pixel_error(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate mean Euclidean distance error (for gaze estimation)."""
        y = np.array(y)
        y_pred = self.predict(X)

        if y.ndim == 1:
            return np.mean(np.abs(y - y_pred))

        distances = np.sqrt(np.sum((y - y_pred) ** 2, axis=1))
        return np.mean(distances)


# =============================================================================
# Convenience functions
# =============================================================================

def create_gaze_estimator(n_features: int = 10,
                          hidden_layers: List[int] = [64, 32],
                          learning_rate: float = 0.001,
                          l2_lambda: float = 0.001) -> NeuralNetwork:
    """
    Create a neural network for gaze estimation.

    Args:
        n_features: Number of input features (default 10)
        hidden_layers: List of hidden layer sizes
        learning_rate: Learning rate
        l2_lambda: L2 regularization

    Returns:
        Configured NeuralNetwork for regression
    """
    layer_sizes = [n_features] + hidden_layers + [2]  # Output: (x, y)
    return NeuralNetwork(
        layer_sizes=layer_sizes,
        activation='relu',
        output_activation='linear',
        learning_rate=learning_rate,
        l2_lambda=l2_lambda
    )


def create_classifier(n_features: int,
                      n_classes: int,
                      hidden_layers: List[int] = [64, 32],
                      learning_rate: float = 0.001) -> NeuralNetwork:
    """
    Create a neural network for classification.

    Args:
        n_features: Number of input features
        n_classes: Number of output classes
        hidden_layers: List of hidden layer sizes
        learning_rate: Learning rate

    Returns:
        Configured NeuralNetwork for classification
    """
    layer_sizes = [n_features] + hidden_layers + [n_classes]
    output_activation = 'softmax' if n_classes > 2 else 'sigmoid'

    return NeuralNetwork(
        layer_sizes=layer_sizes,
        activation='relu',
        output_activation=output_activation,
        learning_rate=learning_rate
    )


# =============================================================================
# Test functions
# =============================================================================

def test_gaze_estimation():
    """Test neural network for gaze estimation (regression)."""
    print("Testing Neural Network for Gaze Estimation...")

    np.random.seed(42)

    # Generate synthetic gaze data
    n_samples = 500
    n_features = 10

    X = np.random.randn(n_samples, n_features)

    # Non-linear relationship
    y_x = 500 + 200 * X[:, 0] + 100 * np.sin(X[:, 1] * 2) + 50 * X[:, 2] ** 2
    y_y = 400 + 150 * X[:, 3] - 80 * np.cos(X[:, 4] * 2) + 40 * X[:, 5] ** 2
    y = np.column_stack([y_x, y_y]) + np.random.randn(n_samples, 2) * 20

    # Split
    X_train, X_test = X[:400], X[400:]
    y_train, y_test = y[:400], y[400:]

    # Normalize
    X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
    X_train_norm = (X_train - X_mean) / (X_std + 1e-8)
    X_test_norm = (X_test - X_mean) / (X_std + 1e-8)

    # Train
    model = create_gaze_estimator(n_features=10, hidden_layers=[64, 32],
                                   learning_rate=0.01, l2_lambda=0.001)
    model.fit(X_train_norm, y_train, epochs=100, batch_size=32, verbose=1)

    # Evaluate
    print(f"\nR² (train): {model.score(X_train_norm, y_train):.4f}")
    print(f"R² (test): {model.score(X_test_norm, y_test):.4f}")
    print(f"Pixel Error (train): {model.pixel_error(X_train_norm, y_train):.2f}")
    print(f"Pixel Error (test): {model.pixel_error(X_test_norm, y_test):.2f}")


def test_classification():
    """Test neural network for classification."""
    print("\nTesting Neural Network for Classification...")

    np.random.seed(42)

    # Generate 3-class data
    n_per_class = 100
    X0 = np.random.randn(n_per_class, 4) + [0, 0, 0, 0]
    X1 = np.random.randn(n_per_class, 4) + [3, 3, 0, 0]
    X2 = np.random.randn(n_per_class, 4) + [0, 0, 3, 3]

    X = np.vstack([X0, X1, X2])
    y_labels = np.array([0]*n_per_class + [1]*n_per_class + [2]*n_per_class)

    # One-hot encode
    y = np.zeros((len(y_labels), 3))
    y[np.arange(len(y_labels)), y_labels] = 1

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y, y_labels = X[idx], y[idx], y_labels[idx]

    # Split
    X_train, X_test = X[:240], X[240:]
    y_train, y_test = y[:240], y[240:]
    y_labels_test = y_labels[240:]

    # Train
    model = create_classifier(n_features=4, n_classes=3,
                               hidden_layers=[16, 8], learning_rate=0.01)
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

    # Evaluate
    y_pred = model.predict_classes(X_test)
    accuracy = np.mean(y_pred == y_labels_test)
    print(f"\nAccuracy: {accuracy:.4f}")


if __name__ == "__main__":
    test_gaze_estimation()
    test_classification()
