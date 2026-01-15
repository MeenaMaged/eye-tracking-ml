"""
Dataset management: train/test split, k-fold cross-validation, save/load, preprocessing.

Provides utilities for managing ML datasets with proper train/test splitting,
cross-validation, and persistence.
"""

import numpy as np
import os
import json
from typing import Tuple, Dict, List, Optional, Generator
from datetime import datetime
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import DATA_DIR


class DatasetManager:
    """
    Manage datasets for ML training and evaluation.

    Features:
    - Stratified train/test splitting
    - K-fold cross-validation
    - Dataset persistence (save/load)
    - Feature normalization
    - Dataset statistics and info
    """

    def __init__(self, data_dir: str = DATA_DIR):
        """
        Initialize DatasetManager.

        Args:
            data_dir: Root directory for dataset storage
        """
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")

        # Create directories if they don't exist
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def train_test_split(self,
                         X: np.ndarray,
                         y: np.ndarray,
                         test_size: float = 0.2,
                         random_state: int = 42,
                         stratify: bool = True) -> Tuple[np.ndarray, np.ndarray,
                                                          np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets.

        Implements stratified splitting to maintain class proportions
        in both train and test sets.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Label vector of shape (n_samples,)
            test_size: Fraction of data to use for testing (0.0-1.0)
            random_state: Random seed for reproducibility
            stratify: If True, maintain class proportions in split

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        np.random.seed(random_state)

        if stratify and len(np.unique(y)) > 1:
            # Stratified split: maintain class proportions
            train_idx, test_idx = [], []

            for cls in np.unique(y):
                cls_idx = np.where(y == cls)[0]
                np.random.shuffle(cls_idx)

                n_test = max(1, int(len(cls_idx) * test_size))
                test_idx.extend(cls_idx[:n_test])
                train_idx.extend(cls_idx[n_test:])

            train_idx = np.array(train_idx)
            test_idx = np.array(test_idx)
        else:
            # Random split
            indices = np.arange(len(X))
            np.random.shuffle(indices)

            n_test = int(len(X) * test_size)
            test_idx = indices[:n_test]
            train_idx = indices[n_test:]

        # Shuffle final indices
        np.random.shuffle(train_idx)
        np.random.shuffle(test_idx)

        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    def k_fold_split(self,
                     X: np.ndarray,
                     y: np.ndarray,
                     n_folds: int = 5,
                     random_state: int = 42,
                     stratify: bool = True) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate k-fold cross-validation splits.

        Args:
            X: Feature matrix
            y: Label vector
            n_folds: Number of folds
            random_state: Random seed for reproducibility
            stratify: If True, maintain class proportions in each fold

        Returns:
            List of (train_indices, test_indices) tuples for each fold
        """
        np.random.seed(random_state)
        n_samples = len(X)

        if stratify and len(np.unique(y)) > 1:
            # Stratified k-fold
            return self._stratified_k_fold(y, n_folds, random_state)
        else:
            # Standard k-fold
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            fold_size = n_samples // n_folds
            folds = []

            for i in range(n_folds):
                start = i * fold_size
                end = start + fold_size if i < n_folds - 1 else n_samples

                test_idx = indices[start:end]
                train_idx = np.concatenate([indices[:start], indices[end:]])

                folds.append((train_idx, test_idx))

            return folds

    def _stratified_k_fold(self,
                           y: np.ndarray,
                           n_folds: int,
                           random_state: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate stratified k-fold splits maintaining class proportions."""
        np.random.seed(random_state)

        # Get indices for each class
        class_indices = {}
        for cls in np.unique(y):
            idx = np.where(y == cls)[0]
            np.random.shuffle(idx)
            class_indices[cls] = idx

        # Distribute each class across folds
        fold_indices = [[] for _ in range(n_folds)]

        for cls, indices in class_indices.items():
            fold_size = len(indices) // n_folds
            remainder = len(indices) % n_folds

            start = 0
            for i in range(n_folds):
                # Add one extra sample to first 'remainder' folds
                end = start + fold_size + (1 if i < remainder else 0)
                fold_indices[i].extend(indices[start:end])
                start = end

        # Convert to train/test splits
        folds = []
        for i in range(n_folds):
            test_idx = np.array(fold_indices[i])
            train_idx = np.concatenate([fold_indices[j] for j in range(n_folds) if j != i])
            folds.append((train_idx, test_idx))

        return folds

    def k_fold_iterator(self,
                        X: np.ndarray,
                        y: np.ndarray,
                        n_folds: int = 5,
                        random_state: int = 42,
                        stratify: bool = True) -> Generator:
        """
        Iterate through k-fold splits yielding actual data.

        Args:
            X: Feature matrix
            y: Label vector
            n_folds: Number of folds
            random_state: Random seed
            stratify: If True, maintain class proportions

        Yields:
            Tuple of (X_train, X_test, y_train, y_test, fold_number)
        """
        folds = self.k_fold_split(X, y, n_folds, random_state, stratify)

        for fold_num, (train_idx, test_idx) in enumerate(folds):
            yield (X[train_idx], X[test_idx],
                   y[train_idx], y[test_idx],
                   fold_num)

    def save_dataset(self,
                     X: np.ndarray,
                     y: np.ndarray,
                     name: str,
                     feature_names: Optional[List[str]] = None,
                     class_names: Optional[List[str]] = None,
                     metadata: Optional[Dict] = None) -> str:
        """
        Save dataset to disk with metadata.

        Args:
            X: Feature matrix
            y: Label vector
            name: Dataset name (used as directory name)
            feature_names: List of feature names
            class_names: List of class names (for classification)
            metadata: Additional metadata to store

        Returns:
            Path to saved dataset directory
        """
        path = os.path.join(self.processed_dir, name)
        os.makedirs(path, exist_ok=True)

        # Save arrays
        np.save(os.path.join(path, "X.npy"), X)
        np.save(os.path.join(path, "y.npy"), y)

        # Calculate statistics
        unique, counts = np.unique(y, return_counts=True)

        info = {
            "name": name,
            "created": datetime.now().isoformat(),
            "n_samples": int(len(X)),
            "n_features": int(X.shape[1]) if len(X) > 0 else 0,
            "n_classes": int(len(unique)),
            "class_distribution": {int(k): int(v) for k, v in zip(unique, counts)},
            "feature_names": feature_names or [],
            "class_names": class_names or [],
            "feature_stats": {
                "mean": X.mean(axis=0).tolist() if len(X) > 0 else [],
                "std": X.std(axis=0).tolist() if len(X) > 0 else [],
                "min": X.min(axis=0).tolist() if len(X) > 0 else [],
                "max": X.max(axis=0).tolist() if len(X) > 0 else [],
            },
            "metadata": metadata or {},
        }

        with open(os.path.join(path, "info.json"), 'w') as f:
            json.dump(info, f, indent=2)

        print(f"Saved dataset '{name}': {len(X)} samples, {X.shape[1] if len(X) > 0 else 0} features")
        return path

    def load_dataset(self, name: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Load dataset from disk.

        Args:
            name: Dataset name to load

        Returns:
            Tuple of (X, y, info) where info contains metadata
        """
        path = os.path.join(self.processed_dir, name)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset '{name}' not found at {path}")

        X = np.load(os.path.join(path, "X.npy"))
        y = np.load(os.path.join(path, "y.npy"))

        with open(os.path.join(path, "info.json"), 'r') as f:
            info = json.load(f)

        return X, y, info

    def list_datasets(self) -> List[str]:
        """List all saved datasets."""
        datasets = []

        if os.path.exists(self.processed_dir):
            for name in os.listdir(self.processed_dir):
                path = os.path.join(self.processed_dir, name)
                if os.path.isdir(path) and os.path.exists(os.path.join(path, "X.npy")):
                    datasets.append(name)

        return datasets

    def normalize(self,
                  X_train: np.ndarray,
                  X_test: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
        """
        Normalize features using training set statistics (z-score normalization).

        IMPORTANT: Always fit on training data only to avoid data leakage.

        Args:
            X_train: Training feature matrix
            X_test: Optional test feature matrix

        Returns:
            Tuple of (X_train_normalized, X_test_normalized, stats)
            stats contains 'mean' and 'std' for later use
        """
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)

        # Prevent division by zero for constant features
        std[std == 0] = 1.0

        X_train_norm = (X_train - mean) / std

        X_test_norm = None
        if X_test is not None:
            X_test_norm = (X_test - mean) / std

        stats = {
            "mean": mean,
            "std": std
        }

        return X_train_norm, X_test_norm, stats

    def min_max_normalize(self,
                          X_train: np.ndarray,
                          X_test: Optional[np.ndarray] = None,
                          feature_range: Tuple[float, float] = (0, 1)) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
        """
        Normalize features to a given range using min-max scaling.

        Args:
            X_train: Training feature matrix
            X_test: Optional test feature matrix
            feature_range: Desired range (min, max)

        Returns:
            Tuple of (X_train_normalized, X_test_normalized, stats)
        """
        min_val = np.min(X_train, axis=0)
        max_val = np.max(X_train, axis=0)

        # Prevent division by zero
        range_val = max_val - min_val
        range_val[range_val == 0] = 1.0

        # Scale to [0, 1]
        X_train_scaled = (X_train - min_val) / range_val

        # Scale to desired range
        scale = feature_range[1] - feature_range[0]
        X_train_norm = X_train_scaled * scale + feature_range[0]

        X_test_norm = None
        if X_test is not None:
            X_test_scaled = (X_test - min_val) / range_val
            X_test_norm = X_test_scaled * scale + feature_range[0]

        stats = {
            "min": min_val,
            "max": max_val,
            "feature_range": feature_range
        }

        return X_train_norm, X_test_norm, stats

    def get_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """
        Calculate class weights for imbalanced datasets.

        Uses inverse frequency weighting.

        Args:
            y: Label vector

        Returns:
            Dictionary mapping class label to weight
        """
        unique, counts = np.unique(y, return_counts=True)
        n_samples = len(y)
        n_classes = len(unique)

        weights = {}
        for cls, count in zip(unique, counts):
            weights[int(cls)] = n_samples / (n_classes * count)

        return weights

    def print_dataset_info(self, name: str):
        """Print summary information about a dataset."""
        try:
            X, y, info = self.load_dataset(name)
            print(f"\n{'='*50}")
            print(f"Dataset: {name}")
            print(f"{'='*50}")
            print(f"Samples: {info['n_samples']}")
            print(f"Features: {info['n_features']}")
            print(f"Classes: {info['n_classes']}")
            print(f"\nClass distribution:")
            for cls, count in info['class_distribution'].items():
                pct = count / info['n_samples'] * 100
                cls_name = info['class_names'][int(cls)] if info.get('class_names') else f"Class {cls}"
                print(f"  {cls_name}: {count} ({pct:.1f}%)")
            print(f"\nCreated: {info.get('created', 'Unknown')}")
        except FileNotFoundError as e:
            print(f"Error: {e}")


# =============================================================================
# Test functions
# =============================================================================

def test_dataset_manager():
    """Test DatasetManager functionality."""
    print("Testing DatasetManager...")

    manager = DatasetManager(data_dir="data_test")

    # Create dummy dataset
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.array([0]*40 + [1]*35 + [2]*25)
    np.random.shuffle(y)

    # Test train/test split
    print("\n1. Testing train_test_split (stratified)...")
    X_train, X_test, y_train, y_test = manager.train_test_split(X, y, test_size=0.2)
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Train class dist: {np.unique(y_train, return_counts=True)}")
    print(f"Test class dist: {np.unique(y_test, return_counts=True)}")

    # Test k-fold
    print("\n2. Testing k_fold_split (stratified)...")
    folds = manager.k_fold_split(X, y, n_folds=5)
    for i, (train_idx, test_idx) in enumerate(folds):
        print(f"Fold {i+1}: train={len(train_idx)}, test={len(test_idx)}")

    # Test normalization
    print("\n3. Testing normalize...")
    X_train_norm, X_test_norm, stats = manager.normalize(X_train, X_test)
    print(f"Train mean: {X_train_norm.mean(axis=0)[:3]}...")
    print(f"Train std: {X_train_norm.std(axis=0)[:3]}...")

    # Test save/load
    print("\n4. Testing save/load...")
    manager.save_dataset(X, y, "test_dataset",
                        feature_names=[f"feat_{i}" for i in range(10)],
                        class_names=["Fixation", "Saccade", "Blink"])

    X_loaded, y_loaded, info = manager.load_dataset("test_dataset")
    print(f"Loaded: {X_loaded.shape}, {y_loaded.shape}")

    # Print info
    manager.print_dataset_info("test_dataset")

    # List datasets
    print("\n5. Available datasets:", manager.list_datasets())

    # Class weights
    print("\n6. Class weights:", manager.get_class_weights(y))

    # Cleanup test directory
    import shutil
    shutil.rmtree("data_test", ignore_errors=True)
    print("\nTest cleanup complete.")


if __name__ == "__main__":
    test_dataset_manager()
