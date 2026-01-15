"""
Data collectors for blink detection and movement classification.

Provides labeled data collection with automatic and manual labeling support.
"""

import numpy as np
import time
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import EAR_THRESHOLD, VELOCITY_THRESHOLD, DISPERSION_THRESHOLD, FEATURE_NAMES


@dataclass
class LabeledSample:
    """Single labeled sample for ML training."""
    features: np.ndarray
    label: int
    timestamp: float


class BlinkDataCollector:
    """
    Collect blink detection training data.

    Uses EAR (Eye Aspect Ratio) threshold for automatic labeling with
    optional manual label override. Extracts temporal features for
    improved blink detection accuracy.

    Output feature vector (19 dimensions):
    - 9 temporal EAR features (current, history, statistics)
    - 10 base features from FeaturePipeline

    Labels:
    - 0: No blink (eyes open)
    - 1: Blink (eyes closed)
    """

    # Temporal feature names
    TEMPORAL_FEATURE_NAMES = [
        'ear_current',
        'ear_t-3',
        'ear_t-6',
        'ear_t-9',
        'ear_min_window',
        'ear_max_window',
        'ear_mean_window',
        'ear_std_window',
        'ear_delta',
    ]

    def __init__(self, ear_threshold: float = EAR_THRESHOLD, window_size: int = 15):
        """
        Initialize BlinkDataCollector.

        Args:
            ear_threshold: EAR below this value = blink (default 0.25)
            window_size: Number of frames to track for temporal features
        """
        self.ear_threshold = ear_threshold
        self.window_size = window_size
        self.samples: List[LabeledSample] = []
        self.ear_history: deque = deque(maxlen=window_size)

    def add_frame(self, features: np.ndarray, ear: float,
                  manual_label: Optional[int] = None) -> int:
        """
        Add a frame and return its label.

        Args:
            features: Base 10-dimensional feature vector from FeaturePipeline
            ear: Average EAR value for this frame
            manual_label: Override auto-label if provided (0 or 1)

        Returns:
            Label assigned to this frame (0=no blink, 1=blink)
        """
        self.ear_history.append(ear)

        # Auto-label based on EAR threshold, or use manual override
        if manual_label is not None:
            label = manual_label
        else:
            label = 1 if ear < self.ear_threshold else 0

        # Extract temporal features for better blink detection
        temporal_features = self._extract_temporal_features(features, ear)

        self.samples.append(LabeledSample(
            features=temporal_features,
            label=label,
            timestamp=time.time()
        ))

        return label

    def _extract_temporal_features(self, base_features: np.ndarray, ear: float) -> np.ndarray:
        """
        Extract temporal features for blink detection.

        Combines current EAR with historical statistics for improved
        blink detection that captures the temporal dynamics of blinking.

        Args:
            base_features: 10-dimensional feature vector from FeaturePipeline
            ear: Current average EAR value

        Returns:
            19-dimensional feature vector (9 temporal + 10 base)
        """
        history = list(self.ear_history)

        temporal = [
            ear,  # Current EAR
            history[-4] if len(history) > 3 else ear,   # EAR at t-3 frames
            history[-7] if len(history) > 6 else ear,   # EAR at t-6 frames
            history[-10] if len(history) > 9 else ear,  # EAR at t-9 frames
            min(history) if history else ear,           # Min EAR in window
            max(history) if history else ear,           # Max EAR in window
            np.mean(history) if history else ear,       # Mean EAR in window
            np.std(history) if len(history) > 1 else 0, # Std EAR in window
            ear - history[-2] if len(history) > 1 else 0,  # EAR delta (rate of change)
        ]

        return np.concatenate([temporal, base_features])

    def get_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get collected dataset as numpy arrays.

        Returns:
            Tuple of (X, y) where:
            - X: Feature matrix of shape (n_samples, 19)
            - y: Label vector of shape (n_samples,)
        """
        if not self.samples:
            return np.array([]).reshape(0, 19), np.array([])

        X = np.array([s.features for s in self.samples])
        y = np.array([s.label for s in self.samples])
        return X, y

    def get_balanced_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get class-balanced dataset (equal blink/non-blink samples).

        Useful for training when there's significant class imbalance
        (typically many more non-blink frames than blink frames).

        Returns:
            Tuple of (X, y) with equal class representation
        """
        X, y = self.get_dataset()

        if len(X) == 0:
            return X, y

        n_blink = np.sum(y == 1)
        n_no_blink = np.sum(y == 0)
        min_size = min(n_blink, n_no_blink)

        if min_size == 0:
            return X, y

        # Sample equally from each class
        blink_idx = np.where(y == 1)[0][:min_size]
        no_blink_idx = np.where(y == 0)[0][:min_size]

        selected = np.concatenate([blink_idx, no_blink_idx])
        np.random.shuffle(selected)

        return X[selected], y[selected]

    def get_statistics(self) -> dict:
        """Get collection statistics."""
        X, y = self.get_dataset()
        if len(y) == 0:
            return {'total': 0, 'blinks': 0, 'no_blinks': 0, 'blink_ratio': 0}

        n_blinks = int(np.sum(y == 1))
        n_no_blinks = int(np.sum(y == 0))

        return {
            'total': len(y),
            'blinks': n_blinks,
            'no_blinks': n_no_blinks,
            'blink_ratio': n_blinks / len(y) if len(y) > 0 else 0
        }

    def reset(self):
        """Clear all collected samples."""
        self.samples.clear()
        self.ear_history.clear()

    @property
    def feature_names(self) -> List[str]:
        """Get names of all features in order."""
        return self.TEMPORAL_FEATURE_NAMES + FEATURE_NAMES

    @property
    def num_features(self) -> int:
        """Number of features per sample."""
        return 19  # 9 temporal + 10 base


class MovementDataCollector:
    """
    Collect eye movement classification data.

    Uses I-VT (velocity threshold) and I-DT (dispersion threshold)
    algorithms for automatic movement classification.

    Output feature vector: 14 dimensions

    Labels:
    - 0: Fixation (stable gaze)
    - 1: Saccade (rapid movement)
    - 2: Blink (eyes closed)
    """

    # Movement class constants
    FIXATION = 0
    SACCADE = 1
    BLINK = 2

    # Class names for display
    CLASS_NAMES = ['Fixation', 'Saccade', 'Blink']

    # Feature names for the 14-dimensional output
    FEATURE_NAMES = [
        'velocity_current',
        'velocity_mean',
        'velocity_max',
        'acceleration_current',
        'acceleration_mean_abs',
        'dispersion',
        'rms_deviation',
        'direction_consistency',
        'ear_mean',
        'ear_min',
        'ear_std',
        'velocity_std',
        'x_range',
        'y_range',
    ]

    def __init__(self,
                 velocity_threshold: float = VELOCITY_THRESHOLD,
                 dispersion_threshold: float = DISPERSION_THRESHOLD,
                 ear_threshold: float = EAR_THRESHOLD,
                 window_size: int = 20):
        """
        Initialize MovementDataCollector.

        Args:
            velocity_threshold: Velocity above this = saccade (deg/s)
            dispersion_threshold: Dispersion above this = saccade (pixels)
            ear_threshold: EAR below this = blink
            window_size: Number of frames for feature calculation
        """
        self.velocity_threshold = velocity_threshold
        self.dispersion_threshold = dispersion_threshold
        self.ear_threshold = ear_threshold
        self.window_size = window_size
        self.samples: List[LabeledSample] = []
        self.gaze_history: deque = deque(maxlen=window_size)

    def add_frame(self, gaze_x: float, gaze_y: float, ear: float,
                  timestamp: float, manual_label: Optional[int] = None) -> int:
        """
        Add frame with gaze position and return movement classification.

        Args:
            gaze_x: Gaze X position (screen coordinates or normalized)
            gaze_y: Gaze Y position
            ear: Average EAR value
            timestamp: Frame timestamp in seconds
            manual_label: Override auto-classification if provided

        Returns:
            Movement class (0=Fixation, 1=Saccade, 2=Blink)
        """
        self.gaze_history.append((gaze_x, gaze_y, timestamp, ear))

        # Need at least 3 frames for velocity calculation
        if len(self.gaze_history) < 3:
            return self.FIXATION

        # Auto-classify or use manual label
        if manual_label is not None:
            label = manual_label
        else:
            label = self._auto_classify(ear)

        features = self._extract_features()

        self.samples.append(LabeledSample(
            features=features,
            label=label,
            timestamp=timestamp
        ))

        return label

    def _auto_classify(self, ear: float) -> int:
        """
        Automatic movement classification using I-VT and I-DT algorithms.

        Priority:
        1. Blink (EAR below threshold)
        2. Saccade (high velocity or high dispersion)
        3. Fixation (default)

        Args:
            ear: Current EAR value

        Returns:
            Movement class
        """
        # Blink detection has highest priority
        if ear < self.ear_threshold:
            return self.BLINK

        # Velocity-based saccade detection (I-VT)
        velocity = self._calculate_velocity()
        if velocity > self.velocity_threshold:
            return self.SACCADE

        # Dispersion-based classification (I-DT)
        dispersion = self._calculate_dispersion()
        if dispersion > self.dispersion_threshold:
            return self.SACCADE

        return self.FIXATION

    def _calculate_velocity(self) -> float:
        """Calculate instantaneous gaze velocity (pixels/second)."""
        if len(self.gaze_history) < 2:
            return 0.0

        p1, p2 = self.gaze_history[-2], self.gaze_history[-1]
        dt = p2[2] - p1[2]  # Time difference

        if dt <= 0:
            return 0.0

        # Euclidean distance / time
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        distance = np.sqrt(dx**2 + dy**2)

        return distance / dt

    def _calculate_dispersion(self) -> float:
        """Calculate spatial dispersion in recent window (I-DT metric)."""
        if len(self.gaze_history) < 3:
            return 0.0

        points = list(self.gaze_history)
        x = [p[0] for p in points]
        y = [p[1] for p in points]

        # Dispersion = (max_x - min_x) + (max_y - min_y)
        return (max(x) - min(x)) + (max(y) - min(y))

    def _extract_features(self) -> np.ndarray:
        """
        Extract 14-dimensional movement features.

        Features capture velocity, acceleration, spatial dispersion,
        direction consistency, and EAR statistics.

        Returns:
            14-dimensional feature vector
        """
        points = list(self.gaze_history)

        if len(points) < 5:
            return np.zeros(14)

        # Extract coordinates and time
        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])
        t = np.array([p[2] for p in points])
        ear = np.array([p[3] for p in points])

        # Calculate time differences
        dt = np.diff(t)
        dt[dt == 0] = 1e-6  # Prevent division by zero

        # Velocity features
        distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        velocities = distances / dt

        # Acceleration features
        if len(velocities) > 1:
            accelerations = np.diff(velocities) / dt[1:]
        else:
            accelerations = np.array([0])

        # Spatial features
        dispersion = (np.max(x) - np.min(x)) + (np.max(y) - np.min(y))
        rms = np.sqrt(np.mean((x - np.mean(x))**2 + (y - np.mean(y))**2))

        # Direction consistency (Rayleigh test approximation)
        # High value = consistent direction, low value = random movement
        if len(x) > 1:
            angles = np.arctan2(np.diff(y), np.diff(x))
            r = np.sqrt(np.mean(np.cos(angles))**2 + np.mean(np.sin(angles))**2)
            direction_consistency = r
        else:
            direction_consistency = 0

        return np.array([
            velocities[-1] if len(velocities) > 0 else 0,  # Current velocity
            np.mean(velocities) if len(velocities) > 0 else 0,  # Mean velocity
            np.max(velocities) if len(velocities) > 0 else 0,   # Max velocity
            accelerations[-1] if len(accelerations) > 0 else 0,  # Current acceleration
            np.mean(np.abs(accelerations)) if len(accelerations) > 0 else 0,  # Mean abs acceleration
            dispersion,
            rms,
            direction_consistency,
            np.mean(ear),   # Mean EAR
            np.min(ear),    # Min EAR (for blink detection)
            np.std(ear),    # EAR variation
            np.std(velocities) if len(velocities) > 0 else 0,  # Velocity variation
            np.max(x) - np.min(x),  # X range
            np.max(y) - np.min(y),  # Y range
        ], dtype=np.float32)

    def get_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get collected dataset as numpy arrays.

        Returns:
            Tuple of (X, y) where:
            - X: Feature matrix of shape (n_samples, 14)
            - y: Label vector of shape (n_samples,) with values 0, 1, or 2
        """
        if not self.samples:
            return np.array([]).reshape(0, 14), np.array([])

        X = np.array([s.features for s in self.samples])
        y = np.array([s.label for s in self.samples])
        return X, y

    def get_statistics(self) -> dict:
        """Get collection statistics per class."""
        X, y = self.get_dataset()

        if len(y) == 0:
            return {
                'total': 0,
                'fixations': 0,
                'saccades': 0,
                'blinks': 0
            }

        return {
            'total': len(y),
            'fixations': int(np.sum(y == self.FIXATION)),
            'saccades': int(np.sum(y == self.SACCADE)),
            'blinks': int(np.sum(y == self.BLINK)),
            'class_ratios': {
                'fixation': float(np.sum(y == self.FIXATION) / len(y)),
                'saccade': float(np.sum(y == self.SACCADE) / len(y)),
                'blink': float(np.sum(y == self.BLINK) / len(y)),
            }
        }

    def reset(self):
        """Clear all collected samples."""
        self.samples.clear()
        self.gaze_history.clear()

    @property
    def feature_names(self) -> List[str]:
        """Get names of all features in order."""
        return self.FEATURE_NAMES.copy()

    @property
    def num_features(self) -> int:
        """Number of features per sample."""
        return 14


# =============================================================================
# Test functions
# =============================================================================

def test_blink_collector():
    """Test BlinkDataCollector with simulated data."""
    print("Testing BlinkDataCollector...")

    collector = BlinkDataCollector(ear_threshold=0.2)

    # Simulate open eyes
    for _ in range(50):
        features = np.random.randn(10) * 0.1
        ear = 0.3 + np.random.randn() * 0.02
        collector.add_frame(features, ear)

    # Simulate blink
    for _ in range(10):
        features = np.random.randn(10) * 0.1
        ear = 0.1 + np.random.randn() * 0.02
        collector.add_frame(features, ear)

    # More open eyes
    for _ in range(40):
        features = np.random.randn(10) * 0.1
        ear = 0.3 + np.random.randn() * 0.02
        collector.add_frame(features, ear)

    X, y = collector.get_dataset()
    stats = collector.get_statistics()

    print(f"Collected: {stats}")
    print(f"Feature shape: {X.shape}")
    print(f"Feature names: {collector.feature_names[:5]}...")

    X_bal, y_bal = collector.get_balanced_dataset()
    print(f"Balanced dataset: {X_bal.shape}, class 0: {sum(y_bal==0)}, class 1: {sum(y_bal==1)}")


def test_movement_collector():
    """Test MovementDataCollector with simulated data."""
    print("\nTesting MovementDataCollector...")

    collector = MovementDataCollector()

    t = 0
    # Simulate fixation
    for i in range(30):
        x = 100 + np.random.randn() * 2
        y = 100 + np.random.randn() * 2
        collector.add_frame(x, y, 0.3, t)
        t += 0.033

    # Simulate saccade
    for i in range(10):
        x = 100 + i * 50
        y = 100 + i * 30
        collector.add_frame(x, y, 0.3, t)
        t += 0.033

    # Simulate blink
    for i in range(5):
        x = 600 + np.random.randn() * 2
        y = 400 + np.random.randn() * 2
        collector.add_frame(x, y, 0.1, t)
        t += 0.033

    X, y = collector.get_dataset()
    stats = collector.get_statistics()

    print(f"Collected: {stats}")
    print(f"Feature shape: {X.shape}")
    print(f"Feature names: {collector.feature_names}")


if __name__ == "__main__":
    test_blink_collector()
    test_movement_collector()
