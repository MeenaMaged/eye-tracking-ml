"""
Calibration data storage and management.

Stores feature samples collected during calibration along with
their corresponding screen coordinates.
"""

import numpy as np
import json
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from datetime import datetime
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import CALIBRATION_DATA_DIR


@dataclass
class CalibrationSample:
    """Single calibration sample with features and target position."""
    screen_x: int
    screen_y: int
    features: np.ndarray
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'screen_x': self.screen_x,
            'screen_y': self.screen_y,
            'features': self.features.tolist(),
            'timestamp': self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'CalibrationSample':
        """Create from dictionary."""
        return cls(
            screen_x=data['screen_x'],
            screen_y=data['screen_y'],
            features=np.array(data['features'], dtype=np.float32),
            timestamp=data.get('timestamp', 0.0),
        )


@dataclass
class CalibrationData:
    """
    Complete calibration session data.

    Stores all samples from a calibration session along with metadata.
    Provides methods for conversion to training arrays and persistence.
    """
    samples: List[CalibrationSample] = field(default_factory=list)
    screen_width: int = 1920
    screen_height: int = 1080
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    num_points: int = 9

    def add_sample(self, screen_x: int, screen_y: int, features: np.ndarray):
        """
        Add a calibration sample.

        Args:
            screen_x: Target X coordinate on screen
            screen_y: Target Y coordinate on screen
            features: Feature vector from FeaturePipeline
        """
        sample = CalibrationSample(
            screen_x=screen_x,
            screen_y=screen_y,
            features=features.copy() if features is not None else np.zeros(10),
        )
        self.samples.append(sample)

    def add_samples(self, screen_x: int, screen_y: int, features_list: List[np.ndarray]):
        """
        Add multiple samples for the same calibration point.

        Args:
            screen_x: Target X coordinate
            screen_y: Target Y coordinate
            features_list: List of feature vectors
        """
        for features in features_list:
            self.add_sample(screen_x, screen_y, features)

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert to X, y arrays for training.

        Returns:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target coordinates of shape (n_samples, 2)
        """
        if not self.samples:
            return np.array([]), np.array([])

        X = np.array([s.features for s in self.samples], dtype=np.float32)
        y = np.array([[s.screen_x, s.screen_y] for s in self.samples], dtype=np.float32)
        return X, y

    def get_point_samples(self, point_index: int, samples_per_point: int = 10
                          ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]]:
        """
        Get samples for a specific calibration point.

        Args:
            point_index: Index of calibration point
            samples_per_point: Number of samples per point

        Returns:
            Tuple of (features array, (screen_x, screen_y)) or (None, None)
        """
        start_idx = point_index * samples_per_point
        end_idx = start_idx + samples_per_point

        if end_idx > len(self.samples):
            return None, None

        point_samples = self.samples[start_idx:end_idx]
        features = np.array([s.features for s in point_samples])
        target = (point_samples[0].screen_x, point_samples[0].screen_y)

        return features, target

    def clear(self):
        """Clear all samples."""
        self.samples = []

    def save(self, filename: Optional[str] = None) -> str:
        """
        Save calibration data to JSON file.

        Args:
            filename: Optional filename, auto-generated if not provided

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"calibration_{timestamp}.json"

        filepath = os.path.join(CALIBRATION_DATA_DIR, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        data = {
            'screen_width': self.screen_width,
            'screen_height': self.screen_height,
            'created_at': self.created_at,
            'num_points': self.num_points,
            'samples': [s.to_dict() for s in self.samples],
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Calibration data saved to {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath: str) -> 'CalibrationData':
        """
        Load calibration data from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            CalibrationData instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        calibration = cls(
            screen_width=data['screen_width'],
            screen_height=data['screen_height'],
            created_at=data.get('created_at', 0.0),
            num_points=data.get('num_points', 9),
        )

        for sample_dict in data['samples']:
            sample = CalibrationSample.from_dict(sample_dict)
            calibration.samples.append(sample)

        return calibration

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def is_complete(self) -> bool:
        """Check if calibration has expected number of samples."""
        expected = self.num_points * 10  # Assuming 10 samples per point
        return len(self.samples) >= expected
