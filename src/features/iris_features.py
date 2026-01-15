"""
Extract iris position features from landmarks.

Iris position ratios are distance-invariant features that form
the primary input for gaze estimation.
"""

import numpy as np
from typing import Dict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import Landmarks


class IrisFeatureExtractor:
    """
    Calculate normalized iris position ratios.

    These ratios are distance-invariant and form the primary gaze features:
    - X ratio: 0 (looking at inner corner) to 1 (looking at outer corner)
    - Y ratio: 0 (looking up) to 1 (looking down)

    The ratios are calculated relative to the eye boundaries, making them
    robust to variations in distance from camera and face size.
    """

    def __init__(self):
        """Initialize with landmark indices from config."""
        # Iris centers
        self.LEFT_IRIS = Landmarks.LEFT_IRIS_CENTER
        self.RIGHT_IRIS = Landmarks.RIGHT_IRIS_CENTER

        # Eye corners (horizontal reference)
        self.LEFT_EYE_INNER = Landmarks.LEFT_EYE_INNER
        self.LEFT_EYE_OUTER = Landmarks.LEFT_EYE_OUTER
        self.RIGHT_EYE_INNER = Landmarks.RIGHT_EYE_INNER
        self.RIGHT_EYE_OUTER = Landmarks.RIGHT_EYE_OUTER

        # Eyelid points (vertical reference)
        self.LEFT_EYE_TOP = Landmarks.LEFT_EYE_TOP
        self.LEFT_EYE_BOTTOM = Landmarks.LEFT_EYE_BOTTOM
        self.RIGHT_EYE_TOP = Landmarks.RIGHT_EYE_TOP
        self.RIGHT_EYE_BOTTOM = Landmarks.RIGHT_EYE_BOTTOM

    def extract(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Extract iris position ratios from landmarks.

        Args:
            landmarks: Array of shape (478, 3) with normalized coordinates

        Returns:
            Dictionary with iris features:
            - left_iris_x_ratio: 0 (inner) to 1 (outer)
            - left_iris_y_ratio: 0 (top) to 1 (bottom)
            - right_iris_x_ratio: 0 (inner) to 1 (outer)
            - right_iris_y_ratio: 0 (top) to 1 (bottom)
        """
        # Left eye horizontal ratio
        left_iris = landmarks[self.LEFT_IRIS, :2]
        left_inner = landmarks[self.LEFT_EYE_INNER, :2]
        left_outer = landmarks[self.LEFT_EYE_OUTER, :2]

        left_eye_width = np.linalg.norm(left_outer - left_inner)
        if left_eye_width > 0:
            # Project iris onto the eye axis
            left_iris_to_inner = np.linalg.norm(left_iris - left_inner)
            left_x_ratio = left_iris_to_inner / left_eye_width
        else:
            left_x_ratio = 0.5

        # Left eye vertical ratio
        left_top = landmarks[self.LEFT_EYE_TOP, :2]
        left_bottom = landmarks[self.LEFT_EYE_BOTTOM, :2]

        left_eye_height = np.linalg.norm(left_bottom - left_top)
        if left_eye_height > 0:
            left_iris_to_top = np.linalg.norm(left_iris - left_top)
            left_y_ratio = left_iris_to_top / left_eye_height
        else:
            left_y_ratio = 0.5

        # Right eye horizontal ratio
        right_iris = landmarks[self.RIGHT_IRIS, :2]
        right_inner = landmarks[self.RIGHT_EYE_INNER, :2]
        right_outer = landmarks[self.RIGHT_EYE_OUTER, :2]

        right_eye_width = np.linalg.norm(right_outer - right_inner)
        if right_eye_width > 0:
            right_iris_to_inner = np.linalg.norm(right_iris - right_inner)
            right_x_ratio = right_iris_to_inner / right_eye_width
        else:
            right_x_ratio = 0.5

        # Right eye vertical ratio
        right_top = landmarks[self.RIGHT_EYE_TOP, :2]
        right_bottom = landmarks[self.RIGHT_EYE_BOTTOM, :2]

        right_eye_height = np.linalg.norm(right_bottom - right_top)
        if right_eye_height > 0:
            right_iris_to_top = np.linalg.norm(right_iris - right_top)
            right_y_ratio = right_iris_to_top / right_eye_height
        else:
            right_y_ratio = 0.5

        # Clamp ratios to [0, 1] range
        left_x_ratio = np.clip(left_x_ratio, 0.0, 1.0)
        left_y_ratio = np.clip(left_y_ratio, 0.0, 1.0)
        right_x_ratio = np.clip(right_x_ratio, 0.0, 1.0)
        right_y_ratio = np.clip(right_y_ratio, 0.0, 1.0)

        return {
            'left_iris_x_ratio': float(left_x_ratio),
            'left_iris_y_ratio': float(left_y_ratio),
            'right_iris_x_ratio': float(right_x_ratio),
            'right_iris_y_ratio': float(right_y_ratio),
        }

    @property
    def feature_names(self) -> list:
        """List of feature names this extractor produces."""
        return [
            'left_iris_x_ratio',
            'left_iris_y_ratio',
            'right_iris_x_ratio',
            'right_iris_y_ratio',
        ]
