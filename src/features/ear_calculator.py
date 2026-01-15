"""
Eye Aspect Ratio (EAR) calculation for blink detection.

EAR is a scalar quantity that measures the eye openness.
When the eye is open, EAR is approximately 0.25-0.35.
When the eye is closed, EAR drops to approximately 0.
"""

import numpy as np
from typing import Dict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import Landmarks, EAR_THRESHOLD


class EARCalculator:
    """
    Calculate Eye Aspect Ratio for blink detection.

    EAR Formula:
        EAR = (||p2-p6|| + ||p3-p5||) / (2.0 × ||p1-p4||)

    Where p1-p6 are the 6 eye contour points:
        p1 = inner corner
        p2 = upper-inner lid
        p3 = upper-outer lid
        p4 = outer corner
        p5 = lower-outer lid
        p6 = lower-inner lid

    Reference:
        Soukupová and Čech, "Real-Time Eye Blink Detection using Facial Landmarks"
    """

    def __init__(self, ear_threshold: float = EAR_THRESHOLD):
        """
        Initialize EAR calculator.

        Args:
            ear_threshold: Threshold below which eye is considered closed
        """
        # 6 points around each eye for EAR calculation
        self.LEFT_EYE_INDICES = Landmarks.LEFT_EYE_EAR
        self.RIGHT_EYE_INDICES = Landmarks.RIGHT_EYE_EAR
        self.threshold = ear_threshold

    def _calculate_ear(self, eye_points: np.ndarray) -> float:
        """
        Calculate EAR for a single eye.

        Args:
            eye_points: Array of shape (6, 2) with eye contour points
                        Order: [p1, p2, p3, p4, p5, p6]
                        p1=inner corner, p4=outer corner
                        p2,p3=top, p5,p6=bottom

        Returns:
            EAR value (typically 0.2-0.3 when open, ~0.05 when closed)
        """
        # Vertical distances
        v1 = np.linalg.norm(eye_points[1] - eye_points[5])  # p2-p6
        v2 = np.linalg.norm(eye_points[2] - eye_points[4])  # p3-p5

        # Horizontal distance
        h = np.linalg.norm(eye_points[0] - eye_points[3])   # p1-p4

        if h < 1e-6:  # Avoid division by zero
            return 0.0

        ear = (v1 + v2) / (2.0 * h)
        return float(ear)

    def extract(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Extract EAR for both eyes from landmarks.

        Args:
            landmarks: Array of shape (478, 3) with normalized coordinates

        Returns:
            Dictionary with:
            - left_ear: Left eye EAR
            - right_ear: Right eye EAR
            - avg_ear: Average EAR of both eyes
            - is_blinking: Whether average EAR is below threshold
        """
        # Extract eye points (only x, y coordinates)
        left_eye = landmarks[self.LEFT_EYE_INDICES, :2]
        right_eye = landmarks[self.RIGHT_EYE_INDICES, :2]

        # Calculate EAR
        left_ear = self._calculate_ear(left_eye)
        right_ear = self._calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        # Detect blink
        is_blinking = avg_ear < self.threshold

        return {
            'left_ear': left_ear,
            'right_ear': right_ear,
            'avg_ear': avg_ear,
            'is_blinking': is_blinking,
        }

    def is_blinking(self, landmarks: np.ndarray) -> bool:
        """
        Quick check if eyes are closed (blinking).

        Args:
            landmarks: Landmark array from face detector

        Returns:
            True if average EAR is below threshold
        """
        features = self.extract(landmarks)
        return features['is_blinking']

    @property
    def feature_names(self) -> list:
        """List of feature names this extractor produces."""
        return ['left_ear', 'right_ear']
