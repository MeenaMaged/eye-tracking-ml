"""
Head pose estimation using OpenCV solvePnP.

Provides pitch, yaw, roll angles to compensate for head movement
in gaze estimation. Uses a generic 3D face model and camera matrix.

Note: This uses OpenCV's solvePnP (geometric solution), not ML-based.
"""

import cv2
import numpy as np
from typing import Dict, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import Landmarks, MODEL_POINTS_3D, FRAME_WIDTH, FRAME_HEIGHT


class HeadPoseEstimator:
    """
    Estimate head pose using 6 facial landmarks and a generic 3D face model.

    Uses OpenCV's solvePnP to estimate rotation and translation vectors,
    then converts to Euler angles (pitch, yaw, roll).

    Pitch: Looking up (+) or down (-)
    Yaw: Looking left (-) or right (+)
    Roll: Tilting head left (-) or right (+)
    """

    # Landmark indices for head pose estimation
    LANDMARK_INDICES = [
        Landmarks.NOSE_TIP,                # 0: Nose tip
        Landmarks.CHIN,                    # 1: Chin
        Landmarks.LEFT_EYE_OUTER_CORNER,   # 2: Left eye outer
        Landmarks.RIGHT_EYE_OUTER_CORNER,  # 3: Right eye outer
        Landmarks.LEFT_MOUTH_CORNER,       # 4: Left mouth
        Landmarks.RIGHT_MOUTH_CORNER,      # 5: Right mouth
    ]

    def __init__(self, frame_width: int = FRAME_WIDTH, frame_height: int = FRAME_HEIGHT):
        """
        Initialize with camera parameters.

        Args:
            frame_width: Image width in pixels
            frame_height: Image height in pixels
        """
        self.frame_width = frame_width
        self.frame_height = frame_height

        # 3D model points (from config)
        self.model_points_3d = MODEL_POINTS_3D

        # Approximate camera matrix (assuming no lens distortion)
        # Focal length approximated as frame width
        focal_length = frame_width
        center = (frame_width / 2, frame_height / 2)

        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        # Assume no lens distortion
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    def extract(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Extract head pose angles from landmarks.

        Args:
            landmarks: Normalized landmarks array (478, 3)

        Returns:
            Dictionary with:
            - pitch: Up/down rotation (degrees), + = looking up
            - yaw: Left/right rotation (degrees), + = looking right
            - roll: Tilt rotation (degrees), + = tilting right
        """
        # Get 2D image points (convert normalized to pixel coordinates)
        image_points = np.array([
            landmarks[idx, :2] for idx in self.LANDMARK_INDICES
        ], dtype=np.float64)

        # Scale to pixel coordinates
        image_points[:, 0] *= self.frame_width
        image_points[:, 1] *= self.frame_height

        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points_3d,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}

        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Extract Euler angles from rotation matrix
        pitch, yaw, roll = self._rotation_matrix_to_euler(rotation_matrix)

        return {
            'pitch': float(np.degrees(pitch)),
            'yaw': float(np.degrees(yaw)),
            'roll': float(np.degrees(roll)),
        }

    def _rotation_matrix_to_euler(self, R: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert rotation matrix to Euler angles (pitch, yaw, roll).

        Uses the ZYX Euler angle convention.

        Args:
            R: 3x3 rotation matrix

        Returns:
            Tuple of (pitch, yaw, roll) in radians
        """
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

        singular = sy < 1e-6

        if not singular:
            pitch = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(-R[2, 0], sy)
            roll = np.arctan2(R[1, 0], R[0, 0])
        else:
            pitch = np.arctan2(-R[1, 2], R[1, 1])
            yaw = np.arctan2(-R[2, 0], sy)
            roll = 0

        return pitch, yaw, roll

    def get_pose_direction(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Get 3D direction vector of head pose (for visualization).

        Args:
            landmarks: Normalized landmarks array

        Returns:
            3D direction vector
        """
        pose = self.extract(landmarks)
        pitch = np.radians(pose['pitch'])
        yaw = np.radians(pose['yaw'])

        # Calculate direction vector
        x = -np.sin(yaw)
        y = np.sin(pitch)
        z = -np.cos(pitch) * np.cos(yaw)

        return np.array([x, y, z])

    @property
    def feature_names(self) -> list:
        """List of feature names this extractor produces."""
        return ['pitch', 'yaw', 'roll']
