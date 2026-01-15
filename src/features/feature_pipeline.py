"""
Complete feature extraction pipeline.

Combines all feature extractors into a single pipeline that produces
a fixed-size feature vector for gaze estimation and classification.
"""

import numpy as np
from typing import Optional, Dict, List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import Landmarks, FRAME_WIDTH, FRAME_HEIGHT, FEATURE_NAMES

from .iris_features import IrisFeatureExtractor
from .ear_calculator import EARCalculator
from .head_pose import HeadPoseEstimator


class FeaturePipeline:
    """
    Unified feature extraction pipeline.

    Produces a fixed-size 10-dimensional feature vector:
    [0] left_iris_x_ratio    - Left iris horizontal position (0-1)
    [1] left_iris_y_ratio    - Left iris vertical position (0-1)
    [2] right_iris_x_ratio   - Right iris horizontal position (0-1)
    [3] right_iris_y_ratio   - Right iris vertical position (0-1)
    [4] left_ear             - Left eye aspect ratio (~0.25 open, ~0 closed)
    [5] right_ear            - Right eye aspect ratio
    [6] pitch                - Head pitch in degrees (up/down)
    [7] yaw                  - Head yaw in degrees (left/right)
    [8] roll                 - Head roll in degrees (tilt)
    [9] inter_ocular_dist    - Distance between eyes (normalized)

    All features are designed to be distance-invariant for robust gaze estimation.
    """

    def __init__(self, frame_width: int = FRAME_WIDTH, frame_height: int = FRAME_HEIGHT):
        """
        Initialize all feature extractors.

        Args:
            frame_width: Frame width for head pose estimation
            frame_height: Frame height for head pose estimation
        """
        self.iris_extractor = IrisFeatureExtractor()
        self.ear_calculator = EARCalculator()
        self.head_pose_estimator = HeadPoseEstimator(frame_width, frame_height)

        self.frame_width = frame_width
        self.frame_height = frame_height

        # Feature configuration
        self._feature_names = FEATURE_NAMES
        self._num_features = len(FEATURE_NAMES)

    def extract(self, landmarks: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract complete feature vector from landmarks.

        Args:
            landmarks: Array of shape (478, 3) from MediaPipe FaceMesh

        Returns:
            Feature vector of shape (10,) or None if extraction fails
        """
        try:
            # Extract all features
            iris_features = self.iris_extractor.extract(landmarks)
            ear_features = self.ear_calculator.extract(landmarks)
            head_pose = self.head_pose_estimator.extract(landmarks)

            # Calculate inter-ocular distance (normalized by frame width)
            left_eye_center = landmarks[Landmarks.LEFT_EYE_INNER, :2]
            right_eye_center = landmarks[Landmarks.RIGHT_EYE_INNER, :2]
            inter_ocular = np.linalg.norm(right_eye_center - left_eye_center)
            # Normalize by approximate maximum inter-ocular distance
            inter_ocular_normalized = inter_ocular / 0.15  # Typical max ~0.15 in normalized coords

            # Combine into feature vector (must match FEATURE_NAMES order)
            features = np.array([
                iris_features['left_iris_x_ratio'],     # [0]
                iris_features['left_iris_y_ratio'],     # [1]
                iris_features['right_iris_x_ratio'],    # [2]
                iris_features['right_iris_y_ratio'],    # [3]
                ear_features['left_ear'],               # [4]
                ear_features['right_ear'],              # [5]
                head_pose['pitch'],                     # [6]
                head_pose['yaw'],                       # [7]
                head_pose['roll'],                      # [8]
                inter_ocular_normalized,                # [9]
            ], dtype=np.float32)

            return features

        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None

    def extract_dict(self, landmarks: np.ndarray) -> Optional[Dict[str, float]]:
        """
        Extract features as a dictionary (useful for debugging).

        Args:
            landmarks: Landmark array from face detector

        Returns:
            Dictionary with feature name -> value mapping, or None if failed
        """
        features = self.extract(landmarks)
        if features is None:
            return None

        return dict(zip(self._feature_names, features.tolist()))

    def extract_all(self, landmarks: np.ndarray) -> Optional[Dict]:
        """
        Extract all features including derived ones.

        Args:
            landmarks: Landmark array from face detector

        Returns:
            Dictionary with all extracted features and metadata
        """
        try:
            iris = self.iris_extractor.extract(landmarks)
            ear = self.ear_calculator.extract(landmarks)
            head = self.head_pose_estimator.extract(landmarks)

            # Feature vector
            features = self.extract(landmarks)

            return {
                'feature_vector': features,
                'iris': iris,
                'ear': ear,
                'head_pose': head,
                'is_blinking': ear.get('is_blinking', False),
            }

        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None

    @property
    def feature_names(self) -> List[str]:
        """List of feature names in order."""
        return self._feature_names.copy()

    @property
    def num_features(self) -> int:
        """Number of features in the feature vector."""
        return self._num_features


# =============================================================================
# Test function
# =============================================================================

def test_feature_pipeline():
    """Test feature extraction with live webcam."""
    import cv2
    from src.capture import WebcamCapture
    from src.detection import FaceDetector

    print("Testing FeaturePipeline...")
    print("Press 'q' to quit")

    with WebcamCapture() as cap:
        detector = FaceDetector()
        pipeline = FeaturePipeline()

        while True:
            frame = cap.get_frame()
            if frame is None:
                continue

            landmarks = detector.detect(frame)

            if landmarks is not None:
                # Extract features
                all_features = pipeline.extract_all(landmarks)

                if all_features is not None:
                    features = all_features['feature_vector']
                    is_blinking = all_features['is_blinking']

                    # Draw feature info
                    frame = detector.draw_landmarks(frame, landmarks)

                    y_offset = 30
                    for i, name in enumerate(pipeline.feature_names):
                        value = features[i]
                        text = f"{name}: {value:.3f}"
                        cv2.putText(frame, text, (10, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        y_offset += 20

                    # Blink indicator
                    blink_text = "BLINK!" if is_blinking else ""
                    cv2.putText(frame, blink_text, (frame.shape[1] - 100, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "No face detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Feature Pipeline Test", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_feature_pipeline()
