"""
MediaPipe FaceLandmarker wrapper for face and eye landmark detection.

This is the ONLY pre-trained model in the system (CNNs not covered in course).
Provides 478 facial landmarks including 10 iris landmarks.

Updated for MediaPipe Tasks API (0.10.30+).
"""

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
from typing import Optional, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MAX_NUM_FACES,
)

# Path to the model file
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models",
    "face_landmarker.task"
)

# Face mesh contour connections (for drawing)
# These are the key facial contour connections
FACEMESH_CONTOURS = frozenset([
    # Lips
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405),
    (405, 321), (321, 375), (375, 291), (61, 185), (185, 40), (40, 39), (39, 37),
    (37, 0), (0, 267), (267, 269), (269, 270), (270, 409), (409, 291),
    # Left eye
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154), (154, 155),
    (155, 133), (33, 246), (246, 161), (161, 160), (160, 159), (159, 158), (158, 157),
    (157, 173), (173, 133),
    # Right eye
    (362, 382), (382, 381), (381, 380), (380, 374), (374, 373), (373, 390), (390, 249),
    (249, 263), (263, 466), (466, 388), (388, 387), (387, 386), (386, 385), (385, 384),
    (384, 398), (398, 362),
    # Left eyebrow
    (46, 53), (53, 52), (52, 65), (65, 55), (70, 63), (63, 105), (105, 66), (66, 107),
    # Right eyebrow
    (276, 283), (283, 282), (282, 295), (295, 285), (300, 293), (293, 334), (334, 296), (296, 336),
    # Face oval
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389), (389, 356),
    (356, 454), (454, 323), (323, 361), (361, 288), (288, 397), (397, 365), (365, 379),
    (379, 378), (378, 400), (400, 377), (377, 152), (152, 148), (148, 176), (176, 149),
    (149, 150), (150, 136), (136, 172), (172, 58), (58, 132), (132, 93), (93, 234),
    (234, 127), (127, 162), (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10),
])


class FaceDetector:
    """
    Wrapper for MediaPipe FaceLandmarker (Tasks API).

    Provides 478 facial landmarks including iris landmarks:
    - 468 standard face mesh landmarks
    - 10 iris landmarks (468-477): 5 per eye

    Usage:
        detector = FaceDetector()
        landmarks = detector.detect(frame)
        if landmarks is not None:
            pixel_coords = detector.get_pixel_coordinates(landmarks, width, height)
    """

    def __init__(
        self,
        min_detection_confidence: float = MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = MIN_TRACKING_CONFIDENCE,
        max_num_faces: int = MAX_NUM_FACES,
        model_path: str = MODEL_PATH,
    ):
        """
        Initialize MediaPipe FaceLandmarker.

        Args:
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for landmark tracking
            max_num_faces: Maximum number of faces to detect
            model_path: Path to the face_landmarker.task model file
        """
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                "Download from: https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            )

        # Create options for the FaceLandmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_detection_success = False

    def detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect face landmarks in frame.

        Args:
            frame: BGR image from webcam (H, W, 3)

        Returns:
            landmarks: Array of shape (478, 3) with (x, y, z) normalized
                       coordinates (0-1 range), or None if no face detected
        """
        # Convert BGR to RGB (MediaPipe requirement)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Process frame
        results = self.face_landmarker.detect(mp_image)

        if not results.face_landmarks:
            self._last_detection_success = False
            return None

        self._last_detection_success = True

        # Extract first face landmarks
        face_landmarks = results.face_landmarks[0]

        # Convert to numpy array
        landmarks = np.array([
            [lm.x, lm.y, lm.z]
            for lm in face_landmarks
        ], dtype=np.float32)

        return landmarks

    def get_pixel_coordinates(
        self,
        landmarks: np.ndarray,
        frame_width: int,
        frame_height: int
    ) -> np.ndarray:
        """
        Convert normalized landmarks to pixel coordinates.

        Args:
            landmarks: Normalized landmarks (0-1 range), shape (478, 3)
            frame_width: Image width in pixels
            frame_height: Image height in pixels

        Returns:
            Pixel coordinates array of shape (478, 2) as integers
        """
        pixel_coords = landmarks[:, :2].copy()
        pixel_coords[:, 0] *= frame_width
        pixel_coords[:, 1] *= frame_height
        return pixel_coords.astype(np.int32)

    def get_landmark_point(
        self,
        landmarks: np.ndarray,
        index: int,
        frame_width: int,
        frame_height: int
    ) -> Tuple[int, int]:
        """
        Get a single landmark as pixel coordinates.

        Args:
            landmarks: Normalized landmarks array
            index: Landmark index (0-477)
            frame_width: Image width
            frame_height: Image height

        Returns:
            Tuple of (x, y) pixel coordinates
        """
        x = int(landmarks[index, 0] * frame_width)
        y = int(landmarks[index, 1] * frame_height)
        return (x, y)

    def draw_landmarks(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        draw_iris: bool = True,
        draw_contours: bool = True
    ) -> np.ndarray:
        """
        Draw landmarks on frame for visualization.

        Args:
            frame: BGR image to draw on
            landmarks: Normalized landmarks array
            draw_iris: If True, draw iris landmarks
            draw_contours: If True, draw face mesh contours

        Returns:
            Frame with landmarks drawn
        """
        output = frame.copy()
        h, w = frame.shape[:2]

        if draw_contours:
            # Draw face mesh connections
            for connection in FACEMESH_CONTOURS:
                start_idx, end_idx = connection
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start_point = self.get_landmark_point(landmarks, start_idx, w, h)
                    end_point = self.get_landmark_point(landmarks, end_idx, w, h)
                    cv2.line(output, start_point, end_point, (200, 200, 200), 1)

        if draw_iris and len(landmarks) >= 478:
            # Draw iris landmarks (468-477)
            # Left iris
            for idx in range(468, 473):
                point = self.get_landmark_point(landmarks, idx, w, h)
                color = (0, 255, 0) if idx == 468 else (0, 200, 0)  # Brighter for center
                radius = 3 if idx == 468 else 2
                cv2.circle(output, point, radius, color, -1)

            # Right iris
            for idx in range(473, 478):
                point = self.get_landmark_point(landmarks, idx, w, h)
                color = (0, 255, 0) if idx == 473 else (0, 200, 0)
                radius = 3 if idx == 473 else 2
                cv2.circle(output, point, radius, color, -1)

        return output

    @property
    def last_detection_success(self) -> bool:
        """Whether the last detection was successful."""
        return self._last_detection_success

    @property
    def num_landmarks(self) -> int:
        """Number of landmarks detected (always 478 with FaceLandmarker)."""
        return 478

    def close(self):
        """Release MediaPipe resources."""
        self.face_landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# =============================================================================
# Test function
# =============================================================================

def test_face_detector():
    """Test face detector with live webcam."""
    print("Testing FaceDetector...")
    print("Press 'q' to quit")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    detector = FaceDetector()

    detection_count = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror
        frame_count += 1

        landmarks = detector.detect(frame)

        if landmarks is not None:
            detection_count += 1
            frame = detector.draw_landmarks(frame, landmarks)

            # Show iris centers
            h, w = frame.shape[:2]
            left_iris = detector.get_landmark_point(landmarks, 468, w, h)
            right_iris = detector.get_landmark_point(landmarks, 473, w, h)

            cv2.circle(frame, left_iris, 5, (255, 0, 0), -1)
            cv2.circle(frame, right_iris, 5, (255, 0, 0), -1)

            status = "Face Detected"
        else:
            status = "No Face"

        # Display status
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                   (0, 255, 0) if detector.last_detection_success else (0, 0, 255), 2)

        detection_rate = detection_count / frame_count * 100 if frame_count > 0 else 0
        cv2.putText(frame, f"Detection: {detection_rate:.1f}%", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Face Detector Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

    print(f"Test complete: {detection_count}/{frame_count} frames detected ({detection_rate:.1f}%)")


if __name__ == "__main__":
    test_face_detector()
