"""
Calibration workflow orchestration.

Manages the complete calibration process by coordinating
webcam capture, face detection, feature extraction, and calibration UI.
"""

import numpy as np
import threading
import time
from typing import Optional, Callable, List, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import CALIBRATION_POINTS, SAMPLES_PER_POINT

from src.capture import WebcamCapture
from src.detection import FaceDetector
from src.features import FeaturePipeline
from .calibration_data import CalibrationData
from .calibration_ui import CalibrationUI


class CalibrationManager:
    """
    Orchestrates the calibration workflow.

    Coordinates:
    - Webcam capture in background thread
    - Face detection and feature extraction
    - Calibration UI display
    - Sample collection and storage

    Usage:
        manager = CalibrationManager()
        success = manager.run_calibration()
        if success:
            X, y = manager.get_training_data()
    """

    def __init__(
        self,
        num_points: int = CALIBRATION_POINTS,
        samples_per_point: int = SAMPLES_PER_POINT,
    ):
        """
        Initialize calibration manager.

        Args:
            num_points: Number of calibration points
            samples_per_point: Samples to collect per point
        """
        self.num_points = num_points
        self.samples_per_point = samples_per_point

        # Components
        self.webcam: Optional[WebcamCapture] = None
        self.detector: Optional[FaceDetector] = None
        self.pipeline: Optional[FeaturePipeline] = None

        # State
        self._current_features: Optional[np.ndarray] = None
        self._features_lock = threading.Lock()
        self._capture_thread: Optional[threading.Thread] = None
        self._running = False

        # Results
        self.calibration_data: Optional[CalibrationData] = None
        self._calibration_complete = False

    def run_calibration(self) -> bool:
        """
        Run the complete calibration process.

        Returns:
            True if calibration completed successfully
        """
        print("Starting calibration...")

        # Initialize components
        self.webcam = WebcamCapture()
        if not self.webcam.start():
            print("Failed to start webcam")
            return False

        self.detector = FaceDetector()
        self.pipeline = FeaturePipeline()

        # Start feature extraction thread
        self._running = True
        self._capture_thread = threading.Thread(
            target=self._feature_extraction_loop,
            daemon=True,
        )
        self._capture_thread.start()

        # Wait briefly for first features
        time.sleep(0.5)

        # Create and run calibration UI
        self._calibration_complete = False

        ui = CalibrationUI(
            num_points=self.num_points,
            samples_per_point=self.samples_per_point,
            on_sample=self._get_current_features,
            on_complete=self._on_calibration_complete,
        )

        try:
            ui.start()  # This blocks until UI closes
        except Exception as e:
            print(f"Calibration UI error: {e}")
            self._cleanup()
            return False

        # Cleanup
        self._cleanup()

        return self._calibration_complete

    def _feature_extraction_loop(self):
        """Continuously extract features in background."""
        while self._running:
            frame = self.webcam.get_frame(timeout=0.1)
            if frame is None:
                continue

            landmarks = self.detector.detect(frame)
            if landmarks is None:
                with self._features_lock:
                    self._current_features = None
                continue

            features = self.pipeline.extract(landmarks)

            with self._features_lock:
                self._current_features = features

    def _get_current_features(self) -> Optional[np.ndarray]:
        """Get current features (called by calibration UI)."""
        with self._features_lock:
            if self._current_features is not None:
                return self._current_features.copy()
            return None

    def _on_calibration_complete(self, samples: List[Tuple[int, int, np.ndarray]]):
        """Handle calibration completion."""
        print(f"Calibration complete: {len(samples)} samples collected")

        # Get screen size from first sample (assuming all same screen)
        # We'll estimate from the sample positions
        max_x = max(s[0] for s in samples) if samples else 1920
        max_y = max(s[1] for s in samples) if samples else 1080

        # Create calibration data
        self.calibration_data = CalibrationData(
            screen_width=int(max_x * 1.2),  # Estimate full screen
            screen_height=int(max_y * 1.2),
            num_points=self.num_points,
        )

        for screen_x, screen_y, features in samples:
            self.calibration_data.add_sample(screen_x, screen_y, features)

        self._calibration_complete = True

    def _cleanup(self):
        """Clean up resources."""
        self._running = False

        if self._capture_thread is not None:
            self._capture_thread.join(timeout=1.0)
            self._capture_thread = None

        if self.webcam is not None:
            self.webcam.stop()
            self.webcam = None

        if self.detector is not None:
            self.detector.close()
            self.detector = None

    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get training data from calibration.

        Returns:
            Tuple of (X, y) where:
            - X: Feature matrix (n_samples, n_features)
            - y: Target coordinates (n_samples, 2)
        """
        if self.calibration_data is None:
            return np.array([]), np.array([])

        return self.calibration_data.to_arrays()

    def save_calibration(self, filename: Optional[str] = None) -> Optional[str]:
        """
        Save calibration data to file.

        Args:
            filename: Optional filename

        Returns:
            Path to saved file, or None if no data
        """
        if self.calibration_data is None:
            print("No calibration data to save")
            return None

        return self.calibration_data.save(filename)

    @staticmethod
    def load_calibration(filepath: str) -> CalibrationData:
        """
        Load calibration data from file.

        Args:
            filepath: Path to calibration file

        Returns:
            CalibrationData instance
        """
        return CalibrationData.load(filepath)


# =============================================================================
# Test function
# =============================================================================

def test_calibration_manager():
    """Test the calibration manager."""
    print("Testing CalibrationManager...")

    manager = CalibrationManager(num_points=9)
    success = manager.run_calibration()

    if success:
        X, y = manager.get_training_data()
        print(f"Training data: X shape = {X.shape}, y shape = {y.shape}")

        # Save calibration
        filepath = manager.save_calibration()
        if filepath:
            print(f"Saved to: {filepath}")
    else:
        print("Calibration failed or was cancelled")


if __name__ == "__main__":
    test_calibration_manager()
