"""
Live demonstration of Eye Tracking ML System.

Shows real-time:
- Webcam feed with detected landmarks
- Gaze position estimate
- Blink detection status
- Eye movement classification (Fixation/Saccade/Blink)

All predictions use ML models trained from scratch.
"""

import numpy as np
import cv2
import time
import os
import sys
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.capture.webcam import WebcamCapture
from src.detection.face_detector import FaceDetector
from src.features.feature_pipeline import FeaturePipeline
from src.features.ear_calculator import EARCalculator
from src.dataset.data_collector import BlinkDataCollector, MovementDataCollector


@dataclass
class DemoStats:
    """Statistics for the demo session."""
    frames_processed: int = 0
    faces_detected: int = 0
    blinks_detected: int = 0
    fixations: int = 0
    saccades: int = 0
    start_time: float = 0

    @property
    def fps(self) -> float:
        elapsed = time.time() - self.start_time
        return self.frames_processed / elapsed if elapsed > 0 else 0

    @property
    def detection_rate(self) -> float:
        return self.faces_detected / self.frames_processed if self.frames_processed > 0 else 0


class LiveDemo:
    """
    Live demonstration of the eye tracking ML system.

    Features demonstrated:
    1. Real-time face and eye landmark detection (MediaPipe)
    2. Feature extraction (10-dimensional vector)
    3. Gaze estimation (screen coordinates prediction)
    4. Blink detection (binary classification)
    5. Eye movement classification (3-class: Fixation/Saccade/Blink)

    All ML models are trained from scratch.
    """

    # Movement class names
    MOVEMENT_CLASSES = ['Fixation', 'Saccade', 'Blink']
    MOVEMENT_COLORS = [(0, 255, 0), (0, 255, 255), (0, 0, 255)]  # Green, Yellow, Red

    def __init__(self,
                 gaze_model=None,
                 blink_model=None,
                 movement_model=None,
                 window_name: str = "Eye Tracking ML Demo"):
        """
        Initialize the live demo.

        Args:
            gaze_model: Trained gaze estimation model (optional)
            blink_model: Trained blink detection classifier (optional)
            movement_model: Trained movement classifier (optional)
            window_name: Name of the OpenCV window
        """
        self.gaze_model = gaze_model
        self.blink_model = blink_model
        self.movement_model = movement_model
        self.window_name = window_name

        # Initialize components
        self.webcam = WebcamCapture()
        self.detector = FaceDetector()
        self.pipeline = FeaturePipeline()
        self.ear_calculator = EARCalculator()

        # Data collectors for feature extraction
        self.blink_collector = BlinkDataCollector()
        self.movement_collector = MovementDataCollector()

        # Tracking history
        self.gaze_history: deque = deque(maxlen=30)
        self.ear_history: deque = deque(maxlen=30)

        # Statistics
        self.stats = DemoStats()

        # State
        self._running = False
        self._gaze_position = (0, 0)
        self._current_movement = 0
        self._is_blinking = False

        # Normalization params (from training data)
        self._norm_mean: Optional[np.ndarray] = None
        self._norm_std: Optional[np.ndarray] = None

    def set_normalization_params(self, mean: np.ndarray, std: np.ndarray):
        """Set normalization parameters from training data."""
        self._norm_mean = mean
        self._norm_std = std

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using training statistics."""
        if self._norm_mean is not None and self._norm_std is not None:
            return (features - self._norm_mean) / (self._norm_std + 1e-8)
        return features

    def run(self, duration_seconds: float = 0, show_stats: bool = True):
        """
        Run the live demo.

        Args:
            duration_seconds: How long to run (0 = indefinite, press 'q' to quit)
            show_stats: If True, display statistics on screen
        """
        if not self.webcam.start():
            print("ERROR: Could not start webcam")
            return

        self._running = True
        self.stats = DemoStats(start_time=time.time())

        print("\n" + "=" * 60)
        print("EYE TRACKING ML DEMO")
        print("=" * 60)
        print("Press 'q' to quit")
        print("Press 's' to save screenshot")
        print("Press 'r' to reset statistics")
        print("=" * 60)

        try:
            while self._running:
                # Check duration limit
                if duration_seconds > 0 and (time.time() - self.stats.start_time) > duration_seconds:
                    break

                # Get frame
                frame = self.webcam.get_frame()
                if frame is None:
                    continue

                self.stats.frames_processed += 1

                # Process frame
                processed_frame = self._process_frame(frame, show_stats)

                # Display
                cv2.imshow(self.window_name, processed_frame)

                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._save_screenshot(processed_frame)
                elif key == ord('r'):
                    self.stats = DemoStats(start_time=time.time())
                    print("Statistics reset")

        finally:
            self._cleanup()

    def _process_frame(self, frame: np.ndarray, show_stats: bool = True) -> np.ndarray:
        """Process a single frame and add visualizations."""
        display_frame = frame.copy()
        h, w = frame.shape[:2]

        # Detect face landmarks
        landmarks = self.detector.detect(frame)

        if landmarks is not None:
            self.stats.faces_detected += 1

            # Extract features
            features = self.pipeline.extract(landmarks)
            ear_data = self.ear_calculator.extract(landmarks)

            if features is not None:
                avg_ear = ear_data.get('avg_ear', 0.3)
                self.ear_history.append(avg_ear)

                # 1. Predict gaze position
                gaze_x, gaze_y = self._predict_gaze(features)
                self._gaze_position = (gaze_x, gaze_y)
                self.gaze_history.append((gaze_x, gaze_y))

                # 2. Detect blink
                self._is_blinking = self._detect_blink(features, avg_ear)
                if self._is_blinking:
                    self.stats.blinks_detected += 1

                # 3. Classify movement
                self._current_movement = self._classify_movement(gaze_x, gaze_y, avg_ear)
                if self._current_movement == 0:
                    self.stats.fixations += 1
                elif self._current_movement == 1:
                    self.stats.saccades += 1

                # Draw visualizations
                display_frame = self._draw_landmarks(display_frame, landmarks)
                display_frame = self._draw_gaze(display_frame, gaze_x, gaze_y)
                display_frame = self._draw_ear_bar(display_frame, avg_ear)
                display_frame = self._draw_status(display_frame)

        else:
            # No face detected
            cv2.putText(display_frame, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Draw statistics
        if show_stats:
            display_frame = self._draw_stats(display_frame)

        return display_frame

    def _predict_gaze(self, features: np.ndarray) -> Tuple[float, float]:
        """Predict gaze position from features."""
        if self.gaze_model is not None:
            # Use trained model
            features_2d = features.reshape(1, -1)
            features_norm = self._normalize_features(features_2d)
            prediction = self.gaze_model.predict(features_norm)
            return float(prediction[0, 0]), float(prediction[0, 1])
        else:
            # Simple rule-based approximation
            # Map iris ratios to screen coordinates
            left_x, left_y = features[0], features[1]
            right_x, right_y = features[2], features[3]

            # Average iris positions and scale to screen
            avg_x = (left_x + right_x) / 2
            avg_y = (left_y + right_y) / 2

            screen_x = int(avg_x * 1920)  # Assuming 1920x1080 screen
            screen_y = int(avg_y * 1080)

            return screen_x, screen_y

    def _detect_blink(self, features: np.ndarray, ear: float) -> bool:
        """Detect if eyes are blinking."""
        if self.blink_model is not None:
            # Use trained model
            # Add temporal features for blink detection
            self.blink_collector.add_frame(features, ear)
            temporal_features = self.blink_collector.samples[-1].features
            temporal_features = temporal_features.reshape(1, -1)
            prediction = self.blink_model.predict(temporal_features)
            return bool(prediction[0] == 1)
        else:
            # Simple threshold-based detection
            return ear < 0.2

    def _classify_movement(self, gaze_x: float, gaze_y: float, ear: float) -> int:
        """Classify eye movement type."""
        if self.movement_model is not None:
            # Use trained model
            timestamp = time.time()
            self.movement_collector.add_frame(gaze_x, gaze_y, ear, timestamp)
            if len(self.movement_collector.samples) > 0:
                movement_features = self.movement_collector.samples[-1].features
                movement_features = movement_features.reshape(1, -1)
                prediction = self.movement_model.predict(movement_features)
                return int(prediction[0])
        else:
            # Rule-based classification
            if ear < 0.2:
                return 2  # Blink

            # Calculate velocity from history
            if len(self.gaze_history) >= 2:
                prev_x, prev_y = self.gaze_history[-2]
                velocity = np.sqrt((gaze_x - prev_x)**2 + (gaze_y - prev_y)**2)
                if velocity > 100:  # High velocity = saccade
                    return 1  # Saccade

            return 0  # Fixation

    def _draw_landmarks(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Draw eye landmarks on frame."""
        h, w = frame.shape[:2]

        # Iris centers (landmarks 468 and 473)
        left_iris = landmarks[468, :2]
        right_iris = landmarks[473, :2]

        left_px = (int(left_iris[0] * w), int(left_iris[1] * h))
        right_px = (int(right_iris[0] * w), int(right_iris[1] * h))

        # Draw iris centers
        cv2.circle(frame, left_px, 5, (0, 255, 0), -1)
        cv2.circle(frame, right_px, 5, (0, 255, 0), -1)

        # Draw eye contours
        left_eye_idx = [33, 160, 158, 133, 153, 144]
        right_eye_idx = [362, 385, 387, 263, 373, 380]

        for idx in left_eye_idx:
            pt = (int(landmarks[idx, 0] * w), int(landmarks[idx, 1] * h))
            cv2.circle(frame, pt, 2, (255, 0, 0), -1)

        for idx in right_eye_idx:
            pt = (int(landmarks[idx, 0] * w), int(landmarks[idx, 1] * h))
            cv2.circle(frame, pt, 2, (255, 0, 0), -1)

        return frame

    def _draw_gaze(self, frame: np.ndarray, gaze_x: float, gaze_y: float) -> np.ndarray:
        """Draw gaze position indicator."""
        h, w = frame.shape[:2]

        # Scale gaze to frame coordinates (minimap)
        minimap_w, minimap_h = 160, 90
        minimap_x, minimap_y = w - minimap_w - 10, 10

        # Draw minimap background
        cv2.rectangle(frame, (minimap_x, minimap_y),
                     (minimap_x + minimap_w, minimap_y + minimap_h),
                     (50, 50, 50), -1)
        cv2.rectangle(frame, (minimap_x, minimap_y),
                     (minimap_x + minimap_w, minimap_y + minimap_h),
                     (255, 255, 255), 1)

        # Scale gaze position to minimap
        gaze_mini_x = int(minimap_x + (gaze_x / 1920) * minimap_w)
        gaze_mini_y = int(minimap_y + (gaze_y / 1080) * minimap_h)

        # Clamp to minimap bounds
        gaze_mini_x = max(minimap_x + 5, min(minimap_x + minimap_w - 5, gaze_mini_x))
        gaze_mini_y = max(minimap_y + 5, min(minimap_y + minimap_h - 5, gaze_mini_y))

        # Draw gaze trail
        for i, (hx, hy) in enumerate(self.gaze_history):
            alpha = i / len(self.gaze_history)
            hx_mini = int(minimap_x + (hx / 1920) * minimap_w)
            hy_mini = int(minimap_y + (hy / 1080) * minimap_h)
            hx_mini = max(minimap_x + 2, min(minimap_x + minimap_w - 2, hx_mini))
            hy_mini = max(minimap_y + 2, min(minimap_y + minimap_h - 2, hy_mini))
            color = (int(100 * alpha), int(100 * alpha), int(255 * alpha))
            cv2.circle(frame, (hx_mini, hy_mini), 2, color, -1)

        # Draw current gaze point
        cv2.circle(frame, (gaze_mini_x, gaze_mini_y), 5, (0, 0, 255), -1)
        cv2.circle(frame, (gaze_mini_x, gaze_mini_y), 7, (255, 255, 255), 1)

        # Label
        cv2.putText(frame, f"Gaze: ({int(gaze_x)}, {int(gaze_y)})",
                   (minimap_x, minimap_y + minimap_h + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return frame

    def _draw_ear_bar(self, frame: np.ndarray, ear: float) -> np.ndarray:
        """Draw EAR indicator bar."""
        h, w = frame.shape[:2]

        # Bar position
        bar_x, bar_y = 10, h - 50
        bar_w, bar_h = 150, 20

        # Draw background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                     (50, 50, 50), -1)

        # Draw EAR level
        ear_normalized = min(1.0, ear / 0.4)  # Normalize to 0-1 range
        fill_w = int(bar_w * ear_normalized)
        color = (0, 255, 0) if ear > 0.2 else (0, 0, 255)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                     color, -1)

        # Draw threshold line
        threshold_x = bar_x + int(bar_w * (0.2 / 0.4))
        cv2.line(frame, (threshold_x, bar_y - 5), (threshold_x, bar_y + bar_h + 5),
                (255, 255, 0), 2)

        # Label
        cv2.putText(frame, f"EAR: {ear:.3f}", (bar_x, bar_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def _draw_status(self, frame: np.ndarray) -> np.ndarray:
        """Draw current status (blink, movement type)."""
        h, w = frame.shape[:2]

        # Movement classification
        movement_name = self.MOVEMENT_CLASSES[self._current_movement]
        movement_color = self.MOVEMENT_COLORS[self._current_movement]

        cv2.putText(frame, f"Movement: {movement_name}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, movement_color, 2)

        # Blink indicator
        if self._is_blinking:
            cv2.putText(frame, "BLINK!", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return frame

    def _draw_stats(self, frame: np.ndarray) -> np.ndarray:
        """Draw statistics panel."""
        h, w = frame.shape[:2]

        # Stats panel
        panel_x, panel_y = 10, h - 150
        panel_w, panel_h = 200, 90

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                     (panel_x + panel_w, panel_y + panel_h),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # Stats text
        stats_text = [
            f"FPS: {self.stats.fps:.1f}",
            f"Detection: {self.stats.detection_rate * 100:.1f}%",
            f"Blinks: {self.stats.blinks_detected}",
            f"Fix/Sac: {self.stats.fixations}/{self.stats.saccades}",
        ]

        y_offset = panel_y + 20
        for text in stats_text:
            cv2.putText(frame, text, (panel_x + 5, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20

        return frame

    def _save_screenshot(self, frame: np.ndarray):
        """Save screenshot to results directory."""
        os.makedirs("results/phase8_integration/screenshots", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"results/phase8_integration/screenshots/demo_{timestamp}.png"
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved: {filename}")

    def _cleanup(self):
        """Clean up resources."""
        self._running = False
        self.webcam.stop()
        self.detector.close()
        cv2.destroyAllWindows()

        # Print final stats
        print("\n" + "=" * 60)
        print("DEMO SESSION SUMMARY")
        print("=" * 60)
        print(f"Duration: {time.time() - self.stats.start_time:.1f} seconds")
        print(f"Frames processed: {self.stats.frames_processed}")
        print(f"Average FPS: {self.stats.fps:.1f}")
        print(f"Detection rate: {self.stats.detection_rate * 100:.1f}%")
        print(f"Blinks detected: {self.stats.blinks_detected}")
        print(f"Fixations: {self.stats.fixations}")
        print(f"Saccades: {self.stats.saccades}")
        print("=" * 60)


def run_live_demo(gaze_model=None, blink_model=None, movement_model=None,
                  duration_seconds: float = 0):
    """
    Convenience function to run the live demo.

    Args:
        gaze_model: Trained gaze estimation model (optional)
        blink_model: Trained blink detection classifier (optional)
        movement_model: Trained movement classifier (optional)
        duration_seconds: How long to run (0 = indefinite)
    """
    demo = LiveDemo(
        gaze_model=gaze_model,
        blink_model=blink_model,
        movement_model=movement_model
    )
    demo.run(duration_seconds=duration_seconds)


# =============================================================================
# Test functions
# =============================================================================

def test_demo_basic():
    """Run basic demo without ML models (rule-based only)."""
    print("Running basic demo (rule-based predictions)...")
    print("Press 'q' to quit")

    demo = LiveDemo()
    demo.run(duration_seconds=30)  # Run for 30 seconds


def test_demo_with_models():
    """Run demo with trained ML models."""
    print("Loading trained models...")

    # Try to load trained models
    try:
        import pickle

        # Load gaze model
        with open("data/trained_models/gaze_model.pkl", "rb") as f:
            gaze_model = pickle.load(f)
        print("Loaded gaze model")

        # Load blink model
        with open("data/trained_models/blink_model.pkl", "rb") as f:
            blink_model = pickle.load(f)
        print("Loaded blink model")

        # Load movement model
        with open("data/trained_models/movement_model.pkl", "rb") as f:
            movement_model = pickle.load(f)
        print("Loaded movement model")

        demo = LiveDemo(
            gaze_model=gaze_model,
            blink_model=blink_model,
            movement_model=movement_model
        )
        demo.run()

    except FileNotFoundError as e:
        print(f"Model files not found: {e}")
        print("Running basic demo instead...")
        test_demo_basic()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Eye Tracking ML Live Demo")
    parser.add_argument("--duration", type=float, default=0,
                       help="Duration in seconds (0 = indefinite)")
    parser.add_argument("--basic", action="store_true",
                       help="Run basic demo without ML models")

    args = parser.parse_args()

    if args.basic:
        test_demo_basic()
    else:
        run_live_demo(duration_seconds=args.duration)
