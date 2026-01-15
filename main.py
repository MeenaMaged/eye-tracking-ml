#!/usr/bin/env python3
"""
Eye Tracking System - Main Entry Point
CSE381 Introduction to Machine Learning - Course Project

A webcam-based eye tracking system that calculates eye position, gaze direction,
and eye movements with high accuracy.

Usage:
    python main.py                    # Run demo mode
    python main.py --calibrate        # Run calibration
    python main.py --test-webcam      # Test webcam capture
    python main.py --test-detection   # Test face detection
    python main.py --test-features    # Test feature extraction
"""

import argparse
import sys
import os
import cv2
import numpy as np
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import FRAME_WIDTH, FRAME_HEIGHT, FEATURE_NAMES
from src.capture import WebcamCapture
from src.detection import FaceDetector
from src.features import FeaturePipeline
from src.calibration import CalibrationManager


def test_webcam():
    """Test webcam capture module."""
    print("Testing Webcam Capture...")
    print("Press 'q' to quit")

    with WebcamCapture() as cap:
        if not cap.is_running:
            print("Failed to start webcam")
            return False

        while True:
            frame = cap.get_frame()
            if frame is not None:
                # Display FPS
                fps = cap.actual_fps
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Webcam Test", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    print(f"Test complete. Total frames: {cap.frame_count}")
    return True


def test_detection():
    """Test face detection module."""
    print("Testing Face Detection...")
    print("Press 'q' to quit")

    with WebcamCapture() as cap:
        detector = FaceDetector()

        if not cap.is_running:
            print("Failed to start webcam")
            return False

        detection_count = 0
        frame_count = 0

        while True:
            frame = cap.get_frame()
            if frame is None:
                continue

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

                status = f"Face Detected ({detection_count}/{frame_count})"
                color = (0, 255, 0)
            else:
                status = "No Face"
                color = (0, 0, 255)

            cv2.putText(frame, status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow("Detection Test", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        detector.close()
        cv2.destroyAllWindows()

    rate = detection_count / frame_count * 100 if frame_count > 0 else 0
    print(f"Detection rate: {rate:.1f}%")
    return True


def test_features():
    """Test feature extraction pipeline."""
    print("Testing Feature Extraction...")
    print("Press 'q' to quit")

    with WebcamCapture() as cap:
        detector = FaceDetector()
        pipeline = FeaturePipeline()

        if not cap.is_running:
            print("Failed to start webcam")
            return False

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

                    # Draw landmarks
                    frame = detector.draw_landmarks(frame, landmarks)

                    # Display features
                    y_offset = 30
                    for i, name in enumerate(FEATURE_NAMES):
                        value = features[i]
                        text = f"{name}: {value:.3f}"
                        cv2.putText(frame, text, (10, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        y_offset += 20

                    # Blink indicator
                    if is_blinking:
                        cv2.putText(frame, "BLINK!", (frame.shape[1] - 100, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "No face detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Feature Test", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        detector.close()
        cv2.destroyAllWindows()

    return True


def run_calibration():
    """Run the calibration process."""
    print("Starting Calibration...")
    print("Follow the on-screen instructions.")

    manager = CalibrationManager(num_points=9)
    success = manager.run_calibration()

    if success:
        X, y = manager.get_training_data()
        print(f"\nCalibration successful!")
        print(f"Collected {X.shape[0]} samples with {X.shape[1]} features each")
        print(f"Target coordinates shape: {y.shape}")

        # Save calibration
        filepath = manager.save_calibration()
        if filepath:
            print(f"Calibration data saved to: {filepath}")

        return True
    else:
        print("\nCalibration failed or was cancelled.")
        return False


def run_demo():
    """Run demo mode showing all features."""
    print("Eye Tracking System - Demo Mode")
    print("Press 'q' to quit, 'c' to calibrate")
    print("-" * 40)

    with WebcamCapture() as cap:
        detector = FaceDetector()
        pipeline = FeaturePipeline()

        if not cap.is_running:
            print("Failed to start webcam")
            return

        frame_times = []

        while True:
            start_time = time.time()

            frame = cap.get_frame()
            if frame is None:
                continue

            landmarks = detector.detect(frame)

            if landmarks is not None:
                all_features = pipeline.extract_all(landmarks)

                if all_features is not None:
                    features = all_features['feature_vector']
                    is_blinking = all_features['is_blinking']
                    iris = all_features['iris']
                    head = all_features['head_pose']

                    # Draw face mesh
                    frame = detector.draw_landmarks(frame, landmarks, draw_iris=True)

                    # Create info panel
                    panel_width = 250
                    panel = np.zeros((frame.shape[0], panel_width, 3), dtype=np.uint8)
                    panel[:] = (30, 30, 30)

                    # Title
                    cv2.putText(panel, "Eye Tracking", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.line(panel, (10, 40), (panel_width - 10, 40), (100, 100, 100), 1)

                    # Iris info
                    y = 70
                    cv2.putText(panel, "Iris Position:", (10, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    y += 20
                    cv2.putText(panel, f"  L: ({iris['left_iris_x_ratio']:.2f}, {iris['left_iris_y_ratio']:.2f})",
                               (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                    y += 18
                    cv2.putText(panel, f"  R: ({iris['right_iris_x_ratio']:.2f}, {iris['right_iris_y_ratio']:.2f})",
                               (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

                    # Head pose
                    y += 30
                    cv2.putText(panel, "Head Pose:", (10, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    y += 20
                    cv2.putText(panel, f"  Pitch: {head['pitch']:.1f}",
                               (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
                    y += 18
                    cv2.putText(panel, f"  Yaw: {head['yaw']:.1f}",
                               (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
                    y += 18
                    cv2.putText(panel, f"  Roll: {head['roll']:.1f}",
                               (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

                    # Blink status
                    y += 30
                    blink_color = (0, 0, 255) if is_blinking else (0, 255, 0)
                    blink_text = "BLINKING" if is_blinking else "Eyes Open"
                    cv2.putText(panel, f"Status: {blink_text}", (10, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, blink_color, 1)

                    # Combine frame and panel
                    combined = np.hstack([frame, panel])
                    frame = combined

            else:
                # No face panel
                cv2.putText(frame, "No face detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, "Please position your face in the camera", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Calculate and display FPS
            frame_time = time.time() - start_time
            frame_times.append(frame_time)
            if len(frame_times) > 30:
                frame_times.pop(0)
            avg_fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0

            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (frame.shape[1] - 120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Eye Tracking Demo", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                cv2.destroyAllWindows()
                detector.close()
                run_calibration()
                # Restart demo
                detector = FaceDetector()

        detector.close()
        cv2.destroyAllWindows()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Eye Tracking System - CSE381 ML Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                    # Run demo mode
    python main.py --calibrate        # Run calibration
    python main.py --test-webcam      # Test webcam capture
    python main.py --test-detection   # Test face detection
    python main.py --test-features    # Test feature extraction
        """
    )

    parser.add_argument('--calibrate', action='store_true',
                       help='Run calibration process')
    parser.add_argument('--test-webcam', action='store_true',
                       help='Test webcam capture')
    parser.add_argument('--test-detection', action='store_true',
                       help='Test face detection')
    parser.add_argument('--test-features', action='store_true',
                       help='Test feature extraction')

    args = parser.parse_args()

    print("=" * 50)
    print("Eye Tracking System")
    print("CSE381 Introduction to Machine Learning")
    print("=" * 50)
    print()

    if args.test_webcam:
        test_webcam()
    elif args.test_detection:
        test_detection()
    elif args.test_features:
        test_features()
    elif args.calibrate:
        run_calibration()
    else:
        run_demo()


if __name__ == "__main__":
    main()
