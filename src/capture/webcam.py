"""
Webcam capture with threading for non-blocking frame acquisition.

This module provides threaded webcam capture to prevent frame drops
during heavy processing (MediaPipe, ML prediction).
"""

import cv2
import threading
import queue
import time
from typing import Optional, Tuple
import numpy as np


class WebcamCapture:
    """
    Threaded webcam capture to prevent frame drops during processing.

    Uses a producer-consumer pattern with a small queue (maxsize=2) to
    always provide the latest frame while dropping stale frames.

    Usage:
        # Context manager (recommended)
        with WebcamCapture(camera_index=0) as cap:
            while True:
                frame = cap.get_frame()
                if frame is not None:
                    # Process frame
                    pass

        # Manual control
        cap = WebcamCapture()
        cap.start()
        frame = cap.get_frame()
        cap.stop()
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        flip_horizontal: bool = True
    ):
        """
        Initialize webcam capture.

        Args:
            camera_index: Camera device index (0 for default webcam)
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Target frames per second
            flip_horizontal: If True, flip frame horizontally (mirror effect)
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.flip_horizontal = flip_horizontal

        # Internal state
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._running: bool = False
        self._capture_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Performance tracking
        self._frame_count: int = 0
        self._start_time: float = 0.0
        self._last_frame_time: float = 0.0

    def start(self) -> bool:
        """
        Initialize camera and start capture thread.

        Returns:
            True if started successfully, False otherwise
        """
        with self._lock:
            if self._running:
                return True

            self._cap = cv2.VideoCapture(self.camera_index)

            if not self._cap.isOpened():
                print(f"Error: Could not open camera {self.camera_index}")
                return False

            # Set camera properties
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)

            # Verify actual settings
            actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self._cap.get(cv2.CAP_PROP_FPS)

            print(f"Camera opened: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")

            # Start capture thread
            self._running = True
            self._start_time = time.time()
            self._frame_count = 0
            self._capture_thread = threading.Thread(
                target=self._capture_loop,
                daemon=True,
                name="WebcamCaptureThread"
            )
            self._capture_thread.start()

            return True

    def _capture_loop(self):
        """Continuously capture frames in background thread."""
        while self._running:
            ret, frame = self._cap.read()

            if ret:
                # Apply horizontal flip for mirror effect
                if self.flip_horizontal:
                    frame = cv2.flip(frame, 1)

                # Update timing
                self._last_frame_time = time.time()
                self._frame_count += 1

                # Drop old frame if queue is full (keep latest)
                if self._frame_queue.full():
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        pass

                self._frame_queue.put(frame)
            else:
                # Brief sleep on read failure to prevent busy loop
                time.sleep(0.001)

    def get_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """
        Get the latest frame from the queue.

        Args:
            timeout: Maximum wait time in seconds

        Returns:
            BGR image as numpy array (H, W, 3), or None if no frame available
        """
        if not self._running:
            return None

        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_frame_nowait(self) -> Optional[np.ndarray]:
        """
        Get frame without waiting (non-blocking).

        Returns:
            BGR image or None if no frame available
        """
        if not self._running:
            return None

        try:
            return self._frame_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        """Stop capture and release resources."""
        with self._lock:
            self._running = False

            if self._capture_thread is not None:
                self._capture_thread.join(timeout=1.0)
                self._capture_thread = None

            if self._cap is not None:
                self._cap.release()
                self._cap = None

            # Clear queue
            while not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    break

    @property
    def is_running(self) -> bool:
        """Whether the capture is currently active."""
        return self._running

    @property
    def frame_size(self) -> Tuple[int, int]:
        """Return (width, height) of frames."""
        return (self.width, self.height)

    @property
    def actual_fps(self) -> float:
        """Calculate actual FPS based on captured frames."""
        if self._frame_count == 0 or self._start_time == 0:
            return 0.0
        elapsed = time.time() - self._start_time
        if elapsed <= 0:
            return 0.0
        return self._frame_count / elapsed

    @property
    def frame_count(self) -> int:
        """Total frames captured since start."""
        return self._frame_count

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


# =============================================================================
# Test function
# =============================================================================

def test_webcam_capture():
    """Test webcam capture with live preview."""
    print("Testing WebcamCapture...")
    print("Press 'q' to quit, 's' to show stats")

    with WebcamCapture(camera_index=0, width=640, height=480) as cap:
        if not cap.is_running:
            print("Failed to start webcam")
            return

        frame_count = 0
        start_time = time.time()

        while True:
            frame = cap.get_frame()

            if frame is not None:
                frame_count += 1

                # Add FPS overlay
                fps = cap.actual_fps
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                cv2.imshow("Webcam Test", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                print(f"Stats: {cap.frame_count} frames, {cap.actual_fps:.1f} FPS")

        cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    print(f"Test complete: {frame_count} frames in {elapsed:.1f}s ({frame_count/elapsed:.1f} FPS)")


if __name__ == "__main__":
    test_webcam_capture()
