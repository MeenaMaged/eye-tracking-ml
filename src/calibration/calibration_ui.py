"""
9-point calibration GUI.

Displays animated targets on screen and collects gaze feature samples
for each calibration point. Uses Tkinter for fullscreen display.
"""

import tkinter as tk
from tkinter import ttk
import time
import numpy as np
from typing import Callable, List, Tuple, Optional
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    CALIBRATION_POINTS,
    SAMPLES_PER_POINT,
    CALIBRATION_SETTLE_MS,
    CALIBRATION_COLLECT_MS,
    CALIBRATION_MARGIN,
    CALIBRATION_POINT_SIZE,
)


@dataclass
class CalibrationPoint:
    """Represents a calibration target position."""
    screen_x: int
    screen_y: int
    index: int
    collected: bool = False


class CalibrationUI:
    """
    Fullscreen calibration interface with animated targets.

    Displays a grid of calibration points and collects feature samples
    at each point. Provides visual feedback with animated shrinking circles.

    Usage:
        ui = CalibrationUI(
            on_sample=get_features,      # Callback to get current features
            on_complete=handle_complete,  # Callback when done
        )
        ui.start()
    """

    def __init__(
        self,
        num_points: int = CALIBRATION_POINTS,
        samples_per_point: int = SAMPLES_PER_POINT,
        settle_ms: int = CALIBRATION_SETTLE_MS,
        collect_ms: int = CALIBRATION_COLLECT_MS,
        margin: float = CALIBRATION_MARGIN,
        on_sample: Optional[Callable[[], Optional[np.ndarray]]] = None,
        on_complete: Optional[Callable[[List], None]] = None,
    ):
        """
        Initialize calibration UI.

        Args:
            num_points: Number of calibration points (9, 16, or 25)
            samples_per_point: Feature samples to collect per point
            settle_ms: Wait time before collecting (eyes settle)
            collect_ms: Total collection time per point
            margin: Screen margin as fraction (0.1 = 10%)
            on_sample: Callback to get current feature vector
            on_complete: Callback when calibration completes
        """
        self.num_points = num_points
        self.samples_per_point = samples_per_point
        self.settle_ms = settle_ms
        self.collect_ms = collect_ms
        self.margin = margin

        self.on_sample = on_sample
        self.on_complete = on_complete

        # UI state
        self.root: Optional[tk.Tk] = None
        self.canvas: Optional[tk.Canvas] = None
        self.screen_width: int = 0
        self.screen_height: int = 0

        self.calibration_points: List[CalibrationPoint] = []
        self.current_point_index: int = 0
        self.collected_samples: List[List[Tuple[int, int, np.ndarray]]] = []

        self.is_collecting: bool = False
        self.is_settling: bool = False
        self._animation_id: Optional[str] = None
        self._collection_start_time: float = 0
        self._current_point_samples: List[np.ndarray] = []

        # Visual settings
        self.target_color = "#00FF00"
        self.target_outline = "#FFFFFF"
        self.bg_color = "#1a1a1a"
        self.max_radius = CALIBRATION_POINT_SIZE
        self.min_radius = 5

    def _generate_grid_points(self) -> List[CalibrationPoint]:
        """Generate calibration point positions based on grid layout."""
        points = []

        # Calculate margins
        margin_x = int(self.screen_width * self.margin)
        margin_y = int(self.screen_height * self.margin)

        # Determine grid dimensions
        if self.num_points == 5:
            # 5-point: center + corners
            positions = [
                (0.5, 0.5),  # Center
                (0, 0), (1, 0),  # Top corners
                (0, 1), (1, 1),  # Bottom corners
            ]
        elif self.num_points == 9:
            # 3x3 grid
            positions = [
                (x / 2, y / 2)
                for y in range(3)
                for x in range(3)
            ]
        elif self.num_points == 16:
            # 4x4 grid
            positions = [
                (x / 3, y / 3)
                for y in range(4)
                for x in range(4)
            ]
        else:
            # Default to 9-point
            positions = [
                (x / 2, y / 2)
                for y in range(3)
                for x in range(3)
            ]

        # Convert relative positions to screen coordinates
        usable_width = self.screen_width - 2 * margin_x
        usable_height = self.screen_height - 2 * margin_y

        for idx, (rel_x, rel_y) in enumerate(positions):
            screen_x = int(margin_x + rel_x * usable_width)
            screen_y = int(margin_y + rel_y * usable_height)
            points.append(CalibrationPoint(
                screen_x=screen_x,
                screen_y=screen_y,
                index=idx,
            ))

        return points

    def start(self):
        """Start the calibration session."""
        # Create fullscreen window
        self.root = tk.Tk()
        self.root.title("Eye Tracking Calibration")
        self.root.attributes('-fullscreen', True)
        self.root.configure(bg=self.bg_color)

        # Get screen dimensions
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()

        # Generate calibration points
        self.calibration_points = self._generate_grid_points()
        self.current_point_index = 0
        self.collected_samples = [[] for _ in range(len(self.calibration_points))]

        # Create canvas
        self.canvas = tk.Canvas(
            self.root,
            width=self.screen_width,
            height=self.screen_height,
            bg=self.bg_color,
            highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bind escape key to exit
        self.root.bind('<Escape>', self._on_escape)
        self.root.bind('<space>', self._on_space)

        # Show instructions
        self._show_instructions()

        # Start main loop
        self.root.mainloop()

    def _show_instructions(self):
        """Show calibration instructions."""
        self.canvas.delete("all")

        instructions = [
            "Eye Tracking Calibration",
            "",
            "Look at each target as it appears.",
            "Keep your head still during calibration.",
            "",
            f"Total points: {self.num_points}",
            "",
            "Press SPACE to begin",
            "Press ESC to cancel",
        ]

        y_offset = self.screen_height // 3
        for line in instructions:
            self.canvas.create_text(
                self.screen_width // 2,
                y_offset,
                text=line,
                fill="white",
                font=("Helvetica", 24 if line == instructions[0] else 18),
            )
            y_offset += 40

    def _on_space(self, event):
        """Handle space key press to start/continue calibration."""
        if self.current_point_index == 0 and not self.is_collecting:
            self._start_point_collection()

    def _on_escape(self, event):
        """Handle escape key press to cancel calibration."""
        self._cleanup()

    def _start_point_collection(self):
        """Start collecting samples for current point."""
        if self.current_point_index >= len(self.calibration_points):
            self._complete_calibration()
            return

        self.is_settling = True
        self.is_collecting = False
        self._current_point_samples = []
        self._collection_start_time = time.time()

        # Draw initial target
        self._draw_target(self.max_radius)

        # Schedule settle phase end
        self.root.after(self.settle_ms, self._start_sample_collection)

    def _start_sample_collection(self):
        """Start actual sample collection after settle time."""
        self.is_settling = False
        self.is_collecting = True
        self._collection_start_time = time.time()

        # Start collection loop
        self._collect_sample()

    def _collect_sample(self):
        """Collect a single sample and animate target."""
        if not self.is_collecting:
            return

        elapsed = (time.time() - self._collection_start_time) * 1000  # ms
        progress = min(1.0, elapsed / self.collect_ms)

        # Collect sample if callback available
        if self.on_sample is not None and len(self._current_point_samples) < self.samples_per_point:
            features = self.on_sample()
            if features is not None:
                self._current_point_samples.append(features.copy())

        # Animate target (shrink as progress increases)
        radius = self.max_radius - (self.max_radius - self.min_radius) * progress
        self._draw_target(radius, progress)

        # Check if collection complete
        if progress >= 1.0:
            self._finish_point()
        else:
            # Continue collecting (aim for ~30 fps animation)
            self.root.after(33, self._collect_sample)

    def _draw_target(self, radius: float, progress: float = 0):
        """Draw the calibration target."""
        self.canvas.delete("target")

        point = self.calibration_points[self.current_point_index]
        x, y = point.screen_x, point.screen_y

        # Outer circle
        self.canvas.create_oval(
            x - radius, y - radius,
            x + radius, y + radius,
            fill=self.target_color,
            outline=self.target_outline,
            width=2,
            tags="target",
        )

        # Inner dot
        inner_radius = 3
        self.canvas.create_oval(
            x - inner_radius, y - inner_radius,
            x + inner_radius, y + inner_radius,
            fill=self.target_outline,
            outline=self.target_outline,
            tags="target",
        )

        # Progress text
        point_num = self.current_point_index + 1
        total = len(self.calibration_points)
        status = "Settle..." if self.is_settling else f"{int(progress * 100)}%"

        self.canvas.create_text(
            self.screen_width // 2,
            50,
            text=f"Point {point_num}/{total} - {status}",
            fill="white",
            font=("Helvetica", 16),
            tags="target",
        )

    def _finish_point(self):
        """Finish collection for current point and move to next."""
        self.is_collecting = False

        # Store collected samples
        point = self.calibration_points[self.current_point_index]
        for features in self._current_point_samples:
            self.collected_samples[self.current_point_index].append(
                (point.screen_x, point.screen_y, features)
            )

        point.collected = True
        self.current_point_index += 1

        # Small delay before next point
        self.root.after(200, self._start_point_collection)

    def _complete_calibration(self):
        """Complete calibration and call callback."""
        # Flatten all samples
        all_samples = []
        for point_samples in self.collected_samples:
            all_samples.extend(point_samples)

        # Callback
        if self.on_complete is not None:
            self.on_complete(all_samples)

        # Show completion message briefly
        self.canvas.delete("all")
        self.canvas.create_text(
            self.screen_width // 2,
            self.screen_height // 2,
            text=f"Calibration Complete!\n{len(all_samples)} samples collected",
            fill="white",
            font=("Helvetica", 24),
        )

        self.root.after(2000, self._cleanup)

    def _cleanup(self):
        """Clean up and close window."""
        if self.root is not None:
            self.root.destroy()
            self.root = None

    def stop(self):
        """Stop and close calibration UI."""
        self._cleanup()

    @property
    def screen_size(self) -> Tuple[int, int]:
        """Get screen dimensions."""
        return (self.screen_width, self.screen_height)
