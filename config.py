"""
Eye Tracking System - Global Configuration
CSE381 Introduction to Machine Learning - Course Project

This module contains all configuration constants for the eye tracking system.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

# =============================================================================
# WEBCAM SETTINGS
# =============================================================================
WEBCAM_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30

# =============================================================================
# MEDIAPIPE SETTINGS
# =============================================================================
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
REFINE_LANDMARKS = True  # CRITICAL: Enables iris detection (landmarks 468-477)
MAX_NUM_FACES = 1

# =============================================================================
# CALIBRATION SETTINGS
# =============================================================================
CALIBRATION_POINTS = 9  # 9-point grid (3x3)
CALIBRATION_MARGIN = 0.1  # 10% margin from screen edges
SAMPLES_PER_POINT = 10  # Feature samples to collect per point
CALIBRATION_SETTLE_MS = 400  # Wait time before collecting (eyes settle)
CALIBRATION_COLLECT_MS = 600  # Collection time per point
CALIBRATION_POINT_SIZE = 30  # Target circle radius in pixels

# =============================================================================
# FEATURE EXTRACTION
# =============================================================================
# Eye Aspect Ratio threshold for blink detection
EAR_THRESHOLD = 0.25
BLINK_CONSEC_FRAMES = 3  # Consecutive frames below threshold = blink

# Feature vector size
FEATURE_VECTOR_SIZE = 10

# Feature names for reference
FEATURE_NAMES = [
    'left_iris_x_ratio',
    'left_iris_y_ratio',
    'right_iris_x_ratio',
    'right_iris_y_ratio',
    'left_ear',
    'right_ear',
    'pitch',
    'yaw',
    'roll',
    'inter_ocular_distance',
]

# =============================================================================
# GAZE ESTIMATION
# =============================================================================
POLYNOMIAL_DEGREE = 2  # Degree for polynomial regression
SMOOTHING_ALPHA = 0.2  # EMA smoothing factor (0.1=smooth, 0.3=responsive)

# =============================================================================
# CLICK DETECTION
# =============================================================================
DWELL_TIME_MS = 700  # Dwell time for click trigger
DWELL_RADIUS_PX = 30  # Radius for dwell detection

# =============================================================================
# MOVEMENT CLASSIFICATION
# =============================================================================
VELOCITY_THRESHOLD = 30  # degrees/second for saccade detection
DISPERSION_THRESHOLD = 50  # pixels for fixation detection
MIN_FIXATION_DURATION_MS = 100

# =============================================================================
# MEDIAPIPE LANDMARK INDICES
# =============================================================================

class Landmarks:
    """MediaPipe FaceMesh landmark indices."""

    # Iris centers (requires refine_landmarks=True)
    LEFT_IRIS_CENTER = 468
    RIGHT_IRIS_CENTER = 473

    # Iris points (5 per iris)
    LEFT_IRIS = [468, 469, 470, 471, 472]  # center, right, top, left, bottom
    RIGHT_IRIS = [473, 474, 475, 476, 477]

    # Eye corners
    LEFT_EYE_INNER = 33
    LEFT_EYE_OUTER = 133
    RIGHT_EYE_INNER = 362
    RIGHT_EYE_OUTER = 263

    # Eyelid points for EAR (6 points per eye)
    # Order: inner corner, top-inner, top-outer, outer corner, bottom-outer, bottom-inner
    LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]

    # Eyelid vertical points
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374

    # Head pose estimation points
    NOSE_TIP = 1
    CHIN = 152
    LEFT_EYE_OUTER_CORNER = 33
    RIGHT_EYE_OUTER_CORNER = 263
    LEFT_MOUTH_CORNER = 61
    RIGHT_MOUTH_CORNER = 291

    # Head pose landmark indices list
    HEAD_POSE_INDICES = [1, 152, 33, 263, 61, 291]


# 3D model points for head pose estimation (generic face model in mm)
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye outer corner
    (225.0, 170.0, -135.0),   # Right eye outer corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0, -150.0, -125.0),  # Right mouth corner
], dtype=np.float64)

# =============================================================================
# PERFORMANCE TARGETS
# =============================================================================
TARGET_PIPELINE_LATENCY_MS = 20
MAX_PIPELINE_LATENCY_MS = 40
MIN_FPS = 20

# =============================================================================
# FILE PATHS
# =============================================================================
DATA_DIR = "data"
CALIBRATION_DATA_DIR = f"{DATA_DIR}/calibration_data"
TRAINED_MODELS_DIR = f"{DATA_DIR}/trained_models"
LOGS_DIR = f"{DATA_DIR}/logs"

# =============================================================================
# DATACLASSES FOR CONFIGURATION
# =============================================================================

@dataclass
class CaptureConfig:
    """Webcam capture configuration."""
    camera_index: int = WEBCAM_INDEX
    frame_width: int = FRAME_WIDTH
    frame_height: int = FRAME_HEIGHT
    target_fps: int = TARGET_FPS


@dataclass
class DetectionConfig:
    """Face detection configuration."""
    min_detection_confidence: float = MIN_DETECTION_CONFIDENCE
    min_tracking_confidence: float = MIN_TRACKING_CONFIDENCE
    refine_landmarks: bool = REFINE_LANDMARKS
    max_num_faces: int = MAX_NUM_FACES


@dataclass
class CalibrationConfig:
    """Calibration configuration."""
    num_points: int = CALIBRATION_POINTS
    margin: float = CALIBRATION_MARGIN
    samples_per_point: int = SAMPLES_PER_POINT
    settle_ms: int = CALIBRATION_SETTLE_MS
    collect_ms: int = CALIBRATION_COLLECT_MS
    point_size: int = CALIBRATION_POINT_SIZE


@dataclass
class GazeConfig:
    """Gaze estimation configuration."""
    polynomial_degree: int = POLYNOMIAL_DEGREE
    smoothing_alpha: float = SMOOTHING_ALPHA


@dataclass
class ControlConfig:
    """Control configuration."""
    dwell_time_ms: int = DWELL_TIME_MS
    dwell_radius_px: int = DWELL_RADIUS_PX
    ear_threshold: float = EAR_THRESHOLD


def get_default_config():
    """Get default configuration objects."""
    return {
        'capture': CaptureConfig(),
        'detection': DetectionConfig(),
        'calibration': CalibrationConfig(),
        'gaze': GazeConfig(),
        'control': ControlConfig(),
    }
