"""
File: config.py
Description: Central configuration for synthesis, display/capture, and processing.
"""
from datetime import datetime, timedelta
import os

# --- Time ---
# J2000 epoch and observation offset in seconds.
J2000_EPOCH = datetime(2000, 1, 1, 12, 0, 0)
OBSERVATION_SECONDS = 757339269.184
OBSERVATION_DATE = J2000_EPOCH + timedelta(seconds=OBSERVATION_SECONDS)

# --- Paths ---
# Use relative paths so it works on any computer.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
IMAGE_DIR = os.path.join(DATA_DIR, 'generated')
CAPTURED_DIR = os.path.join(DATA_DIR, 'captured')
TRUTH_FILE = os.path.join(DATA_DIR, 'scenarios.csv')


# =============================================================================
# 1) SYNTHETIC STAR IMAGE GENERATION (GROUND TRUTH RENDER)
# =============================================================================
# Synthetic image raster size (pixels).
SYNTH_IMAGE_WIDTH = 1280
SYNTH_IMAGE_HEIGHT = 1024

# Synthetic camera model used to project catalog stars.
SYNTH_CAM_FOCAL_LENGTH_PX = 1.1568e3
SYNTH_CAM_CENTER_X = 633.4
SYNTH_CAM_CENTER_Y = 513.5

# Distortion [k1, k2, p1, p2, k3] used for synthetic generation.
# Keep zeros for first real-camera display/capture smoke test.
SYNTH_DISTORTION_COEFFS = [0.0, 0.0, 0.0, 0.0, 0.0]

# Synthetic star rendering controls.
SYNTH_MAX_MAGNITUDE = 8
SYNTH_PSF_SIGMA = 0.5


# =============================================================================
# 2) DISPLAY/CAPTURE PRESENTATION SETTINGS (MONITOR OUTPUT)
# =============================================================================
# Default FITS image displayed on monitor for camera capture.
DISPLAY_IMAGE_PATH = os.path.join(IMAGE_DIR, 'Single_Scenario.fits')
DISPLAY_SCENARIO_CSV = TRUTH_FILE
DISPLAY_SERIES_START_INDEX = 0

# Monitor selection policy for display_single_image.py.
# If None, script will prompt when multiple monitors are detected.
DISPLAY_MONITOR_INDEX = None
DISPLAY_PROMPT_IF_MULTI_MONITOR = True

# Fallback monitor window geometry when monitor index is not used.
DISPLAY_MONITOR_WIDTH = 1920
DISPLAY_MONITOR_HEIGHT = 1080
DISPLAY_OFFSET_X = 0
DISPLAY_OFFSET_Y = 0
DISPLAY_DPI = 100

# Display transfer function.
DISPLAY_APPLY_GAMMA = True
DISPLAY_GAMMA = 2.2

# =============================================================================
# SHARED MODE SELECTOR
# =============================================================================
# Controls both display (single image vs series) and processing (single vs batch).
# 'single' = one image  |  'batch' = all scenarios from scenarios.csv
PROC_MODE = 'single'  # 'single' | 'batch'


# =============================================================================
# 3) PROCESSING PIPELINE SETTINGS (CALIBRATED REAL CAMERA MODEL)
# =============================================================================
# Input image used by processing pipeline (captured frame).
# Update this path per capture session.
PROCESS_IMAGE_PATH = os.path.join(CAPTURED_DIR, 'Single_Scenario_Capture.fits')

# Calibrated camera model for processing.
PROC_CAM_N_COLS = 1280
PROC_CAM_N_ROWS = 1024
PROC_CAM_FOCAL_LENGTH_PX = 1.1386e3
PROC_CAM_CENTER_X = 642.0725
PROC_CAM_CENTER_Y = 515.6352
# Distortion order is [k1, k2, p1, p2, k3].
# From calibration: radial=[-0.2042, 0.3663], tangential=[0, 0].
#PROC_DISTORTION_COEFFS = [-0.2042, 0.3663, 0.0, 0.0, 0.0]
PROC_DISTORTION_COEFFS = [-0.2429, 0.2943, 0.0, 0.0, 0.0]  # Use this for no distortion in processing.

# Star ID tuning for processing captured images.
PROC_MAX_MAGNITUDE = 6.5
PROC_STARID_TOLERANCE = 25.0
PROC_RANSAC_TOLERANCE = 20.0
PROC_MAX_COMBOS = 0

# Console verbosity and warning filtering.
PROC_VERBOSE = False
PROC_SUPPRESS_GIANT_RUNTIME_WARNINGS = True
PROC_SUPPRESS_ALL_RUNTIME_WARNINGS = False

# Point-of-interest extraction tuning for noisy monitor-captured images.
# threshold is in units of image noise sigma; lower finds more stars.
PROC_POI_THRESHOLD = 5.0
PROC_POI_MIN_SIZE = 1
PROC_POI_MAX_SIZE = 50
PROC_POI_CENTROID_SIZE = 2
PROC_POI_REJECT_SATURATION = False

# A-priori attitude for the solver.
PROC_INITIAL_QUATERNION = [-0.704416026, 0.061628417, 0.0, 0.707106781]
PROC_USE_SCENARIO_CSV_INITIAL_QUAT = False
PROC_SCENARIO_CSV = TRUTH_FILE
# Optional explicit scenario filename key from scenarios.csv (for single-image runs).
PROC_SCENARIO_FILENAME = None

# Ground-truth attitude for error reporting.
# If PROC_GT_QUATERNION is None, processing will try to read Q_X/Q_Y/Q_Z/Q_W
# from PROC_GT_IMAGE_PATH FITS header (default: displayed image path).
PROC_GT_QUATERNION = None
PROC_GT_IMAGE_PATH = DISPLAY_IMAGE_PATH
PROC_REPORT_ATTITUDE_ERROR = False

# Optional boresight correction (for HITL): q^B_I = (q^C_B)^-1 q^C_I
PROC_USE_BORESIGHT = False
PROC_BORESIGHT_Q_CB = [0.0, 0.0, 0.0, 1.0]

# Runtime behavior controls.
PROC_INTERACTIVE_PROMPTS = False
PROC_ENABLE_PLOTS = False
PROC_MIN_MATCHES = 3

# Euler-angle constants for the J2000 -> true HCI transform in process_star_image.
# C_J2000^HCI = R_x(i) @ R_z(Omega) @ R_x(epsilon)
PROC_J2000_TO_HCI_OBLIQUITY_DEG = 23.439291111
PROC_J2000_TO_HCI_ASCENDING_NODE_LONGITUDE_DEG = 75.76
PROC_J2000_TO_HCI_SOLAR_INCLINATION_DEG = 7.25

# Show a blocking popup of matched stars after each OpNav id_stars() call.
# Can be overridden at runtime from StateMachine.py via OPNAV_SHOW_MATCHED_STARS.
PROC_SHOW_MATCHED_STARS_POPUP = False


# =============================================================================
# 4) BATCH EVALUATION SETTINGS (MULTI-IMAGE STATISTICS)
# =============================================================================
BATCH_SCENARIO_CSV = TRUTH_FILE
BATCH_CAPTURE_DIR = CAPTURED_DIR
# If your captured file names differ, add a captured_filename column to scenarios.csv.
BATCH_CAPTURE_FILENAME_TEMPLATE = "{stem}.fits"
BATCH_OUTPUT_CSV = os.path.join(DATA_DIR, 'batch_attitude_results.csv')
BATCH_REQUIRE_MIN_MATCHES = 3


# =============================================================================
# Legacy aliases kept for compatibility with existing scripts
# =============================================================================
CAM_FOCAL_LENGTH = SYNTH_CAM_FOCAL_LENGTH_PX
CAM_CENTER_X = SYNTH_CAM_CENTER_X
CAM_CENTER_Y = SYNTH_CAM_CENTER_Y
IMG_RES = SYNTH_IMAGE_WIDTH
DISTORTION_COEFFS = SYNTH_DISTORTION_COEFFS
MAX_MAGNITUDE = SYNTH_MAX_MAGNITUDE